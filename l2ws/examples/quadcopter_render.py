"""Paper/talk-quality 3D quadcopter renderer.

Standalone matplotlib animation utilities for rendering closed-loop rollout
data captured by `Workspace.run_closed_loop_rollouts` (state arrays saved to
`rollouts/{col}/rollout_{i}_states.npz`). Produces a side-by-side comparison
GIF — typically nearest-neighbor warm-start vs. learned warm-start — with a
realistic quadcopter mesh, fading position trail, ground shadow, and a
gently orbiting camera.
"""

from __future__ import annotations

import os
import shutil
import tempfile

import imageio.v2 as imageio
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Helvetica Neue is the primary font (macOS-installed by default) with sane
# fallbacks. We set this on the global rcParams once at import; the renderer
# also re-applies inside `make_compare_gif` so this still works if a caller
# has tweaked rcParams elsewhere.
_FONT_STACK = [
    "Helvetica Neue",
    "Helvetica",
    "Arial",
    "DejaVu Sans",
]
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = _FONT_STACK + list(
    mpl.rcParams.get("font.sans-serif", [])
)

# ----------------------------------------------------------------------------
# Visual constants — tweak here, not in the rendering code below.
# ----------------------------------------------------------------------------
BODY_COLOR = "#1a1a1f"
ARM_COLOR = "#2c2c34"
ROTOR_HUB_COLOR = "#3a3a44"
ROTOR_RIM_COLOR = "#5a5a66"
BLADE_COLOR = "#cfcfd6"
BLADE_BLUR_COLOR = "#bdbdc6"
SHADOW_COLOR = "#000000"
FLOOR_COLOR = "#f4f4f6"
FLOOR_GRID_COLOR = "#cdd0d6"
REF_COLOR = "#0d0d10"
TRAIL_COLOR_NN = "#d2691e"        # warm orange
TRAIL_COLOR_LEARNED = "#1f77b4"   # paper blue
TRAIL_COLOR_PREV = "#5b3a8c"      # muted purple
ERR_BAR_BG = "#e8e8ec"
TEXT_COLOR = "#1a1a1f"


def quat_to_R(q: np.ndarray) -> np.ndarray:
    """[qw, qx, qy, qz] -> 3x3 rotation matrix (standard convention).

    Tolerates non-unit quaternions by normalizing.
    """
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = q / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _box_faces(center: np.ndarray, R: np.ndarray, size: tuple[float, float, float]):
    """6 face polygons of an oriented rectangular prism."""
    sx, sy, sz = size
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    corners_local = np.array(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ]
    )
    corners_world = (R @ corners_local.T).T + center
    faces_idx = [
        [0, 1, 2, 3],  # bottom
        [4, 5, 6, 7],  # top
        [0, 1, 5, 4],  # -y
        [2, 3, 7, 6],  # +y
        [1, 2, 6, 5],  # +x
        [0, 3, 7, 4],  # -x
    ]
    return [corners_world[idx] for idx in faces_idx]


def _disk(center_local: np.ndarray, R: np.ndarray, world_origin: np.ndarray,
          radius: float, n: int = 24) -> np.ndarray:
    """Disk in body-XY plane around `center_local`, transformed to world."""
    theta = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    pts_local = np.stack(
        [center_local[0] + radius * np.cos(theta),
         center_local[1] + radius * np.sin(theta),
         np.full_like(theta, center_local[2])],
        axis=1,
    )
    return (R @ pts_local.T).T + world_origin


def draw_quadcopter(ax, position: np.ndarray, R: np.ndarray, scale: float,
                    blade_phase: float):
    """Draw a stylized quadcopter at `position` with rotation `R`.

    Returns the list of artists added so caller can clear them next frame.
    """
    artists = []

    # --- body (compact rectangular prism, smaller than rotors) -------------
    body_size = (0.85 * scale, 0.85 * scale, 0.18 * scale)
    body_faces = _box_faces(position, R, body_size)
    body = Poly3DCollection(
        body_faces,
        facecolor=BODY_COLOR,
        edgecolor=ARM_COLOR,
        linewidths=0.4,
        alpha=1.0,
    )
    body.set_zorder(10)
    ax.add_collection3d(body)
    artists.append(body)

    # --- four arms + rotor hubs + rotor rims + spinning blades --------------
    arm_offsets_local = np.array(
        [
            [scale, scale, 0.16 * scale],
            [scale, -scale, 0.16 * scale],
            [-scale, -scale, 0.16 * scale],
            [-scale, scale, 0.16 * scale],
        ]
    )
    rotor_radius = 0.62 * scale

    for k, off_local in enumerate(arm_offsets_local):
        # arm: thin oriented box from body center to rotor hub
        arm_center_local = off_local / 2.0
        arm_dir_local = off_local / np.linalg.norm(off_local[:2])
        arm_len = float(np.linalg.norm(off_local[:2]))
        # build a rotation that aligns the arm's local x-axis with arm_dir_local
        ax_local = arm_dir_local.copy()
        ax_local[2] = 0
        ax_local /= np.linalg.norm(ax_local)
        ay_local = np.array([-ax_local[1], ax_local[0], 0.0])
        az_local = np.array([0.0, 0.0, 1.0])
        R_arm_local = np.stack([ax_local, ay_local, az_local], axis=1)
        arm_size = (arm_len, 0.13 * scale, 0.09 * scale)
        # arm corners in arm-local frame
        corners_arm = np.array(
            [
                [-arm_size[0] / 2, -arm_size[1] / 2, -arm_size[2] / 2],
                [arm_size[0] / 2, -arm_size[1] / 2, -arm_size[2] / 2],
                [arm_size[0] / 2, arm_size[1] / 2, -arm_size[2] / 2],
                [-arm_size[0] / 2, arm_size[1] / 2, -arm_size[2] / 2],
                [-arm_size[0] / 2, -arm_size[1] / 2, arm_size[2] / 2],
                [arm_size[0] / 2, -arm_size[1] / 2, arm_size[2] / 2],
                [arm_size[0] / 2, arm_size[1] / 2, arm_size[2] / 2],
                [-arm_size[0] / 2, arm_size[1] / 2, arm_size[2] / 2],
            ]
        )
        # transform arm corners: arm-local -> body-local -> world
        corners_body = (R_arm_local @ corners_arm.T).T + arm_center_local
        corners_world = (R @ corners_body.T).T + position
        faces_idx = [
            [0, 1, 2, 3], [4, 5, 6, 7],
            [0, 1, 5, 4], [2, 3, 7, 6],
            [1, 2, 6, 5], [0, 3, 7, 4],
        ]
        arm_faces = [corners_world[idx] for idx in faces_idx]
        arm_poly = Poly3DCollection(
            arm_faces,
            facecolor=ARM_COLOR,
            edgecolor=BODY_COLOR,
            linewidths=0.3,
        )
        arm_poly.set_zorder(9)
        ax.add_collection3d(arm_poly)
        artists.append(arm_poly)

        # rotor hub (small filled disk in body-XY plane at off_local)
        hub_pts = _disk(off_local, R, position, 0.16 * scale, n=20)
        hub = Poly3DCollection(
            [hub_pts],
            facecolor=ROTOR_HUB_COLOR,
            edgecolor=BODY_COLOR,
            linewidths=0.4,
        )
        hub.set_zorder(11)
        ax.add_collection3d(hub)
        artists.append(hub)

        # rotor rim (outer circle, thin)
        rim_pts = _disk(off_local, R, position, rotor_radius, n=40)
        rim_pts_closed = np.vstack([rim_pts, rim_pts[:1]])
        (rim_line,) = ax.plot(
            rim_pts_closed[:, 0],
            rim_pts_closed[:, 1],
            rim_pts_closed[:, 2],
            color=ROTOR_RIM_COLOR,
            linewidth=0.9,
            zorder=12,
        )
        artists.append(rim_line)

        # spinning blade pair: 2 thin lines at blade_phase + (k flips direction)
        spin_dir = 1.0 if k % 2 == 0 else -1.0
        phase = spin_dir * blade_phase + (k * np.pi / 4.0)
        for blade_offset in (0.0, np.pi / 2.0):
            ang = phase + blade_offset
            tip_a_local = off_local + np.array(
                [rotor_radius * np.cos(ang), rotor_radius * np.sin(ang), 0.0]
            )
            tip_b_local = off_local + np.array(
                [-rotor_radius * np.cos(ang), -rotor_radius * np.sin(ang), 0.0]
            )
            tips_local = np.stack([tip_a_local, tip_b_local], axis=0)
            tips_world = (R @ tips_local.T).T + position
            (blade_line,) = ax.plot(
                tips_world[:, 0],
                tips_world[:, 1],
                tips_world[:, 2],
                color=BLADE_COLOR,
                linewidth=1.4,
                alpha=0.9,
                zorder=13,
            )
            artists.append(blade_line)

        # blur disk: faint translucent fill suggesting rotor sweep
        blur_pts = _disk(off_local, R, position, rotor_radius * 0.92, n=28)
        blur = Poly3DCollection(
            [blur_pts],
            facecolor=BLADE_BLUR_COLOR,
            edgecolor="none",
            alpha=0.18,
        )
        blur.set_zorder(11)
        ax.add_collection3d(blur)
        artists.append(blur)

    return artists


def draw_shadow(ax, position: np.ndarray, scale: float, z_floor: float,
                alpha: float = 0.22):
    """Filled black ellipse on the floor plane underneath the quadcopter.

    Shadow softens with altitude; below 1.5 m it is fully visible, then fades.
    """
    altitude = max(position[2] - z_floor, 0.0)
    fade = max(0.0, 1.0 - altitude / 2.5)
    if fade <= 0.02:
        return []
    n = 36
    theta = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    radius = 1.5 * scale * (1.0 + 0.4 * altitude)
    pts = np.stack(
        [
            position[0] + radius * np.cos(theta),
            position[1] + radius * np.sin(theta) * 0.6,
            np.full(n, z_floor + 1e-3),
        ],
        axis=1,
    )
    poly = Poly3DCollection(
        [pts],
        facecolor=SHADOW_COLOR,
        edgecolor="none",
        alpha=alpha * fade,
    )
    poly.set_zorder(2)
    ax.add_collection3d(poly)
    return [poly]


def draw_trail(ax, positions: np.ndarray, color: str, n_segments: int = 40,
               max_alpha: float = 0.95, base_lw: float = 2.4):
    """Recent positions as a polyline with alpha gradient (older = fainter)."""
    if len(positions) < 2:
        return []
    seg_count = min(n_segments, len(positions) - 1)
    artists = []
    for j in range(seg_count):
        idx_end = len(positions) - j
        idx_start = idx_end - 2
        seg = positions[idx_start:idx_end]
        a = max_alpha * (1.0 - j / seg_count) ** 1.4
        (line,) = ax.plot(
            seg[:, 0], seg[:, 1], seg[:, 2],
            color=color, linewidth=base_lw, alpha=a, solid_capstyle="round",
            zorder=5,
        )
        artists.append(line)
    return artists


def draw_floor(ax, x_lim, y_lim, z_floor):
    """Subtle floor plane + a sparse grid for visual depth cues."""
    # plane
    floor_pts = np.array(
        [
            [x_lim[0], y_lim[0], z_floor],
            [x_lim[1], y_lim[0], z_floor],
            [x_lim[1], y_lim[1], z_floor],
            [x_lim[0], y_lim[1], z_floor],
        ]
    )
    plane = Poly3DCollection(
        [floor_pts],
        facecolor=FLOOR_COLOR,
        edgecolor="none",
        alpha=0.55,
    )
    plane.set_zorder(0)
    ax.add_collection3d(plane)
    # gridlines
    n_grid = 8
    xs = np.linspace(x_lim[0], x_lim[1], n_grid + 1)
    ys = np.linspace(y_lim[0], y_lim[1], n_grid + 1)
    for x in xs:
        ax.plot(
            [x, x], [y_lim[0], y_lim[1]], [z_floor, z_floor],
            color=FLOOR_GRID_COLOR, linewidth=0.5, alpha=0.65, zorder=1,
        )
    for y in ys:
        ax.plot(
            [x_lim[0], x_lim[1]], [y, y], [z_floor, z_floor],
            color=FLOOR_GRID_COLOR, linewidth=0.5, alpha=0.65, zorder=1,
        )


def _setup_axes(ax, x_lim, y_lim, z_lim, title, label_color=TEXT_COLOR):
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_zlim(z_lim)
    ax.set_box_aspect(
        (x_lim[1] - x_lim[0], y_lim[1] - y_lim[0], z_lim[1] - z_lim[0])
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    # transparent panes
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis.line.set_color((0, 0, 0, 0))
    ax.yaxis.line.set_color((0, 0, 0, 0))
    ax.zaxis.line.set_color((0, 0, 0, 0))
    if title:
        # title sits low in the axes bbox so it visually belongs to its
        # 3D plot rather than floating between adjacent panes
        ax.set_title(title, fontsize=20, color=label_color,
                     fontweight="bold", y=0.83)


def _tracking_error(position: np.ndarray, reference: np.ndarray) -> float:
    """L2 distance from `position` to nearest reference waypoint (in meters)."""
    diffs = reference[:, :3] - position[None, :3]
    return float(np.min(np.linalg.norm(diffs, axis=1)))


def _scene_bounds(states_list, reference, pad=0.2):
    pts = [reference[:, :3]] + [s[:, :3] for s in states_list]
    all_pts = np.concatenate(pts, axis=0)
    lo = all_pts.min(axis=0) - pad
    hi = all_pts.max(axis=0) + pad
    # symmetric about midpoint with equal extents for nicer 3D framing
    span = float(np.max(hi - lo))
    mid = (hi + lo) / 2.0
    half = span / 2.0
    return (
        (mid[0] - half, mid[0] + half),
        (mid[1] - half, mid[1] + half),
        (mid[2] - half, mid[2] + half),
    )


def render_pane(
    ax,
    states: np.ndarray,
    reference: np.ndarray,
    title: str,
    frame_idx: int,
    bounds,
    *,
    scale: float,
    trail_color: str,
    trail_segments: int,
    blade_phase: float,
    elev: float,
    azim: float,
    show_tracking: bool,
    dt: float,
):
    """Render one pane of the comparison at `frame_idx`.

    Per-pane HUD: only the tracking error is drawn (upper-right of the
    subplot). The shared time stamp is drawn once at the figure level by
    `make_compare_gif`.
    """
    x_lim, y_lim, z_lim = bounds
    z_floor = z_lim[0]
    _setup_axes(ax, x_lim, y_lim, z_lim, title)
    ax.view_init(elev=elev, azim=azim)
    draw_floor(ax, x_lim, y_lim, z_floor)

    # reference path
    ref_xyz = reference[:, :3]
    ax.plot(
        ref_xyz[:, 0], ref_xyz[:, 1], ref_xyz[:, 2],
        color=REF_COLOR, linewidth=1.0, linestyle=(0, (4, 3)),
        alpha=0.55, zorder=3,
    )

    pos = states[frame_idx, :3]
    quat = states[frame_idx, 6:10]
    R = quat_to_R(quat)

    draw_trail(ax, states[: frame_idx + 1, :3], trail_color, trail_segments)
    draw_shadow(ax, pos, scale, z_floor)
    draw_quadcopter(ax, pos, R, scale, blade_phase)

    if show_tracking:
        err = _tracking_error(pos, reference)
        ax.text2D(
            1.00, 0.5, f"error = {err * 100:5.1f} cm",
            transform=ax.transAxes,
            fontsize=15, color=TEXT_COLOR, va="center", ha="left",
            bbox=dict(boxstyle="round,pad=0.36", fc=ERR_BAR_BG, ec="none",
                      alpha=0.9),
        )


def _layout_grid(n_panes: int, layout: str) -> tuple[int, int, tuple[float, float]]:
    """Return (nrows, ncols, figsize) for the requested layout.

    Vertical layout reserves a small top band for the shared `time = ...`
    label and uses overlapping vertical spacing so panes feel close together.
    """
    if layout == "vertical":
        # slide-friendly: panes are short so the figure fits a slide column
        # without the fonts becoming too small; right ~20% reserved for the
        # error-box gutter (handled in subplots_adjust below)
        return n_panes, 1, (6.6, 2.7 * n_panes + 0.4)
    if layout == "horizontal":
        return 1, n_panes, (5.6 * n_panes + 0.4, 5.4)
    raise ValueError(f"unknown layout: {layout!r}")


def make_compare_gif(
    states_list: list[np.ndarray],
    reference: np.ndarray,
    labels: list[str],
    output_path: str,
    *,
    layout: str = "vertical",
    trail_colors: list[str] | None = None,
    fps: int = 20,
    blade_rpm: float = 2400.0,
    scale: float = 0.12,
    camera_elev: float = 22.0,
    camera_azim_start: float = -65.0,
    camera_orbit_deg_per_sec: float = 6.0,
    trail_segments: int = 40,
    show_tracking: bool = True,
    dt: float = 0.05,
    frame_stride: int = 1,
    keep_frames: bool = False,
    write_mp4: bool = True,
    final_png: bool = True,
    dpi: int = 130,
):
    """Render an N-pane comparison GIF (vertical or horizontal layout).

    All panes share the same reference trajectory and the same camera
    (gently orbiting) so the visual comparison is synchronized frame-for-frame.
    """
    states_list = [np.asarray(s) for s in states_list]
    reference = np.asarray(reference)
    n_panes = len(states_list)
    assert n_panes >= 1, "need at least one pane"
    assert len(labels) == n_panes, "labels must match states list"
    if trail_colors is None:
        defaults = [TRAIL_COLOR_NN, TRAIL_COLOR_PREV, TRAIL_COLOR_LEARNED]
        trail_colors = defaults[:n_panes] if n_panes <= 3 else (
            defaults + [TRAIL_COLOR_LEARNED] * (n_panes - 3)
        )
    assert len(trail_colors) == n_panes, "trail_colors must match panes"

    T = states_list[0].shape[0]
    for s in states_list[1:]:
        assert s.shape[0] == T, "panes must have matching frame counts"

    frame_indices = list(range(0, T, frame_stride))
    if frame_indices[-1] != T - 1:
        frame_indices.append(T - 1)

    bounds = _scene_bounds(states_list, reference)

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix="quad_render_")

    blade_phase_per_frame = 2 * np.pi * blade_rpm / 60.0 / fps * frame_stride
    azim_per_frame = camera_orbit_deg_per_sec / fps * frame_stride

    nrows, ncols, figsize = _layout_grid(n_panes, layout)

    # Re-apply font stack each render in case rcParams was changed elsewhere.
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = _FONT_STACK + list(
        plt.rcParams.get("font.sans-serif", [])
    )

    frame_paths = []
    for k, i in enumerate(frame_indices):
        fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="white")
        blade_phase = k * blade_phase_per_frame
        azim = camera_azim_start + k * azim_per_frame

        for j in range(n_panes):
            ax = fig.add_subplot(
                nrows, ncols, j + 1, projection="3d", computed_zorder=False,
            )
            render_pane(
                ax, states_list[j], reference, labels[j], i, bounds,
                scale=scale, trail_color=trail_colors[j],
                trail_segments=trail_segments,
                blade_phase=blade_phase, elev=camera_elev, azim=azim,
                show_tracking=show_tracking, dt=dt,
            )

        if layout == "vertical":
            # right gutter (~22%) reserved for the per-pane error box and
            # the shared time stamp; left gutter unused so the panes are
            # left-aligned with the title. hspace<0 overlaps panes so the
            # figure stays compact for slides.
            fig.subplots_adjust(
                left=0.0, right=0.78, bottom=0.0, top=0.94, hspace=-0.07,
            )
        else:
            fig.subplots_adjust(
                left=0.0, right=1.0, bottom=0.0, top=0.92, wspace=0.0,
            )

        # Single shared timestamp — placed in the top band above the panes,
        # aligned with the right gutter.
        if show_tracking:
            t_now = i * dt
            fig.text(
                0.81, 0.9, f"time = {t_now:5.2f} s",
                ha="right", va="top",
                fontsize=15, color=TEXT_COLOR,
                bbox=dict(boxstyle="round,pad=0.36", fc=ERR_BAR_BG, ec="none",
                          alpha=0.85),
            )

        path = os.path.join(tmp_dir, f"frame_{k:04d}.png")
        fig.savefig(path, facecolor=fig.get_facecolor())
        plt.close(fig)
        frame_paths.append(path)

    # ------------------------------------------------------------------
    # Auto-crop the constant-size white margin off every frame.
    # We compute the union of each frame's non-white bbox so the output
    # has tight margins but every frame remains the same pixel size
    # (required for clean GIF / MP4 assembly).
    # ------------------------------------------------------------------
    try:
        from PIL import Image
    except ImportError:
        Image = None

    if Image is not None:
        union = None
        per_frame_bboxes = []
        for p in frame_paths:
            with Image.open(p) as img:
                arr = np.asarray(img.convert("RGB"))
                # "non-white" = any channel below threshold
                mask = (arr < 248).any(axis=2)
                if not mask.any():
                    bbox = (0, 0, arr.shape[1], arr.shape[0])
                else:
                    rows = np.where(mask.any(axis=1))[0]
                    cols = np.where(mask.any(axis=0))[0]
                    bbox = (cols.min(), rows.min(), cols.max() + 1, rows.max() + 1)
                per_frame_bboxes.append(bbox)
                if union is None:
                    union = list(bbox)
                else:
                    union[0] = min(union[0], bbox[0])
                    union[1] = min(union[1], bbox[1])
                    union[2] = max(union[2], bbox[2])
                    union[3] = max(union[3], bbox[3])

        if union is not None:
            pad = 6  # tiny white border so antialiased edges aren't clipped
            with Image.open(frame_paths[0]) as img0:
                W, H = img0.size
            crop_box = (
                max(union[0] - pad, 0),
                max(union[1] - pad, 0),
                min(union[2] + pad, W),
                min(union[3] + pad, H),
            )
            for p in frame_paths:
                with Image.open(p) as img:
                    cropped = img.crop(crop_box)
                cropped.save(p)
            print(f"[render] cropped frames to {crop_box[2]-crop_box[0]}x"
                  f"{crop_box[3]-crop_box[1]} (was {W}x{H})")

    with imageio.get_writer(output_path, mode="I", duration=1.0 / fps,
                            loop=0) as writer:
        for p in frame_paths:
            writer.append_data(imageio.imread(p))

    if write_mp4:
        mp4_path = os.path.splitext(output_path)[0] + ".mp4"
        try:
            with imageio.get_writer(mp4_path, fps=fps, codec="libx264",
                                    quality=8) as writer:
                for p in frame_paths:
                    writer.append_data(imageio.imread(p))
        except Exception as exc:  # noqa: BLE001
            print(f"[render] MP4 export skipped: {exc}")

    if final_png:
        png_path = os.path.splitext(output_path)[0] + "_final.png"
        shutil.copyfile(frame_paths[-1], png_path)

    if not keep_frames:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        print(f"[render] frames kept at {tmp_dir}")

    return output_path


def make_side_by_side_gif(
    states_left: np.ndarray,
    states_right: np.ndarray,
    reference: np.ndarray,
    labels: tuple[str, str],
    output_path: str,
    *,
    trail_colors: tuple[str, str] = (TRAIL_COLOR_NN, TRAIL_COLOR_LEARNED),
    **kwargs,
):
    """Two-pane horizontal comparison (back-compat wrapper for make_compare_gif)."""
    return make_compare_gif(
        states_list=[np.asarray(states_left), np.asarray(states_right)],
        reference=reference,
        labels=list(labels),
        output_path=output_path,
        layout="horizontal",
        trail_colors=list(trail_colors),
        **kwargs,
    )


def make_single_pane_gif(
    states: np.ndarray,
    reference: np.ndarray,
    label: str,
    output_path: str,
    **kwargs,
):
    """Convenience: single-pane GIF (uses the vertical-layout path)."""
    return make_compare_gif(
        states_list=[np.asarray(states)],
        reference=reference,
        labels=[label],
        output_path=output_path,
        layout="vertical",
        **kwargs,
    )
