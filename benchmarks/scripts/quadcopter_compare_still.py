"""Render a static side-by-side comparison image for one rollout.

Reads the `rollout_<i>_states.npz` files written by the patched launcher (see
`l2ws/launcher.py:run_closed_loop_rollouts`) for each method and emits a
single-frame comparison image (PNG + PDF) with the full trajectory drawn in
every pane, sized for a LaTeX wrapfig. Uses
`l2ws.examples.quadcopter_render.make_compare_still`.

    python benchmarks/scripts/quadcopter_compare_still.py
    python benchmarks/scripts/quadcopter_compare_still.py \
        --rollouts-dir benchmarks/outputs/quadcopter/restyle_rollouts/2026-04-27/.../rollouts \
        --rollout-index 3 --frame-frac 0.75
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def _latest_rollouts_dir() -> Path:
    base = REPO_ROOT / "benchmarks" / "outputs" / "quadcopter" / "restyle_rollouts"
    candidates = sorted(base.glob("*/*/rollouts"))
    if not candidates:
        raise FileNotFoundError(
            f"no rollouts/ found under {base}; run rerun_quadcopter_rollouts.py first"
        )
    return candidates[-1]


METHOD_LABELS = {
    "nearest_neighbor": "nearest neighbor",
    "prev_sol": "previous solution",
    "learned": "learned",
    "no_train": "cold start",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rollouts-dir", default=None,
        help="path to a `rollouts/` dir containing method subdirs",
    )
    parser.add_argument("--rollout-index", type=int, default=3)
    parser.add_argument(
        "--methods", nargs="+",
        default=["nearest_neighbor", "prev_sol", "learned"],
        help="ordered list of method subdir names; each becomes one pane",
    )
    parser.add_argument(
        "--labels", nargs="+", default=None,
        help="display labels for each pane (default: derived from method name)",
    )
    parser.add_argument("--layout", choices=["vertical", "horizontal"],
                        default="horizontal")
    parser.add_argument(
        "--frame-frac", type=float, default=0.75,
        help="quadcopter position as a fraction of the rollout (0..1)",
    )
    parser.add_argument(
        "--frame-idx", type=int, default=None,
        help="explicit frame index (overrides --frame-frac)",
    )
    parser.add_argument("--scale", type=float, default=0.12)
    parser.add_argument("--elev", type=float, default=22.0)
    parser.add_argument("--azim", type=float, default=-65.0)
    parser.add_argument(
        "--trail-fade", action="store_true",
        help="use the GIF-style faded trail instead of a solid polyline",
    )
    parser.add_argument(
        "--trail-min-alpha", type=float, default=0.35,
        help="alpha floor for the oldest trail segments (with --trail-fade)",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--output", default=None,
        help="output PNG path (default: alongside the rollouts/ dir as "
             "<rollouts-dir>/../rollout_<i>_compare_still.png)",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT))
    from l2ws.examples.quadcopter_render import (
        TRAIL_COLOR_LEARNED,
        TRAIL_COLOR_NN,
        TRAIL_COLOR_PREV,
        make_compare_still,
    )

    rollouts_dir = Path(args.rollouts_dir) if args.rollouts_dir else _latest_rollouts_dir()
    if not rollouts_dir.exists():
        print(f"[still] no such rollouts dir: {rollouts_dir}", file=sys.stderr)
        return 1
    print(f"[still] using rollouts dir: {rollouts_dir}")

    def _load(method: str) -> tuple[np.ndarray, np.ndarray]:
        npz_path = rollouts_dir / method / f"rollout_{args.rollout_index}_states.npz"
        if not npz_path.exists():
            raise FileNotFoundError(npz_path)
        data = np.load(npz_path)
        return np.asarray(data["states"]), np.asarray(data["reference"])

    states_list = []
    references = []
    for method in args.methods:
        s, r = _load(method)
        states_list.append(s)
        references.append(r)

    # all panes should share the same reference path
    reference = references[0]
    for r in references[1:]:
        if r.shape != reference.shape or not np.allclose(r, reference, atol=1e-6):
            print("[still] WARNING: reference trajectories differ between panes; using first.")
            break

    if args.labels is None:
        labels = [METHOD_LABELS.get(m, m) for m in args.methods]
    else:
        if len(args.labels) != len(args.methods):
            print(
                f"[still] --labels count ({len(args.labels)}) "
                f"!= --methods count ({len(args.methods)})",
                file=sys.stderr,
            )
            return 1
        labels = args.labels

    color_map = {
        "nearest_neighbor": TRAIL_COLOR_NN,
        "prev_sol": TRAIL_COLOR_PREV,
        "learned": TRAIL_COLOR_LEARNED,
    }
    trail_colors = [color_map.get(m, TRAIL_COLOR_LEARNED) for m in args.methods]

    if args.frame_idx is not None:
        frame_idx = args.frame_idx
    else:
        T = min(s.shape[0] for s in states_list)
        frame_idx = int(round(args.frame_frac * (T - 1)))

    out_path = (
        Path(args.output)
        if args.output
        else rollouts_dir.parent / f"rollout_{args.rollout_index}_compare_still.png"
    )

    print(
        f"[still] rendering {out_path}\n"
        f"        frame : {frame_idx}/{len(states_list[0]) - 1}\n"
        f"        panes : {' | '.join(labels)}\n"
        f"        layout: {args.layout}"
    )

    make_compare_still(
        states_list=states_list,
        reference=reference,
        labels=labels,
        output_path=str(out_path),
        layout=args.layout,
        trail_colors=trail_colors,
        frame_idx=frame_idx,
        scale=args.scale,
        camera_elev=args.elev,
        camera_azim=args.azim,
        trail_solid=not args.trail_fade,
        trail_min_alpha=args.trail_min_alpha,
        dpi=args.dpi,
    )
    print(f"[still] done: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
