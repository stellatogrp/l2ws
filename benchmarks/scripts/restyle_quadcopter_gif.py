"""Render the restyled side-by-side comparison GIF for one rollout.

Reads `rollout_<i>_states.npz` files written by the patched launcher (see
`l2ws/launcher.py:run_closed_loop_rollouts`) for two methods (typically
`nearest_neighbor` and `learned`) and emits a single side-by-side GIF + MP4
+ PNG using the renderer in `l2ws.examples.quadcopter_render`.

    python benchmarks/scripts/restyle_quadcopter_gif.py
    python benchmarks/scripts/restyle_quadcopter_gif.py \
        --rollouts-dir benchmarks/outputs/quadcopter/restyle_rollouts/2026-04-27/.../rollouts \
        --rollout-index 3
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
                        default="vertical")
    parser.add_argument(
        "--output", default=None,
        help="output GIF path (default: paper_outputs/quadcopter/restyled/rollout_<i>_compare.gif)",
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--scale", type=float, default=0.12)
    parser.add_argument("--blade-rpm", type=float, default=2400.0)
    parser.add_argument("--keep-frames", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT))
    from l2ws.examples.quadcopter_render import (
        TRAIL_COLOR_LEARNED,
        TRAIL_COLOR_NN,
        TRAIL_COLOR_PREV,
        make_compare_gif,
    )

    rollouts_dir = Path(args.rollouts_dir) if args.rollouts_dir else _latest_rollouts_dir()
    if not rollouts_dir.exists():
        print(f"[restyle] no such rollouts dir: {rollouts_dir}", file=sys.stderr)
        return 1
    print(f"[restyle] using rollouts dir: {rollouts_dir}")

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
            print("[restyle] WARNING: reference trajectories differ between panes; using first.")
            break

    if args.labels is None:
        labels = [METHOD_LABELS.get(m, m) for m in args.methods]
    else:
        if len(args.labels) != len(args.methods):
            print(
                f"[restyle] --labels count ({len(args.labels)}) "
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

    out_path = (
        Path(args.output)
        if args.output
        else REPO_ROOT
        / "paper_outputs"
        / "quadcopter"
        / "restyled"
        / f"rollout_{args.rollout_index}_compare.gif"
    )

    print(
        f"[restyle] rendering {out_path}\n"
        f"          frames: {len(states_list[0])} (stride={args.frame_stride})\n"
        f"          panes : {' | '.join(labels)}\n"
        f"          layout: {args.layout}"
    )

    make_compare_gif(
        states_list=states_list,
        reference=reference,
        labels=labels,
        output_path=str(out_path),
        layout=args.layout,
        trail_colors=trail_colors,
        fps=args.fps,
        scale=args.scale,
        blade_rpm=args.blade_rpm,
        frame_stride=args.frame_stride,
        keep_frames=args.keep_frames,
    )
    print(f"[restyle] done: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
