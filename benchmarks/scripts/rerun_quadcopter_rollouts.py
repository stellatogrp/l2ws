"""Re-run quadcopter closed-loop rollouts using a previously trained model.

This script bypasses the full Hydra-launcher path. It:
  1) Reconstructs a `Workspace` exactly as `quadcopter.run` does.
  2) Loads the saved NN weights from a previous training run's
     `nn_weights/layer_*_params.npz`.
  3) Calls `Workspace.run_closed_loop_rollouts` for `nearest_neighbor` and
     `learned`, which (with the launcher patch) dumps state arrays as
     `rollouts/{col}/rollout_{i}_states.npz`.

We need 4 rollouts so rollout-index 3 exists.

Run from anywhere; defaults assume the repo layout.

    python benchmarks/scripts/rerun_quadcopter_rollouts.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime as _dt
from functools import partial
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"


def _patch_hydra_cwd(orig_cwd: Path) -> None:
    """Monkey-patch hydra.utils.get_original_cwd before any launcher import.

    The launcher uses `hydra.utils.get_original_cwd()` inside `load_setup_data`
    and `load_weights` to resolve paths. Outside an actual Hydra context this
    raises. We patch it to return our chosen path so the rest of the launcher
    works unchanged.
    """
    import hydra
    import hydra.utils

    fixed = str(orig_cwd)

    def _get(_=None):
        return fixed

    hydra.utils.get_original_cwd = _get
    # Some code paths import the function directly:
    hydra.get_original_cwd = _get  # type: ignore[attr-defined]


def _load_run_cfg(run_yaml: Path, *,
                  data_datetime: str,
                  num_rollouts: int) -> "DictConfig":
    """Load the run-config YAML, override the fields we care about.

    Returns an OmegaConf DictConfig that the launcher can use as `cfg`.
    """
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(run_yaml)
    # overrides for a rollout-only execution
    cfg.skip_startup = True
    cfg.save_weights_flag = False
    cfg.solve_c_num = 0
    cfg.num_rollouts = num_rollouts
    cfg.data.datetime = data_datetime
    # we will skip training entirely
    cfg.nn_cfg.epochs = 0
    return cfg


def _load_setup_cfg(data_setup_yaml: Path) -> dict:
    import yaml

    with open(data_setup_yaml, "r") as f:
        return yaml.safe_load(f)


def _build_workspace(setup_cfg: dict, run_cfg) -> "Workspace":
    """Mirror the workspace construction inside `l2ws.examples.quadcopter.run`.

    Without entering hydra; assumes _patch_hydra_cwd has already run.
    """
    import jax.numpy as jnp
    from jax import vmap

    from l2ws.examples.quadcopter import (
        QUADCOPTER_NX,
        QUADCOPTER_NU,
        plot_traj_3d,
        quadcopter_dynamics,
        shifted_sol,
    )
    from l2ws.launcher import Workspace
    from l2ws.utils.mpc_utils import static_canon_mpc_osqp

    T, dt = setup_cfg["T"], setup_cfg["dt"]
    nx, nu = QUADCOPTER_NX, QUADCOPTER_NU

    x_min = jnp.array(setup_cfg["x_min"])
    x_max = jnp.array(setup_cfg["x_max"])
    u_min = jnp.array(setup_cfg["u_min"])
    u_max = jnp.array(setup_cfg["u_max"])

    rollout_length = setup_cfg["rollout_length"]
    obstacle_tol = setup_cfg["obstacle_tol"]

    Q_diag_ref = jnp.zeros(nx)
    Q_diag_ref = Q_diag_ref.at[:3].set(1)
    Q_ref = jnp.diag(Q_diag_ref)

    n_dim = T * (nx + nu)
    m_dim = T * (2 * nx + nu)
    delta_u_list = setup_cfg.get("delta_u", None)
    if delta_u_list is not None:
        delta_u = jnp.array(setup_cfg["delta_u"])
        m_dim = m_dim + T * nu
    else:
        delta_u = None

    static_dict = dict(m=m_dim, n=n_dim)

    partial_shifted_sol_fn = partial(shifted_sol, T=T, nx=nx, nu=nu, m=m_dim, n=n_dim)
    batch_shifted_sol_fn = vmap(partial_shifted_sol_fn, in_axes=0, out_axes=0)

    system_constants = dict(
        T=T, nx=nx, nu=nu, dt=dt,
        u_min=u_min, u_max=u_max, x_min=x_min, x_max=x_max,
        cd0=jnp.zeros(nx), delta_u=delta_u,
    )

    Q = jnp.diag(jnp.array(setup_cfg["Q_diag"]))
    QT = setup_cfg["QT_factor"] * Q
    R = jnp.diag(jnp.array(setup_cfg["R_diag"]))
    static_canon_mpc_osqp_partial = partial(
        static_canon_mpc_osqp, T=T, nx=nx, nu=nu,
        x_min=x_min, x_max=x_max, u_min=u_min, u_max=u_max,
        Q=Q, QT=QT, R=R, delta_u=delta_u,
    )
    plot_traj_3d_partial = partial(
        plot_traj_3d, T=T, goal_bound=setup_cfg["goal_bound"],
    )

    closed_loop_rollout_dict = dict(
        rollout_length=rollout_length,
        num_rollouts=run_cfg.num_rollouts,
        closed_loop_budget=run_cfg.closed_loop_budget,
        dynamics=quadcopter_dynamics,
        u_init_traj=jnp.zeros(nu),
        system_constants=system_constants,
        Q_ref=Q_ref,
        obstacle_tol=obstacle_tol,
        static_canon_mpc_osqp_partial=static_canon_mpc_osqp_partial,
        plot_traj=plot_traj_3d_partial,
    )

    workspace = Workspace(
        algo="osqp",
        cfg=run_cfg,
        static_flag=False,
        static_dict=static_dict,
        example="quadcopter",
        closed_loop_rollout_dict=closed_loop_rollout_dict,
        shifted_sol_fn=batch_shifted_sol_fn,
        traj_length=rollout_length,
    )
    return workspace


def _load_weights_into(workspace, weights_dir: Path) -> None:
    """Manually load saved NN weights into the workspace's l2ws_model."""
    import jax.numpy as jnp

    npz_files = sorted(weights_dir.glob("layer_*_params.npz"))
    if not npz_files:
        raise FileNotFoundError(f"no layer_*_params.npz found in {weights_dir}")
    params = []
    for p in npz_files:
        loaded = jnp.load(p)
        params.append((loaded["weight"], loaded["bias"]))
    workspace.l2ws_model.params = params
    print(f"[rerun] loaded {len(params)} NN layer weight/bias pairs from {weights_dir}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--setup-datetime", default="2026-04-26/18-26-20",
        help="data_setup_outputs/<DATE>/<TIME> to load training problems from",
    )
    parser.add_argument(
        "--train-datetime", default="2026-04-26/19-17-31",
        help="train_outputs/<DATE>/<TIME> to load NN weights from",
    )
    parser.add_argument(
        "--num-rollouts", type=int, default=4,
        help="how many closed-loop rollouts to run (rollout #3 is the target)",
    )
    parser.add_argument(
        "--out-stamp", default=None,
        help="folder name under benchmarks/outputs/quadcopter/restyle_rollouts/",
    )
    args = parser.parse_args()

    setup_dir = BENCHMARKS_DIR / "outputs" / "quadcopter" / "data_setup_outputs" / args.setup_datetime
    train_dir = BENCHMARKS_DIR / "outputs" / "quadcopter" / "train_outputs" / args.train_datetime
    weights_dir = train_dir / "nn_weights"
    data_setup_yaml = train_dir / "data_setup_copied.yaml"

    if not data_setup_yaml.exists():
        print(f"[rerun] missing {data_setup_yaml}", file=sys.stderr)
        return 1
    if not weights_dir.exists():
        print(f"[rerun] missing {weights_dir}", file=sys.stderr)
        return 1

    stamp = args.out_stamp or _dt.now().strftime("%Y-%m-%d/%H-%M-%S")
    out_dir = BENCHMARKS_DIR / "outputs" / "quadcopter" / "restyle_rollouts" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[rerun] working dir: {out_dir}")

    # the launcher reads `data_setup_copied.yaml` from cwd
    shutil.copy(data_setup_yaml, out_dir / "data_setup_copied.yaml")

    # patch hydra cwd before importing launcher
    _patch_hydra_cwd(BENCHMARKS_DIR)

    # cd into our output dir so all relative writes (rollouts/, plots/) land here
    os.chdir(out_dir)

    setup_cfg = _load_setup_cfg(out_dir / "data_setup_copied.yaml")
    run_cfg = _load_run_cfg(
        BENCHMARKS_DIR / "configs" / "quadcopter" / "quadcopter_run.yaml",
        data_datetime=args.setup_datetime,
        num_rollouts=args.num_rollouts,
    )

    print(f"[rerun] building workspace (factor 11k KKT systems — a few minutes)...")
    workspace = _build_workspace(setup_cfg, run_cfg)
    _load_weights_into(workspace, weights_dir)

    print("[rerun] rollouts: nearest_neighbor")
    workspace.run_closed_loop_rollouts("nearest_neighbor")

    print("[rerun] rollouts: prev_sol")
    workspace.run_closed_loop_rollouts("prev_sol")

    print("[rerun] rollouts: learned")
    workspace.run_closed_loop_rollouts("learned")

    print(f"[rerun] done; outputs at {out_dir}/rollouts/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
