# L2WS — Claude session guide

This is the codebase for the paper *Learning to Warm-Start Fixed-Point Optimization Algorithms* (Sambharya, Hall, Amos, Stellato). Main package is `l2ws/`; numerical experiments are driven from `benchmarks/`. There are roughly ten examples — quadcopter MPC is one.

## Environment (Mac M4, CPU-only)

A `uv`-managed virtual environment lives at `.venv` (Python 3.10.19). Activate with `source .venv/bin/activate` from the repo root. Everything below assumes the venv is active.

The dependency stack is **deliberately pinned to a 2023-era jax** so the numerics match the paper. Don't bump these without a strong reason:

- `jax==0.4.20`, `jaxlib==0.4.20`, `optax==0.1.5`, `jaxopt==0.6`
- `numpy<2` (jax 0.4.20 predates numpy 2.0)
- `scipy<1.13` (jax 0.4.20 references `scipy.linalg.tril`, removed in scipy 1.13)
- `cvxpy>=1.3.0`, `matplotlib`, `hydra-core`, `imageio`, `pyyaml`, `pandas`
- `osqp`, `scs`, `emnist` — imported (emnist via the launcher's unconditional `import l2ws.examples.mnist`) but **not declared** in `pyproject.toml`; install manually
- `trajax @ git+https://github.com/google/trajax` — used by `l2ws/utils/mpc_utils.py`
- `l2ws` itself installed via `uv pip install -e . --no-deps`

**Why jaxopt is pinned to 0.6 (not the latest):** newer jaxopt jit-wraps `optimizer.update`, which causes the dynamic `iters=train_unrolls` kwarg to be traced as a JAX value. That breaks `jnp.zeros(k)` inside `algo_steps.k_steps_train_osqp` (`Shapes must be 1D sequences of concrete values of integer type`). jaxopt 0.6 doesn't do that wrap and accepts dynamic kwargs cleanly.

To recreate the venv from scratch — `pyproject.toml` now declares all the pins, so a one-shot install is enough:

```bash
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -e .
```

Verify: `python -c "import jax, optax, trajax, l2ws.examples.quadcopter; print(jax.devices())"` should print `[CpuDevice(id=0)]`.

`emnist` is only directly used by `l2ws/examples/mnist.py`, but `benchmarks/l2ws_setup.py` and `l2ws_train.py` `import` every example unconditionally, so it is included in `dependencies` even for non-MNIST workflows.

**Source patches applied** (already in the working tree, kept across the session):
- `l2ws/examples/quadcopter.py:1348` — `elif title[:5] == 'train':` → `elif title is not None and title[:5] == 'train':`. The original code crashes on the setup-phase GIFs because `title=None` is the default and is not subscriptable. Harmless for all other call sites.
- `benchmarks/plot.py:773–787` — resolved a pre-existing unresolved merge-conflict block (the rest of the file was already merged; just one block had `<<<<<<< HEAD:plot_script.py` / `>>>>>>> 6f924435...:benchmarks/plot.py` markers left in the committed file). Kept the marker-styled `axes[0].plot(...)` variant, consistent with the rest of the function.

## Quadcopter pipeline (3 commands, run from `benchmarks/`)

```bash
cd benchmarks
python l2ws_setup.py quadcopter local      # ~45 min at full scale on M4 CPU (5 rollouts ≈ 2 min in smoke test)
python l2ws_train.py quadcopter local      # ~hours–days at full scale; smoke test (200 epochs, 400 train probs) finishes in ~3 min
python plot.py       quadcopter local      # paper-style aggregate plots only (optional)
```

The `local` argument tells the Hydra launcher to write under `outputs/` (vs. a Princeton cluster path).

### What each step produces

1. **Setup** — generates 110 random smooth 3D reference trajectories, runs closed-loop MPC of 100 steps × 500 OSQP iterations on each, and writes:
   - `outputs/quadcopter/data_setup_outputs/<date>/<time>/rollouts/rollout_{i}_flight.gif` — **the optimal-trajectory GIFs** (110 of them when `save_gif: true`).
   - `data_setup.npz` and `data_setup_q.npz` — 11,000 compiled MPC problems (this is the training set).
   - `z_stars.pdf`, `q.pdf`, `thetas.pdf` — sanity-check visualizations.

2. **Train** — auto-finds the most recent `data_setup`, loads it, factors the 11k KKT systems, runs cold-start / nearest-neighbor / prev-sol baselines, then trains a NN [θ → 100 → 500 → z₀] minimizing the supervised loss at `train_unrolls=5` algorithm steps. Stops when `plateau_decay` drives the LR to `min_lr`. Then runs final closed-loop rollouts that produce **the comparison GIFs**:
   - `outputs/quadcopter/train_outputs/<date>/<time>/rollouts/{init}/rollout_{i}_flight.gif` where `init ∈ {learned, nearest_neighbor, prev_sol, no_train}`.
   - `plots/eval_iters_test.pdf`, `losses_over_training.pdf`, `train_test_results.csv`, etc.

3. **Plot** — aggregates several training runs (datetimes listed in `quadcopter_plot.yaml`'s `output_datetimes`) into the figures used in the paper. Not needed for GIFs.

### Configs to scale down for testing

`benchmarks/configs/quadcopter/quadcopter_setup.yaml`:
- `num_rollouts: 110` (full) → e.g. `5` for smoke tests. This is also the number of GIFs.
- `rollout_length: 100`, `rollout_osqp_iters: 500` — leave alone (controls GIF length and quality).
- `save_gif: true` — must stay true to produce GIFs.

`benchmarks/configs/quadcopter/quadcopter_run.yaml`:
- `N_train: 10000`, `N_test: 1000` — drop to e.g. 400 / 100 for a smoke test.
- `nn_cfg.epochs: 1e6` — drop to e.g. 200 for a smoke test (the run normally stops via plateau decay, not this cap).
- `eval_every_x_epochs: 200` — drop to e.g. 50 for smoke tests so you see eval output sooner; **at full scale this controls the dominant cost** (each eval evaluates 11k problems × `eval_unrolls=500` steps).
- `num_rollouts: 5` — closed-loop rollouts at end of training; equals number of comparison GIFs per init type. **Constraint:** must be ≤ `setup_cfg.num_rollouts − ceil(N_train / rollout_length)`. At full scale: 110 − 100 = 10 trajectories held out, so the configured 5 is fine. For a 5-rollout smoke test with N_train=400 (4 train trajectories), only 1 trajectory is held out → set this to 1.
- `closed_loop_budget: 15` — solver iterations per closed-loop step in the rollout phase.

### Output layout (gitignored)

```
outputs/quadcopter/
├── data_setup_outputs/<YYYY-MM-DD>/<HH-MM-SS>/
│   ├── rollouts/rollout_{i}_flight.gif       ← setup-phase GIFs
│   ├── data_setup.npz, data_setup_q.npz
│   └── *.pdf
├── train_outputs/<YYYY-MM-DD>/<HH-MM-SS>/
│   ├── rollouts/{learned,nearest_neighbor,prev_sol,...}/rollout_{i}_flight.gif   ← training-phase GIFs
│   ├── plots/{eval_iters_test.pdf, losses_over_training.pdf, ...}
│   └── train_test_results.csv, accuracies/, solve_c/
└── plots/<YYYY-MM-DD>/<HH-MM-SS>/            (only from plot.py)
```

Each invocation produces a new dated folder. The run-phase auto-picks the **most recent** setup folder unless `data.datetime` is set in `quadcopter_run.yaml`.

## Key files

- `l2ws/examples/quadcopter.py` (1483 lines):
  - `setup_probs(setup_cfg)` — data-generation entrypoint; calls `plot_traj_3d()` at line 301 to write each setup-phase GIF.
  - `run(run_cfg)` — training entrypoint; constructs `Workspace` and calls `.run()`.
  - `plot_traj_3d()` — line 1159; the actual GIF writer is at line 1363 (`imageio.get_writer(...).append_data(...)`).
  - `quadcopter_dynamics()`, `shifted_sol()`, helper plotters.
- `l2ws/examples/solve_script.py` — `osqp_setup_script`, `save_results_dynamic` (compiles 11k problems to npz).
- `l2ws/examples/osc_mass.py` — `static_canon_osqp`, `static_canon_mpc_osqp` (canonical QP form for MPC).
- `l2ws/utils/mpc_utils.py` — `closed_loop_rollout()` is the per-rollout simulation driver.
- `l2ws/launcher.py` (~2000 lines) — `Workspace` class: training loop, baselines, eval, final closed-loop rollout GIFs.
- `l2ws/algo_steps.py` — JAX-jit'd OSQP / SCS / ISTA / GD step functions.
- `l2ws/l2ws_model.py`, `l2ws/osqp_model.py`, `l2ws/scs_model.py` — neural-network architecture and per-algorithm wrappers.

## Runtime hot spots / gotchas (M4 CPU)

- **Setup**: ~5.5M total OSQP iterations (110 rollouts × 100 steps × 500 iters) plus 110 × 100 = 11,000 matplotlib frames. Plan for 3–6 hours.
- **Training**: dominated by the `eval_every_x_epochs=200` checkpoints, each of which runs `eval_unrolls=500` algorithm unrolls on all 11k problems. On CPU this is ~10 min per eval. Total to plateau is typically 1–4 days.
- **RAM**: the LU-factor cache for the 11k KKT systems is ~9–10 GB. Keep ≥16 GB free.
- **Long runs**: use `tmux` / `nohup` so a closed laptop lid doesn't kill the run.
- **Plateau-detection**: there is no hard epoch cap that fires in practice. The run ends when `plateau_decay` drives LR below `min_lr=1e-7`.

## Reproducing the GIFs from scratch

The plan written by Claude lives at `~/.claude/plans/i-would-like-you-smooth-zebra.md`. Recommended sequence:

1. Activate venv: `source .venv/bin/activate`.
2. Smoke test: edit `quadcopter_setup.yaml` (`num_rollouts: 5`) and `quadcopter_run.yaml` (`N_train: 500`, `N_test: 100`, `nn_cfg.epochs: 200`, `eval_every_x_epochs: 50`, `num_rollouts: 2`). Run setup, then train. ~30–60 min total. Verify a GIF exists and looks right.
3. Restore configs (revert from git) and run the full pipeline.

No source-code edits are needed at any point.
