"""Run a MuJoCo sim2sim evaluation suite: many rollouts over a command grid with
per-rollout domain randomization, written as NPZ for the analysis layer.

Each rollout holds a single (vx, vy, vyaw) command constant (after a short
settle) so steady-state tracking error is clean per command. Seed, ground
friction, base mass, link mass, and initial joint noise vary per rollout
(sampled from the training DR ranges) to probe robustness (spec §4.2).

Usage:
    python -m holosoma.eval.run_suite --model-path <onnx> --out-dir <dir> \
        --n-per-command 100 --duration 12

    # quick smoke:
    python -m holosoma.eval.run_suite --model-path <onnx> --out-dir /tmp/mj_eval \
        --n-per-command 2 --duration 6
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from loguru import logger

from holosoma.eval.mujoco_runner import (
    CommandSegment,
    DRParams,
    MujocoEvalRunner,
    RolloutConfig,
)

# Command grid covering the spec ranges (vx 0-1.0, vy +-0.3, vyaw +-0.5),
# including stationary and mixed conditions.
DEFAULT_COMMANDS: list[tuple[float, float, float, str]] = [
    (0.0, 0.0, 0.0, "stand"),
    (0.25, 0.0, 0.0, "fwd025"),
    (0.5, 0.0, 0.0, "fwd05"),
    (0.8, 0.0, 0.0, "fwd08"),
    (1.0, 0.0, 0.0, "fwd10"),
    (-0.3, 0.0, 0.0, "back03"),
    (0.0, 0.3, 0.0, "left03"),
    (0.0, -0.3, 0.0, "right03"),
    (0.0, 0.0, 0.5, "yawL05"),
    (0.0, 0.0, -0.5, "yawR05"),
    (0.5, 0.2, 0.3, "mixed_a"),
    (0.8, -0.2, -0.3, "mixed_b"),
]

# Training DR ranges (from config_values/loco/statue/randomization.py)
DR_FRICTION = (0.5, 1.25)
DR_ADDED_MASS = (-1.0, 3.0)
DR_LINK_MASS = (0.9, 1.2)
DR_INIT_JOINT_NOISE = 0.05  # rad


def sample_dr(rng: np.random.Generator, enable: bool) -> DRParams:
    if not enable:
        return DRParams(friction=1.0)
    return DRParams(
        friction=float(rng.uniform(*DR_FRICTION)),
        added_base_mass=float(rng.uniform(*DR_ADDED_MASS)),
        link_mass_scale=float(rng.uniform(*DR_LINK_MASS)),
        init_joint_noise=DR_INIT_JOINT_NOISE,
    )


def _run_jobs(model_path, jobs, out_dir, duration, settle, terrain="flat", push_force=0.0) -> int:
    """Build one runner and execute a list of (vx,vy,vyaw,label,i,seed,DRParams) jobs.

    Used by each worker process. Each rollout is independent, so workers never share
    state. ONNX/OMP threads are capped to 1 to avoid oversubscription across workers.
    """
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    runner = MujocoEvalRunner(model_path=model_path, terrain=terrain)
    out = Path(out_dir)
    for vx, vy, vyaw, label, i, seed, dr in jobs:
        rc = RolloutConfig(
            seed=seed, duration_s=duration, settle_s=settle,
            segments=[CommandSegment(vx, vy, vyaw, duration)], dr=dr, label=label,
            push_force=push_force,
        )
        runner.run_rollout(out / f"{label}_{i:04d}.npz", rc)
    return len(jobs)


def main() -> None:
    ap = argparse.ArgumentParser(description="MuJoCo sim2sim eval suite")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-per-command", type=int, default=100)
    ap.add_argument("--duration", type=float, default=12.0)
    ap.add_argument("--settle", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel worker processes (rollouts are independent). Keeps the full "
                         "n-per-command; just splits the work. Try 8 on a multi-core box.")
    ap.add_argument("--no-dr", action="store_true", help="Disable domain randomization (nominal model)")
    ap.add_argument("--commands", default=None,
                    help="Optional 'vx,vy,vyaw;...' list overriding the default grid")
    ap.add_argument("--push", type=float, default=0.0,
                    help="External push force (N) applied to the torso for 0.2s mid-rollout (robustness test)")
    ap.add_argument("--terrain", choices=["flat", "rough"], default="flat",
                    help="Ground terrain: flat (default) or rough heightfield (robustness test)")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.commands:
        commands = []
        for i, tok in enumerate(args.commands.split(";")):
            vx, vy, vyaw = (float(x) for x in tok.split(","))
            commands.append((vx, vy, vyaw, f"cmd{i}"))
    else:
        commands = DEFAULT_COMMANDS

    # Build the full, deterministic job list in the parent (seeds independent of worker
    # count, so results are reproducible regardless of --workers).
    rng = np.random.default_rng(args.seed)
    jobs = []
    for vx, vy, vyaw, label in commands:
        for i in range(args.n_per_command):
            seed = int(rng.integers(0, 2**31 - 1))
            dr = sample_dr(np.random.default_rng(seed), enable=not args.no_dr)
            jobs.append((vx, vy, vyaw, label, i, seed, dr))

    total = len(jobs)
    logger.info(f"Running {total} rollouts ({len(commands)} commands x {args.n_per_command}), "
                f"DR={'off' if args.no_dr else 'on'}, dur={args.duration}s, workers={args.workers}, "
                f"terrain={args.terrain}, push={args.push}N")
    t0 = time.time()

    if args.workers <= 1:
        runner = MujocoEvalRunner(model_path=args.model_path, terrain=args.terrain)
        for n, (vx, vy, vyaw, label, i, seed, dr) in enumerate(jobs, 1):
            rc = RolloutConfig(seed=seed, duration_s=args.duration, settle_s=args.settle,
                               segments=[CommandSegment(vx, vy, vyaw, args.duration)], dr=dr, label=label,
                               push_force=args.push)
            runner.run_rollout(out / f"{label}_{i:04d}.npz", rc)
            if n % 10 == 0 or n == total:
                rate = n / (time.time() - t0)
                logger.info(f"  {n}/{total}  ({rate:.1f}/s, ETA {(total-n)/max(rate,1e-9)/60:.1f} min)")
    else:
        import multiprocessing as mp
        chunks = [jobs[w::args.workers] for w in range(args.workers)]  # round-robin split
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers) as pool:
            pool.starmap(_run_jobs, [(args.model_path, c, str(out), args.duration, args.settle,
                                      args.terrain, args.push) for c in chunks])

    logger.info(f"Done. {total} rollouts in {out}/  ({(time.time()-t0)/60:.1f} min)")
    logger.info(f"Next: python -m holosoma.eval.analyze --rollout-dir {out}")


if __name__ == "__main__":
    main()
