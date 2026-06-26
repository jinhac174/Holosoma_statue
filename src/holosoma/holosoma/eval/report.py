"""Generate a complete per-run evaluation report.

Given a run directory containing ``mujoco/`` and (optionally) ``isaacgym/``
sub-directories of NPZ rollouts, this writes — into the same run directory —

    scorecard.txt          spec PASS/FAIL table + per-command fall rate + sim2sim gap
    metrics_mujoco.csv      per-rollout metrics
    metrics_isaacgym.csv
    plots/                  all six metric figures (IG vs MuJoCo overlaid)
    run_info.md             template (created once) for recording what changed

This is the single command run after each (re)train + eval cycle, so every
iteration produces a comparable, threshold-checked scorecard.

Usage:
    python -m holosoma.eval.report --run-dir eval/runs/run01_baseline
    # or point at dirs explicitly:
    python -m holosoma.eval.report --run-dir eval/runs/run02 \
        --mujoco-dir eval/runs/run02/mujoco --isaacgym-dir eval/runs/run02/isaacgym
"""

from __future__ import annotations

import argparse
import datetime as _dt
from pathlib import Path

import numpy as np
import pandas as pd

from holosoma.eval.analyze import build_table, sim2sim_gap, spec_summary
from holosoma.eval.metrics import SPEC


def _cmd_col(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cmd"] = df.apply(lambda r: f"vx{r.cmd_vx:+.2f} vy{r.cmd_vy:+.2f} yaw{r.cmd_vyaw:+.2f}", axis=1)
    return df


def _line(mark: str, name: str, val: float, op: str, thr: float, fmt="{:.3f}") -> str:
    v = fmt.format(val) if np.isfinite(val) else "n/a"
    return f"  [{mark}] {name:34s} {v:>9s}   (target {op} {thr})"


def _check(val, thr, less=True) -> str:
    if not np.isfinite(val):
        return "  --"
    ok = (val <= thr) if less else (val >= thr)
    return "PASS" if ok else "FAIL"


def build_scorecard(run_name: str, mj: pd.DataFrame, ig: pd.DataFrame | None) -> str:
    """Return the scorecard text for a run (MuJoCo is the deployment-facing sim)."""
    s = spec_summary(mj)
    L: list[str] = []
    L.append("=" * 66)
    L.append(f"  EVAL SCORECARD — {run_name}")
    L.append(f"  generated {_dt.date.today().isoformat()}")
    L.append("=" * 66)
    L.append(f"  MuJoCo rollouts: {len(mj)}" + (f" | IsaacGym rollouts: {len(ig)}" if ig is not None else ""))
    L.append("")
    L.append("SPEC COMPLIANCE  (MuJoCo sim2sim — the deployment-facing numbers)")
    L.append(_line(_check(s["mean_rms_vx"], SPEC["tracking_rms_lin"]), "tracking RMS vx (m/s)", s["mean_rms_vx"], "<=", SPEC["tracking_rms_lin"]))
    L.append(_line(_check(s["mean_rms_vy"], SPEC["tracking_rms_lin"]), "tracking RMS vy (m/s)", s["mean_rms_vy"], "<=", SPEC["tracking_rms_lin"]))
    L.append(_line(_check(s["mean_rms_vyaw"], SPEC["tracking_rms_yaw"]), "tracking RMS vyaw (rad/s)", s["mean_rms_vyaw"], "<=", SPEC["tracking_rms_yaw"]))
    L.append(_line(_check(s["mean_symmetry"], SPEC["symmetry_index"]), "symmetry index", s["mean_symmetry"], "<=", SPEC["symmetry_index"]))
    L.append(_line(_check(s["min_torque_safety_factor"], SPEC["torque_safety_factor"], less=False), "min torque safety factor", s["min_torque_safety_factor"], ">=", SPEC["torque_safety_factor"]))
    L.append(_line(_check(s["mean_cot"], SPEC["cot_at_0p8"]), "cost of transport", s["mean_cot"], "<=", SPEC["cot_at_0p8"]))
    L.append(_line(_check(s["min_foot_clearance"], 0.0, less=False), "min foot clearance (m)", s["min_foot_clearance"], ">", 0.0))
    L.append(f"  [{'PASS' if s['total_pos_limit_violations']==0 else 'FAIL'}] {'joint pos-limit violations':34s} {s['total_pos_limit_violations']:>9d}   (target == 0)")
    L.append(f"  [INFO] {'fall rate':34s} {s['fall_rate']*100:>8.1f}%")
    L.append(f"  [INFO] {'torque-limit violations (step*joint)':34s} {s['total_torque_violations']:>9d}")
    L.append("")

    # per-command fall rate + tracking
    L.append("PER-COMMAND  (MuJoCo)")
    mjc = _cmd_col(mj)
    g = mjc.groupby("cmd").agg(fall=("fell", "mean"), rms_vx=("rms_vx", "mean"),
                               rms_vy=("rms_vy", "mean"), sym=("symmetry_index", "mean"),
                               sf=("torque_safety_factor", "mean")).round(3)
    L.append("  " + g.to_string().replace("\n", "\n  "))
    L.append("")

    # sim2sim gap
    if ig is not None:
        L.append("SIM2SIM GAP  (IsaacGym -> MuJoCo, tracking-RMS degradation; spec < 30%)")
        gap = sim2sim_gap(mj=mj, ig=ig)
        for axis in ("rms_vx", "rms_vy", "rms_vyaw"):
            gp = gap[f"{axis}_gap_pct"]
            mark = _check(abs(gp), 30.0) if np.isfinite(gp) else "  --"
            L.append(f"  [{mark}] {axis:10s}  IG={gap[f'{axis}_isaacgym']:.3f}  MJ={gap[f'{axis}_mujoco']:.3f}  gap={gp:+.0f}%")
        L.append(f"  fall rate: IsaacGym {gap['fall_rate_isaacgym']*100:.1f}%  vs  MuJoCo {gap['fall_rate_mujoco']*100:.1f}%")
        # per-command fall comparison highlights transfer failures
        igc = _cmd_col(ig)
        fall_cmp = pd.DataFrame({
            "fall_IG": igc.groupby("cmd")["fell"].mean(),
            "fall_MJ": mjc.groupby("cmd")["fell"].mean(),
        }).round(3)
        transfer_fail = fall_cmp[(fall_cmp.fall_MJ - fall_cmp.fall_IG) > 0.5]
        if len(transfer_fail):
            L.append("")
            L.append("  TRANSFER FAILURES (survives in IG, falls in MuJoCo):")
            L.append("  " + transfer_fail.to_string().replace("\n", "\n  "))
    L.append("=" * 66)
    return "\n".join(L)


_RUN_INFO_TEMPLATE = """# {run_name}

- **Date:** {date}
- **Checkpoint:** <path to .pt / .onnx, training step>
- **W&B run:** <url>

## Changes from previous run
<what reward weights / DR / config you changed and why — cite the metric from the
previous run's scorecard that motivated it>

## Hypothesis
<what you expect this change to improve>

## Result (fill after reading scorecard.txt)
<did it work? which thresholds moved? next lever>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a per-run eval report")
    ap.add_argument("--run-dir", required=True, help="Run directory (also default location of mujoco/ and isaacgym/)")
    ap.add_argument("--mujoco-dir", default=None, help="MuJoCo NPZ dir (default: <run-dir>/mujoco)")
    ap.add_argument("--isaacgym-dir", default=None, help="IsaacGym NPZ dir (default: <run-dir>/isaacgym)")
    ap.add_argument("--name", default=None, help="Run name for the scorecard (default: run-dir name)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or run_dir.name
    mj_dir = Path(args.mujoco_dir) if args.mujoco_dir else run_dir / "mujoco"
    ig_dir = Path(args.isaacgym_dir) if args.isaacgym_dir else run_dir / "isaacgym"

    print(f"[report] building MuJoCo table from {mj_dir}")
    mj = build_table(mj_dir)
    mj.to_csv(run_dir / "metrics_mujoco.csv", index=False)
    ig = None
    if ig_dir.exists() and any(ig_dir.glob("*.npz")):
        print(f"[report] building IsaacGym table from {ig_dir}")
        ig = build_table(ig_dir)
        ig.to_csv(run_dir / "metrics_isaacgym.csv", index=False)

    scorecard = build_scorecard(name, mj, ig)
    (run_dir / "scorecard.txt").write_text(scorecard + "\n")
    print(scorecard)

    # plots
    from holosoma.eval import plots as P
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    P.plot_tracking_vs_command(mj, plots_dir, ig)
    P.plot_symmetry_distribution(mj, plots_dir, ig)
    P.plot_cot_vs_velocity(mj, plots_dir, ig)
    P.plot_fall_rate(mj, plots_dir, ig)
    P.plot_foot_clearance(mj, plots_dir, ig)
    worst = mj.sort_values("torque_safety_factor").head(3)["name"].tolist()
    for p in sorted(mj_dir.glob("*.npz")):
        if p.stem in worst:
            P.plot_torque_timeseries(p, plots_dir)

    # run_info template (don't overwrite if it exists)
    info = run_dir / "run_info.md"
    if not info.exists():
        info.write_text(_RUN_INFO_TEMPLATE.format(run_name=name, date=_dt.date.today().isoformat()))

    print(f"\n[report] wrote scorecard.txt, metrics CSVs, plots/ to {run_dir}/")
    print(f"[report] edit {info} to record what changed for this run.")


if __name__ == "__main__":
    main()
