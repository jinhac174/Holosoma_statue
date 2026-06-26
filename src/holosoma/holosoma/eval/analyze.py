"""Aggregate a directory of rollouts into a metrics table + spec report.

Usage:
    python -m holosoma.eval.analyze --rollout-dir <dir> [--out <dir>]
    # Cross-sim gap (spec §4 / sim2sim < 30%):
    python -m holosoma.eval.analyze --rollout-dir mj_dir --compare-dir ig_dir

Outputs ``metrics.csv`` (one row per rollout) and prints a spec pass/fail
summary. Pair with ``plots.py`` for figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from holosoma.eval.metrics import SPEC, compute_metrics
from holosoma.eval.schema import load_rollout_dir


def build_table(rollout_dir: str | Path) -> pd.DataFrame:
    """Load all rollouts in a dir and return a per-rollout metrics DataFrame."""
    rollouts = load_rollout_dir(rollout_dir)
    rows = [compute_metrics(r).to_row() for r in rollouts]
    return pd.DataFrame(rows)


def spec_summary(df: pd.DataFrame) -> dict:
    """Aggregate spec compliance across rollouts (NaNs ignored per metric)."""
    ok = df[~df["fell"]]  # spec metrics evaluated on non-fallen rollouts
    fall_rate = float(df["fell"].mean()) if len(df) else float("nan")

    def _mean(col):
        v = ok[col].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        return float(v.mean()) if v.size else float("nan")

    def _frac_pass(col, thresh, less_is_better=True):
        v = ok[col].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if not v.size:
            return float("nan")
        return float(np.mean(v <= thresh if less_is_better else v >= thresh))

    return {
        "n_rollouts": int(len(df)),
        "fall_rate": fall_rate,
        "mean_rms_vx": _mean("rms_vx"),
        "mean_rms_vy": _mean("rms_vy"),
        "mean_rms_vyaw": _mean("rms_vyaw"),
        "pass_rms_vx": _frac_pass("rms_vx", SPEC["tracking_rms_lin"]),
        "pass_rms_vy": _frac_pass("rms_vy", SPEC["tracking_rms_lin"]),
        "pass_rms_vyaw": _frac_pass("rms_vyaw", SPEC["tracking_rms_yaw"]),
        "mean_cot": _mean("cost_of_transport"),
        "mean_symmetry": _mean("symmetry_index"),
        "pass_symmetry": _frac_pass("symmetry_index", SPEC["symmetry_index"]),
        "min_torque_safety_factor": float(
            np.nanmin(ok["torque_safety_factor"].to_numpy(dtype=float)) if len(ok) else float("nan")
        ),
        "pass_torque_safety": _frac_pass("torque_safety_factor", SPEC["torque_safety_factor"], less_is_better=False),
        "total_torque_violations": int(df["n_torque_violations"].sum()),
        "total_pos_limit_violations": int(df["n_pos_limit_violations"].sum()),
        "min_foot_clearance": float(
            np.nanmin(ok["foot_clearance_min_m"].to_numpy(dtype=float)) if len(ok) else float("nan")
        ),
        "mean_scuff_fraction": _mean("scuff_fraction"),
    }


def sim2sim_gap(mj: pd.DataFrame, ig: pd.DataFrame) -> dict:
    """Tracking-error degradation MuJoCo vs IsaacGym (spec: < 30%).

    gap = (rms_mujoco - rms_isaacgym) / rms_isaacgym, per axis.
    """
    def _m(df, col):
        v = df[~df["fell"]][col].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        return float(v.mean()) if v.size else float("nan")

    out = {}
    for axis in ("rms_vx", "rms_vy", "rms_vyaw"):
        ig_v, mj_v = _m(ig, axis), _m(mj, axis)
        out[f"{axis}_isaacgym"] = ig_v
        out[f"{axis}_mujoco"] = mj_v
        out[f"{axis}_gap_pct"] = (mj_v - ig_v) / ig_v * 100.0 if ig_v > 1e-6 else float("nan")
    out["fall_rate_isaacgym"] = float(ig["fell"].mean())
    out["fall_rate_mujoco"] = float(mj["fell"].mean())
    return out


def _print_report(df: pd.DataFrame, summary: dict, label: str) -> None:
    logger.info(f"\n{'='*60}\n  EVAL REPORT — {label}  ({summary['n_rollouts']} rollouts)\n{'='*60}")
    def chk(name, val, thresh, less=True, fmt="{:.3f}"):
        if not np.isfinite(val):
            mark = "  --"
        else:
            ok = (val <= thresh) if less else (val >= thresh)
            mark = "PASS" if ok else "FAIL"
        logger.info(f"  [{mark}] {name:30s} {fmt.format(val) if np.isfinite(val) else 'n/a':>10s}  (target {'<=' if less else '>='} {thresh})")
    logger.info(f"  fall rate: {summary['fall_rate']*100:.1f}%")
    chk("tracking RMS vx (m/s)", summary["mean_rms_vx"], SPEC["tracking_rms_lin"])
    chk("tracking RMS vy (m/s)", summary["mean_rms_vy"], SPEC["tracking_rms_lin"])
    chk("tracking RMS vyaw (rad/s)", summary["mean_rms_vyaw"], SPEC["tracking_rms_yaw"])
    chk("symmetry index", summary["mean_symmetry"], SPEC["symmetry_index"])
    chk("min torque safety factor", summary["min_torque_safety_factor"], SPEC["torque_safety_factor"], less=False)
    logger.info(f"  mean CoT: {summary['mean_cot']:.3f} (target < {SPEC['cot_at_0p8']} at 0.8 m/s)")
    logger.info(f"  torque violations (steps*joints): {summary['total_torque_violations']}")
    logger.info(f"  pos-limit violations: {summary['total_pos_limit_violations']}")
    logger.info(f"  min foot clearance: {summary['min_foot_clearance']:.3f} m")


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate rollout metrics + spec report")
    ap.add_argument("--rollout-dir", required=True, help="Directory of .npz rollouts (primary, e.g. MuJoCo)")
    ap.add_argument("--compare-dir", default=None, help="Second dir (e.g. IsaacGym) for sim2sim gap")
    ap.add_argument("--out", default=None, help="Output dir for metrics.csv (default: rollout-dir)")
    args = ap.parse_args()

    out = Path(args.out or args.rollout_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = build_table(args.rollout_dir)
    df.to_csv(out / "metrics.csv", index=False)
    logger.info(f"Wrote {out/'metrics.csv'} ({len(df)} rows)")
    _print_report(df, spec_summary(df), label=Path(args.rollout_dir).name)

    if args.compare_dir:
        df_cmp = build_table(args.compare_dir)
        df_cmp.to_csv(out / "metrics_compare.csv", index=False)
        _print_report(df_cmp, spec_summary(df_cmp), label=Path(args.compare_dir).name)
        gap = sim2sim_gap(mj=df, ig=df_cmp)
        logger.info(f"\n{'='*60}\n  SIM2SIM GAP (MuJoCo vs IsaacGym) — target < 30%\n{'='*60}")
        for axis in ("rms_vx", "rms_vy", "rms_vyaw"):
            g = gap[f"{axis}_gap_pct"]
            mark = "PASS" if (np.isfinite(g) and abs(g) < 30) else ("FAIL" if np.isfinite(g) else " --")
            logger.info(f"  [{mark}] {axis}: IG={gap[f'{axis}_isaacgym']:.3f} "
                        f"MJ={gap[f'{axis}_mujoco']:.3f}  gap={g:+.1f}%")


if __name__ == "__main__":
    main()
