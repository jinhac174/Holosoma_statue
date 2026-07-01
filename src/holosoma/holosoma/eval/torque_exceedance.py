"""Torque-exceedance diagnostic: WHICH joints demand torque past the actuator limit,
HOW FAR past, on WHAT commands, and at WHICH moments.

This is the deep-dive behind the stuck "min torque safety factor = 1.000". The headline
metric only sees POST-clip torque (clipped at the effort limit), so a saturated joint always
reads exactly 1.0 and we cannot tell an actuator that is 1% over from one that is 80% over.
This tool reads the PRE-clip PD demand (``torques_substep_raw``, added to the runner) and
quantifies the gap between what the policy ASKED for and what the hardware can deliver.

Four cuts (spec: user request):
  A. which joints + saturation rate   -- per-joint fraction of substeps demand > 0.8 and > 1.0
  B. how far past 100%                 -- per-joint p50/p99/max of |demand|/limit
  C. on what occasions (per command)   -- saturation rate grouped by command family
  D. which moments (time series)       -- worst rollout: demand vs clipped-applied over time

Usage:
    python -m holosoma.eval.torque_exceedance --rollout-dir <dir> --out-dir <dir>
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from holosoma.eval.schema import load_rollout_dir  # noqa: E402

# command family from the rollout filename stem (e.g. "fwd08_0089" -> "fwd08")
_LABEL_RE = re.compile(r"^(.*?)_\d+$")


def _label_of(r) -> str:
    m = _LABEL_RE.match(Path(r.path).stem)
    return m.group(1) if m else Path(r.path).stem


def _raw_substep(r) -> np.ndarray | None:
    """[T, D, N] pre-clip demand if present, else None (older NPZ without pre-clip logging)."""
    if r.has("torques_substep_raw"):
        return np.asarray(r.get("torques_substep_raw"), dtype=np.float64)
    return None


def collect(rollouts):
    """Aggregate |demand|/limit ratios across all rollouts.

    Returns (names, limits, ratios_flat [M, N] over all substeps, per_label dict, rollouts_used).
    """
    names = rollouts[0].dof_names
    limits = np.asarray(rollouts[0].metadata.get("effort_limits", []), dtype=np.float64)
    N = len(names)
    all_ratios = []
    per_label_sat = defaultdict(list)   # label -> list of per-rollout saturation fractions (any joint)
    used = 0
    missing = 0
    for r in rollouts:
        raw = _raw_substep(r)
        if raw is None:
            missing += 1
            continue
        used += 1
        rr = np.abs(raw).reshape(-1, N) / limits[None, :]   # [T*D, N] ratio to limit
        all_ratios.append(rr.astype(np.float32))
        # saturation fraction for this rollout = frac of substeps with ANY joint demand >= 1.0
        sat_any = (rr >= 1.0).any(axis=1).mean()
        per_label_sat[_label_of(r)].append(float(sat_any))
    ratios = np.concatenate(all_ratios, axis=0) if all_ratios else np.zeros((0, N))
    return names, limits, ratios, per_label_sat, used, missing


def worst_rollout(rollouts):
    """Rollout with the highest fraction of saturated substeps (for the time-series panel)."""
    best, best_frac = None, -1.0
    for r in rollouts:
        raw = _raw_substep(r)
        if raw is None:
            continue
        N = len(r.dof_names)
        limits = np.asarray(r.metadata.get("effort_limits", []), dtype=np.float64)
        rr = np.abs(raw).reshape(-1, N) / limits[None, :]
        frac = float((rr >= 1.0).any(axis=1).mean())
        if frac > best_frac:
            best, best_frac = r, frac
    return best, best_frac


def make_figure(names, limits, ratios, per_label_sat, worst, out_dir: Path):
    N = len(names)
    over08 = (ratios > 0.8).mean(axis=0) * 100.0      # % substeps demand > 0.8*limit
    sat = (ratios >= 1.0).mean(axis=0) * 100.0        # % substeps demand >= limit (saturated)
    p99 = np.percentile(ratios, 99, axis=0) if ratios.size else np.zeros(N)
    pmax = ratios.max(axis=0) if ratios.size else np.zeros(N)
    p50 = np.percentile(ratios, 50, axis=0) if ratios.size else np.zeros(N)

    # short joint labels
    short = [n.replace("_joint", "").replace("joint_", "") for n in names]
    order = np.argsort(sat)                            # ascending; worst at top when barh

    fig, axes = plt.subplots(2, 2, figsize=(17, 12))

    # --- A: which joints + saturation rate ---
    ax = axes[0, 0]
    y = np.arange(N)
    ax.barh(y, over08[order], color="tab:orange", alpha=0.85, label="demand > 0.8·limit")
    ax.barh(y, sat[order], color="tab:red", alpha=0.95, label="demand ≥ limit (saturated)")
    ax.set_yticks(y)
    ax.set_yticklabels([short[i] for i in order], fontsize=7)
    ax.set_xlabel("% of substeps")
    ax.set_title("A. Which joints exceed — % of time demand is over 0.8·limit / saturated")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    # --- B: how far past 100% ---
    ax = axes[0, 1]
    ax.barh(y, pmax[order], color="0.8", label="max")
    ax.barh(y, p99[order], color="tab:red", alpha=0.85, label="p99")
    ax.scatter(p50[order], y, s=12, color="black", zorder=5, label="median")
    ax.axvline(1.0, color="red", ls="--", lw=1.2, label="1.0 = actuator limit")
    ax.axvline(0.8, color="orange", ls="--", lw=1.0, label="0.8 = spec")
    ax.set_yticks(y)
    ax.set_yticklabels([short[i] for i in order], fontsize=7)
    ax.set_xlabel("|demanded torque| / effort limit")
    ax.set_title("B. How far past the limit — demand/limit (median, p99, max)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    # --- C: on what occasions (per command family) ---
    ax = axes[1, 0]
    labels = sorted(per_label_sat.keys())
    means = [100.0 * np.mean(per_label_sat[l]) for l in labels]
    colors = ["tab:red" if m > 0 else "tab:green" for m in means]
    ax.bar(range(len(labels)), means, color=colors, alpha=0.85)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("% substeps with ≥1 joint saturated")
    ax.set_title("C. On what commands saturation happens (mean over rollouts)")
    ax.grid(axis="y", alpha=0.3)

    # --- D: which moments (worst rollout, top saturating joints, demand vs applied) ---
    ax = axes[1, 1]
    if worst is not None:
        r = worst
        raw = np.abs(_raw_substep(r))                 # [T, D, N]
        appl = np.abs(np.asarray(r.get("torques_substep")))  # [T, D, N] post-clip
        Tn, D, _ = raw.shape
        lim = limits
        # flatten substeps to a fine time axis
        raw_f = raw.reshape(Tn * D, N)
        appl_f = appl.reshape(Tn * D, N)
        tt = np.arange(Tn * D) * (r.dt / D)
        # pick the 2 joints with the largest peak demand/limit in THIS rollout
        peak = (raw_f / lim[None, :]).max(axis=0)
        top = np.argsort(peak)[::-1][:2]
        cols = ["tab:blue", "tab:purple"]
        for k, j in enumerate(top):
            ax.plot(tt, raw_f[:, j] / lim[j], color=cols[k], lw=0.9,
                    label=f"{short[j]} demand (peak {peak[j]:.2f})")
            ax.plot(tt, appl_f[:, j] / lim[j], color=cols[k], lw=0.9, ls=":", alpha=0.7,
                    label=f"{short[j]} applied (clipped)")
        ax.axhline(1.0, color="red", ls="--", lw=1.2)
        ax.axhline(0.8, color="orange", ls="--", lw=1.0)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("|torque| / limit")
        ax.set_title(f"D. Which moments — worst rollout {Path(r.path).stem} "
                     f"(demand solid, clipped-applied dotted)")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.3)

    fig.suptitle("Torque exceedance diagnostic — pre-clip PD demand vs actuator limit",
                 fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = out_dir / "torque_exceedance.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out, over08, sat, p50, p99, pmax


def write_csv(names, limits, over08, sat, p50, p99, pmax, per_label_sat, out_dir: Path):
    per_joint = out_dir / "torque_exceedance_per_joint.csv"
    with open(per_joint, "w") as f:
        f.write("joint,effort_limit,pct_over_0.8,pct_saturated,median_ratio,p99_ratio,max_ratio\n")
        for i, n in enumerate(names):
            f.write(f"{n},{limits[i]:.3f},{over08[i]:.3f},{sat[i]:.3f},"
                    f"{p50[i]:.4f},{p99[i]:.4f},{pmax[i]:.4f}\n")
    per_cmd = out_dir / "torque_exceedance_per_command.csv"
    with open(per_cmd, "w") as f:
        f.write("command,mean_pct_substeps_saturated,n_rollouts\n")
        for l in sorted(per_label_sat.keys()):
            v = per_label_sat[l]
            f.write(f"{l},{100.0*np.mean(v):.4f},{len(v)}\n")
    return per_joint, per_cmd


def main() -> None:
    ap = argparse.ArgumentParser(description="Torque-exceedance diagnostic (pre-clip demand)")
    ap.add_argument("--rollout-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rollouts = load_rollout_dir(args.rollout_dir)
    if not rollouts:
        raise SystemExit(f"no rollouts in {args.rollout_dir}")

    names, limits, ratios, per_label_sat, used, missing = collect(rollouts)
    if used == 0:
        raise SystemExit(
            "No rollout has 'torques_substep_raw' (pre-clip demand). Re-run the eval with the "
            "updated mujoco_runner that records it, then re-run this tool.")
    worst, worst_frac = worst_rollout(rollouts)

    png, over08, sat, p50, p99, pmax = make_figure(names, limits, ratios, per_label_sat, worst, out_dir)
    cj, cc = write_csv(names, limits, over08, sat, p50, p99, pmax, per_label_sat, out_dir)

    # console summary: joints that ever saturate, ranked
    order = np.argsort(sat)[::-1]
    print(f"\nRollouts used: {used} (skipped {missing} without pre-clip data)")
    print(f"Worst rollout (most saturated substeps): {Path(worst.path).stem}  "
          f"{100*worst_frac:.1f}% of substeps saturated")
    print("\nPer-joint (ranked by % saturated):")
    print(f"  {'joint':<28} {'limit':>6} {'%>0.8':>7} {'%sat':>7} {'p99':>6} {'max':>6}")
    for i in order:
        if sat[i] == 0 and over08[i] == 0:
            continue
        print(f"  {names[i]:<28} {limits[i]:>6.0f} {over08[i]:>7.2f} {sat[i]:>7.2f} "
              f"{p99[i]:>6.2f} {pmax[i]:>6.2f}")
    print(f"\nWrote:\n  {png}\n  {cj}\n  {cc}")


if __name__ == "__main__":
    main()
