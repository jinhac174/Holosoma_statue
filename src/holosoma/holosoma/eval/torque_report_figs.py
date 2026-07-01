"""Report-ready torque figures built on top of the exceedance diagnostic.

Complements torque_exceedance.py (the 4-panel) with three narrative figures for the
write-up:

  1. torque_reveal.png        -- THE headline: per-joint PEAK applied (post-clip, capped
                                 at 1.0) vs PEAK demand (pre-clip). Shows the effort-limit
                                 clip hiding how hard ankle_roll is actually pushed.
  2. torque_distribution.png  -- ankle_roll demand/limit distribution, flat+DR vs rough,
                                 with the clipped tail (>1.0) shaded. Quantifies "how often
                                 / how far" in one view.
  3. torque_by_command.png    -- saturation rate per command, flat+DR vs rough side by side.

Usage:
    python -m holosoma.eval.torque_report_figs \
        --flat-dir <flat_rollouts> --rough-dir <rough_rollouts> --out-dir <dir>
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from holosoma.eval.schema import load_rollout_dir  # noqa: E402
from holosoma.eval.torque_exceedance import _label_of, _raw_substep  # noqa: E402


def _short(names):
    return [n.replace("_joint", "").replace("joint_", "") for n in names]


def scan(rollouts, ankle_subsample=8):
    """One memory-light pass. Returns per-joint peak/rate stats + ankle demand samples."""
    names = rollouts[0].dof_names
    limits = np.asarray(rollouts[0].metadata.get("effort_limits", []), dtype=np.float64)
    N = len(names)
    pre_max = np.zeros(N)
    post_max = np.zeros(N)
    over08 = np.zeros(N)
    sat = np.zeros(N)
    total = 0
    per_label_sat = defaultdict(list)
    ankle_idx = [i for i, n in enumerate(names) if "ankle_roll" in n]
    ankle_samples = []
    for r in rollouts:
        raw = _raw_substep(r)
        if raw is None:
            continue
        K = raw.shape[0] * raw.shape[1]
        pre = np.abs(raw).reshape(K, N) / limits[None, :]
        post = np.abs(np.asarray(r.get("torques_substep"))).reshape(K, N) / limits[None, :]
        pre_max = np.maximum(pre_max, pre.max(axis=0))
        post_max = np.maximum(post_max, post.max(axis=0))
        over08 += (pre > 0.8).sum(axis=0)
        sat += (pre >= 1.0).sum(axis=0)
        total += K
        per_label_sat[_label_of(r)].append(float((pre >= 1.0).any(axis=1).mean()))
        if ankle_idx:
            ankle_samples.append(pre[::ankle_subsample][:, ankle_idx].reshape(-1))
    over08 = 100.0 * over08 / max(total, 1)
    sat = 100.0 * sat / max(total, 1)
    ankle = np.concatenate(ankle_samples) if ankle_samples else np.zeros(0)
    return dict(names=names, limits=limits, pre_max=pre_max, post_max=post_max,
                over08=over08, sat=sat, per_label_sat=per_label_sat, ankle=ankle)


def fig_reveal(flat, out: Path):
    names, N = flat["names"], len(flat["names"])
    short = _short(names)
    order = np.argsort(flat["pre_max"])
    y = np.arange(N)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.barh(y, flat["pre_max"][order], color="tab:red", alpha=0.85,
            label="peak DEMAND (pre-clip)")
    ax.barh(y, flat["post_max"][order], color="0.5", alpha=0.9,
            label="peak APPLIED (post-clip, what the metric saw)")
    ax.axvline(1.0, color="red", ls="--", lw=1.3, label="1.0 = actuator limit")
    ax.axvline(0.8, color="orange", ls="--", lw=1.1, label="0.8 = spec")
    ax.set_yticks(y)
    ax.set_yticklabels([short[i] for i in order], fontsize=8)
    ax.set_xlabel("peak |torque| / effort limit")
    ax.set_title("What the effort-limit clip hid — peak demand vs peak applied, per joint\n"
                 "(applied flatlines at 1.0 where saturated; only ankle_roll demands past it)")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    p = out / "torque_reveal.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def fig_distribution(flat, rough, out: Path):
    fig, ax = plt.subplots(figsize=(11, 6))
    bins = np.linspace(0, 2.2, 90)
    for data, label, color in [(flat["ankle"], "flat + DR", "tab:blue"),
                               (rough["ankle"], "rough", "tab:orange")]:
        if data.size:
            frac_sat = 100.0 * np.mean(data >= 1.0)
            frac_08 = 100.0 * np.mean(data >= 0.8)
            ax.hist(data, bins=bins, histtype="step", density=True, lw=1.8, color=color,
                    label=f"{label}  (>1.0: {frac_sat:.2f}%,  >0.8: {frac_08:.2f}%)")
    ax.axvline(0.8, color="orange", ls="--", lw=1.1)
    ax.axvline(1.0, color="red", ls="--", lw=1.3)
    ax.axvspan(1.0, 2.2, color="red", alpha=0.06)
    ax.text(1.02, ax.get_ylim()[1] * 0.6, "clipped\n(lost authority)", color="red", fontsize=9)
    ax.set_yscale("log")
    ax.set_xlabel("ankle_roll |demand| / effort limit")
    ax.set_ylabel("density (log)")
    ax.set_title("ankle_roll torque-demand distribution (pre-clip) — the tail past 1.0 is what "
                 "the 60 Nm actuator cannot deliver")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out / "torque_distribution.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def fig_by_command(flat, rough, out: Path):
    labels = sorted(set(flat["per_label_sat"]) | set(rough["per_label_sat"]))
    fmean = [100.0 * np.mean(flat["per_label_sat"].get(l, [0])) for l in labels]
    rmean = [100.0 * np.mean(rough["per_label_sat"].get(l, [0])) for l in labels]
    x = np.arange(len(labels))
    w = 0.4
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x - w / 2, fmean, w, color="tab:blue", alpha=0.85, label="flat + DR")
    ax.bar(x + w / 2, rmean, w, color="tab:orange", alpha=0.85, label="rough")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("% substeps with ankle_roll saturated")
    ax.set_title("Where saturation happens — per command, flat+DR vs rough "
                 "(lateral & mixed dominate; stand/back/yaw are clean)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = out / "torque_by_command.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description="Report-ready torque figures")
    ap.add_argument("--flat-dir", required=True)
    ap.add_argument("--rough-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    flat = scan(load_rollout_dir(args.flat_dir))
    rough = scan(load_rollout_dir(args.rough_dir))
    if flat["ankle"].size == 0:
        raise SystemExit("no pre-clip data ('torques_substep_raw') in flat dir — re-run the eval")

    p1 = fig_reveal(flat, out)
    p2 = fig_distribution(flat, rough, out)
    p3 = fig_by_command(flat, rough, out)
    print("Wrote:")
    for p in (p1, p2, p3):
        print(f"  {p}")


if __name__ == "__main__":
    main()
