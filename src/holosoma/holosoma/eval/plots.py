"""Tuning-focused evaluation plots.

Designed to answer, at a glance, the questions the iteration loop asks (spec §4.4):
  1. spec_dashboard      — what passes / fails (every metric vs its threshold)
  2. fall_rate           — WHERE it fails, per named command, IsaacGym vs MuJoCo
  3. tracking            — commanded vs achieved velocity (under/over-tracking)
  4. torque_per_joint    — WHICH joints saturate (drives the torque-penalty lever)
  5. gait_quality        — symmetry / scuffing / CoT per command
  (+ torque_timeseries   — single-rollout deep-dive for a flagged joint)

All comparisons overlay IsaacGym (training dist) vs MuJoCo (sim2sim).

Usage:
    python -m holosoma.eval.plots --rollout-dir <mj_dir> [--compare-dir <ig_dir>]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from holosoma.eval.analyze import build_table, spec_summary  # noqa: E402
from holosoma.eval.metrics import SPEC  # noqa: E402
from holosoma.eval.schema import load_rollout  # noqa: E402

MJ_C, IG_C = "#d62728", "#1f77b4"  # MuJoCo red, IsaacGym blue


def _cmd_order(df):
    """Stable, readable command labels in a sensible order."""
    d = df.copy()
    d["cmd"] = d.apply(lambda r: f"vx{r.cmd_vx:+.2f}\nvy{r.cmd_vy:+.2f}\nyaw{r.cmd_vyaw:+.2f}", axis=1)
    return d


# 1 -----------------------------------------------------------------------------
def plot_spec_dashboard(mj, out: Path, ig=None) -> None:
    """Headline: every spec metric as value/threshold ratio. Bar past the 1.0 line = FAIL.

    All metrics normalized so '1.0 = exactly at threshold'; <1 passes (green), >1 fails (red).
    """
    s = spec_summary(mj)
    # (label, value, threshold, less_is_better)
    rows = [
        ("tracking RMS vx", s["mean_rms_vx"], SPEC["tracking_rms_lin"], True),
        ("tracking RMS vy", s["mean_rms_vy"], SPEC["tracking_rms_lin"], True),
        ("tracking RMS vyaw", s["mean_rms_vyaw"], SPEC["tracking_rms_yaw"], True),
        ("symmetry index", s["mean_symmetry"], SPEC["symmetry_index"], True),
        ("torque safety factor", s["min_torque_safety_factor"], SPEC["torque_safety_factor"], False),
        ("cost of transport", s["mean_cot"], SPEC["cot_at_0p8"], True),
        ("foot scuff fraction", s["mean_scuff_fraction"], 0.02, True),
        ("fall rate", s["fall_rate"], 0.0, True),  # threshold 0; show absolute below
    ]
    labels, ratios, colors, annots = [], [], [], []
    for label, val, thr, less in rows:
        if not np.isfinite(val):
            ratio, ok = 0.0, False
        elif label == "fall rate":
            ratio = val / 0.05 if val > 0 else 0.0  # scale: 5% fall = 1.0 reference
            ok = val < 0.05
        elif less:
            ratio, ok = (val / thr if thr > 0 else val), (val <= thr)
        else:  # greater-is-better -> invert so <1 = pass
            ratio, ok = (thr / val if val > 0 else 9), (val >= thr)
        labels.append(label)
        ratios.append(min(ratio, 4.0))  # clip for readability
        colors.append("#2ca02c" if ok else "#d62728")
        tgt = "0" if label == "fall rate" else f"{thr:g}"
        annots.append(f"{val:.3f}  (target {'<=' if less else '>='} {tgt})" if np.isfinite(val) else "n/a")

    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, ratios, color=colors, alpha=0.85)
    ax.axvline(1.0, color="k", ls="--", lw=1.2, label="threshold")
    for yi, a in zip(y, annots):
        ax.text(0.05, yi, a, va="center", ha="left", fontsize=8, color="white", fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("value / threshold   (bar crossing dashed line = FAIL)")
    ax.set_xlim(0, 4)
    ax.set_title("Spec dashboard — MuJoCo sim2sim (green=pass, red=fail)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "1_spec_dashboard.png", dpi=120)
    plt.close(fig)


# 2 -----------------------------------------------------------------------------
def plot_fall_rate(mj, out: Path, ig=None) -> None:
    """Fall rate per command, IsaacGym vs MuJoCo — the transfer-failure map."""
    m = _cmd_order(mj)
    fr = m.groupby("cmd", sort=False)["fell"].mean() * 100
    fr = fr.groupby(level=0).mean()
    order = fr.index
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(13, 4.5))
    w = 0.4 if ig is not None else 0.7
    if ig is not None:
        g = _cmd_order(ig).groupby("cmd")["fell"].mean().reindex(order).fillna(0) * 100
        ax.bar(x - w / 2, g.values, w, label="IsaacGym", color=IG_C)
        ax.bar(x + w / 2, fr.reindex(order).values, w, label="MuJoCo", color=MJ_C)
    else:
        ax.bar(x, fr.reindex(order).values, w, color=MJ_C, label="MuJoCo")
    ax.set_xticks(x)
    ax.set_xticklabels(order, fontsize=6)
    ax.set_ylabel("fall rate (%)")
    ax.set_title("Fall rate per command  (gap = sim2sim transfer failure)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / "2_fall_rate_per_command.png", dpi=120)
    plt.close(fig)


# 3 -----------------------------------------------------------------------------
def plot_tracking(mj, out: Path, ig=None) -> None:
    """Commanded vs achieved velocity per axis. Points below the diagonal = under-tracking.
    Only non-fallen rollouts; marker ringed red if that command falls a lot."""
    axes_spec = [("cmd_vx", "mean_vx", "forward vx (m/s)", SPEC["tracking_rms_lin"])]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    panels = [
        ("cmd_vx", "mean_vx", "forward vx (m/s)"),
        ("cmd_vy", None, "lateral vy (m/s)"),
        ("cmd_vyaw", None, "yaw rate (rad/s)"),
    ]
    for ax, (cmdc, achc, title) in zip(axs, panels):
        for df, c, name, mk in ((ig, IG_C, "IsaacGym", "x"), (mj, MJ_C, "MuJoCo", "o")):
            if df is None:
                continue
            d = df[~df["fell"]]
            # achieved: vx uses mean_vx; vy/yaw approximate via cmd - rms isn't signed, so
            # use the per-command mean of achieved where available (vx). For vy/yaw show RMS error bars.
            if cmdc == "cmd_vx":
                ax.scatter(d[cmdc], d["mean_vx"], s=14, alpha=0.4, color=c, marker=mk, label=name)
            else:
                # plot RMS error vs commanded magnitude (signed cmd, |error|)
                rms = "rms_vy" if cmdc == "cmd_vy" else "rms_vyaw"
                ax.scatter(d[cmdc], d[rms], s=14, alpha=0.4, color=c, marker=mk, label=name)
        if cmdc == "cmd_vx":
            lims = [-0.4, 1.1]
            ax.plot(lims, lims, "k--", lw=1, label="perfect")
            ax.set_ylabel("achieved vx (m/s)")
            ax.set_xlim(*lims)
        else:
            thr = SPEC["tracking_rms_lin"] if cmdc == "cmd_vy" else SPEC["tracking_rms_yaw"]
            ax.axhline(thr, color="k", ls="--", lw=1, label="spec RMS")
            ax.set_ylabel("RMS error")
        ax.set_xlabel(f"commanded {title}")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Velocity tracking (vx: achieved vs commanded; vy/yaw: RMS error vs command)")
    fig.tight_layout()
    fig.savefig(out / "3_tracking.png", dpi=120)
    plt.close(fig)


# 4 -----------------------------------------------------------------------------
def _peak_torque_over_stall(npz_paths, max_n=80):
    """Worst-case peak|torque|/stall per joint across a sample of rollouts.
    Returns (ratio[N], dof_names) or (None, None)."""
    paths = list(npz_paths)[:max_n]
    if not paths:
        return None, None
    worst = None
    names = None
    for p in paths:
        r = load_rollout(p)
        eff = np.asarray(r.metadata.get("effort_limits", []), dtype=float)
        if eff.size == 0:
            continue
        tau = r.get("torques_substep") if r.has("torques_substep") else r.get("torques")
        peak = np.abs(tau).reshape(-1, eff.size).max(axis=0)
        ratio = peak / eff
        worst = ratio if worst is None else np.maximum(worst, ratio)
        names = r.dof_names
    return worst, names


def plot_torque_per_joint(mj_paths, out: Path, ig_paths=None) -> None:
    """Worst-case peak torque / stall, per joint. Bars over 0.8 fail the hardware-safety
    spec; over 1.0 means the actuator saturated. Directly identifies joints to penalize."""
    mj_r, names = _peak_torque_over_stall(mj_paths)
    if mj_r is None:
        return
    ig_r, _ = _peak_torque_over_stall(ig_paths) if ig_paths else (None, None)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(14, 5))
    w = 0.4 if ig_r is not None else 0.7
    if ig_r is not None:
        ax.bar(x - w / 2, ig_r, w, label="IsaacGym", color=IG_C)
        ax.bar(x + w / 2, mj_r, w, label="MuJoCo", color=MJ_C)
    else:
        ax.bar(x, mj_r, w, color=MJ_C, label="MuJoCo")
    ax.axhline(0.8, color="orange", ls="--", lw=1.3, label="0.8 = spec limit")
    ax.axhline(1.0, color="red", ls="--", lw=1.3, label="1.0 = actuator saturation")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_joint", "") for n in names], rotation=90, fontsize=6)
    ax.set_ylabel("peak |torque| / stall torque (worst case)")
    ax.set_title("Per-joint torque margin  (bars > 0.8 fail hardware safety)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / "4_torque_per_joint.png", dpi=120)
    plt.close(fig)


# 5 -----------------------------------------------------------------------------
def plot_gait_quality(mj, out: Path, ig=None) -> None:
    """Symmetry, scuffing, CoT per command — the gait-quality levers."""
    m = _cmd_order(mj)
    order = m.groupby("cmd", sort=False).size().index
    x = np.arange(len(order))
    fig, axs = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    specs = [("symmetry_index", "symmetry index", SPEC["symmetry_index"]),
             ("scuff_fraction", "foot scuff fraction", 0.02),
             ("cost_of_transport", "cost of transport", SPEC["cot_at_0p8"])]
    for ax, (col, title, thr) in zip(axs, specs):
        mv = m[~m["fell"]].groupby("cmd")[col].mean().reindex(order)
        w = 0.4 if ig is not None else 0.7
        if ig is not None:
            iv = _cmd_order(ig)
            iv = iv[~iv["fell"]].groupby("cmd")[col].mean().reindex(order)
            ax.bar(x - w / 2, iv.values, w, label="IsaacGym", color=IG_C)
            ax.bar(x + w / 2, mv.values, w, label="MuJoCo", color=MJ_C)
        else:
            ax.bar(x, mv.values, w, color=MJ_C, label="MuJoCo")
        ax.axhline(thr, color="k", ls="--", lw=1, label=f"spec ({thr:g})")
        ax.set_ylabel(title)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=7)
    axs[0].set_title("Gait quality per command (NaN bars = command fell / not scored)")
    axs[-1].set_xticks(x)
    axs[-1].set_xticklabels(order, fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "5_gait_quality.png", dpi=120)
    plt.close(fig)


# deep-dive ---------------------------------------------------------------------
def plot_torque_timeseries(rollout_path: Path, out: Path) -> None:
    """Single-rollout torque vs time for the worst leg joints (deep dive)."""
    r = load_rollout(rollout_path)
    tau = np.abs(r.get("torques"))
    t = np.arange(r.T) * r.dt
    limits = np.asarray(r.metadata.get("effort_limits", []), dtype=float)
    names = r.dof_names
    leg = r.leg_joint_indices("left")[:6] or list(range(min(6, tau.shape[1])))
    fig, axes = plt.subplots(len(leg), 1, figsize=(11, 1.6 * len(leg)), sharex=True)
    if len(leg) == 1:
        axes = [axes]
    for ax, j in zip(axes, leg):
        ax.plot(t, tau[:, j], lw=0.8, color=MJ_C)
        if j < limits.size:
            ax.axhline(limits[j], color="red", ls="--", lw=1)
            ax.axhline(0.8 * limits[j], color="orange", ls=":", lw=1)
            over = tau[:, j] > 0.8 * limits[j]
            if over.any():
                ax.scatter(t[over], tau[over, j], color="red", s=8, zorder=5)
        ax.set_ylabel(f"{names[j].replace('_joint','') if j < len(names) else j}\nNm", fontsize=7)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time (s)")
    axes[0].set_title(f"Torque vs time — {rollout_path.stem} (red=stall, orange=0.8·stall)")
    fig.tight_layout()
    fig.savefig(out / f"torque_timeseries_{rollout_path.stem}.png", dpi=120)
    plt.close(fig)


def generate_all(mj, ig, mj_dir, ig_dir, out: Path) -> None:
    """Generate the full tuning-focused plot set into ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    plot_spec_dashboard(mj, out, ig)
    plot_fall_rate(mj, out, ig)
    plot_tracking(mj, out, ig)
    plot_torque_per_joint(sorted(Path(mj_dir).glob("*.npz")), out,
                          sorted(Path(ig_dir).glob("*.npz")) if ig_dir else None)
    plot_gait_quality(mj, out, ig)
    # deep-dive on the worst torque-margin rollouts AMONG SURVIVORS (so the torque
    # trace reflects walking, not a fall). Fall back to all if none survive.
    survivors = mj[~mj["fell"]]
    pool = survivors if len(survivors) else mj
    worst = pool.sort_values("torque_safety_factor").head(2)["name"].tolist()
    for p in sorted(Path(mj_dir).glob("*.npz")):
        if p.stem in worst:
            plot_torque_timeseries(p, out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate tuning-focused eval plots")
    ap.add_argument("--rollout-dir", required=True)
    ap.add_argument("--compare-dir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out or args.rollout_dir) / "plots"
    mj = build_table(args.rollout_dir)
    ig = build_table(args.compare_dir) if args.compare_dir else None
    generate_all(mj, ig, args.rollout_dir, args.compare_dir, out)
    print(f"Saved plots to {out}/")


if __name__ == "__main__":
    main()
