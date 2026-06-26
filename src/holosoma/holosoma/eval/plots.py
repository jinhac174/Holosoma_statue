"""Plotting for the eval framework (spec §4.3).

Figures:
  1. tracking error vs commanded velocity, per axis
  2. joint torque vs time with stall-torque reference (per rollout), over-limit highlighted
  3. symmetry index distribution across rollouts
  4. cost of transport vs commanded forward velocity
  (+ sim2sim overlay when two metric tables are given)

Usage:
    python -m holosoma.eval.plots --rollout-dir <dir> [--compare-dir <dir>] [--out <dir>]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from holosoma.eval.analyze import build_table  # noqa: E402
from holosoma.eval.metrics import SPEC  # noqa: E402
from holosoma.eval.schema import load_rollout  # noqa: E402


def plot_tracking_vs_command(df, out: Path, compare=None) -> None:
    """Scatter: achieved tracking RMS vs commanded velocity magnitude, per axis."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    specs = [("cmd_vx", "rms_vx", SPEC["tracking_rms_lin"], "forward vx"),
             ("cmd_vy", "rms_vy", SPEC["tracking_rms_lin"], "lateral vy"),
             ("cmd_vyaw", "rms_vyaw", SPEC["tracking_rms_yaw"], "yaw")]
    for ax, (cmd_c, rms_c, thr, label) in zip(axes, specs):
        d = df[~df["fell"]]
        ax.scatter(d[cmd_c], d[rms_c], alpha=0.5, s=18, label="MuJoCo")
        if compare is not None:
            c = compare[~compare["fell"]]
            ax.scatter(c[cmd_c], c[rms_c], alpha=0.5, s=18, marker="x", label="IsaacGym")
            ax.legend()
        ax.axhline(thr, color="r", ls="--", lw=1, label="spec")
        ax.set_xlabel(f"commanded {label}")
        ax.set_ylabel(f"RMS error ({rms_c})")
        ax.set_title(f"Tracking error — {label}")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "tracking_vs_command.png", dpi=120)
    plt.close(fig)


def plot_torque_timeseries(rollout_path: Path, out: Path) -> None:
    """Joint torque vs time with per-joint stall reference; over-limit highlighted."""
    r = load_rollout(rollout_path)
    tau = np.abs(r.get("torques"))  # [T, N]
    t = np.arange(r.T) * r.dt
    limits = np.asarray(r.metadata.get("effort_limits", []), dtype=float)
    names = r.dof_names
    # show the 6 leg joints of the worst-margin leg for readability
    leg = r.leg_joint_indices("left")[:6] or list(range(min(6, tau.shape[1])))
    fig, axes = plt.subplots(len(leg), 1, figsize=(11, 1.7 * len(leg)), sharex=True)
    if len(leg) == 1:
        axes = [axes]
    for ax, j in zip(axes, leg):
        ax.plot(t, tau[:, j], lw=0.8, color="C0")
        if j < limits.size:
            stall = limits[j]
            ax.axhline(stall, color="r", ls="--", lw=1)
            ax.axhline(0.8 * stall, color="orange", ls=":", lw=1)
            over = tau[:, j] > 0.8 * stall
            if over.any():
                ax.scatter(t[over], tau[over, j], color="red", s=8, zorder=5)
            ax.set_ylim(0, max(stall * 1.1, tau[:, j].max() * 1.1))
        ax.set_ylabel(f"{names[j] if j < len(names) else j}\n|τ| (Nm)", fontsize=7)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time (s)")
    axes[0].set_title(f"Joint torque vs time — {rollout_path.stem}\n"
                      "red dash = stall, orange = 0.8×stall, red dots = over-limit")
    fig.tight_layout()
    fig.savefig(out / f"torque_timeseries_{rollout_path.stem}.png", dpi=120)
    plt.close(fig)


def plot_symmetry_distribution(df, out: Path, compare=None) -> None:
    """Histogram of symmetry index across rollouts."""
    fig, ax = plt.subplots(figsize=(7, 4))
    d = df[~df["fell"]]["symmetry_index"].to_numpy(dtype=float)
    d = d[np.isfinite(d)]
    ax.hist(d, bins=25, alpha=0.6, label="MuJoCo")
    if compare is not None:
        c = compare[~compare["fell"]]["symmetry_index"].to_numpy(dtype=float)
        c = c[np.isfinite(c)]
        ax.hist(c, bins=25, alpha=0.6, label="IsaacGym")
        ax.legend()
    ax.axvline(SPEC["symmetry_index"], color="r", ls="--", label="spec")
    ax.set_xlabel("symmetry index")
    ax.set_ylabel("count")
    ax.set_title("Left-right symmetry index distribution")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "symmetry_distribution.png", dpi=120)
    plt.close(fig)


def plot_cot_vs_velocity(df, out: Path, compare=None) -> None:
    """CoT vs achieved forward velocity."""
    fig, ax = plt.subplots(figsize=(7, 4))
    d = df[~df["fell"]]
    ax.scatter(d["mean_vx"], d["cost_of_transport"], alpha=0.5, s=18, label="MuJoCo")
    if compare is not None:
        c = compare[~compare["fell"]]
        ax.scatter(c["mean_vx"], c["cost_of_transport"], alpha=0.5, s=18, marker="x", label="IsaacGym")
    ax.axhline(SPEC["cot_at_0p8"], color="r", ls="--", label=f"spec ({SPEC['cot_at_0p8']})")
    ax.axvline(0.8, color="gray", ls=":", lw=1)
    ax.set_xlabel("achieved forward velocity (m/s)")
    ax.set_ylabel("cost of transport")
    ax.set_title("CoT vs forward velocity")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "cot_vs_velocity.png", dpi=120)
    plt.close(fig)


def _cmd_label(df):
    return df.apply(lambda r: f"vx{r.cmd_vx:+.2f}\nvy{r.cmd_vy:+.2f}\nyaw{r.cmd_vyaw:+.2f}", axis=1)


def plot_fall_rate(df, out: Path, compare=None) -> None:
    """Fall rate per command (bar chart, IG vs MuJoCo)."""
    fig, ax = plt.subplots(figsize=(13, 4))
    d = df.copy()
    d["cmd"] = _cmd_label(d)
    fr = d.groupby("cmd")["fell"].mean()
    x = np.arange(len(fr))
    w = 0.4 if compare is not None else 0.7
    ax.bar(x - (w / 2 if compare is not None else 0), fr.values * 100, w, label="MuJoCo", color="C0")
    if compare is not None:
        c = compare.copy()
        c["cmd"] = _cmd_label(c)
        cr = c.groupby("cmd")["fell"].mean().reindex(fr.index).fillna(0)
        ax.bar(x + w / 2, cr.values * 100, w, label="IsaacGym", color="C1")
        ax.legend()
    ax.set_xticks(x)
    ax.set_xticklabels(fr.index, fontsize=6)
    ax.set_ylabel("fall rate (%)")
    ax.set_title("Fall rate per command")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / "fall_rate_per_command.png", dpi=120)
    plt.close(fig)


def plot_foot_clearance(df, out: Path, compare=None) -> None:
    """Foot clearance distribution (min swing clearance per rollout)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    d = df[~df["fell"]]["foot_clearance_min_m"].to_numpy(dtype=float)
    d = d[np.isfinite(d)]
    ax.hist(d, bins=25, alpha=0.6, label="MuJoCo")
    if compare is not None:
        c = compare[~compare["fell"]]["foot_clearance_min_m"].to_numpy(dtype=float)
        c = c[np.isfinite(c)]
        ax.hist(c, bins=25, alpha=0.6, label="IsaacGym")
        ax.legend()
    ax.axvline(0.0, color="r", ls="--", label="scuffing (0)")
    ax.set_xlabel("min foot clearance during swing (m)")
    ax.set_ylabel("count")
    ax.set_title("Foot clearance distribution (lower = scuffing risk)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "foot_clearance.png", dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate eval plots")
    ap.add_argument("--rollout-dir", required=True)
    ap.add_argument("--compare-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--torque-examples", type=int, default=3, help="How many per-rollout torque plots")
    args = ap.parse_args()

    out = Path(args.out or args.rollout_dir) / "plots"
    out.mkdir(parents=True, exist_ok=True)

    df = build_table(args.rollout_dir)
    compare = build_table(args.compare_dir) if args.compare_dir else None

    plot_tracking_vs_command(df, out, compare)
    plot_symmetry_distribution(df, out, compare)
    plot_cot_vs_velocity(df, out, compare)
    plot_fall_rate(df, out, compare)
    plot_foot_clearance(df, out, compare)

    # torque timeseries for a few rollouts (prefer ones with violations)
    paths = sorted(Path(args.rollout_dir).glob("*.npz"))
    worst = df.sort_values("torque_safety_factor").head(args.torque_examples)["name"].tolist()
    chosen = [p for p in paths if p.stem in worst] or paths[: args.torque_examples]
    for p in chosen:
        plot_torque_timeseries(p, out)

    print(f"Saved plots to {out}/")


if __name__ == "__main__":
    main()
