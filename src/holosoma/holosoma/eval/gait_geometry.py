"""Gait-geometry diagnostic: torso roll, stance width, foot clearance, scuff.

Quantifies the things the scorecard doesn't: how much the torso rolls (the lateral
rocking), how wide the stance is (legs vertical vs A-framed inward), and foot-lift
height. Meant for before/after comparison across policies on the forward-walk commands.

Usage:
    python -m holosoma.eval.gait_geometry --rollout-dir <mujoco_npz_dir> [--label NAME] [--fwd-only]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from holosoma.eval.schema import load_rollout_dir, quat_yaw
from holosoma.eval.metrics import foot_clearance, foot_scuffing, _foot_pos


def _base_roll_pitch_deg(r) -> tuple[np.ndarray, np.ndarray]:
    """Per-step base roll and pitch (deg) from root quat (xyzw), via projected gravity."""
    q = np.asarray(r.get("root_quat_xyzw"), dtype=np.float64)  # [T,4] x,y,z,w
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    # body-frame UP vector (= 3rd col of R). Upright -> [0,0,1] -> roll=pitch=0.
    ux = 2 * (x * z + w * y)
    uy = 2 * (y * z - w * x)
    uz = 1 - 2 * (x * x + y * y)
    roll = np.degrees(np.arctan2(uy, uz))
    pitch = np.degrees(np.arctan2(-ux, uz))
    return roll, pitch


def _stance_width(r) -> float:
    """Mean lateral (perpendicular-to-heading) foot separation in metres."""
    fp = _foot_pos(r)
    if fp is None:
        return float("nan")
    _, xy = fp  # [T,2,2] (foot, x/y)
    yaw = quat_yaw(np.asarray(r.get("root_quat_xyzw")))
    lx, ly = xy[:, 0, 0], xy[:, 0, 1]
    rx, ry = xy[:, 1, 0], xy[:, 1, 1]
    perp = np.abs(np.cos(yaw) * (ly - ry) - np.sin(yaw) * (lx - rx))
    return float(np.mean(perp))


def analyze(rollouts, fwd_only: bool):
    roll_rms, pitch_rms, widths, cl_mean, cl_min, scuffs = [], [], [], [], [], []
    used = 0
    for r in rollouts:
        stem = Path(r.path).stem
        if fwd_only and not stem.startswith("fwd"):
            continue
        used += 1
        roll, pitch = _base_roll_pitch_deg(r)
        roll_rms.append(float(np.sqrt(np.mean(roll ** 2))))
        pitch_rms.append(float(np.sqrt(np.mean(pitch ** 2))))
        widths.append(_stance_width(r))
        cm, ci = foot_clearance(r)
        cl_mean.append(cm)
        cl_min.append(ci)
        scuffs.append(foot_scuffing(r))
    f = lambda a: float(np.nanmean(a)) if a else float("nan")
    return dict(
        n=used,
        roll_rms_deg=f(roll_rms),
        pitch_rms_deg=f(pitch_rms),
        stance_width_m=f(widths),
        foot_clearance_mean_m=f(cl_mean),
        foot_clearance_min_m=float(np.nanmin(cl_min)) if cl_min else float("nan"),
        scuff_fraction=f(scuffs),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Gait-geometry diagnostic")
    ap.add_argument("--rollout-dir", required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--fwd-only", action="store_true", help="Only forward-walk (fwd*) rollouts")
    args = ap.parse_args()
    rollouts = load_rollout_dir(args.rollout_dir)
    if not rollouts:
        raise SystemExit(f"no rollouts in {args.rollout_dir}")
    s = analyze(rollouts, args.fwd_only)
    label = args.label or Path(args.rollout_dir).name
    print(f"\n=== gait geometry: {label} "
          f"({'fwd-only' if args.fwd_only else 'all cmds'}, n={s['n']}) ===")
    print(f"  roll RMS           {s['roll_rms_deg']:.2f} deg   <- lateral rocking")
    print(f"  pitch RMS          {s['pitch_rms_deg']:.2f} deg")
    print(f"  stance width       {s['stance_width_m']:.3f} m   <- legs vertical vs inward")
    print(f"  foot clearance mean{s['foot_clearance_mean_m']:.3f} m   <- lift height")
    print(f"  foot clearance min {s['foot_clearance_min_m']:.3f} m")
    print(f"  scuff fraction     {s['scuff_fraction']:.4f}")


if __name__ == "__main__":
    main()
