"""Quantitative gait metrics computed from a :class:`RolloutData`.

All functions are pure (rollout in, numbers out) and simulator-agnostic, so the
same code grades IsaacGym and MuJoCo rollouts. Metric definitions follow the
assignment spec (§4.1).

Spec targets (for reference; thresholds live in ``SPEC``):
    tracking RMS  < 0.15 m/s (lin), < 0.2 rad/s (yaw)
    torque safety factor  > 1.25   (peak torque <= 0.8 * stall)
    symmetry index  < 0.1
    CoT  < 2.5 at 0.8 m/s forward
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from holosoma.eval.schema import (
    GRAVITY,
    RolloutData,
    quat_tilt_angle,
    quat_yaw,
    world_to_body_xy,
)

SPEC = {
    "tracking_rms_lin": 0.15,   # m/s
    "tracking_rms_yaw": 0.20,   # rad/s
    "torque_safety_factor": 1.25,
    "symmetry_index": 0.10,
    "cot_at_0p8": 2.5,
}


@dataclass
class RolloutMetrics:
    """Per-rollout metric bundle (one row of the aggregate table)."""

    # identity / context
    name: str
    simulator: str
    cmd_vx: float
    cmd_vy: float
    cmd_vyaw: float
    # outcome
    fell: bool
    fall_time_s: float
    duration_s: float
    # tracking (steady-state RMS)
    rms_vx: float
    rms_vy: float
    rms_vyaw: float
    mean_vx: float          # achieved mean forward speed (body frame)
    # energetics
    cost_of_transport: float
    mean_mech_power_w: float
    # gait quality
    symmetry_index: float
    foot_clearance_mean_m: float       # mean swing-apex height (lift quality)
    foot_clearance_min_m: float        # lowest swing apex
    scuff_fraction: float              # fraction of time a foot drags near the ground
    # hardware safety
    torque_safety_factor: float       # min over joints of stall/peak
    worst_torque_joint: str
    n_torque_violations: int          # joints exceeding 0.8*stall
    n_pos_limit_violations: int       # steps*joints outside URDF limits
    n_self_collision_steps: int       # steps with any robot-robot (self) contact

    def to_row(self) -> dict:
        return asdict(self)


# ---- helpers ------------------------------------------------------------------
def _command_change_mask(cmd: np.ndarray, settle_steps: int) -> np.ndarray:
    """Boolean mask over T that is False for ``settle_steps`` after each command
    change (transient) and True elsewhere (steady state)."""
    T = cmd.shape[0]
    keep = np.ones(T, dtype=bool)
    changes = np.any(np.abs(np.diff(cmd, axis=0)) > 1e-6, axis=1)
    change_idx = np.flatnonzero(changes) + 1
    for idx in np.concatenate([[0], change_idx]):
        keep[idx : idx + settle_steps] = False
    return keep


def _body_lin_vel(r: RolloutData) -> np.ndarray:
    """Base linear velocity in the yaw frame [T, 2] = (forward, lateral)."""
    yaw = quat_yaw(r.get("root_quat_xyzw"))
    return world_to_body_xy(r.get("root_lin_vel")[:, :2], yaw)


def _mech_power(r: RolloutData) -> np.ndarray:
    """Instantaneous mechanical power [T] = sum_j |tau_j * dq_j|.

    Uses physics-substep resolution when available (more accurate), else the
    per-policy-step torques/velocities.
    """
    if r.has("torques_substep") and r.has("dof_vel_substep"):
        tau = r.get("torques_substep")          # [T, D, N]
        dq = r.get("dof_vel_substep")           # [T, D, N]
        p = np.abs(tau * dq).sum(axis=-1)       # [T, D]
        return p.mean(axis=1)                   # [T]
    tau = r.get("torques")
    dq = r.get("dof_vel")
    return np.abs(tau * dq).sum(axis=-1)


def _peak_torque_per_joint(r: RolloutData) -> np.ndarray:
    """Peak |torque| per joint over the rollout [N]."""
    if r.has("torques_substep"):
        return np.abs(r.get("torques_substep")).reshape(-1, r.get("dof_pos").shape[1]).max(axis=0)
    return np.abs(r.get("torques")).max(axis=0)


# ---- individual metrics -------------------------------------------------------
def detect_fall(r: RolloutData, fall_height: float = 0.4, tilt_deg: float = 50.0) -> tuple[bool, float]:
    """Return (fell, fall_time_s). Fall = base too low OR tilted past threshold."""
    z = r.get("root_pos")[:, 2]
    tilt = np.degrees(quat_tilt_angle(r.get("root_quat_xyzw")))
    bad = (z < fall_height) | (tilt > tilt_deg)
    if not bad.any():
        return False, float("nan")
    return True, float(np.argmax(bad) * r.dt)


def tracking_rms(r: RolloutData, settle_s: float = 0.5, end_idx: int | None = None) -> tuple[float, float, float, float]:
    """Steady-state tracking RMS error (vx, vy, vyaw) and achieved mean vx.

    Transients in the first ``settle_s`` after each command change are excluded.
    If the robot fell, only the pre-fall window is scored.
    """
    cmd = r.get("commanded_velocity")
    body_v = _body_lin_vel(r)
    act = np.column_stack([body_v[:, 0], body_v[:, 1], r.get("root_ang_vel")[:, 2]])
    keep = _command_change_mask(cmd, int(round(settle_s / r.dt)))
    if end_idx is not None:
        keep[end_idx:] = False
    if keep.sum() < 5:
        return float("nan"), float("nan"), float("nan"), float("nan")
    err = act[keep] - cmd[keep]
    rms = np.sqrt(np.mean(err**2, axis=0))
    return float(rms[0]), float(rms[1]), float(rms[2]), float(np.mean(act[keep, 0]))


def cost_of_transport(r: RolloutData, end_idx: int | None = None) -> tuple[float, float]:
    """(CoT, mean_mech_power_W). CoT = P / (m g v).

    Uses achieved forward speed. Returns nan CoT if mass missing or speed ~0.
    """
    power = _mech_power(r)
    body_v = _body_lin_vel(r)
    sl = slice(0, end_idx) if end_idx else slice(None)
    mean_p = float(np.mean(power[sl]))
    speed = float(np.mean(np.abs(body_v[sl, 0])))
    mass = r.metadata.get("total_mass")
    if not mass or speed < 0.05:
        return float("nan"), mean_p
    return mean_p / (mass * GRAVITY * speed), mean_p


def _foot_pos(r: RolloutData) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (height [T,2], xy [T,2,2]) for (left,right) feet, ground-referenced,
    or None if body positions / foot bodies aren't available."""
    if not r.has("body_pos_w"):
        return None
    li = r.body_index("left_foot_contact") or r.body_index("left_ankle_roll")
    ri = r.body_index("right_foot_contact") or r.body_index("right_ankle_roll")
    if li is None or ri is None:
        return None
    bp = r.get("body_pos_w")  # [T, B, 3]
    h = np.stack([bp[:, li, 2], bp[:, ri, 2]], axis=-1)  # [T, 2]
    h = h - np.percentile(h, 2)  # robust ground reference
    xy = np.stack([bp[:, li, :2], bp[:, ri, :2]], axis=1)  # [T, 2, 2]
    return h, xy


def _foot_contact_and_height(r: RolloutData, contact_h: float = 0.07) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (foot_height [T,2], in_contact [T,2]) for (left,right) feet."""
    fp = _foot_pos(r)
    if fp is None:
        return None
    h, _ = fp
    return h, h < contact_h


def symmetry_index(r: RolloutData) -> float:
    """Left-right symmetry index on per-leg mechanical energy.

    SI = |E_L - E_R| / (0.5 (E_L + E_R)). 0 = perfectly symmetric.
    Falls back to stance-fraction symmetry if leg joints can't be resolved.
    """
    left = r.leg_joint_indices("left")
    right = r.leg_joint_indices("right")
    if left and right and r.has("torques") and r.has("dof_vel"):
        p = np.abs(r.get("torques") * r.get("dof_vel"))
        eL = float(p[:, left].sum())
        eR = float(p[:, right].sum())
        denom = 0.5 * (eL + eR)
        return abs(eL - eR) / denom if denom > 1e-6 else float("nan")
    fc = _foot_contact_and_height(r)
    if fc is not None:
        _, contact = fc
        sL, sR = contact[:, 0].mean(), contact[:, 1].mean()
        denom = 0.5 * (sL + sR)
        return abs(sL - sR) / denom if denom > 1e-6 else float("nan")
    return float("nan")


def foot_clearance(r: RolloutData) -> tuple[float, float]:
    """(mean swing-peak clearance, min swing-peak clearance) in metres.

    For each contiguous swing phase per foot, take the peak height; report the
    mean and min of those peaks. Low min => foot-scuffing risk.
    """
    fc = _foot_contact_and_height(r)
    if fc is None:
        return float("nan"), float("nan")
    h, contact = fc
    peaks: list[float] = []
    for foot in range(2):
        swing = ~contact[:, foot]
        if not swing.any():
            continue
        # split swing into contiguous runs
        idx = np.flatnonzero(swing)
        splits = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        for run in splits:
            if run.size:
                peaks.append(float(h[run, foot].max()))
    if not peaks:
        return float("nan"), float("nan")
    return float(np.mean(peaks)), float(np.min(peaks))


def foot_scuffing(r: RolloutData, scuff_h: float = 0.03, scuff_speed: float = 0.2) -> float:
    """Fraction of timesteps a foot is *dragging*: moving horizontally faster than
    ``scuff_speed`` (m/s) while its height is below ``scuff_h`` (m).

    This is the direct "no foot-scuffing during swing" detector — unlike swing-apex,
    it catches a foot that clips/drags the ground at toe-off or touchdown. 0 = clean.
    """
    fp = _foot_pos(r)
    if fp is None:
        return float("nan")
    h, xy = fp
    horiz_speed = np.linalg.norm(np.diff(xy, axis=0), axis=-1) / r.dt  # [T-1, 2]
    low = h[1:] < scuff_h
    drag = low & (horiz_speed > scuff_speed)
    return float(drag.mean())


def torque_safety(r: RolloutData) -> tuple[float, str, int]:
    """(min safety factor, worst joint name, n joints over 0.8*stall).

    safety_factor_j = stall_j / peak_j ; spec requires > 1.25 (peak <= 0.8 stall).
    """
    limits = np.asarray(r.metadata.get("effort_limits", []), dtype=float)
    peak = _peak_torque_per_joint(r)
    if limits.size != peak.size or limits.size == 0:
        return float("nan"), "", 0
    with np.errstate(divide="ignore"):
        sf = limits / np.maximum(peak, 1e-9)
    worst = int(np.argmin(sf))
    n_viol = int(np.sum(peak > 0.8 * limits))
    names = r.dof_names
    return float(sf[worst]), (names[worst] if worst < len(names) else str(worst)), n_viol


def self_collision_steps(r: RolloutData, end_idx: int | None = None) -> int:
    """Number of timesteps with at least one self-collision (robot-robot contact).
    Spec: no self-collision events during a rollout."""
    if not r.has("self_collision"):
        return 0
    sc = r.get("self_collision")
    sl = slice(0, end_idx) if end_idx else slice(None)
    return int(np.sum(sc[sl] > 0))


def pos_limit_violations(r: RolloutData) -> int:
    """Count (step, joint) pairs where dof_pos is outside URDF limits."""
    lo = np.asarray(r.metadata.get("dof_pos_lower_limits", []), dtype=float)
    hi = np.asarray(r.metadata.get("dof_pos_upper_limits", []), dtype=float)
    q = r.get("dof_pos")
    if lo.size != q.shape[1] or hi.size != q.shape[1]:
        return 0
    return int(np.sum((q < lo) | (q > hi)))


# ---- top-level ----------------------------------------------------------------
def compute_metrics(r: RolloutData) -> RolloutMetrics:
    """Compute the full metric bundle for one rollout."""
    fell, fall_t = detect_fall(r)
    end_idx = int(fall_t / r.dt) if fell else None
    rms_vx, rms_vy, rms_vyaw, mean_vx = tracking_rms(r, end_idx=end_idx)
    cot, mean_p = cost_of_transport(r, end_idx=end_idx)
    si = symmetry_index(r)
    fc_mean, fc_min = foot_clearance(r)
    scuff = foot_scuffing(r)
    sf, worst, n_tv = torque_safety(r)
    cmd = r.get("commanded_velocity")
    cmd_mode = cmd[min(int(0.6 / r.dt), cmd.shape[0] - 1)]  # representative command

    return RolloutMetrics(
        name=r.path.stem,
        simulator=str(r.metadata.get("simulator", "unknown")),
        cmd_vx=float(cmd_mode[0]),
        cmd_vy=float(cmd_mode[1]),
        cmd_vyaw=float(cmd_mode[2]),
        fell=fell,
        fall_time_s=fall_t,
        duration_s=r.T * r.dt,
        rms_vx=rms_vx,
        rms_vy=rms_vy,
        rms_vyaw=rms_vyaw,
        mean_vx=mean_vx,
        cost_of_transport=cot,
        mean_mech_power_w=mean_p,
        symmetry_index=si,
        foot_clearance_mean_m=fc_mean,
        foot_clearance_min_m=fc_min,
        scuff_fraction=scuff,
        torque_safety_factor=sf,
        worst_torque_joint=worst,
        n_torque_violations=n_tv,
        n_pos_limit_violations=pos_limit_violations(r),
        n_self_collision_steps=self_collision_steps(r, end_idx=end_idx),
    )
