"""Canonical rollout-log schema shared by the MuJoCo and IsaacGym eval paths.

A *rollout* is one episode of a policy driving a single robot under a known
command schedule. Both simulators write the same NPZ layout so the metrics and
plotting code (``metrics.py``, ``plots.py``) is simulator-agnostic.

NPZ channels (T = number of policy steps, N = num DOF, B = num bodies,
D = physics substeps per policy step):

    dof_pos_target   [T, N]      PD setpoint (default_angles + action*scale)
    dof_pos          [T, N]      measured joint positions
    dof_vel          [T, N]      measured joint velocities
    torques          [T, N]      applied joint torques (last substep)
    torques_substep  [T, D, N]   per-substep torques (for CoT / peak torque)
    dof_vel_substep  [T, D, N]   per-substep joint velocities
    actions          [T, N]      raw policy actions
    root_pos         [T, 3]      base position, world frame
    root_quat_xyzw   [T, 4]      base orientation, world frame, [x,y,z,w]
    root_lin_vel     [T, 3]      base linear velocity, WORLD frame
    root_ang_vel     [T, 3]      base angular velocity, WORLD frame
    body_pos_w       [T, B, 3]   all body positions, world frame
    body_quat_xyzw   [T, B, 4]   all body orientations, world frame
    commanded_velocity [T, 3]    [vx, vy, vyaw] in the base (yaw) frame

Metadata (stored as a JSON string under ``_metadata_json``):

    dt, fps, sim_dt, sim_fps, control_decimation,
    dof_names [N], body_names [B],
    effort_limits [N], dof_pos_lower_limits [N], dof_pos_upper_limits [N],
    velocity_limits [N], total_mass (kg), simulator ("mujoco"|"isaacgym"),
    plus any per-rollout fields (seed, friction, added_mass, command label, ...).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

GRAVITY = 9.81

# Channels every rollout must contain for metrics to run.
REQUIRED_CHANNELS = (
    "dof_pos",
    "dof_vel",
    "torques",
    "root_pos",
    "root_quat_xyzw",
    "root_lin_vel",
    "root_ang_vel",
    "commanded_velocity",
)


@dataclass
class RolloutData:
    """One loaded rollout: all channels as arrays + parsed metadata."""

    channels: dict[str, np.ndarray]
    metadata: dict[str, Any]
    path: Path

    # ---- convenience accessors -------------------------------------------------
    @property
    def T(self) -> int:
        return self.channels["dof_pos"].shape[0]

    @property
    def dt(self) -> float:
        return float(self.metadata.get("dt", 0.02))

    @property
    def dof_names(self) -> list[str]:
        return list(self.metadata.get("dof_names", []))

    @property
    def body_names(self) -> list[str]:
        return list(self.metadata.get("body_names", []))

    def has(self, name: str) -> bool:
        return name in self.channels and self.channels[name].size > 0

    def get(self, name: str) -> np.ndarray:
        return self.channels[name]

    def body_index(self, substr: str) -> int | None:
        """First body whose name contains ``substr`` (case-insensitive)."""
        for i, n in enumerate(self.body_names):
            if substr.lower() in n.lower():
                return i
        return None

    def leg_joint_indices(self, side: str) -> list[int]:
        """Indices of leg joints for 'left' or 'right' (hip/knee/ankle)."""
        out = []
        for i, n in enumerate(self.dof_names):
            ln = n.lower()
            if ln.startswith(side) and any(k in ln for k in ("hip", "knee", "ankle")):
                out.append(i)
        return out


def load_rollout(path: str | Path) -> RolloutData:
    """Load an NPZ rollout written by either simulator's logger."""
    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        channels = {k: data[k] for k in data.files if k != "_metadata_json"}
        meta: dict[str, Any] = {}
        if "_metadata_json" in data.files:
            meta = json.loads(str(data["_metadata_json"]))
    missing = [c for c in REQUIRED_CHANNELS if c not in channels]
    if missing:
        raise ValueError(f"{path.name}: missing required channels {missing}")
    return RolloutData(channels=channels, metadata=meta, path=path)


def load_rollout_dir(directory: str | Path) -> list[RolloutData]:
    """Load every ``*.npz`` rollout in a directory (sorted by name)."""
    directory = Path(directory)
    paths = sorted(directory.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz rollouts found in {directory}")
    return [load_rollout(p) for p in paths]


# ---- quaternion / frame helpers (all [x,y,z,w]) -------------------------------
def quat_yaw(quat_xyzw: np.ndarray) -> np.ndarray:
    """Yaw angle (rad) from a batch of [x,y,z,w] quaternions. Shape [...,4]->[...]."""
    x, y, z, w = quat_xyzw[..., 0], quat_xyzw[..., 1], quat_xyzw[..., 2], quat_xyzw[..., 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)


def world_to_body_xy(vel_world_xy: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """Rotate world-frame horizontal velocity into the base yaw frame.

    vel_world_xy: [T, 2], yaw: [T] -> [T, 2] (forward, lateral).
    """
    c, s = np.cos(yaw), np.sin(yaw)
    vx, vy = vel_world_xy[:, 0], vel_world_xy[:, 1]
    fwd = c * vx + s * vy
    lat = -s * vx + c * vy
    return np.stack([fwd, lat], axis=-1)


def quat_tilt_angle(quat_xyzw: np.ndarray) -> np.ndarray:
    """Angle (rad) between the body +z axis and world +z. Shape [...,4]->[...].

    Used for fall detection: large tilt = robot has toppled.
    """
    x, y, z, w = quat_xyzw[..., 0], quat_xyzw[..., 1], quat_xyzw[..., 2], quat_xyzw[..., 3]
    # body z-axis expressed in world = R[:,2]; its world-z component:
    cos_tilt = 1.0 - 2.0 * (x * x + y * y)
    cos_tilt = np.clip(cos_tilt, -1.0, 1.0)
    return np.arccos(cos_tilt)
