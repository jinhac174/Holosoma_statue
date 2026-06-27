"""Single-process MuJoCo sim2sim eval runner.

Reuses the *real* inference ``LocomotionPolicy`` (zero observation divergence)
driven against an in-process MuJoCo model via a swapped ``BaseInterface``. The
runner owns a seeded command schedule and per-rollout domain randomization, and
writes the canonical NPZ schema (``schema.py``) for the metrics layer.

Why single-process: reproducible (seeded), fast (no ZMQ), perfectly aligned
command logging, and trivial per-rollout DR (mass/friction/initial state).

Run env: ``hsmujoco`` (has mujoco + onnxruntime + holosoma_inference).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
from loguru import logger

from holosoma.config_values.robot import statue_28dof
from holosoma.simulator.mujoco.tensor_views import quat_mujoco_to_holosoma


# NOTE: MuJoCo's freejoint angular velocity (qvel[3:6]) is already in the body frame,
# which is what the policy expects. We pass it through directly — see zmq_bridge.py for
# why a world->body rotation here is a bug (corrupts balance feedback while turning).


@dataclass
class DRParams:
    """Per-rollout domain randomization (sampled by the suite)."""

    friction: float = 1.0                 # ground sliding friction
    added_base_mass: float = 0.0          # kg added to pelvis
    link_mass_scale: float = 1.0          # scale on all link masses
    init_joint_noise: float = 0.0         # rad, uniform per-joint at reset


@dataclass
class CommandSegment:
    vx: float
    vy: float
    vyaw: float
    duration_s: float


@dataclass
class RolloutConfig:
    seed: int = 0
    duration_s: float = 12.0
    settle_s: float = 0.5                  # stand before commands start
    segments: list[CommandSegment] = field(default_factory=list)
    dr: DRParams = field(default_factory=DRParams)
    label: str = ""
    # Robustness: external push (spec: 100 N for 0.2 s at the torso). 0 force = no push.
    push_force: float = 0.0               # N
    push_duration_s: float = 0.2
    push_time_s: float = 4.0              # when the push starts (after settle, mid-walk)


# ---- in-process interface -----------------------------------------------------
class MujocoEvalInterface:
    """BaseInterface-compatible shim that steps MuJoCo in send_low_command and
    exposes full state for logging. Duck-typed (not subclassed) to avoid SDK deps."""

    def __init__(self, runner: "MujocoEvalRunner"):
        self.r = runner
        self._kp_level = 1.0
        self._kd_level = 1.0
        # substep buffers, filled each policy step
        self.substep_tau: list[np.ndarray] = []
        self.substep_dqv: list[np.ndarray] = []
        self.last_tau = np.zeros(runner.n_dof, dtype=np.float32)

    # ---- methods the policy calls ----
    def get_low_state(self) -> np.ndarray:
        r = self.r
        d = r.data
        q = d.qpos[r.dof_qpos_addrs].astype(np.float32)
        dq = d.qvel[r.dof_qvel_addrs].astype(np.float32)
        quat_wxyz = d.qpos[r.fj_qpos + 3 : r.fj_qpos + 7].astype(np.float32)
        omega_body = d.qvel[r.fj_qvel + 3 : r.fj_qvel + 6].astype(np.float32)  # already body-frame
        base_pos = d.qpos[r.fj_qpos : r.fj_qpos + 3].astype(np.float32)
        base_lin = d.qvel[r.fj_qvel : r.fj_qvel + 3].astype(np.float32)
        return np.concatenate([base_pos, quat_wxyz, q, base_lin, omega_body, dq]).reshape(1, -1)

    def send_low_command(self, cmd_q, cmd_dq, cmd_tau, dof_pos_latest=None, kp_override=None, kd_override=None):
        r = self.r
        kp = np.asarray(kp_override if kp_override is not None else r.motor_kp, dtype=np.float32) * self._kp_level
        kd = np.asarray(kd_override if kd_override is not None else r.motor_kd, dtype=np.float32) * self._kd_level
        cmd_q = np.asarray(cmd_q, dtype=np.float32).reshape(-1)
        cmd_dq = np.asarray(cmd_dq, dtype=np.float32).reshape(-1)
        cmd_tau = np.asarray(cmd_tau, dtype=np.float32).reshape(-1)

        self.substep_tau = []
        self.substep_dqv = []
        for _ in range(r.decimation):
            q = r.data.qpos[r.dof_qpos_addrs]
            dq = r.data.qvel[r.dof_qvel_addrs]
            tau = kp * (cmd_q - q) + kd * (cmd_dq - dq) + cmd_tau
            tau = np.clip(tau, -r.effort_limits, r.effort_limits).astype(np.float32)
            r.data.ctrl[r.actuator_ids] = tau
            mujoco.mj_step(r.model, r.data)
            self.substep_tau.append(tau.copy())
            self.substep_dqv.append(r.data.qvel[r.dof_qvel_addrs].astype(np.float32).copy())
        self.last_tau = tau

    def get_joystick_msg(self):
        return None

    def get_joystick_key(self, wc_msg=None):
        return None

    def update_config(self, robot_config):
        pass

    @property
    def kp_level(self):
        return self._kp_level

    @kp_level.setter
    def kp_level(self, v):
        self._kp_level = v

    @property
    def kd_level(self):
        return self._kd_level

    @kd_level.setter
    def kd_level(self, v):
        self._kd_level = v


# ---- runner -------------------------------------------------------------------
class MujocoEvalRunner:
    """Builds the MuJoCo model + inference policy once; runs many rollouts."""

    def __init__(
        self,
        model_path: str,
        physics_fps: int = 2000,
        policy_fps: int = 50,
        floor_friction: float = 1.0,
        terrain: str = "flat",            # "flat" or "rough" (heightfield)
        rough_height: float = 0.05,       # peak-to-peak height of rough terrain (m)
    ):
        self.robot_cfg = statue_28dof
        self.n_dof = self.robot_cfg.dof_obs_size
        self.physics_fps = physics_fps
        self.policy_fps = policy_fps
        self.decimation = physics_fps // policy_fps
        self.dt = 1.0 / policy_fps
        self.terrain = terrain
        self.rough_height = rough_height

        self._build_model(floor_friction)
        self._resolve_addressing()
        self._build_policy(model_path)

    # ---- model assembly ----
    def _build_model(self, floor_friction: float) -> None:
        from holosoma.utils.module_utils import get_holosoma_root

        asset = self.robot_cfg.asset
        asset_root = asset.asset_root
        if asset_root.startswith("@holosoma/"):
            asset_root = asset_root.replace("@holosoma", get_holosoma_root())
        xml_path = Path(asset_root) / asset.xml_file
        spec = mujoco.MjSpec.from_file(str(xml_path))
        spec.option.timestep = 1.0 / self.physics_fps
        spec.option.gravity = [0, 0, -9.81]
        if self.terrain == "rough":
            # Rough terrain via a heightfield (robustness test, spec §1).
            nrow = ncol = 64
            rng = np.random.default_rng(0)
            hdata = rng.random(nrow * ncol).astype(np.float64)  # 0..1, scaled by elevation z
            spec.add_hfield(name="rough", size=[8, 8, self.rough_height, 0.1],
                            nrow=nrow, ncol=ncol, userdata=list(hdata))
            floor = spec.worldbody.add_geom(
                name="eval_floor", type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname="rough",
                pos=[0, 0, 0], friction=[floor_friction, 0.005, 0.001],
            )
        else:
            # Flat ground plane — friction + contact softness matched to the live sim.
            floor = spec.worldbody.add_geom(
                name="eval_floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                size=[0, 0, 0.05], pos=[0, 0, 0], friction=[floor_friction, 0.005, 0.001],
            )
        floor.solimp = [0.99, 0.99, 0.01, 0.5, 2]
        floor.solref = [0.001, 1]
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self._floor_friction = floor_friction
        self.floor_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "eval_floor")

    def _resolve_addressing(self) -> None:
        m = self.model
        # freejoint
        fj_id = next(i for i in range(m.njnt) if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE)
        self.fj_qpos = m.jnt_qposadr[fj_id]
        self.fj_qvel = m.jnt_dofadr[fj_id]
        # dof joints in robot_config order
        self.dof_qpos_addrs = []
        self.dof_qvel_addrs = []
        self.actuator_ids = []
        for name in self.robot_cfg.dof_names:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.dof_qpos_addrs.append(m.jnt_qposadr[jid])
            self.dof_qvel_addrs.append(m.jnt_dofadr[jid])
            aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            self.actuator_ids.append(aid)
        self.dof_qpos_addrs = np.array(self.dof_qpos_addrs)
        self.dof_qvel_addrs = np.array(self.dof_qvel_addrs)
        self.actuator_ids = np.array(self.actuator_ids)
        # body names + foot ids
        self.body_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
        self.effort_limits = np.array(self.robot_cfg.dof_effort_limit_list, dtype=np.float32)
        self.motor_kp, self.motor_kd = self._resolve_gains()
        self.default_angles = np.array(
            [self.robot_cfg.init_state.default_joint_angles[n] for n in self.robot_cfg.dof_names], dtype=np.float32
        )
        self.base_total_mass = float(m.body_mass.sum())
        self.pelvis_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis_link")
        self._base_body_mass = m.body_mass.copy()
        self._base_body_inertia = m.body_inertia.copy()
        self._base_geom_friction = m.geom_friction.copy()

    def _resolve_gains(self) -> tuple[np.ndarray, np.ndarray]:
        stiffness = self.robot_cfg.control.stiffness
        damping = self.robot_cfg.control.damping
        kp, kd = [], []
        for n in self.robot_cfg.dof_names:
            key = n.replace("_joint", "")
            kp.append(_suffix_lookup(stiffness, key))
            kd.append(_suffix_lookup(damping, key))
        return np.array(kp, dtype=np.float32), np.array(kd, dtype=np.float32)

    # ---- policy ----
    def _build_policy(self, model_path: str) -> None:
        import dataclasses as _dc

        from holosoma_inference.config.config_values.inference import statue_28dof_loco
        from holosoma_inference.policies.locomotion import LocomotionPolicy

        task = _dc.replace(
            statue_28dof_loco.task,
            model_path=model_path,
            use_joystick=False,
            interface="lo",
            auto_start=True,
            velocity_input="keyboard",   # inert without a TTY; we inject commands manually
            state_input="keyboard",
        )
        cfg = _dc.replace(statue_28dof_loco, task=task)
        self.policy = LocomotionPolicy(config=cfg)
        # swap in our in-process interface
        self.interface = MujocoEvalInterface(self)
        self.policy.interface = self.interface
        self.policy.use_policy_action = True

    # ---- rollout ----
    def _reset(self, rc: RolloutConfig) -> None:
        m, d = self.model, self.data
        # restore base model props, then apply DR
        m.body_mass[:] = self._base_body_mass * rc.dr.link_mass_scale
        m.body_inertia[:] = self._base_body_inertia * rc.dr.link_mass_scale
        m.body_mass[self.pelvis_body_id] += rc.dr.added_base_mass
        # ground friction (geom 'eval_floor' is last geom)
        floor_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "eval_floor")
        m.geom_friction[floor_gid, 0] = rc.dr.friction

        mujoco.mj_resetData(m, d)
        d.qpos[self.fj_qpos : self.fj_qpos + 3] = self.robot_cfg.init_state.pos
        d.qpos[self.fj_qpos + 3 : self.fj_qpos + 7] = [1, 0, 0, 0]  # wxyz identity
        rng = np.random.default_rng(rc.seed)
        noise = rng.uniform(-rc.dr.init_joint_noise, rc.dr.init_joint_noise, size=self.n_dof) if rc.dr.init_joint_noise else 0.0
        d.qpos[self.dof_qpos_addrs] = self.default_angles + noise
        mujoco.mj_forward(m, d)
        # reset policy internal state (phase, last action)
        if hasattr(self.policy, "last_policy_action"):
            self.policy.last_policy_action[:] = 0.0
        if hasattr(self.policy, "scaled_policy_action"):
            self.policy.scaled_policy_action[:] = 0.0

    def _command_at(self, t: float, rc: RolloutConfig) -> tuple[float, float, float]:
        if t < rc.settle_s or not rc.segments:
            return 0.0, 0.0, 0.0
        tt = t - rc.settle_s
        acc = 0.0
        for seg in rc.segments:
            acc += seg.duration_s
            if tt < acc:
                return seg.vx, seg.vy, seg.vyaw
        last = rc.segments[-1]
        return last.vx, last.vy, last.vyaw

    def run_rollout(self, out_path: str | Path, rc: RolloutConfig) -> Path:
        from holosoma_inference.inputs.api.commands import StateCommand, VelCmd

        self._reset(rc)
        # enter walk mode
        try:
            self.policy._dispatch_command(StateCommand.WALK)
        except Exception:
            pass

        n_steps = int(round(rc.duration_s / self.dt))
        buffers: dict[str, list] = {k: [] for k in (
            "dof_pos_target", "dof_pos", "dof_vel", "torques", "torques_substep", "dof_vel_substep",
            "actions", "root_pos", "root_quat_xyzw", "root_lin_vel", "root_ang_vel",
            "body_pos_w", "body_quat_xyzw", "commanded_velocity", "self_collision",
        )}
        # External push (robustness): horizontal force on the pelvis during the push window.
        push_dir = np.zeros(3, dtype=np.float64)
        if rc.push_force > 0:
            ang = np.random.default_rng(rc.seed + 7).uniform(0, 2 * np.pi)
            push_dir[:2] = [np.cos(ang), np.sin(ang)]

        for step in range(n_steps):
            t = step * self.dt
            vx, vy, vyaw = self._command_at(t, rc)
            self.policy._apply_velocity(VelCmd(lin_vel=(vx, vy), ang_vel=vyaw))
            if getattr(self.policy, "use_phase", False):
                self.policy.update_phase_time()
            # apply push force for its window (cleared otherwise)
            in_push = rc.push_force > 0 and rc.push_time_s <= t < rc.push_time_s + rc.push_duration_s
            self.data.xfrc_applied[self.pelvis_body_id, :3] = push_dir * rc.push_force if in_push else 0.0
            self.policy.policy_action()  # steps physics `decimation` times via interface

            d = self.data
            # self-collision: contacts not involving the floor = robot geom vs robot geom
            nsc = sum(1 for c in range(d.ncon)
                      if d.contact[c].geom1 != self.floor_gid and d.contact[c].geom2 != self.floor_gid)
            buffers["self_collision"].append(np.float32(nsc))
            buffers["dof_pos"].append(d.qpos[self.dof_qpos_addrs].astype(np.float32).copy())
            buffers["dof_vel"].append(d.qvel[self.dof_qvel_addrs].astype(np.float32).copy())
            buffers["torques"].append(self.interface.last_tau.copy())
            buffers["torques_substep"].append(np.stack(self.interface.substep_tau))
            buffers["dof_vel_substep"].append(np.stack(self.interface.substep_dqv))
            buffers["actions"].append(self.policy.scaled_policy_action[0].astype(np.float32).copy())
            buffers["dof_pos_target"].append((self.policy.scaled_policy_action[0] + self.default_angles).astype(np.float32))
            buffers["root_pos"].append(d.qpos[self.fj_qpos : self.fj_qpos + 3].astype(np.float32).copy())
            quat_wxyz = d.qpos[self.fj_qpos + 3 : self.fj_qpos + 7].astype(np.float32)
            buffers["root_quat_xyzw"].append(quat_mujoco_to_holosoma(quat_wxyz).copy())
            buffers["root_lin_vel"].append(d.qvel[self.fj_qvel : self.fj_qvel + 3].astype(np.float32).copy())
            buffers["root_ang_vel"].append(d.qvel[self.fj_qvel + 3 : self.fj_qvel + 6].astype(np.float32).copy())
            buffers["body_pos_w"].append(d.xpos.astype(np.float32).copy())
            buffers["body_quat_xyzw"].append(quat_mujoco_to_holosoma(d.xquat.astype(np.float32)).copy())
            buffers["commanded_velocity"].append(np.array([vx, vy, vyaw], dtype=np.float32))

        arrays = {k: np.stack(v, axis=0) for k, v in buffers.items() if v}
        meta = {
            "dt": self.dt, "fps": self.policy_fps,
            "sim_dt": 1.0 / self.physics_fps, "sim_fps": self.physics_fps,
            "control_decimation": self.decimation,
            "dof_names": list(self.robot_cfg.dof_names),
            "body_names": list(self.body_names),
            "effort_limits": list(map(float, self.effort_limits)),
            "dof_pos_lower_limits": list(map(float, self.robot_cfg.dof_pos_lower_limit_list)),
            "dof_pos_upper_limits": list(map(float, self.robot_cfg.dof_pos_upper_limit_list)),
            "velocity_limits": list(map(float, self.robot_cfg.dof_vel_limit_list)),
            "total_mass": self.base_total_mass * rc.dr.link_mass_scale + rc.dr.added_base_mass,
            "simulator": "mujoco",
            "seed": rc.seed, "label": rc.label,
            "friction": rc.dr.friction, "added_base_mass": rc.dr.added_base_mass,
            "link_mass_scale": rc.dr.link_mass_scale,
            "terrain": self.terrain, "push_force": rc.push_force,
        }
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, _metadata_json=np.array(json.dumps(meta)), **arrays)
        return out_path


def _suffix_lookup(d: dict, key: str) -> float:
    """Match a gain dict keyed by joint substrings (e.g. 'hip_pitch')."""
    for k, v in d.items():
        if key.endswith(k) or k in key:
            return v
    raise KeyError(f"No gain for {key} in {list(d)}")
