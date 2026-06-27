"""IsaacGym evaluation suite — the IsaacGym counterpart to ``run_suite.py``.

Loads a trained checkpoint once, then for each command in the grid runs
``num_envs`` parallel environments (each a different DR / initial-state draw =
one rollout) on FLAT terrain, holding that command constant. Records every env
and writes the canonical NPZ schema (``schema.py``), so the metrics/plots layer
grades IsaacGym and MuJoCo identically.

This gives matched per-command rollouts for the sim2sim gap (spec §3.5 / §4).

Run env: the IsaacGym training env (``hsgym``). Pin the GPU with
``CUDA_VISIBLE_DEVICES``.

Usage:
    CUDA_VISIBLE_DEVICES=6 python -m holosoma.eval.ig_suite \
        --checkpoint logs/.../model_0050000.pt \
        --out-dir ~/eval/isaacgym_baseline \
        --n-per-command 100 --duration 12
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
from pathlib import Path

import numpy as np
from loguru import logger

# Command grid (mirrors run_suite.DEFAULT_COMMANDS; duplicated to avoid importing the
# MuJoCo runner here — `mujoco` isn't installed in the IsaacGym env).
DEFAULT_COMMANDS = [
    (0.0, 0.0, 0.0, "stand"),
    (0.25, 0.0, 0.0, "fwd025"),
    (0.5, 0.0, 0.0, "fwd05"),
    (0.8, 0.0, 0.0, "fwd08"),
    (1.0, 0.0, 0.0, "fwd10"),
    (-0.3, 0.0, 0.0, "back03"),
    (0.0, 0.3, 0.0, "left03"),
    (0.0, -0.3, 0.0, "right03"),
    (0.0, 0.0, 0.5, "yawL05"),
    (0.0, 0.0, -0.5, "yawR05"),
    (0.5, 0.2, 0.3, "mixed_a"),
    (0.8, -0.2, -0.3, "mixed_b"),
]


def _to_np(t):
    return t.detach().cpu().numpy().copy()


def main() -> None:
    ap = argparse.ArgumentParser(description="IsaacGym eval suite (matched to MuJoCo run_suite)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-per-command", type=int, default=100)
    ap.add_argument("--duration", type=float, default=12.0)
    ap.add_argument("--settle", type=float, default=0.5)
    ap.add_argument("--terrain", choices=["flat", "rough"], default="flat",
                    help="flat = plane (sim2sim gap); rough = Holosoma's real terrain term (robustness)")
    args = ap.parse_args()

    # IsaacGym MUST be imported before torch; do it first, before any holosoma import
    # that pulls torch transitively.
    import isaacgym  # noqa: F401

    # heavy imports deferred so --help is fast and isaacgym loads in order
    from holosoma.agents.base_algo.base_algo import BaseAlgo
    from holosoma.config_types.terrain import MeshType
    from holosoma.utils.eval_utils import CheckpointConfig, load_checkpoint, load_saved_experiment_config
    from holosoma.utils.helpers import get_class
    from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- load saved config, force flat terrain + num_envs ---
    ckpt_cfg = CheckpointConfig(checkpoint=args.checkpoint)
    saved_cfg, saved_wandb = load_saved_experiment_config(ckpt_cfg)
    cfg = saved_cfg.get_eval_config()
    if args.terrain == "flat":
        # flat plane for the sim2sim gap measurement (plane needs empty terrain_config)
        terrain_term = dataclasses.replace(cfg.terrain.terrain_term, mesh_type=MeshType.PLANE, terrain_config={})
        cfg = dataclasses.replace(cfg, terrain=dataclasses.replace(cfg.terrain, terrain_term=terrain_term))
    else:
        # rough: use Holosoma's REAL terrain term (terrain_locomotion_mix, trimesh rough 0.6 —
        # the term the policy trained on). This is the spec's "use Holosoma's existing terrain term".
        import holosoma.config_values.terrain as terr
        cfg = dataclasses.replace(cfg, terrain=terr.terrain_locomotion_mix)
    cfg = dataclasses.replace(cfg, training=dataclasses.replace(cfg.training, num_envs=args.n_per_command, headless=True))

    logger.info(f"Building IsaacGym env: num_envs={args.n_per_command}, terrain={args.terrain}")
    env, device, sim_app = setup_simulation_environment(cfg)

    # --- build algo + load policy ---
    algo: BaseAlgo = get_class(cfg.algo._target_)(
        device=device, env=env, config=cfg.algo.config, log_dir=str(out), multi_gpu_cfg=None
    )
    algo.setup()
    algo.load(str(load_checkpoint(args.checkpoint, str(out))))

    actor = algo.actor
    normalize = getattr(algo, "obs_normalization", False)
    normalizer = getattr(algo, "obs_normalizer", None)
    rl_env = algo.env  # wrapped env the actor steps (e.g. FastSACEnv)
    uenv = algo._unwrap_env()
    sim = uenv.simulator
    n = args.n_per_command
    n_steps = int(round(args.duration / float(uenv.dt)))
    settle_steps = int(round(args.settle / float(uenv.dt)))

    # static metadata (shared)
    robot_cfg = uenv.robot_config
    try:
        props = sim.gym.get_actor_rigid_body_properties(sim.envs[0], sim.robot_handles[0])
        total_mass = float(sum(p.mass for p in props))
    except Exception as e:  # noqa: BLE001
        logger.warning(f"total_mass via gym API failed: {e}")
        total_mass = None
    meta_base = {
        "dt": float(uenv.dt), "fps": round(1.0 / float(uenv.dt)),
        "sim_dt": float(uenv.sim_dt), "sim_fps": round(1.0 / float(uenv.sim_dt)),
        "control_decimation": sim.simulator_config.sim.control_decimation,
        "dof_names": list(sim.dof_names), "body_names": list(sim.body_names),
        "effort_limits": list(map(float, robot_cfg.dof_effort_limit_list)),
        "dof_pos_lower_limits": list(map(float, robot_cfg.dof_pos_lower_limit_list)),
        "dof_pos_upper_limits": list(map(float, robot_cfg.dof_pos_upper_limit_list)),
        "velocity_limits": list(map(float, robot_cfg.dof_vel_limit_list)),
        "total_mass": total_mass, "simulator": "isaacgym",
    }

    def _torque_term():
        for _n, term in uenv.action_manager.iter_terms():
            if hasattr(term, "torques"):
                return term
        raise RuntimeError("no torque term")

    vel_term = uenv.command_manager.get_state("locomotion_command")

    for vx, vy, vyaw, label in DEFAULT_COMMANDS:
        logger.info(f"Command {label}: vx={vx} vy={vy} vyaw={vyaw}  ({n} envs)")
        buf = {k: [] for k in (
            "dof_pos", "dof_vel", "torques", "torques_substep", "dof_vel_substep", "actions",
            "root_pos", "root_quat_xyzw", "root_lin_vel", "root_ang_vel",
            "body_pos_w", "body_quat_xyzw", "commanded_velocity",
        )}
        # Pin the command by collapsing the sampling ranges: any (re)sample — including
        # the reset-after-fall path — now yields exactly this command.
        vel_term.command_ranges = {
            "lin_vel_x": (vx, vx), "lin_vel_y": (vy, vy), "ang_vel_yaw": (vyaw, vyaw),
        }
        obs = rl_env.reset()
        uenv.set_is_evaluating()           # eval mode (no step resampling) + gait eval mode
        cmds = uenv.command_manager.commands
        cmds[:, 0], cmds[:, 1], cmds[:, 2] = vx, vy, vyaw   # overwrite the zeroed command
        for step in itertools.islice(itertools.count(), n_steps):
            nobs = normalizer(obs, update=False) if (normalize and normalizer is not None) else obs
            actions = actor(nobs)[0]
            obs, _, _, _ = rl_env.step(actions)

            term = _torque_term()
            root = sim.robot_root_states            # [n,13] pos,quat_xyzw,lin,ang
            buf["dof_pos"].append(_to_np(sim.dof_pos))
            buf["dof_vel"].append(_to_np(sim.dof_vel))
            buf["torques"].append(_to_np(term.torques))
            buf["torques_substep"].append(_to_np(term.torques_substep))
            buf["dof_vel_substep"].append(_to_np(term.dof_vel_substep))
            buf["actions"].append(_to_np(actions))
            buf["root_pos"].append(_to_np(root[:, 0:3]))
            buf["root_quat_xyzw"].append(_to_np(root[:, 3:7]))
            buf["root_lin_vel"].append(_to_np(root[:, 7:10]))
            buf["root_ang_vel"].append(_to_np(root[:, 10:13]))
            buf["body_pos_w"].append(_to_np(sim._rigid_body_pos))
            buf["body_quat_xyzw"].append(_to_np(sim._rigid_body_rot))
            buf["commanded_velocity"].append(_to_np(uenv.command_manager.commands[:, :3]))

        # stack -> [T, n, ...]; write one NPZ per env
        stacked = {k: np.stack(v, axis=0) for k, v in buf.items() if v}
        for e in range(n):
            arrays = {k: stacked[k][:, e] for k in stacked}
            meta = dict(meta_base, seed=e, label=label,
                        cmd_vx=vx, cmd_vy=vy, cmd_vyaw=vyaw)
            np.savez_compressed(out / f"{label}_{e:04d}.npz",
                                _metadata_json=np.array(json.dumps(meta)), **arrays)
        logger.info(f"  wrote {n} rollouts for {label}")

    if sim_app:
        close_simulation_app(sim_app)
    logger.info(f"Done. {len(DEFAULT_COMMANDS)*n} rollouts in {out}/")
    logger.info(f"Next: python -m holosoma.eval.analyze --rollout-dir {args.out_dir.replace(' ','')} "
                f"--compare-dir <mujoco_dir>")


if __name__ == "__main__":
    main()
