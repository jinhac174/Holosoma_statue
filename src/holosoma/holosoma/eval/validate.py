"""Validate that the single-process MuJoCo eval runner is a faithful stand-in
for the live ZMQ sim2sim pipeline.

Three checks:
  1. Determinism      - same seed -> bit-identical trajectory (reproducible).
  2. Model fidelity   - the runner's compiled MuJoCo physics params match the
                        robot config / live-sim ground settings.
  3. ONNX wiring      - the reused LocomotionPolicy is actually executing the
                        supplied ONNX (independent onnxruntime reproduces the
                        action from the policy's own observation).

Usage:
    python -m holosoma.eval.validate --model-path <onnx>
"""

from __future__ import annotations

import argparse

import numpy as np
from loguru import logger

from holosoma.eval.mujoco_runner import (
    CommandSegment,
    DRParams,
    MujocoEvalRunner,
    RolloutConfig,
)


def check_determinism(runner: MujocoEvalRunner, tmp="/tmp/eval_validate") -> bool:
    rc = lambda: RolloutConfig(  # noqa: E731
        seed=12345, duration_s=4.0, settle_s=0.5,
        segments=[CommandSegment(0.6, 0.0, 0.0, 3.5)], dr=DRParams(friction=1.0),
    )
    a = np.load(runner.run_rollout(f"{tmp}/a.npz", rc()), allow_pickle=True)
    b = np.load(runner.run_rollout(f"{tmp}/b.npz", rc()), allow_pickle=True)
    dmax = float(np.max(np.abs(a["dof_pos"] - b["dof_pos"])))
    rmax = float(np.max(np.abs(a["root_pos"] - b["root_pos"])))
    ok = dmax == 0.0 and rmax == 0.0
    logger.info(f"  [{'PASS' if ok else 'FAIL'}] determinism: max dof_pos diff={dmax:.2e}, root_pos diff={rmax:.2e}")
    return ok


def check_model_fidelity(runner: MujocoEvalRunner) -> bool:
    import mujoco

    m = runner.model
    cfg = runner.robot_cfg
    ok = True

    def expect(name, cond, detail):
        nonlocal ok
        ok = ok and cond
        logger.info(f"  [{'PASS' if cond else 'FAIL'}] {name}: {detail}")

    expect("timestep", abs(m.opt.timestep - 1.0 / runner.physics_fps) < 1e-9,
           f"{m.opt.timestep:.2e}s (={runner.physics_fps}Hz)")
    expect("gravity", abs(m.opt.gravity[2] + 9.81) < 1e-6, f"{m.opt.gravity[2]:.3f}")
    expect("n actuators", m.nu == cfg.dof_obs_size, f"{m.nu} (expect {cfg.dof_obs_size})")

    floor_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "eval_floor")
    expect("floor friction", abs(m.geom_friction[floor_gid, 0] - 1.0) < 1e-6,
           f"{m.geom_friction[floor_gid, 0]:.3f} (training default 1.0)")
    expect("floor solref", abs(m.geom_solref[floor_gid, 0] - 0.001) < 1e-9,
           f"timeconst={m.geom_solref[floor_gid, 0]:.4f} (live sim 0.001)")

    # actuator force ranges should equal the config effort limits
    franges = m.actuator_forcerange[runner.actuator_ids, 1]
    eff = np.array(cfg.dof_effort_limit_list, dtype=float)
    expect("actuator force ranges == effort limits", np.allclose(franges, eff),
           f"max abs diff {np.max(np.abs(franges - eff)):.2e}")

    # armature should be present (from statue.xml) and match config
    arm = m.dof_armature[runner.dof_qvel_addrs]
    cfg_arm = np.array(cfg.dof_armature_list, dtype=float)
    expect("joint armature matches config", np.allclose(arm, cfg_arm, atol=1e-6),
           f"max abs diff {np.max(np.abs(arm - cfg_arm)):.2e}")

    # total mass sanity
    expect("total mass > 0", runner.base_total_mass > 0, f"{runner.base_total_mass:.2f} kg")
    return ok


def check_onnx_wiring(runner: MujocoEvalRunner, model_path: str) -> bool:
    """Capture one real (obs, action) from the policy and reproduce the action
    with an independent onnxruntime session on the same observation."""
    import onnxruntime as ort
    from holosoma_inference.inputs.api.commands import StateCommand, VelCmd

    # capture the obs the policy feeds to its network
    captured = {}
    orig = runner.policy.rl_inference

    def spy(robot_state_data):
        out = orig(robot_state_data)
        # the policy keeps its built observation in obs_buf_dict[group]
        grp = next(iter(runner.policy.obs_buf_dict))
        captured["obs"] = np.asarray(runner.policy.obs_buf_dict[grp], dtype=np.float32).copy()
        captured["action"] = np.asarray(out, dtype=np.float32).copy()
        return out

    runner.policy.rl_inference = spy
    rc = RolloutConfig(seed=7, duration_s=2.0, settle_s=0.5,
                       segments=[CommandSegment(0.6, 0.0, 0.0, 1.5)], dr=DRParams(friction=1.0))
    runner.run_rollout("/tmp/eval_validate/wire.npz", rc)
    runner.policy.rl_inference = orig

    if "obs" not in captured:
        logger.info("  [FAIL] onnx wiring: never captured an inference call")
        return False

    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out = sess.run(None, {in_name: captured["obs"].reshape(1, -1)})[0]
    # policy output is action*scale; the raw onnx output is the action — compare shapes + correlation
    pol = captured["action"].reshape(-1)
    raw = np.asarray(out).reshape(-1)
    same_dim = raw.size == pol.size or raw.size * 1 == pol.size
    # the policy scales by policy_action_scale; recover and compare
    scale = getattr(runner.policy, "policy_action_scale", 0.25)
    err = float(np.max(np.abs(raw * scale - pol))) if raw.size == pol.size else float("nan")
    ok = same_dim and (np.isfinite(err) and err < 1e-4)
    logger.info(f"  [{'PASS' if ok else 'FAIL'}] onnx wiring: independent onnxruntime reproduces "
                f"policy action (max diff {err:.2e}, obs dim {captured['obs'].size})")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate the MuJoCo eval runner")
    ap.add_argument("--model-path", required=True)
    args = ap.parse_args()

    logger.info("Building runner...")
    runner = MujocoEvalRunner(model_path=args.model_path)

    logger.info("\n=== 1. Determinism ===")
    d = check_determinism(runner)
    logger.info("\n=== 2. Model fidelity (runner physics == config / live sim) ===")
    f = check_model_fidelity(runner)
    logger.info("\n=== 3. ONNX wiring (policy runs the given model) ===")
    w = check_onnx_wiring(runner, args.model_path)

    logger.info(f"\n{'='*50}")
    logger.info(f"  VALIDATION: determinism={'OK' if d else 'X'}  "
                f"model={'OK' if f else 'X'}  onnx={'OK' if w else 'X'}")
    logger.info(f"{'='*50}")


if __name__ == "__main__":
    main()
