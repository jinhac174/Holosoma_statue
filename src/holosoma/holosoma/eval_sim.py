#!/usr/bin/env python3
"""
Sim2Sim Evaluation Script

Launches sim (headless + video) and policy (scripted velocity commands) as
coordinated subprocesses, waits for the desired duration, then shuts both
down cleanly so the video is flushed to disk.

Usage:
    python src/holosoma/holosoma/eval_sim.py \\
        --model-path path/to/model.onnx \\
        --duration 30 \\
        --save-dir ~/eval_videos

All underlying sim/policy arguments can be forwarded after '--'.
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


SIM_PYTHON = "/home/jinhac174/.holosoma_deps/miniconda3/envs/hsmujoco/bin/python3"
POLICY_PYTHON = "/home/jinhac174/.holosoma_deps/miniconda3/envs/hsinference/bin/python3"
REPO_ROOT = Path(__file__).resolve().parents[3]  # holosoma/


def kill_ports():
    os.system("fuser -k 5555/tcp 5556/tcp 2>/dev/null")
    time.sleep(0.5)


def launch_sim(args):
    env = os.environ.copy()
    env["PATH"] = "/home/jinhac174/.holosoma_deps/miniconda3/envs/hsmujoco/bin:" + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = "/home/jinhac174/.holosoma_deps/miniconda3/envs/hsmujoco/lib"
    env["MUJOCO_GL"] = "egl"

    cmd = [
        SIM_PYTHON,
        str(REPO_ROOT / "src/holosoma/holosoma/run_sim.py"),
        "robot:statue-28dof",
        "--training.headless", "True",
        "--logger.video.enabled", "True",
        "--logger.video.save-dir", args.save_dir,
        "--logger.video.camera.tracking-body-name", "pelvis_link",
    ]
    return subprocess.Popen(cmd, env=env, cwd=str(REPO_ROOT))


def launch_policy(args):
    cmd = [
        POLICY_PYTHON,
        str(REPO_ROOT / "src/holosoma_inference/holosoma_inference/run_policy.py"),
        "inference:statue-28dof-loco",
        "--task.model-path", args.model_path,
        "--task.no-use-joystick",
        "--task.interface", "lo",
        "--task.auto-start",
        "--task.velocity-input", "scripted",
        "--task.state-input", "scripted",
        "--task.scripted-vx-range", str(args.vx_min), str(args.vx_max),
        "--task.scripted-vy-range", str(args.vy_min), str(args.vy_max),
        "--task.scripted-vyaw-range", str(args.vyaw_min), str(args.vyaw_max),
        "--task.scripted-command-interval", str(args.command_interval),
    ]
    return subprocess.Popen(cmd, cwd=str(REPO_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Sim2Sim eval: headless sim + scripted policy + video")
    parser.add_argument("--model-path", required=True, help="Path to ONNX model")
    parser.add_argument("--duration", type=float, default=30.0, help="Recording duration in seconds")
    parser.add_argument("--save-dir", default=str(Path.home() / "statue_eval"), help="Video output directory")
    parser.add_argument("--vx-min", type=float, default=0.3)
    parser.add_argument("--vx-max", type=float, default=0.8)
    parser.add_argument("--vy-min", type=float, default=-0.3)
    parser.add_argument("--vy-max", type=float, default=0.3)
    parser.add_argument("--vyaw-min", type=float, default=-0.5)
    parser.add_argument("--vyaw-max", type=float, default=0.5)
    parser.add_argument("--command-interval", type=float, default=10.0,
                        help="Seconds between random velocity resamples")
    args = parser.parse_args()

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    kill_ports()

    print(f"[eval_sim] Starting sim (headless + video → {args.save_dir})")
    sim_proc = launch_sim(args)

    print("[eval_sim] Waiting 6s for sim to initialize...")
    time.sleep(6)

    print("[eval_sim] Starting policy (scripted walk)")
    policy_proc = launch_policy(args)

    print(f"[eval_sim] Running for {args.duration}s...")
    try:
        time.sleep(args.duration)
    except KeyboardInterrupt:
        print("\n[eval_sim] Interrupted by user")

    print("[eval_sim] Stopping policy...")
    policy_proc.terminate()
    policy_proc.wait(timeout=5)

    print("[eval_sim] Stopping sim (flushing video)...")
    sim_proc.send_signal(signal.SIGTERM)

    # Wait up to 60s for ffmpeg to finish encoding
    for i in range(60):
        if sim_proc.poll() is not None:
            print(f"[eval_sim] Sim exited after {i+1}s")
            break
        time.sleep(1)
    else:
        print("[eval_sim] Sim took too long, force-killing")
        sim_proc.kill()

    print(f"[eval_sim] Done. Video saved to: {args.save_dir}/")
    for f in Path(args.save_dir).glob("episode_*.mp4"):
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
