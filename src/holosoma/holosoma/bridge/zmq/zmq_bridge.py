"""ZMQ bridge for sim2sim with custom robots (e.g., Statue)."""

import numpy as np
import zmq
from loguru import logger

from holosoma.bridge.base.basic_sdk2py_bridge import BasicSdk2Bridge

_STATE_PORT = 5555  # sim PUSH -> policy PULL
_CMD_PORT = 5556    # policy PUSH -> sim PULL


class ZmqBridge(BasicSdk2Bridge):
    """ZMQ-based bridge for robots without a hardware DDS SDK."""

    def _init_sdk_components(self):
        self._ctx = zmq.Context()

        # Sim side binds; policy side connects.
        self._state_sock = self._ctx.socket(zmq.PUSH)
        self._state_sock.setsockopt(zmq.SNDHWM, 1)  # drop stale states
        self._state_sock.bind(f"tcp://*:{_STATE_PORT}")

        self._cmd_sock = self._ctx.socket(zmq.PULL)
        self._cmd_sock.setsockopt(zmq.RCVHWM, 1)
        self._cmd_sock.bind(f"tcp://*:{_CMD_PORT}")

        self._last_cmd = None
        logger.info(f"ZMQ bridge initialized: state→{_STATE_PORT}, cmd←{_CMD_PORT}")

    def low_cmd_handler(self, msg=None):
        try:
            raw = self._cmd_sock.recv(zmq.NOBLOCK)
            self._last_cmd = np.frombuffer(raw, dtype=np.float32).copy()
        except zmq.Again:
            pass

    @staticmethod
    def _world_to_body_angvel(quat_wxyz: np.ndarray, ang_vel_world: np.ndarray) -> np.ndarray:
        """Rotate angular velocity from world frame to body frame.

        MuJoCo's freejoint qvel[3:6] is in world frame; the policy expects body frame
        (matching what a physical IMU on the robot would measure).
        """
        q_w = quat_wxyz[0]
        q_xyz = quat_wxyz[1:]
        v = ang_vel_world
        a = v * (2.0 * q_w**2 - 1.0)
        b = np.cross(q_xyz, v) * (q_w * 2.0)
        c = q_xyz * (np.dot(q_xyz, v) * 2.0)
        return (a - b + c).astype(np.float32)

    def publish_low_state(self):
        positions, velocities, _ = self._get_dof_states()
        quaternion, gyro, _ = self._get_base_imu_data()

        # quaternion is already [w, x, y, z] (converted in base class)
        quat_np = quaternion.detach().cpu().numpy().astype(np.float32)
        # gyro from MuJoCo freejoint qvel is in world frame; rotate to body frame so it
        # matches what the policy saw during training (IMU = body-frame angular velocity).
        omega_np = self._world_to_body_angvel(quat_np, gyro.detach().cpu().numpy())
        pos_np = positions.astype(np.float32)
        vel_np = velocities.astype(np.float32)

        # Message layout: [quat(4), omega(3), motor_q(N), motor_dq(N)]
        msg = np.concatenate([quat_np, omega_np, pos_np, vel_np])
        try:
            self._state_sock.send(msg.tobytes(), zmq.NOBLOCK)
        except zmq.Again:
            pass

    def compute_torques(self):
        if self._last_cmd is None:
            return self.torques

        n = self.num_motor
        # Command layout: [q_target(N), dq_target(N), tau_ff(N), kp(N), kd(N)]
        q_target = self._last_cmd[0:n]
        dq_target = self._last_cmd[n : 2 * n]
        tau_ff = self._last_cmd[2 * n : 3 * n]
        kp = self._last_cmd[3 * n : 4 * n]
        kd = self._last_cmd[4 * n : 5 * n]

        return self._compute_pd_torques(tau_ff, kp, kd, q_target, dq_target)
