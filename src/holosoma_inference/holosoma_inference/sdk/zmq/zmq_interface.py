"""ZMQ robot interface for sim2sim with custom robots (e.g., Statue)."""

import numpy as np
import zmq

from holosoma_inference.config.config_types import RobotConfig
from holosoma_inference.sdk.base.base_interface import BaseInterface

_STATE_PORT = 5555  # sim PUSH -> policy PULL
_CMD_PORT = 5556    # policy PUSH -> sim PULL


class ZmqInterface(BaseInterface):
    """Interface for custom robots using ZMQ for sim2sim communication."""

    def __init__(self, robot_config: RobotConfig, domain_id=0, interface_str=None, use_joystick=True):
        super().__init__(robot_config, domain_id, interface_str, use_joystick)
        self._kp_level = 1.0
        self._kd_level = 1.0
        self._init_zmq()

    def _init_zmq(self):
        # Always connect to loopback for sim2sim.
        host = "127.0.0.1"

        self._ctx = zmq.Context()

        # Policy side connects; sim side binds.
        self._state_sock = self._ctx.socket(zmq.PULL)
        self._state_sock.setsockopt(zmq.CONFLATE, 1)  # keep only latest state
        self._state_sock.connect(f"tcp://{host}:{_STATE_PORT}")

        self._cmd_sock = self._ctx.socket(zmq.PUSH)
        self._cmd_sock.setsockopt(zmq.SNDHWM, 1)
        self._cmd_sock.connect(f"tcp://{host}:{_CMD_PORT}")

    def get_low_state(self) -> np.ndarray:
        """Receive state from sim and return as [base_pos(3), quat(4), joint_pos(N), lin_vel(3), ang_vel(3), joint_vel(N)]."""
        raw = self._state_sock.recv()
        arr = np.frombuffer(raw, dtype=np.float32).copy()

        n = self.robot_config.num_joints
        # State layout from sim: [quat(4), omega(3), motor_q(N), motor_dq(N)]
        quat = arr[0:4]          # [w, x, y, z]
        omega = arr[4:7]         # angular velocity
        motor_q = arr[7 : 7 + n]
        motor_dq = arr[7 + n : 7 + 2 * n]

        base_pos = np.zeros(3, dtype=np.float32)
        base_lin_vel = np.zeros(3, dtype=np.float32)

        return np.concatenate([base_pos, quat, motor_q, base_lin_vel, omega, motor_dq]).reshape(1, -1)

    def send_low_command(
        self,
        cmd_q: np.ndarray,
        cmd_dq: np.ndarray,
        cmd_tau: np.ndarray,
        dof_pos_latest: np.ndarray = None,
        kp_override: np.ndarray = None,
        kd_override: np.ndarray = None,
    ):
        """Send joint command to sim."""
        kp = np.array(kp_override if kp_override is not None else self.robot_config.motor_kp, dtype=np.float32)
        kd = np.array(kd_override if kd_override is not None else self.robot_config.motor_kd, dtype=np.float32)
        kp = (kp * self._kp_level).astype(np.float32)
        kd = (kd * self._kd_level).astype(np.float32)

        # Command layout: [q_target(N), dq_target(N), tau_ff(N), kp(N), kd(N)]
        msg = np.concatenate([
            np.asarray(cmd_q, dtype=np.float32),
            np.asarray(cmd_dq, dtype=np.float32),
            np.asarray(cmd_tau, dtype=np.float32),
            kp,
            kd,
        ])
        try:
            self._cmd_sock.send(msg.tobytes(), zmq.NOBLOCK)
        except zmq.Again:
            pass

    def get_joystick_msg(self):
        return None

    def get_joystick_key(self, wc_msg=None):
        return None

    @property
    def kp_level(self):
        return self._kp_level

    @kp_level.setter
    def kp_level(self, value):
        self._kp_level = value

    @property
    def kd_level(self):
        return self._kd_level

    @kd_level.setter
    def kd_level(self, value):
        self._kd_level = value
