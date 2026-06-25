from __future__ import annotations

from typing import TYPE_CHECKING

from holosoma_inference.config.config_types.task import InputSource
from holosoma_inference.inputs.api.base import StateCommandProvider, VelCmdProvider
from holosoma_inference.inputs.impl.interface import InterfaceInput
from holosoma_inference.inputs.impl.keyboard import KEYBOARD_VELOCITY_LOCOMOTION, KeyboardInput
from holosoma_inference.inputs.impl.ros2 import Ros2Input

if TYPE_CHECKING:
    from holosoma_inference.policies.base import BasePolicy


def create_input(policy: BasePolicy, source: InputSource, role: str) -> VelCmdProvider | StateCommandProvider:
    """Create an input provider for the given source and role ("velocity" or "command")."""
    if not policy.use_joystick and source in ("interface", "joystick"):
        source = "keyboard"

    if source == "interface":
        return InterfaceInput(policy.interface)

    if source == "joystick":
        from holosoma_inference.inputs.impl.usb_joystick import UsbJoystickInput

        return UsbJoystickInput(device_index=policy.config.task.joystick_device)

    if source == "keyboard":
        vel_keys = KEYBOARD_VELOCITY_LOCOMOTION if role == "velocity" else None
        return KeyboardInput.create(velocity_keys=vel_keys)

    if source == "ros2":
        return Ros2Input(
            policy.config.task.ros_cmd_vel_topic,
            policy.config.task.ros_state_input_topic,
            vel_timeout=policy.config.task.ros_vel_timeout,
        )

    if source == "scripted":
        from holosoma_inference.inputs.impl.scripted import ScriptedInput

        t = policy.config.task
        return ScriptedInput(
            vx_range=t.scripted_vx_range,
            vy_range=t.scripted_vy_range,
            vyaw_range=t.scripted_vyaw_range,
            command_interval=t.scripted_command_interval,
        )

    raise ValueError(f"Unknown input source: {source}")
