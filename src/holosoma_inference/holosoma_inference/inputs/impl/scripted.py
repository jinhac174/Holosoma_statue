"""Scripted velocity input for automated evaluation.

Samples random velocity commands at a fixed interval and emits WALK on startup.
Velocity is drawn uniformly from configurable ranges.
"""

from __future__ import annotations

import random
import threading
import time

from holosoma_inference.inputs.api.commands import StateCommand, VelCmd
from holosoma_inference.inputs.api.base import InputProvider


class ScriptedInput(InputProvider):
    """Randomly samples walk commands at a fixed interval.

    Emits ``StateCommand.WALK`` on the first ``poll_commands`` call so the
    policy enters walk mode without any key press.  Velocity is resampled
    every ``command_interval`` seconds.
    """

    def __init__(
        self,
        vx_range: tuple[float, float] = (0.3, 0.8),
        vy_range: tuple[float, float] = (-0.3, 0.3),
        vyaw_range: tuple[float, float] = (-0.5, 0.5),
        command_interval: float = 3.0,
    ) -> None:
        self._vx_range = vx_range
        self._vy_range = vy_range
        self._vyaw_range = vyaw_range
        self._command_interval = command_interval

        self._lock = threading.Lock()
        self._current_vel = VelCmd(lin_vel=(0.0, 0.0), ang_vel=0.0)
        self._pending_commands: list[StateCommand] = []
        self._timer: threading.Timer | None = None
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        # Queue WALK command so policy enters walk mode immediately
        with self._lock:
            self._pending_commands.append(StateCommand.WALK)
        self._resample()

    def _resample(self) -> None:
        vx = random.uniform(*self._vx_range)
        vy = random.uniform(*self._vy_range)
        vyaw = random.uniform(*self._vyaw_range)
        with self._lock:
            self._current_vel = VelCmd(lin_vel=(vx, vy), ang_vel=vyaw)
        self._timer = threading.Timer(self._command_interval, self._resample)
        self._timer.daemon = True
        self._timer.start()

    def poll_velocity(self) -> VelCmd | None:
        with self._lock:
            return self._current_vel

    def zero(self) -> None:
        with self._lock:
            self._current_vel = VelCmd(lin_vel=(0.0, 0.0), ang_vel=0.0)

    def poll_commands(self) -> list[StateCommand]:
        with self._lock:
            cmds = self._pending_commands[:]
            self._pending_commands.clear()
        return cmds

    def stop(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
