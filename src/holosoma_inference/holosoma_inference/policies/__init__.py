from .base import BasePolicy
from .dual_mode import DualModePolicy
from .locomotion import LocomotionPolicy

__all__ = ["BasePolicy", "DualModePolicy", "LocomotionPolicy", "WholeBodyTrackingPolicy"]


def __getattr__(name):
    # Lazy import: WBT needs pinocchio, which isn't installed in every env
    # (e.g. the MuJoCo eval env). Only import it on demand.
    if name == "WholeBodyTrackingPolicy":
        from .wbt import WholeBodyTrackingPolicy

        return WholeBodyTrackingPolicy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
