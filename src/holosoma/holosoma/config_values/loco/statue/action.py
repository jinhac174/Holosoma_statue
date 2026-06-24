"""Locomotion action presets for the Statue robot."""

from holosoma.config_types.action import ActionManagerCfg, ActionTermCfg

statue_28dof_joint_pos = ActionManagerCfg(
    terms={
        "joint_control": ActionTermCfg(
            func="holosoma.managers.action.terms.joint_control:JointPositionActionTerm",
            params={},
            scale=1.0,
            clip=None,
        ),
    }
)

__all__ = ["statue_28dof_joint_pos"]
