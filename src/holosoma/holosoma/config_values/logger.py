from holosoma.config_types.logger import DisabledLoggerConfig, WandbLoggerConfig
from holosoma.config_types.video import VideoConfig

disabled = DisabledLoggerConfig()

wandb = WandbLoggerConfig(mode="online")

wandb_offline = WandbLoggerConfig(mode="offline")

wandb_fast = WandbLoggerConfig(mode="online", video=VideoConfig(interval=2))

DEFAULTS = {
    "disabled": disabled,
    "wandb": wandb,
    "wandb_offline": wandb_offline,
    "wandb-fast": wandb_fast,
}
