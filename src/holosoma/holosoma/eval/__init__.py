"""Quantitative gait-evaluation framework for Holosoma locomotion policies.

Simulator-agnostic: MuJoCo (sim2sim) and IsaacGym rollouts write the same NPZ
schema (``schema.py``), graded by the same metrics (``metrics.py``).
"""

from holosoma.eval.metrics import SPEC, RolloutMetrics, compute_metrics
from holosoma.eval.schema import RolloutData, load_rollout, load_rollout_dir

__all__ = [
    "SPEC",
    "RolloutData",
    "RolloutMetrics",
    "compute_metrics",
    "load_rollout",
    "load_rollout_dir",
]
