"""Orbit Wars 策略网络模块。

动作空间: (target_index: int, ship_ratio: float)
- target_index: 在决策矩阵中选择候选行（含 no-op）
- ship_ratio: Beta 分布, 全局 per-source-planet 决策
"""

from .action_head import (
    TargetHead,
    ShipAlphaHead,
    ShipBetaHead,
    sample_action,
    compute_ships_to_send,
)
from .value_head import ValueHead
from .model import (
    SelfEncoder,
    CandidateEncoder,
    GlobalEncoder,
    PolicyNetwork,
)

__all__ = [
    "TargetHead",
    "ShipAlphaHead",
    "ShipBetaHead",
    "ValueHead",
    "SelfEncoder",
    "CandidateEncoder",
    "GlobalEncoder",
    "PolicyNetwork",
    "sample_action",
    "compute_ships_to_send",
]
