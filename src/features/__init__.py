"""Orbit Wars 特征工程模块。

将世界模型的输出编码为神经网络可消费的特征向量。
维度规格: self(21) + candidate(29) + global(16) = 66 维总计。
"""

from .self_features import build_self_features
from .candidate_features import build_candidate_features, INVALID_CANDIDATE_VECTOR
from .global_features import build_global_features
from .builder import build_decision_matrix, DecisionRow

__all__ = [
    "build_self_features",
    "build_candidate_features",
    "INVALID_CANDIDATE_VECTOR",
    "build_global_features",
    "build_decision_matrix",
    "DecisionRow",
]
