"""对手模块 —— 课程学习对手池。"""

from .base import Opponent, OpponentLike
from .sniper import SniperOpponent
from .heuristic import HeuristicOpponent
from .pool import OpponentPool
from .self_play import SelfPlayOpponent
from .lb1200 import LB1200Opponent
from .v4_hybrid import V4HybridOpponent

__all__ = [
    "Opponent",
    "OpponentLike",
    "SniperOpponent",
    "HeuristicOpponent",
    "OpponentPool",
    "SelfPlayOpponent",
    "LB1200Opponent",
    "V4HybridOpponent",
]
