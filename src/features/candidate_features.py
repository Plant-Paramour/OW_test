"""候选目标特征编码 —— 29 维特征向量。

编码一个 (source_planet → target_planet) 决策对的所有博弈相关信息。
第 29 维 (min_eta_from_tgt_to_enemy) 编码目标作为进攻跳板的价值。
"""

import math
import numpy as np

from ..engine.constants import BOARD_SIZE, CENTER_X, CENTER_Y, SIM_HORIZON
from ..engine.physics import dist, segment_hits_sun, travel_time
from ..engine.prediction import is_static_planet, comet_remaining_life
from ..engine.interception import aim_at

# 无效候选的标记向量（全零，mask=False 时使用）
INVALID_CANDIDATE_VECTOR = np.zeros(29, dtype=np.float32)


def _dist_to_sun(planet) -> float:
    return dist(planet.x, planet.y, CENTER_X, CENTER_Y)


def build_candidate_features(src, tgt, state, ships_to_send: int,
                             ledger: dict, tgt_timeline: dict,
                             aim_result=None) -> np.ndarray:
    """构建候选目标的特征向量。

    Args:
        src: 源行星 (己方 Planet)
        tgt: 候选目标行星 (Planet)
        state: 全局游戏状态 (GameState)
        ships_to_send: 拟发送的舰船数
        ledger: 到达账本 {planet_id: [(eta, owner, ships), ...]}
        tgt_timeline: 目标行星的时间线 dict
        aim_result: 可选, 预计算的 aim_at 结果 (angle, eta, pred_x, pred_y),
            避免重复计算拦截求解

    Returns:
        np.ndarray shape (29,) 或 INVALID_CANDIDATE_VECTOR (全零, 29维)
    """
    # ── 拦截求解 ──
    if aim_result is not None:
        angle, eta, pred_x, pred_y = aim_result
    else:
        aim = aim_at(
            src, tgt, max(1, ships_to_send),
            state.initial_by_id, state.angular_velocity,
            state.comets, state.comet_ids,
        )
        if aim is None:
            return INVALID_CANDIDATE_VECTOR
        angle, eta, pred_x, pred_y = aim

    # ── 飞行中舰队对目标的影响 ──
    incoming = ledger.get(tgt.id, [])
    enemy_to_tgt = sum(s for eta2, own, s in incoming if own not in (-1, state.player))
    my_to_tgt = sum(s for eta2, own, s in incoming if own == state.player)
    hostile_etas = [eta2 for eta2, own, s in incoming if own not in (-1, state.player)]
    first_hostile_eta = min(hostile_etas) if hostile_etas else 999
    enemy_etas_to_tgt = [eta2 for eta2, own, s in incoming if own != state.player and own != -1]
    eta_enemy_min = min(enemy_etas_to_tgt) if enemy_etas_to_tgt else 999

    # ── 到达时刻状态 ──
    from ..world.combat import state_at_timeline
    owner_on_arrival, garrison_on_arrival = state_at_timeline(tgt_timeline, eta)
    owner_after, garrison_after = state_at_timeline(tgt_timeline, eta + 1)

    # ── 防御缺口（仅对己方行星有意义）──
    keep_needed = tgt_timeline.get("keep_needed", 0)
    if tgt.owner == state.player:
        defense_shortfall = max(0.0, keep_needed - tgt.ships) / max(1.0, tgt.ships)
    else:
        defense_shortfall = 0.0

    # ── 承诺比：含本次拟发送的舰船 ──
    committed_ratio = min(1.0, (my_to_tgt + ships_to_send) / max(1.0, tgt.ships))

    # ── 直达路径太阳阻挡 ──
    hits_sun = segment_hits_sun(src.x, src.y, tgt.x, tgt.y)

    # ── 彗星剩余生命 ──
    comet_life = comet_remaining_life(tgt.id, state.comets) if tgt.id in state.comet_ids else 0

    # ── 生产性回合 ──
    if tgt.id in state.comet_ids:
        productive_turns = comet_life
    else:
        productive_turns = state.remaining_steps

    # ── 战略目标标记 ──
    is_strategic = float(tgt.production >= 3 or tgt.id in state.comet_ids)

    # ── 削弱敌方比例 ──
    if tgt.owner not in (-1, state.player) and state.enemy_total_production > 0:
        enemy_weakened = min(1.0, tgt.production / max(1.0, state.enemy_total_production))
    else:
        enemy_weakened = 0.0

    # ── 从目标到敌方的进攻跳板价值 ──
    enemy_etas_from_tgt = [
        travel_time(tgt.x, tgt.y, tgt.radius, e.x, e.y, e.radius, max(1, tgt.ships))
        for e in state.enemy_planets if e.id != tgt.id
    ]
    min_eta_from_tgt_to_enemy = min(enemy_etas_from_tgt) if enemy_etas_from_tgt else 999

    feats = np.array([
        # ── 所有权 one-hot (4 维) ──
        float(tgt.owner == -1),
        float(tgt.owner == state.player),
        float(tgt.owner not in (-1, state.player)),
        float(tgt.id in state.comet_ids),

        # ── 目标属性 (5 维) ──
        min(1.0, tgt.production / 5.0),
        min(tgt.ships, 400) / 400.0,
        min(1.0, tgt.radius / 5.0),
        float(not is_static_planet(tgt)),
        min(1.0, comet_life / 500.0),

        # ── 空间关系 (4 维) ──
        math.cos(angle),
        math.sin(angle),
        float(hits_sun),
        min(1.0, _dist_to_sun(tgt) / 70.0),

        # ── 时序特征 (5 维) ──
        min(1.0, eta / 100.0),
        min(1.0, eta_enemy_min / 100.0),
        min(1.0, garrison_on_arrival / 400.0),
        min(1.0, defense_shortfall),
        committed_ratio,

        # ── 飞行中舰队对目标的影响 (4 维) ──
        min(1.0, enemy_to_tgt / max(1.0, tgt.ships + tgt.production * max(1, first_hostile_eta))),
        min(1.0, my_to_tgt / max(1.0, tgt.ships)),
        1.0 / (1.0 + first_hostile_eta),
        float(first_hostile_eta < eta),

        # ── 到达后状态 (3 维) ──
        min(1.0, garrison_after / 400.0),
        float(owner_after == state.player),
        float(owner_after != tgt.owner),

        # ── 战略价值 (4 维) ──
        min(1.0, productive_turns / 500.0),
        is_strategic,
        enemy_weakened,
        1.0 / (1.0 + min_eta_from_tgt_to_enemy),
    ], dtype=np.float32)

    return feats
