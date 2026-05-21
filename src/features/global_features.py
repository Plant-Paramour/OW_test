"""全局特征编码 —— 16 维特征向量。

编码整个游戏状态的宏观信息：回合进度、版图分布、经济军事对比、攻防态势。
不编码无意义的"天上总舰船数"等聚合量。
"""

import numpy as np


def build_global_features(state, ledger: dict) -> np.ndarray:
    """构建全局特征向量。

    Args:
        state: 全局游戏状态 (GameState)
        ledger: 到达账本 {planet_id: [(eta, owner, ships), ...]}

    Returns:
        np.ndarray shape (16,) 全部分数归一化到 [0, 1]
    """
    # ── 受攻击的我方行星数量 ──
    my_planets_under_attack = 0
    for pid in ledger:
        incoming = ledger[pid]
        has_enemy = any(own != state.player and own != -1 for _, own, _ in incoming)
        if has_enemy:
            is_mine = any(p.id == pid and p.owner == state.player for p in state.my_planets)
            if is_mine:
                my_planets_under_attack += 1

    # ── 最危急防御点压力比 ──
    max_defense_pressure = 0.0
    for p in state.my_planets:
        incoming = ledger.get(p.id, [])
        enemy_ships = sum(s for _, own, s in incoming if own != state.player and own != -1)
        if enemy_ships <= 0:
            continue
        enemy_etas = [eta for eta, own, _ in incoming if own != state.player and own != -1]
        first_eta = min(enemy_etas)
        defense_capacity = p.ships + p.production * first_eta
        pressure = enemy_ships / max(1.0, defense_capacity)
        if pressure > max_defense_pressure:
            max_defense_pressure = pressure
    max_defense_pressure = min(5.0, max_defense_pressure) / 5.0  # clamp to [0, 1]

    # ── 被瞄准的敌方行星数量 ──
    enemy_planets_targeted = 0
    for pid in ledger:
        incoming = ledger[pid]
        has_mine = any(own == state.player for _, own, _ in incoming)
        if has_mine:
            is_enemy = any(p.id == pid and p.owner not in (-1, state.player) for p in state.planets)
            if is_enemy:
                enemy_planets_targeted += 1

    # ── 驻军统计（仅行星上的舰船，不含飞行中舰队）──
    my_garrison = sum(p.ships for p in state.my_planets)
    enemy_garrison = sum(p.ships for p in state.enemy_planets)
    my_max_prod = max((p.production for p in state.my_planets), default=0)
    enemy_max_prod = max((p.production for p in state.enemy_planets), default=0)

    feats = np.array([
        # ── 回合进度 (2 维) ──
        state.step / 500.0,
        1.0 - state.step / 500.0,

        # ── 版图 (3 维) ──
        len(state.my_planets) / 48.0,
        len(state.enemy_planets) / 48.0,
        len(state.neutral_planets) / 48.0,

        # ── 经济 (2 维) ──
        state.my_total_production / 240.0,   # 48 * 5 = 240
        state.enemy_total_production / 240.0,

        # ── 军力 (3 维) ──
        min(1.0, state.my_total_ships / 19200.0),
        min(1.0, state.enemy_total_ships / 19200.0),
        (state.my_total_ships - state.enemy_total_ships)
            / max(1.0, state.my_total_ships + state.enemy_total_ships),

        # ── 攻防态势 & 特殊事件 (6 维) ──
        my_planets_under_attack / max(1, len(state.my_planets)),
        max_defense_pressure,
        enemy_planets_targeted / max(1, len(state.enemy_planets)),
        float(len(state.comet_ids) > 0),
        min(5.0, my_max_prod / max(1.0, enemy_max_prod)) / 5.0,
        min(5.0, my_garrison / max(1.0, enemy_garrison)) / 5.0,
    ], dtype=np.float32)

    return feats
