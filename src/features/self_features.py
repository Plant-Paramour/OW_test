"""源行星特征编码 —— 21 维特征向量。

每个特征描述一颗己方行星的状态，包括其生产、防御、受威胁程度和全局地位。
"""

import numpy as np

from ..engine.constants import BOARD_SIZE, CENTER_X, CENTER_Y
from ..engine.physics import dist, travel_time, segment_hits_sun, fleet_speed
from ..engine.prediction import is_static_planet


def _dist_to_sun(planet) -> float:
    return dist(planet.x, planet.y, CENTER_X, CENTER_Y)


def _find_nearest_eta(from_planets: list, to_planet, ships_override: int = None) -> float:
    """计算 from_planets 中各行星到 to_planet 的最短 ETA。"""
    best = 1e9
    for p in from_planets:
        if p.id == to_planet.id:
            continue
        s = ships_override if ships_override is not None else max(1, p.ships)
        eta = travel_time(p.x, p.y, p.radius, to_planet.x, to_planet.y, to_planet.radius, s)
        if eta < best:
            best = eta
    return best if best < 1e8 else 999


def _sun_block_to_nearest(from_planet, to_planets: list) -> bool:
    """检查从 from_planet 到最近行星的路径是否被太阳阻挡。"""
    nearest = None
    nearest_dist = 1e9
    for p in to_planets:
        if p.id == from_planet.id:
            continue
        d = dist(from_planet.x, from_planet.y, p.x, p.y)
        if d < nearest_dist:
            nearest_dist = d
            nearest = p
    if nearest is None:
        return False
    return segment_hits_sun(from_planet.x, from_planet.y, nearest.x, nearest.y)


def build_self_features(src, state, ledger: dict, timeline: dict) -> np.ndarray:
    """构建源行星的特征向量。

    Args:
        src: 己方行星 (Planet 命名元组)
        state: 全局游戏状态 (GameState)
        ledger: 到达账本 {planet_id: [(eta, owner, ships), ...]}
        timeline: 行星时间线 {planet_id: dict}

    Returns:
        np.ndarray shape (21,) 全部分数归一化到 [0, 1]
    """
    t = timeline.get(src.id, {})
    incoming = ledger.get(src.id, [])

    # ── 飞行中舰队分析 ──
    my_incoming = sum(s for eta, own, s in incoming if own == state.player)
    enemy_incoming = sum(s for eta, own, s in incoming if own != state.player and own != -1)
    neutral_incoming = sum(s for eta, own, s in incoming if own == -1)

    enemy_etas = [eta for eta, own, s in incoming if own != state.player and own != -1]
    first_enemy_eta = min(enemy_etas) if enemy_etas else 999

    # ── ETA 可达性 ──
    min_enemy_eta_to_me = _find_nearest_eta(state.enemy_planets, src)
    min_neutral_eta_to_me = _find_nearest_eta(state.neutral_planets, src)

    # ── 防御风险：keep_needed / current_ships（锁定比例）──
    keep_needed = t.get("keep_needed", 0) if t else 0
    defense_risk = min(1.0, keep_needed / max(1.0, src.ships))

    # ── 太阳阻挡检查 ──
    has_sun_block_enemy = _sun_block_to_nearest(src, state.enemy_planets) if state.enemy_planets else False
    has_sun_block_neutral = _sun_block_to_nearest(src, state.neutral_planets) if state.neutral_planets else False

    feats = np.array([
        # ── 基本属性 (6 维) ──
        src.x / BOARD_SIZE,
        src.y / BOARD_SIZE,
        min(1.0, src.radius / 5.0),
        min(src.ships, 400) / 400.0,
        min(1.0, src.production / 5.0),
        float(not is_static_planet(src)),

        # ── 空间特征 (3 维) —— ETA 替代距离 ──
        min(1.0, _dist_to_sun(src) / 70.0),
        1.0 / (1.0 + min_enemy_eta_to_me),
        1.0 / (1.0 + min_neutral_eta_to_me),

        # ── 防御态势 (5 维) ──
        defense_risk,
        float(has_sun_block_enemy),
        float(has_sun_block_neutral),
        min(1.0, enemy_incoming / max(1.0, src.ships + src.production * max(1, first_enemy_eta))),
        1.0 / (1.0 + first_enemy_eta),

        # ── 增援状态 (3 维) ──
        min(1.0, my_incoming / max(1.0, src.ships)),
        min(1.0, neutral_incoming / max(1.0, src.ships)),
        float(len(incoming) > 0),

        # ── 战略特征 (4 维) ──
        len(state.my_planets) / 48.0,
        len(state.enemy_planets) / 48.0,
        min(1.0, state.my_total_ships / 19200.0),
        min(1.0, state.enemy_total_ships / 19200.0),
    ], dtype=np.float32)

    return feats
