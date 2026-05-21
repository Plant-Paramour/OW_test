"""决策矩阵组装器 —— 将世界模型输出组装为神经网络可消费的决策矩阵。

为每颗己方行星评估 Top-K 候选目标，构建 (source × candidate) 特征矩阵。
"""

import math
from dataclasses import dataclass

import numpy as np

from ..engine.constants import SIM_HORIZON
from ..engine.physics import travel_time
from ..engine.interception import aim_at, check_path_blocked
from ..world.fleet_tracker import build_arrival_ledger
from ..world.combat import simulate_planet_timeline
from .self_features import build_self_features
from .candidate_features import build_candidate_features, INVALID_CANDIDATE_VECTOR
from .global_features import build_global_features

MIN_FLEET = 12


@dataclass
class DecisionRow:
    """一个 (source_planet, candidate_target) 决策对。"""
    source_id: int
    candidate_id: int        # -1 表示 no-op
    self_feat: np.ndarray    # (21,)
    cand_feat: np.ndarray    # (29,)
    global_feat: np.ndarray  # (16,)
    mask: bool               # 是否有效
    action_info: dict        # {angle, ships, eta, ...} 用于执行


def _score_candidate(src, tgt, ships: int) -> float:
    """候选目标评分（用于 Top-K 排序）。

    综合考虑产量价值和到达时间。
    """
    eta = travel_time(src.x, src.y, src.radius, tgt.x, tgt.y, tgt.radius, max(1, ships))
    eta = max(1, eta)
    return tgt.production / (eta + 1.0) + tgt.ships * 0.001


def _select_top_candidates(src, state, k: int) -> list:
    """从所有非己方行星中选择 Top-K 候选目标。

    早期游戏使用渐进式 ETA 过滤：随游戏推进逐步放宽航程限制，
    防止智能体前期向极远行星派遣舰队（"飞跃银河"）。

    Args:
        src: 源行星（己方 Planet）
        state: 全局游戏状态
        k: 候选数量上限

    Returns:
        按评分降序排列的候选行星列表
    """
    candidates = []
    ships = max(MIN_FLEET, src.ships // 2)
    all_targets = state.enemy_planets + state.neutral_planets

    # 渐进式 ETA 过滤：前期只考虑近处目标，随游戏推进逐步放宽
    game_progress = state.step / max(1, state.episode_steps)
    max_eta = 40.0 + game_progress * 85.0  # 开局 40 回合航程限制，终局 100（无限制）

    for tgt in all_targets:
        if tgt.id == src.id:
            continue
        eta = travel_time(src.x, src.y, src.radius, tgt.x, tgt.y, tgt.radius, max(1, ships))
        if eta > max_eta:
            continue
        score = _score_candidate(src, tgt, ships)
        candidates.append((score, tgt))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return [tgt for _, tgt in candidates[:k]]


def _estimate_ships_to_send(src, tgt, available: int, tgt_timeline: dict,
                            state) -> int:
    """估算发送到目标的合理舰船数，用于特征编码。

    对敌方/中立目标：发送足够攻占的兵力
    对己方目标：增援所需防御缺口
    """
    if tgt.owner not in (-1, state.player):
        garrison_est = max(0, int(tgt_timeline.get("ships_at", {}).get(0, tgt.ships)))
        needed = garrison_est + 10
    elif tgt.owner == -1:
        needed = int(tgt.ships) + 10
    else:
        keep_needed = tgt_timeline.get("keep_needed", 0)
        needed = max(0, keep_needed - int(tgt.ships))

    return max(MIN_FLEET, min(available, needed))


def build_decision_matrix(state, candidate_count: int = 20,
                          sim_horizon: int = SIM_HORIZON) -> list:
    """为每颗己方行星构建 (source × top-K candidate) 决策矩阵。

    内部流程：
    1. 构建到达账本 → ledger
    2. 为每颗行星模拟时间线 → timelines
    3. 构建全局特征（一次）
    4. 对每颗己方行星：
       a. 构建自身特征
       b. 选择 Top-K 候选目标
       c. 为每个候选构建候选特征
       d. 生成 DecisionRow

    Args:
        state: 全局游戏状态 (GameState)
        candidate_count: 每个源行星的最大候选目标数
        sim_horizon: 时间线模拟前瞻回合数

    Returns:
        list[DecisionRow] —— 可直接送入策略网络
    """
    ledger = build_arrival_ledger(state.fleets, state.planets)

    timelines = {}
    for planet in state.planets:
        arrivals = ledger.get(planet.id, [])
        timelines[planet.id] = simulate_planet_timeline(
            planet, arrivals, state.player, sim_horizon
        )

    global_feat = build_global_features(state, ledger)
    rows = []

    for src in state.my_planets:
        self_feat = build_self_features(src, state, ledger, timelines)

        keep_needed = timelines.get(src.id, {}).get("keep_needed", 0)
        reserve = min(int(src.ships), int(keep_needed))
        available = max(0, int(src.ships) - reserve)

        # no-op 行：始终排在第一位 (target_index=0)
        noop_action = {"angle": 0.0, "ships": 0, "eta": 999, "available": available}
        rows.append(DecisionRow(
            source_id=src.id, candidate_id=-1,
            self_feat=self_feat,
            cand_feat=np.zeros_like(INVALID_CANDIDATE_VECTOR),
            global_feat=global_feat, mask=True,
            action_info=noop_action,
        ))

        candidates = _select_top_candidates(src, state, candidate_count)

        if not candidates:
            continue

        for tgt in candidates:
            ships_est = _estimate_ships_to_send(
                src, tgt, max(MIN_FLEET, available), timelines[tgt.id], state
            )
            # 预计算 aim_at，供特征编码和 action_info 共用
            aim = aim_at(
                src, tgt, max(1, ships_est),
                state.initial_by_id, state.angular_velocity,
                state.comets, state.comet_ids,
            )
            cand_feat = build_candidate_features(
                src, tgt, state, ships_est, ledger, timelines[tgt.id],
                aim_result=aim,
            )

            valid = aim is not None
            if valid:
                blocked, blocker_id, _ = check_path_blocked(
                    src, tgt, max(1, ships_est),
                    aim[0], float(aim[1]), state,
                )
                if blocked:
                    valid = False
            action_info = {}
            if valid:
                action_info = {
                    "angle": aim[0], "ships": ships_est,
                    "eta": aim[1], "target_id": tgt.id,
                    "available": available,
                }

            rows.append(DecisionRow(
                source_id=src.id, candidate_id=tgt.id,
                self_feat=self_feat, cand_feat=cand_feat,
                global_feat=global_feat, mask=valid,
                action_info=action_info,
            ))

    return rows
