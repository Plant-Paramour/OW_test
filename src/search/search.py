"""动作搜索算法 —— 枚举并评估所有 (source, target, ships) 组合，最大化占领价值。

流程:
1. 对每颗己方行星 (source)，枚举候选目标
2. 对每个 (source, target) 对，二分搜索最小攻占舰船数
3. 尝试多种舰船规模，运行 what-if 模拟
4. 计算每种组合的净价值，排序返回

支持单步搜索 (search_best_actions) 和浅层束搜索 (beam_search)。
"""

from dataclasses import dataclass

from ..engine.physics import travel_time
from ..engine.interception import aim_at, check_path_blocked
from .simulator import simulate_fleet_launch, find_min_ships_to_capture
from .valuation import compute_action_value, value_of_capture

SEARCH_MIN_SHIPS = 5


@dataclass
class ScoredAction:
    """评分后的候选动作。"""
    source_id: int
    target_id: int
    ships: int
    value: float
    capture_turn: int | None
    hold_until: int | None
    eta: int
    source_available: int
    source_at_risk: bool
    is_enemy: bool


def _try_ship_amounts(src, tgt, ship_amounts, state, ledger, timelines,
                      delays=None):
    """对给定的舰船数量列表逐一模拟，返回最佳结果。

    Args:
        ship_amounts: 要尝试的舰船数列表 (去重排序)
        delays: {ships: wait_turns} 映射，表示该舰船数需要攒几回合
        state, ledger, timelines: 世界模型状态

    Returns:
        (best_value, best_ships, best_outcome) 或 (None, None, None)
    """
    best_value = float("-inf")
    best_ships = None
    best_outcome = None

    for ships in ship_amounts:
        if ships < SEARCH_MIN_SHIPS:
            continue
        outcome = simulate_fleet_launch(src, tgt, ships, state, ledger, timelines)
        if outcome.blocked:
            continue
        if outcome.aim is None:
            continue

        delay = (delays or {}).get(ships, 0)

        if delay > 0 and outcome.capture_turn is not None:
            adj_capture = outcome.capture_turn + delay
            adj_hold = outcome.hold_until + delay
            is_enemy = tgt.owner not in (-1, state.player)
            value = value_of_capture(
                tgt, adj_capture, adj_hold,
                state.remaining_steps, ships, is_enemy,
                source_at_risk=outcome.source_at_risk,
            )
        else:
            value = compute_action_value(
                outcome, tgt, state.player, state.remaining_steps,
                ships, state.comet_ids,
            )

        if value > best_value:
            best_value = value
            best_ships = ships
            best_outcome = outcome

    return best_value, best_ships, best_outcome


def _evaluate_candidate(src, tgt, available, state, ledger, timelines):
    """评估单个 (source, target) 候选对的最佳舰船数。

    策略:
    1. 二分搜索找到最小攻占舰船数 min_ships
    2. 尝试 [min_ships, min_ships×1.3, min(min_ships×2, available), available]
    3. 返回最佳结果

    Args:
        src: 源行星
        tgt: 目标行星
        available: src 可用的舰船数上限
        state, ledger, timelines: 世界模型状态

    Returns:
        ScoredAction 或 None
    """
    if available < SEARCH_MIN_SHIPS:
        return None

    cap_available = available

    # Step 1: 找到最小攻占舰船数
    min_ships, _min_outcome = find_min_ships_to_capture(
        src, tgt, cap_available, state, ledger, timelines,
    )

    # Step 2: 构建尝试列表
    if min_ships is not None:
        candidates = [
            min_ships,
            min(int(min_ships * 1.3), cap_available),
            min(min_ships * 2, cap_available),
            cap_available,
        ]
    else:
        # 无法占领 → 仍尝试最大可用（可能用于削弱敌方）
        candidates = [cap_available]

    # 攒兵前瞻: 尝试等待 1~5 回合积累更多舰船后发射
    # 更多船 = 更快速度, 等待 + 快速飞行可能比立刻慢飞更早到达
    delays = {}
    src_prod = src.production
    if src_prod > 0:
        for wait in range(1, 6):
            future_ships = available + wait * src_prod
            if future_ships > cap_available:
                candidates.append(future_ships)
                delays[future_ships] = wait

    # 去重 + 排序
    candidates = sorted(set(c for c in candidates if c >= SEARCH_MIN_SHIPS))

    if not candidates:
        return None

    # Step 3: 逐一模拟
    best_value, best_ships, best_outcome = _try_ship_amounts(
        src, tgt, candidates, state, ledger, timelines, delays,
    )

    if best_ships is None:
        return None

    aim = best_outcome.aim
    eta = int(aim[1]) if aim else 999

    is_enemy = tgt.owner not in (-1, state.player)

    return ScoredAction(
        source_id=src.id,
        target_id=tgt.id,
        ships=best_ships,
        value=best_value,
        capture_turn=best_outcome.capture_turn,
        hold_until=best_outcome.hold_until,
        eta=eta,
        source_available=available,
        source_at_risk=best_outcome.source_at_risk,
        is_enemy=is_enemy,
    )


def search_best_actions(state, ledger, timelines, top_k=20,
                        include_comets=False, eta_progressive=True):
    """搜索最佳单步动作 —— 枚举所有候选，返回按价值排序的结果。

    Args:
        state: GameState
        ledger: 到达账本
        timelines: 行星时间线
        top_k: 返回前 K 个最佳动作
        include_comets: 是否包含彗星目标（默认跳过，由 warmup 处理）
        eta_progressive: 是否使用渐进式 ETA 过滤（前期限制远距离目标）

    Returns:
        list[ScoredAction]: 按 value 降序排列
    """
    results = []
    game_progress = state.step / max(1, state.episode_steps)

    for src in state.my_planets:
        src_timeline = timelines.get(src.id, {})
        keep_needed = src_timeline.get("keep_needed", 0)
        reserve = min(int(src.ships), int(keep_needed))
        available = max(0, int(src.ships) - reserve)

        if available < SEARCH_MIN_SHIPS:
            continue

        all_targets = state.enemy_planets + state.neutral_planets

        # 渐进式 ETA 过滤
        max_eta = 40.0 + game_progress * 85.0 if eta_progressive else 999.0

        for tgt in all_targets:
            if tgt.id == src.id:
                continue
            if not include_comets and tgt.id in state.comet_ids:
                continue

            # 快速 ETA 预筛选
            eta_quick = travel_time(
                src.x, src.y, src.radius,
                tgt.x, tgt.y, tgt.radius,
                max(SEARCH_MIN_SHIPS, available),
            )
            if eta_quick > max_eta:
                continue

            scored = _evaluate_candidate(
                src, tgt, available, state, ledger, timelines,
            )
            if scored is not None:
                results.append(scored)

    results.sort(key=lambda a: a.value, reverse=True)
    return results[:top_k]


def beam_search(state, ledger, timelines, beam_width=5, max_depth=2):
    """束搜索 —— 浅层多步前瞻，考虑占领后的连锁收益。

    搜索树:
    - 节点: 游戏状态快照 + 累计价值
    - 边: (source, target, ships) 动作
    - 深度限制: max_depth 层

    由于舰队到达需要 time，深层的模拟会涉及"时间推进"的复杂性。
    当前实现为 1 层深度的最佳动作搜索（等效于 search_best_actions）。

    未来可扩展为真正的多步搜索（推进 ledger 到舰队到达时刻）。

    Args:
        state: GameState
        ledger: 到达账本
        timelines: 行星时间线
        beam_width: 束宽度
        max_depth: 最大搜索深度

    Returns:
        list[ScoredAction]: 最佳动作序列
    """
    # 当前版本: 单层搜索（等价于 search_best_actions）
    # 多层束搜索需要处理"虚拟时间推进"的复杂语义
    # —— 在舰队到达前，源行星持续产兵、可能有新舰队到达等
    return search_best_actions(state, ledger, timelines, top_k=beam_width)
