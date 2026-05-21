"""动作价值计算 —— 评估占领一颗行星的净价值。

公式: value = productive_turns × production × swing_factor − ships_sent

- 中立行星: swing=1.0（仅我方获得产能）
- 敌方行星: swing=2.0（我方获得 + 敌方失去 = 净 swing）
- 彗星: 不使用此公式（自然价值太小，由 warmup 引导探索）
"""


def value_of_capture(target, capture_turn, hold_until, remaining_steps,
                     ships_sent, is_enemy, source_at_risk=False):
    """计算占领一颗行星的净价值。

    Args:
        target: 目标行星 (Planet)，需含 .production 和 .id 属性
        capture_turn: 首次占领回合数 (None = 未占领)
        hold_until: 最后持续拥有的回合数
        remaining_steps: 游戏剩余回合数
        ships_sent: 投入的舰船数
        is_enemy: 目标是否敌方行星（决定 swing 系数）
        source_at_risk: 源行星是否因发兵而陷入危险

    Returns:
        float: 净价值（可为负）
    """
    if capture_turn is None:
        penalty = ships_sent * 1.5 if source_at_risk else ships_sent
        return -float(penalty)

    productive_turns = hold_until - capture_turn + 1
    max_possible = remaining_steps - capture_turn
    productive_turns = min(productive_turns, max_possible)
    productive_turns = max(0, productive_turns)

    swing = 2.0 if is_enemy else 1.0
    gross_value = productive_turns * target.production * swing

    cost = ships_sent
    if source_at_risk:
        cost *= 1.5

    return float(gross_value - cost)


def is_comet_target(target, comet_ids):
    """判断目标是否为彗星（彗星不使用占领价值公式）。"""
    return target.id in comet_ids


def compute_action_value(outcome, target, player_id, remaining_steps,
                         ships_sent, comet_ids):
    """一站式动作价值计算 —— 从 LaunchOutcome 直接得到价值。

    对彗星返回 0.0（由 reward shaping warmup 引导探索）。

    Args:
        outcome: LaunchOutcome from simulate_fleet_launch()
        target: 目标行星
        player_id: 我方玩家 ID
        remaining_steps: 游戏剩余回合数
        ships_sent: 发射舰船数
        comet_ids: 彗星 ID 集合

    Returns:
        float: 动作净价值
    """
    if target.id in comet_ids:
        return 0.0

    if outcome.blocked or outcome.aim is None:
        return -float(ships_sent)

    if outcome.capture_turn is None:
        return -float(ships_sent)

    is_enemy = target.owner not in (-1, player_id)

    return value_of_capture(
        target, outcome.capture_turn, outcome.hold_until,
        remaining_steps, ships_sent, is_enemy,
        source_at_risk=outcome.source_at_risk,
    )
