"""Orbit Wars 奖励塑形 —— 基于统一对称势能函数 Φ 的未来价值优先奖励。

核心设计：
  Φ(s, player_id) = 我方领土现值 − 敌方领土现值（零和）
  reward = Φ(s') − Φ(s) + Δ舰船优势 + warmup + 终局

零和的优势：敌人扩张自动产生惩罚信号，无需单独的"惩罚逻辑"。
Φ 定义可替换——同一套 ΔΦ 框架下，换一个 Φ 即可从 1v1 扩展到多人乱斗。

来源：实施规划.md §6 (2026-05-19 对称 Φ 改版)
"""

from ..engine.prediction import comet_remaining_life


# ── 配置常量 ──

ALPHA = 1.0                    # 领土势能微调系数（默认 1.0，对称 Φ 已足够）
COMET_WARMUP_MAX = 1.5         # 彗星占领最大额外加分
TARGET_UTILIZATION = 0.20      # 目标：20% 的 episode 中有彗星→敌星攻击


# ═══════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════

def planet_remaining_life(planet, state) -> int:
    """行星还能产出多少回合。

    彗星用实际剩余寿命（从 paths[] 计算），
    普通行星用游戏剩余回合。
    """
    if planet.id in state.comet_ids:
        return comet_remaining_life(planet.id, state.comets)
    return max(0, state.remaining_steps)


def capture_present_value(planet, state) -> float:
    """占领行星后的未来总产出（原始值，最大 = 500 × 5 = 2500）。"""
    return planet_remaining_life(planet, state) * planet.production


def normalize_pv(raw_pv: float) -> float:
    """归一化到 [0, 5] 范围，与 Δ舰船优势量级匹配。"""
    return raw_pv / 500.0   # 2500 / 500 = 5.0


def _find_planet(state, planet_id: int):
    """在状态中按 planet_id 查找行星。"""
    for p in state.planets:
        if p.id == planet_id:
            return p
    return None


# ═══════════════════════════════════════════════════════════════════
# 势能函数 Φ（PHI）
# ═══════════════════════════════════════════════════════════════════

def state_potential(state, player: int) -> float:
    """计算玩家在状态中的领土势能。

    零和版本：我方行星 PV 总和 − 敌方行星 PV 总和。
    中立行星 (owner == -1) 不计入任何一方。

    一行覆盖全部六种所有权变化：
      - 我占中立 → +pv（敌方 PV 不变）
      - 敌占中立 → −pv（敌方 PV 增加 → Φ 差缩小）
      - 我抢敌星 → +2pv（我增 + 敌减）
      - 敌抢我星 → −2pv
      - 我彗星过期 → −pv
      - 敌彗星过期 → +pv（敌方 PV 减少 → Φ 差扩大）
    """
    my_pv = 0.0
    enemy_pv = 0.0

    for p in state.planets:
        pv = normalize_pv(capture_present_value(p, state))
        if p.owner == player:
            my_pv += pv
        elif p.owner != -1:
            enemy_pv += pv

    return my_pv - enemy_pv


# ═══════════════════════════════════════════════════════════════════
# 舰船优势（Layer 1 代价信号）
# ═══════════════════════════════════════════════════════════════════

def _ship_ratio(state, owner: int) -> float:
    """计算 owner 的舰船数占双方总舰船数的比例 [0, 1]。

    重要：飞行中的舰队（state.fleets）已纳入统计。
    发送舰队时，舰船从行星驻军转移到舰队列表，总数不变 → Δship_ratio ≈ 0。
    比例仅在以下情况变化：生产差异、战斗损失、行星占领/丢失。

    天然对称（零和）：我方比例 + 敌方比例 = 1.0。
    """
    # 行星驻军 + 飞行中舰队（两者都算"我的资产"）
    my = (sum(p.ships for p in state.planets if p.owner == owner) +
          sum(f.ships for f in state.fleets if f.owner == owner))
    opponent = (sum(p.ships for p in state.planets if p.owner not in (-1, owner)) +
                sum(f.ships for f in state.fleets if f.owner not in (-1, owner)))
    return my / max(1.0, my + opponent)


# ═══════════════════════════════════════════════════════════════════
# 彗星 warmup（探索引导，独立于 PBRS）
# ═══════════════════════════════════════════════════════════════════

def get_comet_warmup(utilization_ema: float) -> float:
    """根据彗星利用率 EMA 计算 warmup 值。

    utilization_ema = 0.0  →  warmup = 1.50（全开，强引导）
    utilization_ema = 0.10 →  warmup = 0.75（半开）
    utilization_ema ≥ 0.20 →  warmup = 0.00（退场，值函数已自举）
    """
    ratio = min(1.0, utilization_ema / TARGET_UTILIZATION)
    return COMET_WARMUP_MAX * (1.0 - ratio)


# ═══════════════════════════════════════════════════════════════════
# 主奖励函数
# ═══════════════════════════════════════════════════════════════════

def soft_clip(value: float, linear_range: float = 5.0,
              soft_coef: float = 0.3, hard_cap: float = 8.0) -> float:
    """软 clip：线性区 + 压缩尾 + 硬上限。

    |x| ≤ linear_range: y = x（不压缩）
    |x| > linear_range: y = sign(x) × (linear_range + soft_coef × (|x| − linear_range))
    |y| capped at hard_cap
    """
    abs_v = abs(value)
    if abs_v <= linear_range:
        return value
    sign = 1.0 if value > 0 else -1.0
    compressed = linear_range + soft_coef * (abs_v - linear_range)
    compressed = min(compressed, hard_cap)
    return sign * compressed


def compute_step_reward(prev_state, curr_state, player: int,
                        comet_warmup: float = 0.0) -> float:
    """计算一步奖励（对称 Φ 框架）。

    两层奖励结构：
      Layer 1: Δ舰船优势 —— 比值变化 × 3.0（对称，天然编码代价）
      Layer 2: ΔΦ_territory —— 领土势能变化（一行覆盖六种所有权变化）
      + comet_warmup: 彗星探索引导加分（训练阶段注入，评估时为 0）

    终局由 ΔΦ 自然处理——若歼灭敌人，Φ 已在此前的占领中给出充分正信号。
    GAE 在 done 步自动 reset，无需额外终局 boost。

    Args:
        prev_state: 前一步的 GameState
        curr_state: 当前的 GameState
        player: 我方玩家编号
        comet_warmup: 彗星 warmup 值，由训练入口根据利用率 EMA 计算后注入

    Returns:
        float 奖励值（调用方应通过 soft_clip 压缩）
    """
    reward = 0.0

    # Layer 1: 舰船优势变化
    prev_ratio = _ship_ratio(prev_state, player)
    curr_ratio = _ship_ratio(curr_state, player)
    reward += 3.0 * (curr_ratio - prev_ratio)

    # Layer 2: 领土势能变化 ΔΦ（奖励 + 惩罚，一体两面）
    prev_phi = state_potential(prev_state, player)
    curr_phi = state_potential(curr_state, player)
    reward += (curr_phi - prev_phi)

    # comet_warmup: 刚占领的彗星 → 额外加分
    for p in curr_state.planets:
        prev = _find_planet(prev_state, p.id)
        if prev is None:
            continue
        if (prev.owner != player and p.owner == player
                and p.id in curr_state.comet_ids):
            reward += comet_warmup

    return reward
