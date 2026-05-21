"""验证第四阶段奖励塑形模块的正确性。"""
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import math
import random
import numpy as np

random.seed(42)
np.random.seed(42)

sys.path.insert(0, r"C:\code\[kaggle]\Orbit Wars")

from src.world.types import GameState
from src.env.reward import (
    planet_remaining_life, capture_present_value, normalize_pv,
    state_potential, get_comet_warmup, compute_step_reward,
    ALPHA, COMET_WARMUP_MAX, TARGET_UTILIZATION,
)
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def check_eq(name, actual, expected, tol=0.001):
    global PASS, FAIL
    if abs(actual - expected) <= tol:
        PASS += 1
        print(f"  PASS  {name}: {actual:.4f} == {expected:.4f}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}: got {actual:.4f}, expected {expected:.4f}")


def check_sign(name, value, expected_sign):
    """检查值的正负符号。"""
    global PASS, FAIL
    actual_sign = "positive" if value > 0 else ("negative" if value < 0 else "zero")
    if actual_sign == expected_sign:
        PASS += 1
        print(f"  PASS  {name}: {value:.4f} is {expected_sign}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}: {value:.4f} is {actual_sign}, expected {expected_sign}")


# ── 辅助: 创建 mock Planet ──

def planet(id, owner, x=50, y=50, radius=2, ships=50, production=3):
    return Planet(id=id, owner=owner, x=float(x), y=float(y),
                  radius=float(radius), ships=ships, production=production)


def fleet(id, owner, x=50, y=50, angle=0, from_planet_id=0, ships=30):
    return Fleet(id=id, owner=owner, x=float(x), y=float(y),
                 angle=float(angle), from_planet_id=from_planet_id, ships=ships)


# ── 辅助: 创建 mock GameState ──

def make_state(step, player, planets, fleets=None, comets=None, comet_ids=None,
               episode_steps=500):
    """快速构造 GameState，自动填充派生字段。"""
    if fleets is None:
        fleets = []
    if comets is None:
        comets = []
    if comet_ids is None:
        comet_ids = set()

    my_planets = [p for p in planets if p.owner == player]
    enemy_planets = [p for p in planets if p.owner not in (-1, player)]
    neutral_planets = [p for p in planets if p.owner == -1]

    my_total_ships = (sum(p.ships for p in my_planets) +
                      sum(f.ships for f in fleets if f.owner == player))
    enemy_total_ships = (sum(p.ships for p in enemy_planets) +
                         sum(f.ships for f in fleets if f.owner not in (-1, player)))
    my_total_production = sum(p.production for p in my_planets)
    enemy_total_production = sum(p.production for p in enemy_planets)

    return GameState(
        step=step, player=player,
        planets=planets, fleets=fleets,
        angular_velocity=0.03,
        initial_by_id={p.id: p for p in planets},
        comets=comets, comet_ids=comet_ids,
        my_planets=my_planets,
        enemy_planets=enemy_planets,
        neutral_planets=neutral_planets,
        remaining_steps=max(1, episode_steps - step),
        episode_steps=episode_steps,
        my_total_ships=my_total_ships,
        enemy_total_ships=enemy_total_ships,
        my_total_production=my_total_production,
        enemy_total_production=enemy_total_production,
    )


# ── 辅助: 创建 mock 彗星数据 ──

def make_comet_data(planet_id, remaining_life):
    """创建最小彗星数据以通过 comet_remaining_life 检查。"""
    path = [(50.0, 50.0)] * (remaining_life + 5)
    return {
        "planet_ids": [planet_id],
        "paths": [path],
        "path_index": 5,  # len(path) - path_index - 5 = remaining_life → remaining_life
    }


def run_tests():
    global PASS, FAIL
    print("=" * 60)
    print("Orbit Wars 奖励塑形验证 (对称势能函数 Φ)")
    print("=" * 60)

    # ═══════════════════════════════════════════════════════════════
    # 1. 辅助函数
    # ═══════════════════════════════════════════════════════════════

    print("\n[1] 辅助函数测试")

    # --- normalize_pv ---
    print("\n  [1a] normalize_pv")
    check_eq("  normalize_pv(0)", normalize_pv(0), 0.0)
    check_eq("  normalize_pv(2500)", normalize_pv(2500), 5.0)
    check_eq("  normalize_pv(1250)", normalize_pv(1250), 2.5)
    check_eq("  normalize_pv(500)", normalize_pv(500), 1.0)

    # --- planet_remaining_life (普通行星) ---
    print("\n  [1b] planet_remaining_life (普通行星)")
    p_normal = planet(1, 0, production=3)
    st = make_state(100, 0, [p_normal])
    check_eq("  普通行星 T=100, remaining_life", planet_remaining_life(p_normal, st), 400)
    st_end = make_state(499, 0, [p_normal])
    check_eq("  普通行星 T=499, remaining_life", planet_remaining_life(p_normal, st_end), 1)

    # --- planet_remaining_life (彗星) ---
    print("\n  [1c] planet_remaining_life (彗星)")
    comet_p = planet(99, -1, production=1)
    comet_data = make_comet_data(99, 30)
    st_comet = make_state(100, 0, [comet_p], comets=[comet_data], comet_ids={99})
    check_eq("  彗星 remaining_life=30", planet_remaining_life(comet_p, st_comet), 30)

    # --- capture_present_value ---
    print("\n  [1d] capture_present_value")
    p_high = planet(2, 0, production=5)
    st_early = make_state(50, 0, [p_high])
    pv_high = capture_present_value(p_high, st_early)
    check_eq("  产能5, T=50 → PV=2250 (450*5)", pv_high, 2250.0)

    p_low = planet(3, 0, production=1)
    st_late = make_state(490, 0, [p_low])
    pv_low = capture_present_value(p_low, st_late)
    check_eq("  产能1, T=490 → PV=10 (10*1)", pv_low, 10.0)

    # --- get_comet_warmup ---
    print("\n  [1e] get_comet_warmup")
    check_eq("  utilization=0.00 → warmup=1.5", get_comet_warmup(0.0), 1.5)
    check_eq("  utilization=0.10 → warmup=0.75", get_comet_warmup(0.10), 0.75)
    check_eq("  utilization=0.20 → warmup=0.0", get_comet_warmup(0.20), 0.0)
    check_eq("  utilization=0.25 → warmup=0.0 (capped)", get_comet_warmup(0.25), 0.0)
    check_eq("  utilization=0.05 → warmup=1.125", get_comet_warmup(0.05), 1.125)
    check("  utilization=0.15 → warmup in (0.25, 0.50)",
          0.25 < get_comet_warmup(0.15) < 0.50)

    # ═══════════════════════════════════════════════════════════════
    # 2. state_potential Φ 函数
    # ═══════════════════════════════════════════════════════════════

    print("\n[2] state_potential Φ 函数")

    # 场景 2a: 平衡状态
    p_mine = planet(1, 0, production=4, ships=100)
    p_enemy = planet(2, 1, production=4, ships=100)
    st_balanced = make_state(100, 0, [p_mine, p_enemy])
    phi_balanced = state_potential(st_balanced, 0)
    check_eq("  对称状态 Φ ≈ 0", phi_balanced, 0.0, tol=0.01)

    # 场景 2b: 我方优势
    p_my2 = planet(3, 0, production=5)
    st_my_adv = make_state(100, 0, [p_mine, p_my2, p_enemy])
    phi_my_adv = state_potential(st_my_adv, 0)
    check_sign("  我方多一颗星 → Φ > 0", phi_my_adv, "positive")

    # 场景 2c: 敌方优势
    p_en2 = planet(4, 1, production=5)
    st_en_adv = make_state(100, 0, [p_mine, p_enemy, p_en2])
    phi_en_adv = state_potential(st_en_adv, 0)
    check_sign("  敌方多一颗星 → Φ < 0", phi_en_adv, "negative")

    # 场景 2d: 中立行星不影响 Φ
    p_neutral = planet(5, -1, production=5)
    st_with_neutral = make_state(100, 0, [p_mine, p_enemy, p_neutral])
    phi_with_neutral = state_potential(st_with_neutral, 0)
    check_eq("  中立行星不影响 Φ", phi_with_neutral, 0.0, tol=0.01)

    # 场景 2e: 两家都没有行星（全中立）
    st_all_neutral = make_state(100, 0, [p_neutral])
    phi_all_neutral = state_potential(st_all_neutral, 0)
    check_eq("  全中立 → Φ = 0", phi_all_neutral, 0.0)

    # ═══════════════════════════════════════════════════════════════
    # 3. 所有权变化场景 (ΔΦ)
    # ═══════════════════════════════════════════════════════════════

    print("\n[3] 所有权变化场景")

    # --- 场景 3a: 我方占领中立行星 ---
    print("\n  [3a] 我方占领中立")
    p_neut_high = planet(10, -1, production=5, ships=10)
    p_my_base = planet(11, 0, production=3, ships=200)
    p_en_base = planet(12, 1, production=3, ships=200)
    prev_st = make_state(50, 0, [p_neut_high, p_my_base, p_en_base])
    # 占领后
    p_neut_taken = planet(10, 0, production=5, ships=5)  # 战斗后剩5艘
    p_my_after = planet(11, 0, production=3, ships=120)  # 派了80艘
    curr_st = make_state(51, 0, [p_neut_taken, p_my_after, p_en_base])

    # 没有舰队变化时应有的舰船劣势
    reward_3a = compute_step_reward(prev_st, curr_st, 0)
    # ΔΦ: 中立→我, PV = (500-51)*5/500 = 4.49
    # Δship: my_ships decreased by 80 (200→120+5=125? no, 200→120, neut_high was 10→5)
    # prev: my = 200, enemy = 200, ratio = 0.5
    # curr: my = 120+5 = 125, enemy = 200, ratio = 125/325 = 0.3846
    # Δratio = -0.1154, Layer1 = 3*(-0.1154) = -0.346
    # ΔΦ = pv_mine_after - pv_mine_before = pv_neut_taken = (449*5)/500 = 4.49
    # Total ≈ 4.49 - 0.346 = 4.144
    expected_neut_pv = (500 - 51) * 5 / 500.0
    check_sign("  占领中立高产星 → 正奖励", reward_3a, "positive")
    check("  奖励值 ≈ ΔΦ − 舰船代价",
          abs(reward_3a - (expected_neut_pv - 0.346)) < 0.1)

    # --- 场景 3b: 敌方占领中立行星（旧方案盲区！）---
    print("\n  [3b] 敌方占领中立")
    p_neut2 = planet(20, -1, production=5, ships=10)
    p_my2 = planet(21, 0, production=3, ships=200)
    p_en2 = planet(22, 1, production=3, ships=200)
    prev_3b = make_state(50, 0, [p_neut2, p_my2, p_en2])
    p_neut2_en = planet(20, 1, production=5, ships=5)
    curr_3b = make_state(51, 0, [p_neut2_en, p_my2, p_en2])
    reward_3b = compute_step_reward(prev_3b, curr_3b, 0)
    check_sign("  敌方占中立 → 负奖励（自动惩罚）", reward_3b, "negative")
    # ΔΦ = -pv (enemy PV increased)
    # Layer 1 也有微小变化（敌方多了占领行星的驻军 +5 艘）
    expected_pv = (500 - 51) * 5 / 500.0  # = 4.49
    # Layer1: 3.0*(200/405 - 200/400) ≈ -0.019
    layer1_3b = 3.0 * (200.0 / 405.0 - 200.0 / 400.0)
    expected_3b = -expected_pv + layer1_3b
    check_eq("  惩罚含 Φ + Layer1", reward_3b, expected_3b, tol=0.001)

    # --- 场景 3c: 我方抢夺敌方行星 ---
    print("\n  [3c] 我方抢夺敌星")
    p_en_owned = planet(30, 1, production=4, ships=60)
    p_my3 = planet(31, 0, production=3, ships=200)
    prev_3c = make_state(100, 0, [p_en_owned, p_my3])
    p_en_taken = planet(30, 0, production=4, ships=10)
    p_my3_after = planet(31, 0, production=3, ships=120)
    curr_3c = make_state(101, 0, [p_en_taken, p_my3_after])
    reward_3c = compute_step_reward(prev_3c, curr_3c, 0)
    check_sign("  抢夺敌星 → 强正奖励", reward_3c, "positive")
    # ΔΦ = +2pv (敌失 + 我得)
    expected_pv = (500 - 101) * 4 / 500.0  # ≈ 3.192
    # ship_ratio change: my went from 200→130, enemy 60→0, so ratio drops
    # The ΔΦ should be ≈ 2 * 3.192 = 6.384
    # With ship penalty, should still be strongly positive
    check("  抢夺奖励含双重信号 (≈2×PV)",
          reward_3c > expected_pv * 1.5)  # at least > 1.5×PV (should be ~2×)

    # --- 场景 3d: 敌方抢夺我方行星 ---
    print("\n  [3d] 敌方抢我行星")
    p_my_owned = planet(40, 0, production=4, ships=60)
    p_en4 = planet(41, 1, production=3, ships=200)
    prev_3d = make_state(100, 0, [p_my_owned, p_en4])
    p_my_lost = planet(40, 1, production=4, ships=10)
    curr_3d = make_state(101, 0, [p_my_lost, p_en4])
    reward_3d = compute_step_reward(prev_3d, curr_3d, 0)
    check_sign("  被抢行星 → 强负奖励", reward_3d, "negative")

    # 独立计算 ΔΦ（不使用 state_potential，防止循环验证）
    # prev (step=100): 我拥有 prod=4, 敌拥有 prod=3
    #   prev_Φ = (400*4 - 400*3) / 500 = (1600 - 1200) / 500 = 0.8
    # curr (step=101): 我无行星, 敌拥有 prod=4 + prod=3
    #   curr_Φ = (0 - (399*4 + 399*3)) / 500 = (0 - 1596 - 1197) / 500 = -5.586
    # ΔΦ = -5.586 - 0.8 = -6.386
    rem_prev = 500 - 100   # = 400 (prev.remaining_steps)
    rem_curr = 500 - 101   # = 399 (curr.remaining_steps)
    phi_prev_indep = (rem_prev * 4 - rem_prev * 3) / 500.0      # = 0.8
    phi_curr_indep = (0.0 - (rem_curr * 4 + rem_curr * 3)) / 500.0  # = -5.586
    delta_phi_indep = phi_curr_indep - phi_prev_indep            # = -6.386
    # Layer1 独立计算
    layer1_3d = 3.0 * (0.0 - 60.0 / 260.0)  # ≈ -0.6923
    expected_3d = delta_phi_indep + layer1_3d  # ≈ -7.078
    check_eq("  惩罚含 2×Φ + Layer1（独立计算）", reward_3d, expected_3d, tol=0.001)

    # 用 state_potential 做交叉验证：ΔΦ 应 ≈ −2×PV
    prev_phi_3d = state_potential(prev_3d, 0)
    curr_phi_3d = state_potential(curr_3d, 0)
    delta_phi_3d = curr_phi_3d - prev_phi_3d
    approx_2pv = -2 * (500 - 101) * 4 / 500.0
    check("  ΔΦ ≈ −2×PV（回合衰减 ~0.002）",
          abs(delta_phi_3d - approx_2pv) < 0.005,
          f"ΔΦ={delta_phi_3d:.4f}, −2pv={approx_2pv:.4f}, diff={abs(delta_phi_3d - approx_2pv):.4f}")
    # 交叉验证: state_potential 算出的 ΔΦ 应与独立计算一致
    check("  state_potential ΔΦ 与独立计算一致",
          abs(delta_phi_3d - delta_phi_indep) < 0.001,
          f"state_potential ΔΦ={delta_phi_3d:.4f}, independent={delta_phi_indep:.4f}")

    # --- 场景 3e: 彗星过期（我方损失）---
    print("\n  [3e] 彗星过期（我方持有）")
    comet_data_e = make_comet_data(50, 2)  # 剩余2回合
    p_comet_mine = planet(50, 0, production=1, ships=30)
    p_en5 = planet(51, 1, production=3, ships=200)
    prev_3e = make_state(150, 0, [p_comet_mine, p_en5],
                         comets=[comet_data_e], comet_ids={50})
    p_comet_gone = planet(50, -1, production=0, ships=0)  # 过期变中立
    curr_3e = make_state(151, 0, [p_comet_gone, p_en5],
                         comets=[comet_data_e], comet_ids=set())  # 不再标记为彗星
    reward_3e = compute_step_reward(prev_3e, curr_3e, 0)
    check_sign("  彗星过期 → 负奖励", reward_3e, "negative")

    # --- 场景 3f: 敌方彗星过期（旧方案盲区！）---
    print("\n  [3f] 敌方彗星过期")
    comet_data_f = make_comet_data(60, 2)
    p_comet_enemy = planet(60, 1, production=1, ships=30)
    p_my6 = planet(61, 0, production=3, ships=200)
    prev_3f = make_state(150, 0, [p_comet_enemy, p_my6],
                         comets=[comet_data_f], comet_ids={60})
    p_comet_gone2 = planet(60, -1, production=0, ships=0)
    curr_3f = make_state(151, 0, [p_comet_gone2, p_my6],
                         comets=[comet_data_f], comet_ids=set())
    reward_3f = compute_step_reward(prev_3f, curr_3f, 0)
    check_sign("  敌方彗星过期 → 正奖励（自动）", reward_3f, "positive")

    # ═══════════════════════════════════════════════════════════════
    # 4. 舰船优势变化 (Layer 1)
    # ═══════════════════════════════════════════════════════════════

    print("\n[4] 舰船优势变化")

    # 场景 4a: 纯舰船损失（无所有权变化）
    p_a = planet(70, 0, production=3, ships=100)
    p_b = planet(71, 1, production=3, ships=100)
    prev_4a = make_state(100, 0, [p_a, p_b])
    # 我方行星舰船减少（被战斗消耗）
    p_a_less = planet(70, 0, production=3, ships=50)
    curr_4a = make_state(101, 0, [p_a_less, p_b])
    reward_4a = compute_step_reward(prev_4a, curr_4a, 0)
    check_sign("  纯舰船损失 → 负 Layer 1", reward_4a, "negative")
    # prev ratio = 100/200 = 0.5
    # curr ratio = 50/150 = 0.333
    # Δratio = -0.167, Layer1 = -0.5
    # ΔΦ ≈ 0 (both still own 1 planet each)
    check_eq("  舰船损失 ≈ 3.0 × Δratio", reward_4a, 3.0 * (50/150 - 100/200), tol=0.02)

    # 场景 4b: 纯舰船增益
    prev_4b = make_state(100, 0, [p_a_less, p_b])
    p_a_more = planet(70, 0, production=3, ships=100)
    curr_4b = make_state(101, 0, [p_a_more, p_b])
    reward_4b = compute_step_reward(prev_4b, curr_4b, 0)
    check_sign("  纯舰船增益 → 正 Layer 1", reward_4b, "positive")

    # ═══════════════════════════════════════════════════════════════
    # 5. 彗星 warmup
    # ═══════════════════════════════════════════════════════════════

    print("\n[5] 彗星 warmup")

    comet_data_w = make_comet_data(80, 50)
    p_comet_neut = planet(80, -1, production=2, ships=5)
    p_my_w = planet(81, 0, production=3, ships=200)
    p_en_w = planet(82, 1, production=3, ships=200)
    prev_5 = make_state(100, 0, [p_comet_neut, p_my_w, p_en_w],
                        comets=[comet_data_w], comet_ids={80})
    p_comet_taken = planet(80, 0, production=2, ships=20)
    p_my_w_after = planet(81, 0, production=3, ships=170)
    curr_5 = make_state(101, 0, [p_comet_taken, p_my_w_after, p_en_w],
                        comets=[comet_data_w], comet_ids={80})

    # 无 warmup
    r_no_warmup = compute_step_reward(prev_5, curr_5, 0, comet_warmup=0.0)
    # 全 warmup
    r_full_warmup = compute_step_reward(prev_5, curr_5, 0, comet_warmup=1.5)
    # 半 warmup
    r_half_warmup = compute_step_reward(prev_5, curr_5, 0, comet_warmup=0.75)

    warmup_diff = r_full_warmup - r_no_warmup
    check_eq("  warmup=1.5 比 warmup=0 多了 1.5", warmup_diff, 1.5)
    half_diff = r_half_warmup - r_no_warmup
    check_eq("  warmup=0.75 比 warmup=0 多了 0.75", half_diff, 0.75)
    check_sign("  warmup=0 时彗星占领仍为正", r_no_warmup, "positive")

    # 非彗星占领不应受 warmup 影响
    print("\n  [5b] warmup 不泄漏到非彗星场景")
    p_neut_w = planet(90, -1, production=5, ships=10)
    p_my_w2 = planet(91, 0, production=3, ships=200)
    prev_5b = make_state(50, 0, [p_neut_w, p_my_w2])
    p_neut_taken = planet(90, 0, production=5, ships=5)
    p_my_taken = planet(91, 0, production=3, ships=120)
    curr_5b = make_state(51, 0, [p_neut_taken, p_my_taken])
    r5b_0 = compute_step_reward(prev_5b, curr_5b, 0, comet_warmup=0.0)
    r5b_15 = compute_step_reward(prev_5b, curr_5b, 0, comet_warmup=1.5)
    check_eq("  非彗星占领 warmup 不影响", r5b_0, r5b_15)

    # ═══════════════════════════════════════════════════════════════
    # 6. 终局：无额外 boost（ΔΦ 已充分编码胜负）
    # ═══════════════════════════════════════════════════════════════

    print("\n[6] 终局（无额外 boost）")

    p_win_a = planet(100, 0, production=3, ships=500)
    p_win_b = planet(101, 1, production=3, ships=10)
    st_win = make_state(500, 0, [p_win_a, p_win_b])
    st_prev = make_state(499, 0, [p_win_a, p_win_b])
    r_win = compute_step_reward(st_prev, st_win, 0)
    check_eq("  终局胜利无额外 boost", r_win, 0.0, tol=0.01)
    check("  终局 reward 不再 > 5", not (r_win > 5.0), f"got {r_win:.4f}")

    p_lose_a = planet(100, 0, production=3, ships=10)
    p_lose_b = planet(101, 1, production=3, ships=500)
    st_lose = make_state(500, 0, [p_lose_a, p_lose_b])
    st_lprev = make_state(499, 0, [p_lose_a, p_lose_b])
    r_lose = compute_step_reward(st_lprev, st_lose, 0)
    check_eq("  终局失败无额外惩罚", r_lose, 0.0, tol=0.01)
    check("  终局惩罚不再 < -5", not (r_lose < -5.0), f"got {r_lose:.4f}")

    # 平局
    p_tie_a = planet(200, 0, production=3, ships=200)
    p_tie_b = planet(201, 1, production=3, ships=200)
    st_tie = make_state(500, 0, [p_tie_a, p_tie_b])
    st_tprev = make_state(499, 0, [p_tie_a, p_tie_b])
    r_tie = compute_step_reward(st_tprev, st_tie, 0)
    check_eq("  平局 → reward ≈ 0", r_tie, 0.0, tol=0.01)

    # ═══════════════════════════════════════════════════════════════
    # 7. 边界情况
    # ═══════════════════════════════════════════════════════════════

    print("\n[7] 边界情况")

    # 场景 7a: 无己方行星
    p_empty_enemy = planet(300, 1, production=3, ships=100)
    prev_7a = make_state(100, 0, [p_empty_enemy])
    curr_7a = make_state(101, 0, [p_empty_enemy])
    r_7a = compute_step_reward(prev_7a, curr_7a, 0)
    check("  无己方行星不崩溃", True, f"reward={r_7a:.4f}")

    # 场景 7b: 无敌方行星
    p_only_mine = planet(400, 0, production=3, ships=100)
    prev_7b = make_state(100, 0, [p_only_mine])
    curr_7b = make_state(101, 0, [p_only_mine])
    r_7b = compute_step_reward(prev_7b, curr_7b, 0)
    check("  无敌方行星不崩溃", True, f"reward={r_7b:.4f}")

    # 场景 7c: 全中立
    p_all_neut = planet(500, -1, production=3, ships=10)
    prev_7c = make_state(100, 0, [p_all_neut])
    curr_7c = make_state(101, 0, [p_all_neut])
    r_7c = compute_step_reward(prev_7c, curr_7c, 0)
    check_eq("  全中立 → reward ≈ 0", r_7c, 0.0, tol=0.01)

    # 场景 7d: 同一回合多颗行星所有权变化
    print("\n  [7d] 同回合多重所有权变化")
    p_multi_neut1 = planet(600, -1, production=4, ships=10)
    p_multi_neut2 = planet(601, -1, production=3, ships=10)
    p_multi_my = planet(602, 0, production=3, ships=200)
    p_multi_en = planet(603, 1, production=3, ships=200)
    prev_7d = make_state(100, 0, [p_multi_neut1, p_multi_neut2, p_multi_my, p_multi_en])
    p_multi_mine1 = planet(600, 0, production=4, ships=5)
    p_multi_en2 = planet(601, 1, production=3, ships=5)
    curr_7d = make_state(101, 0, [p_multi_mine1, p_multi_en2, p_multi_my, p_multi_en])
    r_7d = compute_step_reward(prev_7d, curr_7d, 0)
    # ΔΦ: (占领600: +pv_600) + (敌人占领601: -pv_601)
    pv_600 = (500 - 101) * 4 / 500.0  # ≈ 3.192
    pv_601 = (500 - 101) * 3 / 500.0  # ≈ 2.394
    expected_7d = pv_600 - pv_601  # ≈ 0.798
    check("  多行星变化奖励 ≈ 各自 ΔΦ 之和",
          abs(r_7d - expected_7d) < 0.05, f"got {r_7d:.4f}, expected {expected_7d:.4f}")

    # 场景 7e: 含飞行中舰队
    print("\n  [7e] 飞行中舰队参与舰船优势")
    p7e_my = planet(700, 0, production=3, ships=100)
    p7e_en = planet(701, 1, production=3, ships=100)
    f7e_my = fleet(1, 0, x=30, y=30, ships=50)
    prev_7e = make_state(100, 0, [p7e_my, p7e_en], fleets=[f7e_my])
    # 敌人也派了舰队
    f7e_en = fleet(2, 1, x=70, y=70, ships=50)
    curr_7e = make_state(101, 0, [p7e_my, p7e_en], fleets=[f7e_my, f7e_en])
    r_7e = compute_step_reward(prev_7e, curr_7e, 0)
    check("  含舰队不崩溃", True, f"reward={r_7e:.4f}")

    # 场景 7f: ALPHA 常量存在且可访问
    print("\n  [7f] 常量可访问")
    check("  ALPHA 已定义", ALPHA is not None)
    check("  COMET_WARMUP_MAX 已定义", COMET_WARMUP_MAX == 1.5)
    check("  TARGET_UTILIZATION 已定义", TARGET_UTILIZATION == 0.20)

    # 场景 7g: 验证奖励不含 NaN/Inf
    print("\n  [7g] 数值稳定性")
    for i, (name, r) in enumerate([
        ("3a", reward_3a), ("3b", reward_3b), ("3c", reward_3c),
        ("3d", reward_3d), ("3e", reward_3e), ("3f", reward_3f),
        ("4a", reward_4a), ("4b", reward_4b), ("5", r_no_warmup),
        ("6_win", r_win), ("6_lose", r_lose), ("7a", r_7a),
    ]):
        check(f"  场景 {name} 无 NaN/Inf", not (math.isnan(r) or math.isinf(r)),
              f"reward={r}")

    # ═══════════════════════════════════════════════════════════════
    # 8. 验证对称性：玩家视角反转
    # ═══════════════════════════════════════════════════════════════

    print("\n[8] 对称性验证（玩家视角反转）")

    p_sym_mine = planet(800, 0, production=4, ships=100)
    p_sym_enemy = planet(801, 1, production=3, ships=150)
    prev_8 = make_state(100, 0, [p_sym_mine, p_sym_enemy])
    # 敌方占领了一颗中立星
    p_sym_new_en = planet(802, 1, production=5, ships=20)
    curr_8 = make_state(101, 0, [p_sym_mine, p_sym_enemy, p_sym_new_en])

    r_player0 = compute_step_reward(prev_8, curr_8, 0)
    r_player1 = compute_step_reward(prev_8, curr_8, 1)
    # 对玩家0：敌扩张 → 负奖励
    # 对玩家1：我扩张 → 正奖励
    check_sign("  玩家0（敌方扩张）→ 负", r_player0, "negative")
    check_sign("  玩家1（我方扩张）→ 正", r_player1, "positive")
    # 零和：r_player0 + r_player1 ≈ 0（Layer 1 项会互相抵消？不完全，因为舰队归属不同）
    # 但 Φ 部分是完全零和的
    check("  两位玩家的 Φ 变化符号相反",
          (r_player0 > 0) != (r_player1 > 0))

    # 终局：无 boost，纯 ΔΦ + Δship
    p_final_a = planet(900, 0, production=3, ships=500)
    p_final_b = planet(901, 1, production=3, ships=100)
    st_final = make_state(500, 0, [p_final_a, p_final_b])
    st_fprev = make_state(499, 0, [p_final_a, p_final_b])
    r_f0 = compute_step_reward(st_fprev, st_final, 0)
    r_f1 = compute_step_reward(st_fprev, st_final, 1)
    check_eq("  终局无 boost: 玩家0 reward ≈ 0", r_f0, 0.0, tol=0.01)
    check_eq("  终局无 boost: 玩家1 reward ≈ 0", r_f1, 0.0, tol=0.01)

    # ═══════════════════════════════════════════════════════════════
    # 总结
    # ═══════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print(f"结果: {PASS} 通过, {FAIL} 失败, 共 {PASS + FAIL} 项")
    print("=" * 60)

    return FAIL == 0


if __name__ == "__main__":
    ok = run_tests()
    sys.exit(0 if ok else 1)
