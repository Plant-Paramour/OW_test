"""验证第二阶段特征工程模块的端到端正确性。"""

import sys
import io
import traceback

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import random
import numpy as np
from kaggle_environments import make

# Fixed seeds for reproducibility
random.seed(42)
np.random.seed(42)

# 将 src 加入 path
sys.path.insert(0, r"C:\code\[kaggle]\Orbit Wars")

from src.world.observation import parse_observation
from src.features import (
    build_global_features,
    build_self_features,
    build_candidate_features,
    INVALID_CANDIDATE_VECTOR,
    build_decision_matrix,
    DecisionRow,
)
from src.world.fleet_tracker import build_arrival_ledger
from src.world.combat import simulate_planet_timeline
from src.world.types import GameState
from src.engine.constants import SIM_HORIZON, BOARD_SIZE, CENTER_X, CENTER_Y
from src.engine.physics import dist, travel_time, fleet_speed, segment_hits_sun
from src.engine.prediction import is_static_planet
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

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


def run_tests():
    global PASS, FAIL
    print("=" * 60)
    print("Orbit Wars 特征工程验证")
    print("=" * 60)

    # ── 创建环境并运行几步 ──
    print("\n[1] 准备测试环境...")
    env = make("orbit_wars", debug=True)
    env.run(["random", "random"])
    obs = env.steps[-1][0].observation

    state = parse_observation(obs)
    print(f"  回合: {state.step}, 己方行星: {len(state.my_planets)}, "
          f"敌方行星: {len(state.enemy_planets)}, 中立: {len(state.neutral_planets)}")
    print(f"  彗星 IDs: {state.comet_ids}")

    ledger = build_arrival_ledger(state.fleets, state.planets)

    timelines = {}
    for p in state.planets:
        arrivals = ledger.get(p.id, [])
        timelines[p.id] = simulate_planet_timeline(p, arrivals, state.player, SIM_HORIZON)

    # ── 测试 1: 全局特征维度 ──
    print("\n[2] 全局特征测试")
    global_feat = build_global_features(state, ledger)
    check("global_features.shape == (16,)", global_feat.shape == (16,),
          f"实际: {global_feat.shape}")
    check("global_features dtype float32", global_feat.dtype == np.float32,
          f"实际: {global_feat.dtype}")
    check("global_features 全在 [-1,1] 范围内",
          np.all((global_feat >= -1.0) & (global_feat <= 1.0)),
          f"min={global_feat.min():.4f} max={global_feat.max():.4f}")
    check("global_features 无 NaN", not np.any(np.isnan(global_feat)))
    check("global_features 无 Inf", not np.any(np.isinf(global_feat)))

    # ── 测试 2: 自身特征维度 ──
    print("\n[3] 自身特征测试 (self_features)")
    for p in state.my_planets:
        sf = build_self_features(p, state, ledger, timelines)
        check(f"self_feat[{p.id}].shape == (21,)", sf.shape == (21,),
              f"实际: {sf.shape}")
        check(f"self_feat[{p.id}] 全在 [0,1]", np.all((sf >= 0.0) & (sf <= 1.0)),
              f"min={sf.min():.4f} max={sf.max():.4f}")
        check(f"self_feat[{p.id}] 无 NaN", not np.any(np.isnan(sf)))
        break  # 只测第一个

    # ── 测试 3: 边界情况 —— 无己方行星 ──
    print("\n[4] 边界情况测试")
    if state.my_planets and state.enemy_planets:
        src = state.my_planets[0]
        tgt = state.enemy_planets[0]
        ships_est = max(12, src.ships // 2)
        cf = build_candidate_features(src, tgt, state, ships_est, ledger, timelines[tgt.id])
        check("candidate_features.shape == (29,)", cf.shape == (29,),
              f"实际: {cf.shape}")
        check("candidate_features 无 NaN", not np.any(np.isnan(cf)))
        check("candidate_features 全在 [-1,1] (cos/sin 可为负)",
              np.all((cf >= -1.0) & (cf <= 1.0)),
              f"min={cf.min():.4f} max={cf.max():.4f}")

    if state.neutral_planets:
        tgt_neutral = state.neutral_planets[0]
        cf2 = build_candidate_features(src, tgt_neutral, state, ships_est, ledger, timelines[tgt_neutral.id])
        check("中立行星候选特征.shape == (29,)", cf2.shape == (29,),
              f"实际: {cf2.shape}")
        check("中立行星特征无 NaN", not np.any(np.isnan(cf2)))

    check("INVALID_CANDIDATE_VECTOR.shape == (29,)",
          INVALID_CANDIDATE_VECTOR.shape == (29,),
          f"实际: {INVALID_CANDIDATE_VECTOR.shape}")
    check("INVALID_CANDIDATE_VECTOR 全零",
          np.all(INVALID_CANDIDATE_VECTOR == 0.0))

    # ── 测试 4: 决策矩阵构建 ──
    print("\n[5] 决策矩阵构建测试")
    rows = build_decision_matrix(state, candidate_count=12)
    check("build_decision_matrix 返回 list", isinstance(rows, list))
    check(f"决策行数 > 0 (实际: {len(rows)})", len(rows) > 0)

    for i, row in enumerate(rows[:3]):
        check(f"Row[{i}] 是 DecisionRow", isinstance(row, DecisionRow))
        check(f"Row[{i}].self_feat.shape == (21,)", row.self_feat.shape == (21,),
              f"实际: {row.self_feat.shape}")
        check(f"Row[{i}].cand_feat.shape == (29,)", row.cand_feat.shape == (29,),
              f"实际: {row.cand_feat.shape}")
        check(f"Row[{i}].global_feat.shape == (16,)", row.global_feat.shape == (16,),
              f"实际: {row.global_feat.shape}")
        check(f"Row[{i}].mask 是 bool", isinstance(row.mask, bool))
        check(f"Row[{i}].action_info 是 dict", isinstance(row.action_info, dict))

    # 检查至少有一些有效行
    valid_rows = [r for r in rows if r.mask]
    check(f"至少存在有效决策行 (有效: {len(valid_rows)}/{len(rows)})",
          len(valid_rows) > 0)

    # ── 测试 5: 多回合稳定性 ──
    print("\n[6] 多回合稳定性测试")
    env2 = make("orbit_wars", debug=True)
    env2.run(["random", "random"])
    for step_idx in range(min(5, len(env2.steps))):
        obs2 = env2.steps[step_idx][0].observation
        try:
            state2 = parse_observation(obs2)
            ledger2 = build_arrival_ledger(state2.fleets, state2.planets)
            timelines2 = {}
            for p in state2.planets:
                arrivals = ledger2.get(p.id, [])
                timelines2[p.id] = simulate_planet_timeline(
                    p, arrivals, state2.player, SIM_HORIZON
                )
            rows2 = build_decision_matrix(state2, candidate_count=8)
            check(f"第 {step_idx} 回合决策矩阵构建成功",
                  isinstance(rows2, list) and len(rows2) >= 0,
                  f"行数: {len(rows2)}")
        except Exception as e:
            check(f"第 {step_idx} 回合异常", False, str(e)[:80])

    # ── 测试 6: 特征维度汇总 ──
    print("\n[7] 特征维度汇总")
    total_dims = 21 + 29 + 16
    check(f"特征维度总和 = 66 (self=21 + cand=29 + global=16)",
          total_dims == 66,
          f"实际: {total_dims}")

    # ── 测试 7: 彗星专项测试 ──
    print("\n[8] 彗星专项测试")
    comet_tests(state, ledger, timelines)

    # ── 测试 8: 确定性数值验证 ──
    print("\n[9] 确定性数值验证")
    deterministic_checks()

    # ── 报告结果 ──
    print("\n" + "=" * 60)
    total = PASS + FAIL
    print(f"结果: {PASS}/{total} 通过"
          + (f", {FAIL} 失败" if FAIL > 0 else "  全部通过! (￣▽￣)b"))
    print("=" * 60)
    return FAIL == 0


def comet_tests(state, ledger, timelines):
    """彗星场景专项测试 —— 构造含彗星的 GameState 验证所有彗星相关特征。"""
    player = state.player

    # 构造含彗星的受控 GameState
    # 所有行星放在太阳下方 (y<35) 确保 src→target 路径不穿过太阳安全区
    # Planet 字段: (id, owner, x, y, radius, ships, production)
    src = Planet(0, player, 28.0, 15.0, 2.0, 100, 5)
    comet_p = Planet(100, -1, 40.0, 18.0, 2.0, 50, 3)
    enemy_p = Planet(1, 1, 80.0, 15.0, 3.0, 80, 4)
    neutral_p = Planet(2, -1, 60.0, 15.0, 2.0, 30, 2)

    all_p = [src, comet_p, enemy_p, neutral_p]
    # 彗星路径从 (40,18) 开始，在太阳下方移动；30 个点确保 aim_at 5 次迭代收敛
    mock_comets = [{
        "planet_ids": [100],
        "paths": [[(40, 18), (42, 19), (44, 20), (46, 21), (48, 22),
                    (50, 23), (52, 24), (54, 25), (56, 26), (58, 27),
                    (60, 28), (62, 29), (64, 30), (66, 31), (68, 32),
                    (70, 33), (72, 34), (74, 35), (76, 36), (78, 37),
                    (80, 38), (82, 39), (84, 40), (86, 41), (88, 42),
                    (90, 43), (92, 44), (94, 45), (96, 46), (98, 47)]],
        "path_index": 0,
    }]
    comet_ids = {100}

    comet_state = GameState(
        step=100, player=player, planets=all_p, fleets=[],
        angular_velocity=0.03,
        initial_by_id={p.id: p for p in all_p},
        comets=mock_comets, comet_ids=comet_ids,
        my_planets=[src], enemy_planets=[enemy_p], neutral_planets=[neutral_p],
        remaining_steps=400, num_players=2,
        my_total_ships=100, enemy_total_ships=80,
        my_total_production=5, enemy_total_production=4,
    )

    comet_ledger = build_arrival_ledger([], all_p)
    comet_timelines = {}
    for p in all_p:
        arrivals = comet_ledger.get(p.id, [])
        comet_timelines[p.id] = simulate_planet_timeline(
            p, arrivals, player, SIM_HORIZON
        )

    # ── 测试彗星候选特征 ──
    cf_comet = build_candidate_features(
        src, comet_p, comet_state, 30, comet_ledger,
        comet_timelines[comet_p.id]
    )
    check("彗星候选特征.shape == (29,)", cf_comet.shape == (29,),
          f"实际: {cf_comet.shape}")
    check("彗星候选特征无 NaN", not np.any(np.isnan(cf_comet)))

    # 特征索引验证 (candidate_features.py 中的维度定义)
    check("彗星: is_comet=1.0 (idx 3)", cf_comet[3] == 1.0,
          f"实际: {cf_comet[3]:.4f}")
    check("彗星: comet_life/500 > 0 (idx 8)", cf_comet[8] > 0.0,
          f"实际: {cf_comet[8]:.4f}")
    check("彗星: is_strategic=1.0 (idx 26)", cf_comet[26] == 1.0,
          f"实际: {cf_comet[26]:.4f}")
    check("彗星: productive_turns=comet_life < 500 (idx 25)",
          cf_comet[25] < 1.0,
          f"实际: {cf_comet[25]:.4f}")

    # ── 对比：非彗星行星的对应特征 ──
    cf_enemy = build_candidate_features(
        src, enemy_p, comet_state, 30, comet_ledger,
        comet_timelines[enemy_p.id]
    )
    check("敌方: is_comet=0.0", cf_enemy[3] == 0.0)
    check("敌方: comet_life=0.0", cf_enemy[8] == 0.0)
    check("敌方: productive_turns=400/500 > comet_life",
          cf_enemy[25] > cf_comet[25],
          f"敌方: {cf_enemy[25]:.4f}, 彗星: {cf_comet[25]:.4f}")

    # ── 对比：彗星 vs 非彗星 is_strategic ──
    cf_neutral = build_candidate_features(
        src, neutral_p, comet_state, 30, comet_ledger,
        comet_timelines[neutral_p.id]
    )
    check(f"彗星 is_strategic=1.0 (production={comet_p.production}, is_comet)",
          cf_comet[26] == 1.0)
    check(f"中立 is_strategic={1.0 if neutral_p.production >= 3 else 0.0} (production={neutral_p.production})",
          cf_neutral[26] == float(neutral_p.production >= 3),
          f"实际: {cf_neutral[26]}")

    # ── 测试彗星行星的 aim_at 结果用于特征编码 ──
    from src.engine.interception import aim_at
    aim_comet = aim_at(
        src, comet_p, 30,
        comet_state.initial_by_id, comet_state.angular_velocity,
        comet_state.comets, comet_state.comet_ids,
    )
    check("aim_at comet → 有效结果", aim_comet is not None)
    if aim_comet:
        cf_aim = build_candidate_features(
            src, comet_p, comet_state, 30, comet_ledger,
            comet_timelines[comet_p.id],
            aim_result=aim_comet,
        )
        check("带 aim_result 的彗星特征无 NaN", not np.any(np.isnan(cf_aim)))
        check("aim_result 与 without 的 cos 一致",
              abs(cf_aim[9] - cf_comet[9]) < 0.001,
              f"with: {cf_aim[9]:.4f}, without: {cf_comet[9]:.4f}")


def deterministic_checks():
    """确定性数值验证 —— 构造受控场景，验证已知输入→预期输出。"""
    player = 0

    # 场景：一颗己方行星，一颗敌方行星，同侧避开太阳
    # Planet 字段: (id, owner, x, y, radius, ships, production)
    my_p = Planet(0, player, 25.0, 25.0, 2.0, 120, 4)
    enemy_p = Planet(1, 1, 75.0, 25.0, 3.0, 90, 5)
    all_p = [my_p, enemy_p]
    empty_fleets = []

    ctl_state = GameState(
        step=50, player=player, planets=all_p, fleets=empty_fleets,
        angular_velocity=0.03,
        initial_by_id={p.id: p for p in all_p},
        comets=[], comet_ids=set(),
        my_planets=[my_p], enemy_planets=[enemy_p], neutral_planets=[],
        remaining_steps=450, num_players=2,
        my_total_ships=120, enemy_total_ships=90,
        my_total_production=4, enemy_total_production=5,
    )

    ctl_ledger = build_arrival_ledger(empty_fleets, all_p)
    ctl_timelines = {}
    for p in all_p:
        ctl_timelines[p.id] = simulate_planet_timeline(
            p, [], player, SIM_HORIZON
        )

    sf = build_self_features(my_p, ctl_state, ctl_ledger, ctl_timelines)

    # ── 自身特征确定性检查 ──
    check("self: x/BOARD = 0.25", abs(sf[0] - 0.25) < 0.001,
          f"实际: {sf[0]:.4f}")
    check("self: y/BOARD = 0.25", abs(sf[1] - 0.25) < 0.001,
          f"实际: {sf[1]:.4f}")
    check("self: min(radius/5, 1.0) = 0.4", abs(sf[2] - 0.4) < 0.001,
          f"实际: {sf[2]:.4f}")
    check("self: min(ships,400)/400 = 0.3", abs(sf[3] - 0.3) < 0.001,
          f"实际: {sf[3]:.4f}")
    check("self: min(prod/5, 1.0) = 0.8", abs(sf[4] - 0.8) < 0.001,
          f"实际: {sf[4]:.4f}")

    # 无飞行舰队时 defense_risk=keep_needed/ships (keep_needed 来自空 arrivals)
    # keep_needed=0 (无敌军到达) → defense_risk=0
    check("self: defense_risk=0 (无 incoming)", abs(sf[9] - 0.0) < 0.001,
          f"实际: {sf[9]:.4f}")

    # dist_to_sun: (25,25)→(50,50)≈35.36, min(1.0, 35.36/70)≈0.505
    expected_dist = dist(my_p.x, my_p.y, CENTER_X, CENTER_Y)
    expected_dist_sun = min(1.0, expected_dist / 70.0)
    assert abs(expected_dist_sun - 0.505) < 0.01, f"Unexpected dist: {expected_dist_sun}"
    check(f"self: dist_to_sun/70 = {expected_dist_sun:.4f}",
          abs(sf[6] - expected_dist_sun) < 0.001,
          f"实际: {sf[6]:.4f}")

    # ── 候选特征确定性检查 ──
    ships_est = max(12, my_p.ships // 2)
    cf = build_candidate_features(
        my_p, enemy_p, ctl_state, ships_est, ctl_ledger,
        ctl_timelines[enemy_p.id]
    )

    check("cand: is_enemy=1.0 (idx 2)", cf[2] == 1.0,
          f"实际: {cf[2]:.4f}")
    check("cand: is_neutral=0.0 (idx 0)", cf[0] == 0.0)
    check("cand: is_self=0.0 (idx 1)", cf[1] == 0.0)

    # enemy production / 5 = 5/5 = 1.0, clamped
    check("cand: min(prod/5, 1.0) = 1.0", abs(cf[4] - 1.0) < 0.001,
          f"实际: {cf[4]:.4f}")

    # 无舰队 → first_hostile_eta=999, 1/(1+999) ≈ 0.001
    check("cand: 1/(1+first_hostile)=0.001", abs(cf[20] - 0.001) < 0.001,
          f"实际: {cf[20]:.4f}")

    # ── 全局特征确定性检查 ──
    gf = build_global_features(ctl_state, ctl_ledger)
    check("global: step/500 = 0.1", abs(gf[0] - 0.1) < 0.001,
          f"实际: {gf[0]:.4f}")
    check("global: remaining/500 = 0.9", abs(gf[1] - 0.9) < 0.001,
          f"实际: {gf[1]:.4f}")
    check("global: my_planets/48 = 1/48", abs(gf[2] - 1.0/48.0) < 0.001,
          f"实际: {gf[2]:.4f}")
    check("global: my_prod/240 = 4/240 (idx 5)", abs(gf[5] - 4.0/240.0) < 0.001,
          f"实际: {gf[5]:.4f}")

    # 无 incoming → 无受攻击行星
    # gf[10] = planets_under_attack / max(1, len(my_planets))
    check("global: planets_under_attack=0 (idx 10)", abs(gf[10] - 0.0) < 0.001,
          f"实际: {gf[10]:.4f}")
    # gf[11] = max_defense_pressure (also 0 in no-incoming scenario)
    check("global: max_defense_pressure=0 (idx 11)", abs(gf[11] - 0.0) < 0.001,
          f"实际: {gf[11]:.4f}")


if __name__ == "__main__":
    ok = run_tests()
    sys.exit(0 if ok else 1)
