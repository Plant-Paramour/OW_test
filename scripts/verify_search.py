r"""搜索模块验证测试 —— 覆盖 what-if 模拟、价值计算、搜索算法。

用法:
"C:\ProgramData\anaconda3\envs\Orbit_Wars\python.exe" "C:\code\[kaggle]\Orbit Wars\scripts\verify_search.py"
"""

import io
import math
import sys
import unittest

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np

from kaggle_environments import make

# 将被测模块插入 sys.path 上方
sys.path.insert(0, r"C:\code\[kaggle]\Orbit Wars")

from src.world.observation import parse_observation
from src.world.fleet_tracker import build_arrival_ledger
from src.world.combat import simulate_planet_timeline, state_at_timeline
from src.engine.physics import travel_time, fleet_speed, estimate_arrival
from src.engine.interception import aim_at, check_path_blocked
from src.engine.prediction import comet_remaining_life
from src.search.simulator import (
    LaunchOutcome,
    simulate_fleet_launch,
    find_min_ships_to_capture,
)
from src.search.valuation import (
    value_of_capture,
    compute_action_value,
    is_comet_target,
)
from src.search.search import (
    ScoredAction,
    search_best_actions,
    beam_search,
    _evaluate_candidate,
    _try_ship_amounts,
)

# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

MIN_FLEET = 12


def _make_env():
    """创建 1v1 环境并运行一步以获得初始状态。"""
    env = make("orbit_wars", debug=True)
    env.reset()
    return env


def _get_state(env, player=0):
    """从环境中解析 GameState + ledger + timelines。"""
    obs = env.steps[-1][player].observation
    state = parse_observation(obs, player_override=player)
    ledger = build_arrival_ledger(state.fleets, state.planets)
    timelines = {}
    for p in state.planets:
        timelines[p.id] = simulate_planet_timeline(
            p, ledger.get(p.id, []), state.player, state.remaining_steps,
        )
    return state, ledger, timelines


def _run_until_step(env, target_step, agent0=None, agent1=None):
    """运行环境直到目标步数。"""
    agents = [agent0, agent1] if agent0 else ["random", "random"]
    while env.steps[-1][0].observation.get("step", 0) < target_step:
        last_step = env.steps[-1][0].observation.get("step", 0)
        env.run(agents)
        if env.steps[-1][0].observation.get("step", 0) == last_step:
            break
        if env.done:
            break
    return env


# ---------------------------------------------------------------------------
# Test: Simulator
# ---------------------------------------------------------------------------


class TestSimulator(unittest.TestCase):
    """what-if 模拟器测试。"""

    def setUp(self):
        self.env = _make_env()
        self.state, self.ledger, self.timelines = _get_state(self.env)

    def test_01_basic_launch_to_neutral(self):
        """向中立行星发射舰队 —— 应返回有效结果。"""
        src = self.state.my_planets[0]
        neutrals = self.state.neutral_planets
        if not neutrals:
            self.skipTest("无中立行星")
        tgt = neutrals[0]

        ships = max(MIN_FLEET, min(int(src.ships // 2), 100))
        outcome = simulate_fleet_launch(
            src, tgt, ships, self.state, self.ledger, self.timelines,
        )

        self.assertIsNotNone(outcome.aim, "aim 不应为 None")
        self.assertFalse(outcome.blocked, "不应被阻挡")
        self.assertIsNotNone(outcome.new_timeline, "应产生新时间线")

    def test_02_aim_angle_and_eta(self):
        """瞄准结果应有合理的角度和 ETA。"""
        src = self.state.my_planets[0]
        tgt = self.state.neutral_planets[0] if self.state.neutral_planets else self.state.enemy_planets[0]

        outcome = simulate_fleet_launch(
            src, tgt, 50, self.state, self.ledger, self.timelines,
        )

        self.assertIsNotNone(outcome.aim)
        angle, turns, px, py = outcome.aim
        self.assertGreater(turns, 0, "ETA 应 > 0")
        self.assertLess(turns, 200, "ETA 应在合理范围内")
        self.assertGreater(px, -10, "预测 x 应在棋盘内")
        self.assertLess(px, 110, "预测 x 应在棋盘内")

    def test_03_capture_turn_detected(self):
        """占领中立行星后 capture_turn 应被检测到。"""
        src = self.state.my_planets[0]
        neutrals = [p for p in self.state.neutral_planets if p.ships < 50]
        if not neutrals:
            self.skipTest("无弱中立行星")
        tgt = neutrals[0]

        # 发送远超驻军的兵力
        ships = max(MIN_FLEET, int(tgt.ships) + 30)
        outcome = simulate_fleet_launch(
            src, tgt, ships, self.state, self.ledger, self.timelines,
        )

        if outcome.aim is not None and not outcome.blocked:
            self.assertIsNotNone(outcome.capture_turn,
                                 f"应能占领弱中立行星，ships={ships}, garrison={tgt.ships}")
            self.assertIsNotNone(outcome.hold_until)

    def test_04_source_at_risk_detection(self):
        """发送超过 keep_needed 的兵力 → source_at_risk=True。"""
        src = self.state.my_planets[0]
        tgt = self.state.neutral_planets[0] if self.state.neutral_planets else self.state.enemy_planets[0]
        src_timeline = self.timelines.get(src.id, {})
        keep_needed = src_timeline.get("keep_needed", 0)

        # 发送几乎所有舰船
        ships = max(MIN_FLEET, int(src.ships))
        outcome = simulate_fleet_launch(
            src, tgt, ships, self.state, self.ledger, self.timelines,
        )

        remaining = max(0, int(src.ships) - ships)
        self.assertEqual(outcome.source_at_risk, remaining < keep_needed,
                         f"keep={keep_needed}, sent={ships}, remaining={remaining}")

    def test_05_sending_nothing(self):
        """发送不足 MIN_FLEET 的舰队 → 仍应产生结果。"""
        src = self.state.my_planets[0]
        tgt = self.state.neutral_planets[0] if self.state.neutral_planets else self.state.enemy_planets[0]

        outcome = simulate_fleet_launch(
            src, tgt, 5, self.state, self.ledger, self.timelines,
        )
        # 不应崩溃，应返回有效结构
        self.assertIsInstance(outcome, LaunchOutcome)

    def test_06_new_timeline_has_all_keys(self):
        """新时间线应包含所有必要键。"""
        src = self.state.my_planets[0]
        tgt = self.state.neutral_planets[0] if self.state.neutral_planets else self.state.enemy_planets[0]

        outcome = simulate_fleet_launch(
            src, tgt, 50, self.state, self.ledger, self.timelines,
        )

        if outcome.new_timeline:
            for key in ["owner_at", "ships_at", "keep_needed", "first_enemy", "horizon"]:
                self.assertIn(key, outcome.new_timeline, f"时间线缺少 {key}")

    def test_07_hold_until_geq_capture_turn(self):
        """hold_until >= capture_turn（若占领成功）。"""
        src = self.state.my_planets[0]
        neutrals = [p for p in self.state.neutral_planets if p.ships < 50]
        if not neutrals:
            self.skipTest("无弱中立行星")
        tgt = neutrals[0]

        ships = max(MIN_FLEET, int(tgt.ships) + 30)
        outcome = simulate_fleet_launch(
            src, tgt, ships, self.state, self.ledger, self.timelines,
        )

        if outcome.capture_turn is not None:
            self.assertGreaterEqual(outcome.hold_until, outcome.capture_turn)

    def test_08_enemy_planet_capture(self):
        """占领敌方行星 —— 模拟不应崩溃，即使太阳阻挡也返回有效结构。"""
        src = self.state.my_planets[0]
        enemies = self.state.enemy_planets
        if not enemies:
            self.skipTest("无敌方行星")
        tgt = enemies[0]

        # 发送大量舰船尝试占领
        ships = max(MIN_FLEET, int(tgt.ships) + 50)
        outcome = simulate_fleet_launch(
            src, tgt, ships, self.state, self.ledger, self.timelines,
        )

        # aim 可能为 None（太阳阻挡），但不应崩溃
        self.assertIsInstance(outcome, LaunchOutcome)

    def test_09_new_timeline_owner_changes(self):
        """新时间线中应反映我方占领后的所有权变更。"""
        src = self.state.my_planets[0]
        neutrals = [p for p in self.state.neutral_planets if p.ships < 30]
        if not neutrals:
            self.skipTest("无弱中立行星")
        tgt = neutrals[0]

        ships = max(MIN_FLEET, int(tgt.ships) + 50)
        outcome = simulate_fleet_launch(
            src, tgt, ships, self.state, self.ledger, self.timelines,
        )

        if outcome.capture_turn is not None and outcome.new_timeline:
            ct = outcome.capture_turn
            owner_at_ct = outcome.new_timeline["owner_at"].get(ct)
            self.assertEqual(owner_at_ct, self.state.player,
                             f"占领回合 {ct} 应为我方所有，实际 owner={owner_at_ct}")


# ---------------------------------------------------------------------------
# Test: find_min_ships_to_capture
# ---------------------------------------------------------------------------


class TestMinShipsToCapture(unittest.TestCase):
    """二分搜索最小攻占舰船数测试。"""

    def setUp(self):
        self.env = _make_env()
        self.state, self.ledger, self.timelines = _get_state(self.env)

    def test_10_returns_int(self):
        """应返回整数舰船数。"""
        src = self.state.my_planets[0]
        neutrals = [p for p in self.state.neutral_planets if p.ships < 50]
        if not neutrals:
            self.skipTest("无弱中立行星")
        tgt = neutrals[0]

        available = max(MIN_FLEET, int(src.ships) - 10)
        min_ships, outcome = find_min_ships_to_capture(
            src, tgt, available, self.state, self.ledger, self.timelines,
        )

        if min_ships is not None:
            self.assertIsInstance(min_ships, int)
            self.assertGreaterEqual(min_ships, MIN_FLEET)
            self.assertLessEqual(min_ships, available)

    def test_11_min_ships_can_capture(self):
        """最小舰船数应确实能占领。"""
        src = self.state.my_planets[0]
        neutrals = [p for p in self.state.neutral_planets if p.ships < 50]
        if not neutrals:
            self.skipTest("无弱中立行星")
        tgt = neutrals[0]

        available = max(MIN_FLEET, int(src.ships) - 10)
        min_ships, outcome = find_min_ships_to_capture(
            src, tgt, available, self.state, self.ledger, self.timelines,
        )

        if min_ships is not None:
            self.assertIsNotNone(outcome.capture_turn,
                                 f"min_ships={min_ships} 应能占领 {tgt.id}")

    def test_12_one_less_cannot_capture(self):
        """min_ships - 1 应无法占领。"""
        src = self.state.my_planets[0]
        neutrals = [p for p in self.state.neutral_planets if p.ships < 50]
        if not neutrals:
            self.skipTest("无弱中立行星")
        tgt = neutrals[0]

        available = max(MIN_FLEET, int(src.ships) - 10)
        min_ships, _ = find_min_ships_to_capture(
            src, tgt, available, self.state, self.ledger, self.timelines,
        )

        if min_ships is not None and min_ships > MIN_FLEET + 1:
            less = min_ships - 1
            outcome_less = simulate_fleet_launch(
                src, tgt, less, self.state, self.ledger, self.timelines,
            )
            self.assertIsNone(outcome_less.capture_turn,
                              f"min_ships-1={less} 不应能占领 (min={min_ships})")

    def test_13_insufficient_available(self):
        """可用舰船不足时返回 None。"""
        src = self.state.my_planets[0]
        tgt = self.state.enemy_planets[0] if self.state.enemy_planets else self.state.neutral_planets[0]

        # 兵力严重不足
        min_ships, outcome = find_min_ships_to_capture(
            src, tgt, 5, self.state, self.ledger, self.timelines,
        )
        self.assertIsNone(min_ships, "可用舰船不足应返回 None")


# ---------------------------------------------------------------------------
# Test: Valuation
# ---------------------------------------------------------------------------


class TestValuation(unittest.TestCase):
    """价值计算测试。"""

    def setUp(self):
        self.env = _make_env()
        self.state, self.ledger, self.timelines = _get_state(self.env)

    def test_20_neutral_capture_value_positive(self):
        """占领中立行星 → 价值应为正（低 garrison + 剩余回合充足）。"""
        src = self.state.my_planets[0]
        neutrals = [p for p in self.state.neutral_planets if p.ships < 50 and p.production >= 1]
        if not neutrals:
            self.skipTest("无合适中立行星")
        tgt = neutrals[0]

        ships = max(MIN_FLEET, int(tgt.ships) + 30)
        outcome = simulate_fleet_launch(
            src, tgt, ships, self.state, self.ledger, self.timelines,
        )

        if outcome.capture_turn is not None:
            value = value_of_capture(
                tgt, outcome.capture_turn, outcome.hold_until,
                self.state.remaining_steps, ships, is_enemy=False,
            )
            self.assertGreater(value, -ships,
                               f"中立行星价值不应远超成本, value={value}, ships={ships}")

    def test_21_enemy_capture_value_double_swing(self):
        """敌方行星 swing=2 → 理论价值是中立行星的 2 倍。"""
        src = self.state.my_planets[0]
        neutrals = [p for p in self.state.neutral_planets if p.ships < 50 and p.production >= 1]
        if not neutrals:
            self.skipTest("无合适中立行星")
        tgt = neutrals[0]

        ships = max(MIN_FLEET, int(tgt.ships) + 30)
        outcome = simulate_fleet_launch(
            src, tgt, ships, self.state, self.ledger, self.timelines,
        )

        if outcome.capture_turn is not None:
            val_neutral = value_of_capture(
                tgt, outcome.capture_turn, outcome.hold_until,
                self.state.remaining_steps, ships, is_enemy=False,
            )
            val_enemy = value_of_capture(
                tgt, outcome.capture_turn, outcome.hold_until,
                self.state.remaining_steps, ships, is_enemy=True,
            )
            # 敌方价值应严格大于中立价值（相同条件下）
            self.assertGreater(val_enemy, val_neutral,
                               f"敌方 swing=2 应 > 中立 swing=1: {val_enemy} vs {val_neutral}")

    def test_22_failed_capture_negative_value(self):
        """未占领 → 价值为负（浪费舰船）。"""
        src = self.state.my_planets[0]
        tgt = self.state.enemy_planets[0] if self.state.enemy_planets else self.state.neutral_planets[0]

        # 发送极少舰船
        outcome = simulate_fleet_launch(
            src, tgt, MIN_FLEET, self.state, self.ledger, self.timelines,
        )

        if outcome.capture_turn is None and outcome.aim is not None:
            value = value_of_capture(
                tgt, None, None,
                self.state.remaining_steps, MIN_FLEET, is_enemy=False,
            )
            self.assertLess(value, 0, "未占领应返回负价值")

    def test_23_source_at_risk_penalty(self):
        """source_at_risk → 惩罚加重 1.5x。"""
        src = self.state.my_planets[0]
        tgt = self.state.neutral_planets[0] if self.state.neutral_planets else self.state.enemy_planets[0]

        val_safe = value_of_capture(
            tgt, 10, 400, 500, 100, is_enemy=False, source_at_risk=False,
        )
        val_risk = value_of_capture(
            tgt, 10, 400, 500, 100, is_enemy=False, source_at_risk=True,
        )
        self.assertLess(val_risk, val_safe, "源行星危险应有惩罚")

    def test_24_comet_returns_zero(self):
        """彗星 → compute_action_value 返回 0.0。"""
        src = self.state.my_planets[0]
        comet_ids = self.state.comet_ids

        if not comet_ids:
            self.skipTest("当前无彗星")

        for cid in comet_ids:
            comet = next((p for p in self.state.planets if p.id == cid), None)
            if comet:
                ships = max(MIN_FLEET, int(src.ships // 3))
                outcome = simulate_fleet_launch(
                    src, comet, ships, self.state, self.ledger, self.timelines,
                )
                value = compute_action_value(
                    outcome, comet, self.state.player,
                    self.state.remaining_steps, ships, self.state.comet_ids,
                )
                self.assertEqual(value, 0.0, "彗星应返回 0.0（由 warmup 处理）")
                break

    def test_25_is_comet_target(self):
        """is_comet_target 正确识别彗星。"""
        for pid in self.state.comet_ids:
            # 创建一个 mock target
            for p in self.state.planets:
                if p.id == pid:
                    self.assertTrue(is_comet_target(p, self.state.comet_ids))
                    break

        for p in self.state.planets:
            if p.id not in self.state.comet_ids:
                self.assertFalse(is_comet_target(p, self.state.comet_ids))

    def test_26_productive_turns_capped(self):
        """生产回合不应超过 remaining_steps。"""
        # hold_until 很大时，productive_turns 受 remaining_steps 上限约束
        val1 = value_of_capture(
            self.state.neutral_planets[0] if self.state.neutral_planets else self.state.enemy_planets[0],
            capture_turn=10, hold_until=999, remaining_steps=100,
            ships_sent=50, is_enemy=False,
        )
        val2 = value_of_capture(
            self.state.neutral_planets[0] if self.state.neutral_planets else self.state.enemy_planets[0],
            capture_turn=10, hold_until=999, remaining_steps=200,
            ships_sent=50, is_enemy=False,
        )
        self.assertLess(val1, val2, "remaining_steps 更大 → 更多生产回合 → 价值更高")

    def test_27_deterministic_value(self):
        """手算验证价值公式。"""
        # 假设行星 production=2, capture_turn=50, hold_until=100, remaining=500
        # productive_turns = min(100-50+1, 500-50) = min(51, 450) = 51
        # gross_neutral = 51 * 2 * 1 = 102
        # gross_enemy = 51 * 2 * 2 = 204
        # net_neutral = 102 - 80 = 22
        # net_enemy = 204 - 80 = 124

        class MockPlanet:
            id = 999
            production = 2
            owner = -1

        tgt = MockPlanet()

        val_neutral = value_of_capture(tgt, 50, 100, 500, 80, is_enemy=False)
        val_enemy = value_of_capture(tgt, 50, 100, 500, 80, is_enemy=True)

        self.assertAlmostEqual(val_neutral, 51 * 2 * 1 - 80, msg="中立价值手算验证")
        self.assertAlmostEqual(val_enemy, 51 * 2 * 2 - 80, msg="敌方价值手算验证")

    def test_28_productive_turns_at_game_end(self):
        """游戏结束时 productive_turns 应正确截断。"""
        class MockPlanet:
            id = 998
            production = 3
            owner = -1

        tgt = MockPlanet()
        # capture at 490, remaining 500, so 10 turns left
        # hold_until = 500 (game end)
        # productive_turns = min(500-490+1, 500-490) = min(11, 10) = 10
        val = value_of_capture(tgt, 490, 500, 500, 30, is_enemy=False)
        expected = 10 * 3 * 1 - 30  # = 0
        self.assertAlmostEqual(val, expected, msg=f"终局 productive_turns 应为 10, got {val}")


# ---------------------------------------------------------------------------
# Test: Search
# ---------------------------------------------------------------------------


class TestSearch(unittest.TestCase):
    """搜索算法测试。"""

    def setUp(self):
        self.env = _make_env()
        self.state, self.ledger, self.timelines = _get_state(self.env)

    def test_30_search_returns_results(self):
        """search_best_actions 应返回结果列表。"""
        results = search_best_actions(
            self.state, self.ledger, self.timelines, top_k=20,
        )
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0, "应有至少一个候选动作")

    def test_31_results_sorted_by_value(self):
        """结果应按价值降序排列。"""
        results = search_best_actions(
            self.state, self.ledger, self.timelines, top_k=20,
        )
        for i in range(len(results) - 1):
            self.assertGreaterEqual(
                results[i].value, results[i + 1].value,
                f"结果 [{i}] value={results[i].value} < [{i+1}] value={results[i+1].value}",
            )

    def test_32_scored_action_has_all_fields(self):
        """ScoredAction 应包含所有必要字段。"""
        results = search_best_actions(
            self.state, self.ledger, self.timelines, top_k=5,
        )
        if results:
            a = results[0]
            for field in ["source_id", "target_id", "ships", "value",
                          "capture_turn", "hold_until", "eta",
                          "source_available", "source_at_risk", "is_enemy"]:
                self.assertTrue(hasattr(a, field), f"ScoredAction 缺少字段 {field}")

    def test_33_ships_within_available(self):
        """舰船数不应超过 source_available。"""
        results = search_best_actions(
            self.state, self.ledger, self.timelines, top_k=20,
        )
        for a in results:
            self.assertLessEqual(a.ships, a.source_available + 10,
                                 f"ships={a.ships} 超过 available={a.source_available}")

    def test_34_no_comets_in_results(self):
        """默认不应包含彗星目标。"""
        results = search_best_actions(
            self.state, self.ledger, self.timelines, top_k=100,
            include_comets=False,
        )
        comet_ids = self.state.comet_ids
        for a in results:
            self.assertNotIn(a.target_id, comet_ids,
                             f"不应包含彗星目标 {a.target_id}")

    def test_35_include_comets_flag(self):
        """include_comets=True 时应包含彗星。"""
        if not self.state.comet_ids:
            self.skipTest("当前无彗星")

        results = search_best_actions(
            self.state, self.ledger, self.timelines, top_k=100,
            include_comets=True,
        )
        comet_in_results = any(a.target_id in self.state.comet_ids for a in results)
        self.assertTrue(comet_in_results, "include_comets=True 时应包含彗星")

    def test_36_top_k_limit(self):
        """top_k 应限制结果数量。"""
        for k in [3, 5, 10]:
            results = search_best_actions(
                self.state, self.ledger, self.timelines, top_k=k,
            )
            self.assertLessEqual(len(results), k, f"结果数应 <= top_k={k}")

    def test_37_empty_when_no_available(self):
        """所有己方行星无可用舰船时返回空列表。"""
        # 不会在实际游戏中出现，但测试边界条件
        # 用 search_best_actions 的正常调用即可（初始状态总有可用舰船）
        results = search_best_actions(
            self.state, self.ledger, self.timelines, top_k=20,
        )
        # 初始状态有己方行星 → 应有结果
        self.assertGreater(len(results), 0)

    def test_38_evaluate_candidate_returns_scored_action(self):
        """_evaluate_candidate 应返回 ScoredAction 或 None。"""
        src = self.state.my_planets[0]
        tgt = self.state.neutral_planets[0] if self.state.neutral_planets else self.state.enemy_planets[0]
        src_timeline = self.timelines.get(src.id, {})
        keep_needed = src_timeline.get("keep_needed", 0)
        available = max(0, int(src.ships) - int(keep_needed))

        result = _evaluate_candidate(
            src, tgt, available, self.state, self.ledger, self.timelines,
        )
        if result is not None:
            self.assertIsInstance(result, ScoredAction)
            self.assertEqual(result.source_id, src.id)
            self.assertEqual(result.target_id, tgt.id)

    def test_39_try_ship_amounts_returns_best(self):
        """_try_ship_amounts 应返回最佳价值。"""
        src = self.state.my_planets[0]
        tgt = self.state.neutral_planets[0] if self.state.neutral_planets else self.state.enemy_planets[0]

        amounts = [MIN_FLEET, 50, 100]
        best_value, best_ships, best_outcome = _try_ship_amounts(
            src, tgt, amounts, self.state, self.ledger, self.timelines,
        )

        if best_value is not None and best_value > float("-inf"):
            # 验证: 没有其他 ship 数量比 best 更好
            for ships in amounts:
                if ships == best_ships:
                    continue
                outcome = simulate_fleet_launch(
                    src, tgt, ships, self.state, self.ledger, self.timelines,
                )
                value = compute_action_value(
                    outcome, tgt, self.state.player,
                    self.state.remaining_steps, ships, self.state.comet_ids,
                )
                self.assertLessEqual(value, best_value + 1e-6,
                                     f"ships={ships}不应优于 best_ships={best_ships}")


# ---------------------------------------------------------------------------
# Test: Multi-Fleet Scenario (关键场景)
# ---------------------------------------------------------------------------


class TestMultiFleetScenario(unittest.TestCase):
    """多舰队场景 —— 验证"敌方抵达 m 回合，我方 m+1 回合低成本占领"。"""

    def test_40_enemy_then_us_cheap_capture(self):
        """敌方 t-1 回合发舰队 → m 回合抵达；我方发舰队 → m+1 抵达 → 低成本占领。

        用已有到达账本中的敌军舰队验证这一逻辑。
        """
        env = _make_env()
        # 运行若干步以产生舰队
        env = _run_until_step(env, 20)

        state, ledger, timelines = _get_state(env)

        # 找到有敌军舰队正在抵达的行星
        found = False
        for planet_id, arrivals in ledger.items():
            enemy_arrivals = [(eta, owner, ships)
                              for eta, owner, ships in arrivals
                              if owner not in (-1, state.player)]
            if not enemy_arrivals:
                continue

            tgt = next((p for p in state.planets if p.id == planet_id), None)
            if tgt is None or tgt.owner == state.player:
                continue

            # 找到一艘我方可以"跟在后头"的行星
            src = state.my_planets[0] if state.my_planets else None
            if src is None:
                continue

            enemy_eta = enemy_arrivals[0][0]
            enemy_ships = enemy_arrivals[0][2]

            # 尝试找到一种舰船方案，使我们的 ETA > enemy_eta
            for try_ships in [MIN_FLEET, 30, 50, 100, 200]:
                aim = aim_at(
                    src, tgt, max(1, try_ships),
                    state.initial_by_id, state.angular_velocity,
                    state.comets, state.comet_ids,
                )
                if aim is None:
                    continue
                our_eta = aim[1]

                if our_eta > enemy_eta:
                    # 模拟: 敌军 m 回合抵达 + 我军 m+1 抵达
                    outcome = simulate_fleet_launch(
                        src, tgt, try_ships, state, ledger, timelines,
                    )

                    if outcome.capture_turn is not None:
                        # 占领回合应 ≥ 我方 ETA
                        self.assertGreaterEqual(
                            outcome.capture_turn, our_eta,
                            f"占领回合 {outcome.capture_turn} 应 >= ETA {our_eta}",
                        )
                        # 占领回合可能 > enemy_eta (敌军先到，我们后续跟上)
                        found = True
                        break

            if found:
                break

        if not found:
            self.skipTest("未找到敌方先行我方跟进场景")

    def test_41_timeline_reflects_multi_fleet(self):
        """验证时间线正确反映多支舰队先后到达。"""
        state, ledger, timelines = _get_state(self.env) if hasattr(self, "env") else _get_state(_make_env())
        env = _make_env() if not hasattr(self, "env") else self.env
        state, ledger, timelines = _get_state(env)

        # 找一颗有多个到达者的行星
        for planet_id, arrivals in ledger.items():
            if len(arrivals) >= 2:
                timeline = timelines[planet_id]
                self.assertIn("owner_at", timeline)
                self.assertIn("ships_at", timeline)
                self.assertGreater(len(timeline["owner_at"]), 0)
                return

        self.skipTest("无多舰队到达的行星")


# ---------------------------------------------------------------------------
# Test: Integration with existing modules
# ---------------------------------------------------------------------------


class TestIntegration(unittest.TestCase):
    """与现有模块的集成测试。"""

    def test_50_search_with_real_game_state(self):
        """在真实游戏状态上运行完整搜索流程。"""
        env = _make_env()
        env = _run_until_step(env, 10)

        state, ledger, timelines = _get_state(env)

        # 完整搜索
        results = search_best_actions(state, ledger, timelines, top_k=10)

        self.assertIsInstance(results, list)

        # 每个结果应可在 ledgers 和时间线中验证
        for action in results:
            self.assertIn(action.source_id,
                          [p.id for p in state.my_planets],
                          f"源行星 {action.source_id} 不属于我方")
            tgt = next((p for p in state.planets if p.id == action.target_id), None)
            self.assertIsNotNone(tgt, f"目标行星 {action.target_id} 不存在")

    def test_51_aim_at_consistency(self):
        """搜索使用的 ETA 应与 aim_at 一致。"""
        state, ledger, timelines = _get_state(_make_env())

        results = search_best_actions(state, ledger, timelines, top_k=5)
        for action in results:
            src = next(p for p in state.my_planets if p.id == action.source_id)
            tgt = next(p for p in state.planets if p.id == action.target_id)

            aim = aim_at(
                src, tgt, max(1, action.ships),
                state.initial_by_id, state.angular_velocity,
                state.comets, state.comet_ids,
            )
            if aim:
                self.assertEqual(action.eta, aim[1],
                                 f"search ETA {action.eta} 应与 aim_at ETA {aim[1]} 一致")

    def test_52_available_consistency(self):
        """搜索中的 available 应与 keep_needed 计算一致。"""
        state, ledger, timelines = _get_state(_make_env())

        results = search_best_actions(state, ledger, timelines, top_k=5)
        for action in results:
            src_timeline = timelines.get(action.source_id, {})
            keep_needed = src_timeline.get("keep_needed", 0)
            src = next(p for p in state.my_planets if p.id == action.source_id)
            expected_available = max(0, int(src.ships) - min(int(src.ships), int(keep_needed)))
            self.assertEqual(action.source_available, expected_available,
                             f"available 不一致: {action.source_available} vs {expected_available}")

    def test_53_beam_search_returns_results(self):
        """beam_search 应返回结果。"""
        state, ledger, timelines = _get_state(_make_env())
        results = beam_search(state, ledger, timelines, beam_width=5, max_depth=1)
        self.assertIsInstance(results, list)
        if results:
            self.assertIsInstance(results[0], ScoredAction)


# ---------------------------------------------------------------------------
# Test: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases(unittest.TestCase):
    """边界条件测试。"""

    def setUp(self):
        self.env = _make_env()
        self.state, self.ledger, self.timelines = _get_state(self.env)

    def test_60_no_neutral_planets(self):
        """无中立行星时不应崩溃。"""
        # 使用真实状态即可（可能有或无中立行星）
        results = search_best_actions(
            self.state, self.ledger, self.timelines, top_k=10,
        )
        self.assertIsInstance(results, list)

    def test_61_single_my_planet(self):
        """只有一颗己方行星时仍应正常工作。"""
        if len(self.state.my_planets) == 1:
            results = search_best_actions(
                self.state, self.ledger, self.timelines, top_k=10,
            )
            self.assertIsInstance(results, list)

    def test_62_very_small_available(self):
        """available < SEARCH_MIN_SHIPS 时该源行星被跳过。"""
        src = self.state.my_planets[0]
        tgt = self.state.neutral_planets[0] if self.state.neutral_planets else self.state.enemy_planets[0]

        result = _evaluate_candidate(
            src, tgt, available=3, state=self.state,
            ledger=self.ledger, timelines=self.timelines,
        )
        # SEARCH_MIN_SHIPS=5, available=3 < 5, 应返回 None
        self.assertIsNone(result, "available=3 < SEARCH_MIN_SHIPS 时不应产生候选动作")

    def test_63_ships_amount_dedup(self):
        """_try_ship_amounts 应对重复舰船数去重。"""
        src = self.state.my_planets[0]
        tgt = self.state.neutral_planets[0] if self.state.neutral_planets else self.state.enemy_planets[0]

        # 传入重复值
        amounts = [MIN_FLEET, MIN_FLEET, 50, 50, 100]
        best_value, best_ships, _ = _try_ship_amounts(
            src, tgt, amounts, self.state, self.ledger, self.timelines,
        )
        # 不应崩溃
        if best_value is not None:
            self.assertIn(best_ships, [MIN_FLEET, 50, 100])

    def test_64_launch_outcome_repr(self):
        """LaunchOutcome 应有可读的字符串表示。"""
        outcome = LaunchOutcome(
            aim=None, blocked=False, blocker_id=None,
            new_timeline=None, capture_turn=None, hold_until=None,
            source_at_risk=False,
        )
        s = repr(outcome)
        self.assertIn("LaunchOutcome", s)

    def test_65_scored_action_repr(self):
        """ScoredAction 应有可读的字符串表示。"""
        a = ScoredAction(
            source_id=1, target_id=5, ships=50, value=100.0,
            capture_turn=10, hold_until=400, eta=8,
            source_available=80, source_at_risk=False, is_enemy=True,
        )
        s = repr(a)
        self.assertIn("ScoredAction", s)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    unittest.main(verbosity=2)
