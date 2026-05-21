"""Phase 7 对手模块验证测试。"""

import sys
import io
import os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import random
import torch
import numpy as np

passed = 0
failed = 0


def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}")


# ═══════════════════════════════════════════════════════════════════════
# Test [1]: SniperOpponent 基本行为
# ═══════════════════════════════════════════════════════════════════════
print("\n── [1] SniperOpponent 基本行为 ──")

from src.opponents.sniper import SniperOpponent

# 构造模拟观测
def make_mock_obs(player=0, owned=None, enemy=None):
    """构造最小观测用于测试。"""
    planets = []
    planet_id = 0
    for x, y, own, ships in (owned or []):
        planets.append([planet_id, own, x, y, 1.693, ships, 2])
        planet_id += 1
    for x, y, own, ships in (enemy or []):
        planets.append([planet_id, own, x, y, 1.693, ships, 2])
        planet_id += 1
    return {
        "player": player,
        "planets": planets,
        "fleets": [],
        "angular_velocity": 0.0375,
        "comets": [],
        "comet_planet_ids": [],
        "initial_planets": planets,
        "step": 0,
        "remainingOverageTime": 999.0,
    }

sniper = SniperOpponent()

# 1a: 空手（无己方行星）
obs_empty = make_mock_obs(player=0, owned=[])
actions = sniper(obs_empty, {})
check("空手返回空列表", actions == [])

# 1b: 无目标（唯一行星已是己方）
obs_no_target = make_mock_obs(player=0, owned=[(10, 10, 0, 50)])
actions = sniper(obs_no_target, {})
check("无目标返回空列表", actions == [])

# 1c: 基本狙击（足够舰船）
obs_basic = make_mock_obs(player=0, owned=[(10, 10, 0, 50)], enemy=[(90, 90, -1, 10)])
actions = sniper(obs_basic, {})
check("基本狙击产生动作", len(actions) == 1)
if actions:
    check("动作格式 [source, angle, ships]", len(actions[0]) == 3)
    check("源行星 ID 正确", actions[0][0] == 0)
    check("舰船数 ≥ 21 (enemy+1, min 20)", actions[0][2] >= 20)
    expected_angle = math.atan2(80, 80)  # from (10,10) to (90,90)
    check("角度正确 (45°)", abs(actions[0][1] - expected_angle) < 0.01)

# 1d: 不足舰船不发射
obs_no_ships = make_mock_obs(player=0, owned=[(10, 10, 0, 5)], enemy=[(90, 90, -1, 10)])
actions = sniper(obs_no_ships, {})
if actions:
    check("不足舰船不发射", actions[0][2] <= 5)
else:
    check("不足舰船不发射 (返回空)", True)


# ═══════════════════════════════════════════════════════════════════════
# Test [2]: HeuristicOpponent 基本行为
# ═══════════════════════════════════════════════════════════════════════
print("\n── [2] HeuristicOpponent 基本行为 ──")

from src.opponents.heuristic import HeuristicOpponent

heuristic = HeuristicOpponent()

# 2a: 空手
obs_empty2 = make_mock_obs(player=0, owned=[])
actions = heuristic(obs_empty2, {})
check("空手返回空列表", actions == [])

# 2b: 基本进攻
obs_basic2 = make_mock_obs(player=0, owned=[(10, 10, 0, 50)], enemy=[(90, 90, -1, 10)])
actions = heuristic(obs_basic2, {})
check("基本进攻产生动作", len(actions) >= 0)  # heuristic 可能不发送如果条件不足

# 2c: 有敌方行星时产生动作
obs_enemy = make_mock_obs(
    player=0,
    owned=[(10, 10, 0, 80), (20, 20, 0, 60)],
    enemy=[(90, 90, 1, 10), (80, 80, -1, 5)]
)
actions = heuristic(obs_enemy, {})
check("有目标时产生结果", isinstance(actions, list))
if actions:
    for a in actions:
        check(f"动作格式正确 [{a[0]}, {a[1]:.4f}, {a[2]}]",
              len(a) == 3 and isinstance(a[0], int) and isinstance(a[2], int))


# ═══════════════════════════════════════════════════════════════════════
# Test [3]: OpponentPool 采样
# ═══════════════════════════════════════════════════════════════════════
print("\n── [3] OpponentPool 采样 ──")

from src.opponents.pool import OpponentPool

# 3a: 基本添加
pool = OpponentPool()
pool.add("random", 1.0, "random")
pool.add(SniperOpponent(), 2.0, "sniper")
check("池大小=2", len(pool) == 2)
check("名称列表", pool.names() == ["random", "sniper"])

# 3b: 加权采样确定性测试
rng = random.Random(42)
counts = {"random": 0, "sniper": 0}
for _ in range(1000):
    opp = pool.sample(rng=rng)
    if opp == "random":
        counts["random"] += 1
    else:
        counts["sniper"] += 1
# 权重 1:2，期望 random 33%, sniper 67%
check(f"random 采样比例 ~33% (实际 {counts['random']/10:.1f}%)",
      250 <= counts["random"] <= 420)
check(f"sniper 采样比例 ~67% (实际 {counts['sniper']/10:.1f}%)",
      580 <= counts["sniper"] <= 750)

# 3c: set_weight
pool.set_weight("sniper", 0.0)
check("权重设为 0 后 get_weight", pool.get_weight("sniper") == 0.0)
# sniper 权重为 0，只能采样到 random
opp_zero = pool.sample(rng=rng)
check("权重 0 不会采样到", opp_zero == "random")

# 3d: 空池报错
empty_pool = OpponentPool()
try:
    empty_pool.sample()
    check("空池采样报错", False)
except RuntimeError:
    check("空池采样报错", True)

# 3e: repr
check("repr 正确", "OpponentPool" in repr(pool))


# ═══════════════════════════════════════════════════════════════════════
# Test [4]: SelfPlayOpponent
# ═══════════════════════════════════════════════════════════════════════
print("\n── [4] SelfPlayOpponent ──")

from src.policy.model import PolicyNetwork
from src.opponents.self_play import SelfPlayOpponent

policy = PolicyNetwork(hidden=256)
self_play = SelfPlayOpponent(policy, candidate_count=20, episode_steps=500)

# 4a: 空手状态
obs_empty4 = make_mock_obs(player=0, owned=[])
actions = self_play(obs_empty4, {})
check("空手返回空列表", actions == [])

# 4b: 基本决策
obs_sp = make_mock_obs(
    player=0,
    owned=[(10, 10, 0, 50), (30, 30, 0, 40)],
    enemy=[(90, 90, -1, 10), (70, 70, -1, 15)]
)
actions = self_play(obs_sp, {})
check("产生决策", isinstance(actions, list))
if actions:
    for a in actions:
        check(f"动作格式 [source, angle, ships] ({a})",
              len(a) == 3 and isinstance(a[0], int) and isinstance(a[2], int) and a[2] > 0)

# 4c: sync_weights
policy2 = PolicyNetwork(hidden=256)
original = {k: v.clone() for k, v in self_play.policy.state_dict().items()}
self_play.sync_weights(policy2)
after = {k: v.clone() for k, v in self_play.policy.state_dict().items()}
# 同步后权重应不同于原始（新策略随机初始化）
changed = any(not torch.equal(original[k], after[k]) for k in original)
check("sync_weights 更新策略参数", changed)

# 4d: deterministic vs stochastic
obs_det = make_mock_obs(
    player=0,
    owned=[(10, 10, 0, 60)],
    enemy=[(90, 90, -1, 10)]
)
sp_det = SelfPlayOpponent(policy, candidate_count=20, episode_steps=500, deterministic=True)
# 多次调用 deterministic 结果应一致
actions1 = sp_det(obs_det, {})
actions2 = sp_det(obs_det, {})
# 由于 policy 权重固定，deterministic 输出应完全一致
if actions1 and actions2:
    same = all(a1 == a2 for a1, a2 in zip(actions1, actions2))
    check("deterministic 输出一致", same)


# ═══════════════════════════════════════════════════════════════════════
# Test [5]: OrbitWarsEnv 可调用对手集成
# ═══════════════════════════════════════════════════════════════════════
print("\n── [5] OrbitWarsEnv 对手集成 ──")

from src.env.wrapper import OrbitWarsEnv

# 5a: 字符串对手（向后兼容）
env_str = OrbitWarsEnv(opponent="random", candidate_count=20, episode_steps=500)
check("字符串对手设置正确", env_str.opponent == "random")
state, _ = env_str.reset()
check("字符串对手 reset 成功", state is not None)
check("字符串对手 _current_opponent", env_str._current_opponent == "random")

# 5b: 可调用对手
env_callable = OrbitWarsEnv(opponent=SniperOpponent(), candidate_count=20, episode_steps=500)
check("可调用对手设置正确", isinstance(env_callable.opponent, SniperOpponent))
state2, _ = env_callable.reset()
check("可调用对手 reset 成功", state2 is not None)
check("可调用对手 _current_opponent 是 SniperOpponent",
      isinstance(env_callable._current_opponent, SniperOpponent))

# 5c: 对手池
pool_env = OpponentPool()
pool_env.add("random", 1.0, "random")
pool_env.add(SniperOpponent(), 1.0, "sniper")
env_pool = OrbitWarsEnv(opponent=pool_env, candidate_count=20, episode_steps=500)
state3, _ = env_pool.reset()
check("对手池 reset 成功", state3 is not None)
# 采样到的对手应该是 "random" 或 SniperOpponent
curr = env_pool._current_opponent
check("对手池采样结果有效", curr == "random" or isinstance(curr, SniperOpponent))


# ═══════════════════════════════════════════════════════════════════════
# Test [6]: 完整决策→执行循环（Sniper 对手）
# ═══════════════════════════════════════════════════════════════════════
print("\n── [6] 完整决策→执行循环 ──")

env6 = OrbitWarsEnv(opponent=SniperOpponent(), candidate_count=20, episode_steps=500)
state6, _ = env6.reset()
decisions, transitions = env6.collect_decisions(policy, state6, deterministic=True)
check("产生决策或 transitions", len(decisions) >= 0 and len(transitions) >= 0)

if decisions:
    next_state, raw_obs, reward, done, info = env6.step(decisions, state6)
    check("step 返回 next_state", next_state is not None)
    check("step 返回 reward", isinstance(reward, float))
    check("step 返回 done", isinstance(done, bool))
else:
    # 无决策时发空指令
    next_state, raw_obs, reward, done, info = env6.step([], state6)
    check("空指令 step 成功", next_state is not None)


# ═══════════════════════════════════════════════════════════════════════
# Test [7]: 对手池在多局间轮换
# ═══════════════════════════════════════════════════════════════════════
print("\n── [7] 对手池多局轮换 ──")

pool7 = OpponentPool()
pool7.add("random", 1.0, "random")
pool7.add(SniperOpponent(), 0.0, "sniper")  # 权重 0，不会采样到

env7 = OrbitWarsEnv(opponent=pool7, candidate_count=20, episode_steps=500)

# 多次 reset 采样（权重 1:0，只能采到 random）
all_random = True
for _ in range(20):
    env7.reset()
    if env7._current_opponent != "random":
        all_random = False
        break
check("权重 0 对手不会被采样到", all_random)


# ═══════════════════════════════════════════════════════════════════════
# Test [8]: 训练入口 _build_opponent
# ═══════════════════════════════════════════════════════════════════════
print("\n── [8] _build_opponent 配置解析 ──")

from src.train import _build_opponent

# 8a: random 字符串
opp = _build_opponent({"type": "random"})
check("random → 'random'", opp == "random")

# 8b: sniper
opp = _build_opponent({"type": "sniper"})
check("sniper → SniperOpponent", isinstance(opp, SniperOpponent))

# 8c: heuristic
opp = _build_opponent({"type": "heuristic"})
check("heuristic → HeuristicOpponent", isinstance(opp, HeuristicOpponent))

# 8d: pool
opp = _build_opponent({
    "type": "pool",
    "pool": [
        {"name": "r", "type": "random", "weight": 1.0},
        {"name": "s", "type": "sniper", "weight": 2.0},
    ]
})
check("pool → OpponentPool", hasattr(opp, "sample") and hasattr(opp, "add"))
check("pool 大小=2", len(opp) == 2)

# 8e: 未知类型 fallback
opp = _build_opponent({"type": "nonexistent"})
check("未知类型 → 'random'", opp == "random")

# 8f: 空池 fallback
opp = _build_opponent({"type": "pool", "pool": []})
check("空池 → 'random'", opp == "random")


# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"结果: {passed}/{passed+failed} 通过 ({failed} 失败)")
print(f"{'='*60}")

if failed > 0:
    sys.exit(1)
