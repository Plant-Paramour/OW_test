# CLAUDE.md

Orbit Wars 智能体项目 —— 物理引擎 + 世界模型 + PPO 强化学习 + 动作搜索。

## 路径 & 环境

- **Python**: `C:\ProgramData\anaconda3\envs\Orbit_Wars\python.exe` (conda 环境)
- **编码**: 任何打印中文的脚本需要 `sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')` 放在文件顶部
- **文件路径**: 所有文件操作使用完整 Windows 绝对路径 + 反斜杠

## 验证命令

```bash
PY="C:\ProgramData\anaconda3\envs\Orbit_Wars\python.exe"
BASE="C:\code\[kaggle]\Orbit Wars"

$PY "$BASE\scripts\verify_engine.py"      # 40 tests — 物理引擎
$PY "$BASE\scripts\verify_features.py"    # 69 tests — 特征工程
$PY "$BASE\scripts\verify_action.py"      # 690 tests — 动作空间
$PY "$BASE\scripts\verify_reward.py"      # 69 tests — 奖励塑形
$PY "$BASE\scripts\verify_model.py"       # 149 tests — 策略网络
$PY "$BASE\scripts\verify_ppo.py"         # 64 tests — PPO 训练
$PY "$BASE\scripts\verify_opponents.py"   # 43 tests — 对手模块
$PY "$BASE\scripts\verify_search.py"      # 44 tests — 动作搜索
```

## 训练 & 工具

```bash
# 训练入口 (默认开启搜索引导 alpha=0.05)
$PY "$BASE\src\train.py"

# Phase 0: 早期发育 (50 回合截断, PV 终局奖励)
$PY "$BASE\src\train.py" --config configs/curriculum_phase0_early.yaml

# Phase 1-3: 完整课程训练
$PY "$BASE\src\train.py" --config configs/curriculum_phase1.yaml
$PY "$BASE\src\train.py" --config configs/curriculum_phase2.yaml --resume checkpoint_500.pt
$PY "$BASE\src\train.py" --config configs/curriculum_phase3.yaml --resume checkpoint_1500.pt

# 生成回放
$PY "$BASE\scripts\gen_replays.py" -n 10 --opponent random --ckpt checkpoint_500.pt

# BC 预训练 (Sniper -> 策略网络)
$PY "$BASE\scripts\bc_sniper.py" --demo-games 200 --epochs 50 --validate

# 回放查看器: 浏览器打开 replays/viewer.html
```

## 架构

```
src/engine/        物理引擎 (纯函数, 无副作用)
  constants.py       棋盘几何, 速度上限, 太阳安全余量
  physics.py         舰队速度曲线, 太阳碰撞, ETA 估算
  prediction.py      行星公转, 彗星轨迹, 统一位置预测
  interception.py    迭代拦截求解器 (浮点 ETA, 50 次迭代) + 中途行星阻挡检测

src/world/         世界模型 (状态解析 & 未来模拟)
  types.py           GameState 数据类
  observation.py     原始 obs -> GameState
  fleet_tracker.py   射线-圆命中判定 -> 到达账本 {planet_id: [(eta, owner, ships)]}
  combat.py          同回合战斗结算, 行星时间线模拟 (含二分 keep_needed)

src/features/      特征工程 (Phase 2, 69/69 tests)
  self_features.py      (21d) 源行星特征
  candidate_features.py (29d) 候选目标特征
  global_features.py    (16d) 全局态势特征
  builder.py            DecisionMatrix 组装器

src/policy/        策略网络 (Phase 5, 149/149 tests)
  model.py           三编码器 (Self/Candidate/Global) -> 联合嵌入 -> 四个头
  action_head.py     Categorical 目标选择 + Beta 舰船比例
  value_head.py      状态价值估计 (218K 参数)

src/env/           环境封装 & 奖励 (Phase 4, 69/69 tests)
  reward.py          对称势能 Phi = my_PV - enemy_PV (PBRS)
  wrapper.py         OrbitWarsEnv + OpponentLike + comet warmup + soft_clip
                     + 搜索 Logit Bias (search_alpha > 0 时偏置策略 logits)

src/ppo/           PPO 训练 (Phase 6, 64/64 tests)
  buffer.py          RolloutBuffer + GAE
  update.py          PPO 更新 (混合离散+连续, masked_mean)
  trainer.py         训练循环 + comet EMA + 搜索 alpha 退火调度

src/opponents/     课程对手 (Phase 7, 43/43 tests)
  sniper.py          最近行星狙击手
  heuristic.py       完整战术启发式
  pool.py            加权随机对手池
  self_play.py       SelfPlay 封装 PolicyNetwork

src/search/        动作搜索 (Phase 9, 44/44 tests)
  simulator.py       What-if 模拟: 注入舰队 -> 重模拟时间线 -> 分析占领
  valuation.py       价值公式: productive_turns * production * swing - ships_sent
  search.py          枚举 (source, target, ships) -> 二分最优舰船 -> 排序
```

## 核心设计决策

- **智能体不输出角度**: 动作为 (source, target, ship_ratio), 物理引擎计算 atan2 和 ETA
- **ETA 替代距离**: fleet_speed(ships) 非线性, 预计算 ETA 比让网络学 dist/speed 更高效
- **对称势能函数**: Phi = my_PV - enemy_PV, 纯 PBRS, 自动涵盖 6 种所有权变化
- **彗星 warmup**: 自适应探索引导, PBRS 证明不扭曲最优策略
- **浮点 ETA 瞄准**: estimate_arrival_float() + 位置插值 + 50 次迭代, 消除远距离偏差
- **搜索 Logit Bias**: 搜索算法评估每个候选的占领价值, 作为偏置加入策略 logits
  - `augmented_logits = target_logits + search_alpha * search_value`
  - 初期 alpha=0.05~0.08 引导探索, 逐步退火至 0.005, 最终关闭
  - 配置: configs/*.yaml 中的 `search` 段

## 数据流

```
原始 obs -> parse_observation() -> GameState
  -> build_arrival_ledger()      -> {planet_id: [(eta, owner, ships)]}
  -> simulate_planet_timeline()  -> {owner_at, ships_at, keep_needed, ...}
  -> build_decision_matrix()     -> [DecisionRow, ...]
  -> policy.forward()            -> target_logits + value
  -> [+ search_bias]             -> augmented_logits  (search_alpha > 0 时)
  -> sample_action()             -> (target_id, ship_ratio)
  -> physics.estimate_arrival()  -> (angle, eta) 用于执行
```

## 进度

| Phase | 状态 | 测试 |
|-------|------|------|
| 0. 早期发育课程 | **完成** | Phase 0 配置 + 终局 Φ 奖励 |
| 1. 物理引擎 | 完成 | 40/40 |
| 2. 特征工程 | 完成 | 69/69 |
| 3. 动作空间 | 完成 | 690/690 |
| 4. 奖励塑形 | 完成 | 69/69 |
| 5. 策略网络 | 完成 | 149/149 |
| 6. PPO 训练 | 完成 | 64/64 |
| 7. 课程对手 | 完成 | 44/44 |
| 8. 评估 & 调优 | **进行中** | Phase 0 就绪, BC Sniper 完成, 回放系统就绪 |
| 9. 动作搜索 + 集成 | 完成 | 44/44 |

总测试: 1169/1169 (100%)

## 课程训练计划 (4 阶段)

| Phase | 配置 | Updates | 回合 | 对手 | 搜索 α |
|-------|------|---------|------|------|--------|
| 0 早期发育 | `curriculum_phase0_early.yaml` | 500 | 50 | Random | 0.15 |
| 1 热身 | `curriculum_phase1.yaml` | 500 | 500 | Random | 0.08 |
| 2 基础对抗 | `curriculum_phase2.yaml` | 1000 | 500 | Random(40%)+Sniper(60%) | 0.03 |
| 3 战术对抗 | `curriculum_phase3.yaml` | 1500 | 500 | Sniper(40%)+Heuristic(60%) | 0.0 |

**Phase 0 核心设计：**
- 50 回合截断对局，终局奖励 = `clip(scale × Φ, ±15)` where `scale=3.0`
- Φ = 对称势能函数 (my_PV − enemy_PV)，综合评估已有舰队 + 飞行舰队 + 行星产能
- 目标：让智能体在前期 0-50 回合学会疯狂扩张，不顾后期战术
- 回合结束后自动 hot-start 到 Phase 1 完整对局

**搜索值归一化：**
- 搜索原始值量级数千（productive_turns × prod × swing），不适合直接当 logit bias
- `_build_search_value_map` 将值除以 `remaining_steps`，归一化到 [−1, 10] 范围
- 归一化后的 alpha 偏置约 0.1~1.5（对 logits 的 10-30% 影响），合理引导探索

## 已知设计问题

- **终局奖励被 soft_clip 部分压缩**: wrapper.py soft_clip 对大额奖励有压缩, 终局 +-10 被压到 ~6.5-8
- **全行星无可用舰船时 reward 丢失**: 极端罕见, 所有己方行星 available==0 时整步无 transition
- **中途行星拦截**: 已通过 check_path_blocked() 修复

## 参考文件

- `docs/episode-json-format.md` — Top 10% episode JSON 数据格式
- `deprecated/` — 历史文档 (实施规划、方案分析、多人规则详解、技术接口文档)
- `Data Explorer/` — Kaggle 竞赛规则原文及中文翻译
- `other's_work/` — 参考 notebooks (含 Structured Baseline v11)
