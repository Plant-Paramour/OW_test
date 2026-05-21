# Top 10% Episodes 数据格式参考文档

> **来源**: `Orbit Wars top 10% episodes 2026-05-04/`  
> **数量**: 2631 个对局文件  
> **对局类型**: 4 人制 (4-player FFA)  
> **选手来源**: Kaggle 排行榜前 10% 提交  
> **游戏版本**: module_version=1.29.0, version=1.0.9

---

## 1. 文件概览

```
Orbit Wars top 10% episodes 2026-05-04/
├── manifest.csv              # 对局元信息 (2631 条)
└── episodes/episodes/
    ├── 75829347.json         # ~3.6MB / 局 (压缩前)
    ├── 75829348.json
    └── ... (2631 个文件)
```

每个 JSON 文件包含一局完整的 4 人对局回放，包括每个玩家每步的观测和行动。

---

## 2. 顶层 JSON 结构

| 键 | 类型 | 说明 |
|----|------|------|
| `configuration` | `dict` | 游戏参数配置 |
| `description` | `str` | 游戏描述 |
| `id` | `str` | 对局 UUID |
| `info` | `dict` | 选手元信息 |
| `module_version` | `str` | Kaggle 环境版本 (1.29.0) |
| `name` | `str` | 游戏名 "orbit_wars" |
| `rewards` | `list[float]` | 4 个玩家的终局奖励 (1=胜, -1=负) |
| `schema_version` | `int` | Schema 版本 (1) |
| `specification` | `dict` | 动作/观测/配置的 Schema |
| `statuses` | `list[str]` | 4 个玩家的终局状态 (均为 "DONE") |
| `steps` | `list[list[dict]]` | **核心数据** — 每步每玩家的观测+行动 |
| `title` | `str` | "Orbit Wars" |
| `version` | `str` | "1.0.9" |

### 2.1 configuration

```python
{
    "actTimeout": 1,        # 单步决策时限 (秒)
    "agentTimeout": 2,      # 已废弃
    "cometSpeed": 4.0,      # 彗星移速
    "episodeSteps": 500,    # 最大步数
    "runTimeout": 1200,     # 单局总时限 (秒)
    "seed": None,           # 随机种子 (已被清除)
    "shipSpeed": 6.0,       # 舰船最大速度
}
```

### 2.2 info

```python
{
    "Agents": [
        {"Name": "bowwowforeach", "ThumbnailUrl": None},
        {"Name": "Vadasz", "ThumbnailUrl": None},
        {"Name": "HY2017", "ThumbnailUrl": None},
        {"Name": "kovi", "ThumbnailUrl": None},
    ],
    "EpisodeId": 75829347,
    "TeamNames": ["bowwowforeach", "Vadasz", "HY2017", "kovi"],
    "seed": 971724317,         # 解析后的实际种子
}
```

### 2.3 rewards

4 元素列表，对应 Player 0-3。Kaggle 4 人制的 rank 奖励：
- `1.0` = 第 1 名
- `-1.0` = 第 2/3/4 名（均等）

> ⚠️ **注意**: 此处的 `rewards` 是排名奖励，不是 manifest.csv 中的游戏分数。  
> manifest.csv 的 `scores` 列是实际游戏得分（浮点数，量级 ~1400-1700）。

---

## 3. steps 结构

### 3.1 整体结构

```python
steps: list[list[dict]]  # steps[step_index][player_index]
```

- `len(steps)` = 对局实际步数 (105~500，中位 ~256)
- `len(steps[t])` = 4（每步 4 个玩家）
- `steps[t][p]` = Player p 在步 t 的数据

### 3.2 每步元素

```python
step[t][p] = {
    "action": list[list],        # 该步发出的舰队指令
    "info": dict,                # 空 {} (未使用)
    "observation": dict,         # 步初的观测数据
    "reward": int,               # 中间步 = 0，终局 = 排名奖励
    "status": str,               # "ACTIVE" 或 "DONE"
}
```

### 3.3 重要约定

1. **Step 0**: 是初始状态，actions 均为空 `[]`
2. **观察时机**: `observation` 是步**初**状态（action 执行前）
3. **对局长度**: 最大 500 步。多数对局在 200-300 步内结束（所有行星被单一玩家占领）
4. **终局**: 最后一帧的 `status` = "DONE"，但中间步是 "ACTIVE"

---

## 4. Observation 详细格式

### 4.1 顶层字段

| 键 | 类型 | 说明 |
|----|------|------|
| `angular_velocity` | `float` | 轨道行星角速度 (≈0.0375 rad/turn) |
| `comet_planet_ids` | `list[int]` | 彗星行星 ID 列表 (4 个) |
| `comets` | `list[dict]` | 彗星状态 |
| `fleets` | `list[list]` | 飞行中舰队列表 |
| `initial_planets` | `list[list]` | 行星初始状态 (不变) |
| `next_fleet_id` | `int` | 下一个舰队 ID 计数器 |
| `planets` | `list[list]` | 行星当前状态 |
| `player` | `int` | 当前玩家编号 (0-3) |
| `remainingOverageTime` | `float` | 剩余超时配额 |
| `step` | `int` | 当前步数 |

### 4.2 planets — 行星数组

```python
# 格式: [id, owner, x, y, radius, ships, production]
#         0   1     2  3    4       5      6
planets = [
    [0,  0, 73.07, 95.92, 1.693, 7,  2],   # Player 0 的轨道行星
    [1,  1,  4.08, 73.07, 1.693, 12, 2],   # Player 1 的轨道行星
    [2,  3, 95.92, 26.93, 1.693, 47, 2],   # Player 3 的轨道行星
    [36, 0, 98.27, 30.96, 1.693,  5, 1],   # 彗星行星 (被 P0 占领)
    ...
]
```

| 索引 | 字段 | 类型 | 说明 |
|------|------|------|------|
| 0 | `id` | `int` | 行星 ID (0-39) |
| 1 | `owner` | `int` | 所有者 (-1=中立, 0-3=玩家) |
| 2 | `x` | `float` | X 坐标 (0-100) |
| 3 | `y` | `float` | Y 坐标 (0-100) |
| 4 | `radius` | `float` | 行星半径 (≈1.693) |
| 5 | `ships` | `int` | 当前驻军数 |
| 6 | `production` | `int` | 每回合产量 (轨道行星=2, 彗星=1) |

**行星分类**:
- **轨道行星**: ID 0-35 (36 个)，分布在以太阳(50,50)为中心的 4 个轨道上，每轨道 9 个
- **彗星行星**: ID 36-39 (4 个)，沿固定路径高速移动
- **初始分配**: 每个玩家占有 9 个轨道行星（对称），彗星初始中立

### 4.3 fleets — 舰队数组

```python
# 格式: [id, owner, x, y, angle, from_planet_id, ships]
#         0   1     2  3    4        5           6
fleets = [
    [63, 1, 13.81, 66.47, -0.5958, 1,  6],   # P1 从行星 1 发出的 6 舰船
    [64, 0, 72.77, 93.54,  1.4461, 26, 15],  # P0 从行星 26 发出的 15 舰船
    ...
]
```

| 索引 | 字段 | 类型 | 说明 |
|------|------|------|------|
| 0 | `id` | `int` | 舰队唯一 ID |
| 1 | `owner` | `int` | 所属玩家 |
| 2 | `x` | `float` | 当前 X 坐标 |
| 3 | `y` | `float` | 当前 Y 坐标 |
| 4 | `angle` | `float` | 行进角度 (弧度) |
| 5 | `from_planet_id` | `int` | 源行星 ID |
| 6 | `ships` | `int` | 舰船数 |

### 4.4 comets — 彗星数组

```python
comet = {
    "path_index": 0,                    # 当前路径索引 (0-31)
    "planet_ids": [36, 37, 38, 39],    # 4 个彗星行星 ID (旋转对称)
    "paths": [                          # 4 条路径 (旋转对称，各 32 点)
        [[x0,y0], [x1,y1], ... [x31,y31]],   # 路径 0
        [[x0,y0], [x1,y1], ... [x31,y31]],   # 路径 1 (旋转 90°)
        [[x0,y0], [x1,y1], ... [x31,y31]],   # 路径 2 (旋转 180°)
        [[x0,y0], [x1,y1], ... [x31,y31]],   # 路径 3 (旋转 270°)
    ],
}
```

**彗星关键参数** (来自游戏引擎):
- 生命期: 5-40 回合 (由 path_index 驱动，非计时器)
- 数量: 4 个 (旋转对称)
- 速度: 4.0
- 产量: 1 (固定)
- 半径: 1.0
- 生成时机: 步 50, 150, 250, 350, 450

**彗星过期判定**: `path_index >= len(paths[0])` 时彗星消失。过期发生在舰队发射**前**——占领者无法在最后一步提取驻军。

### 4.5 initial_planets — 初始行星状态

格式与 `planets` 相同（`[id, owner, x, y, radius, ships, production]`），存储游戏开始时的行星快照。用于判断行星是否在轨道上运行 (`is_rotating`)。

---

## 5. Action 格式

```python
# 格式: [source_planet_id, angle, ships]
#         0                 1      2
action = [8, 3.407355255736652, 12]  # 从行星 8 以角度 3.41 rad 发射 12 舰船
```

| 索引 | 字段 | 类型 | 说明 |
|------|------|------|------|
| 0 | `source_planet_id` | `int` | 源行星 ID |
| 1 | `angle` | `float` | 发射角度 (弧度) |
| 2 | `ships` | `int` | 舰船数 |

> ⚠️ **重要**: 与我们的内部动作表示不同。Kaggle 原生格式用 `(source, angle, ships)`，而我们的 PolicyNetwork 输出 `(source, target, ship_ratio)`，再由 `aim_at()` 计算角度。

### 5.1 空指令

当玩家不发射舰队时，action 为空列表 `[]`。

---

## 6. manifest.csv 格式

```csv
episode_id,create_time,sum_score,min_score,avg_score,scores,submission_ids,size_bytes
75873267,2026-05-04T22:52:04.755106+00:00,6385.0826,1484.2099,1596.2706,
  "[1658.877, 1484.210, 1700.414, 1541.581]",
  "[52318886, 52294319, 52332567, 52292204]",
  15545918
```

| 列 | 说明 |
|----|------|
| `episode_id` | 对局 ID (对应 JSON 文件名) |
| `scores` | 4 个玩家的实际游戏分数 (JSON 数组) |
| `submission_ids` | 4 个提交 ID |
| `sum_score` | 总分 |
| `avg_score` | 平均分 |

> **分数与排名奖励不同**: manifest 的 `scores` 是连续值 (1400-1700)，JSON 的 `rewards` 是离散排名 (1/-1)。

---

## 7. 关键统计数据 (N=300 采样)

### 7.1 基本指标

| 指标 | 值 |
|------|-----|
| 平均对局长度 | 256 步 (min=105, max=500) |
| 平均每步行动数 | 0.58 (人均) |
| 人均每局行动数 | ~148 |

### 7.2 胜者 vs 败者行为对比

| 指标 | 胜者 (Rank 1) | 败者 (Rank 4) |
|------|-------------|-------------|
| 总行动占比 | 74.9% | 25.1% |
| 行动比 | 3.0x | 1.0x |
| 每行动舰船 (中位) | 22 | 20 |
| 每行动舰船 (P75) | 49 | 39 |
| 行星控制 (平均) | 13.7 | 3.6 |
| 彗星进攻占比 | 1.56% | 1.99% |

### 7.3 分阶段行动密度（胜/败比）

| 步数 | 胜者行动 | 败者行动 | 胜/败比 | 解读 |
|------|---------|---------|--------|------|
| 0-49 | 7,622 | 17,117 | **0.45** | 败者更活跃！ |
| 50-99 | 16,971 | 27,648 | **0.61** | 败者仍领先 |
| 100-149 | 22,891 | 19,368 | 1.18 | 胜者开始反超 |
| 150-199 | 20,029 | 10,619 | 1.89 | 胜者拉开差距 |
| 200-249 | 11,331 | 5,440 | 2.08 | 败者衰落 |
| 250-299 | 5,104 | 3,038 | 1.68 | |
| 300-349 | 3,118 | 1,901 | 1.64 | |
| 350-399 | 3,036 | 1,495 | 2.03 | |
| 400-449 | 2,105 | 1,021 | 2.06 | |
| 450-499 | 2,135 | 1,162 | 1.84 | |

### 7.4 分阶段舰船数（中位）

| 步数 | 胜者中位 | 败者中位 | 解读 |
|------|---------|---------|------|
| 0-49 | 23 | 17 | 胜者早期用更多舰船 |
| 50-149 | 24-27 | 20-21 | 相近 |
| 150-299 | 20-22 | 21-28 | 败者逐渐增加 |
| 300-449 | **15-20** | **34-73** | 败者恐慌式大额发射！|

**核心洞察**: 晚期败者舰船数是胜者的 2-5 倍 — 这是绝望的 all-in 行为。

### 7.5 舰船数分位数

| 分位数 | 胜者 | 败者 |
|--------|------|------|
| P10 | 6 | 7 |
| P25 | 11 | 12 |
| P50 | 22 | 20 |
| P75 | 49 | 39 |
| P90 | 97 | 82 |
| P95 | 138 | 131 |

---

## 8. 用于训练的读取方式

### 8.1 提取胜者训练样本 (Behavior Cloning)

```python
import json

def extract_winner_samples(filepath: str) -> list[dict]:
    """提取胜者的 (observation, action) 对用于行为克隆。"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    winner_idx = data['rewards'].index(1.0)
    samples = []
    
    for step in data['steps']:
        winner_entry = step[winner_idx]
        actions = winner_entry.get('action', [])
        if not actions:
            continue
        obs = winner_entry['observation']
        for action in actions:
            samples.append({
                'observation': obs,   # 需要对 planets/fleets/comets 做特征工程
                'action': action,     # [source_id, angle, ships]
            })
    
    return samples
```

### 8.2 特征转换要点

Kaggle 原生 observation 需要先通过 `src/world/observation.py:parse_observation()` 转换为 `GameState`，再用 `src/features/builder.py:build_decision_matrix()` 构建特征，才能送入我们的 PolicyNetwork。

原生 action `[source_id, angle, ships]` 需要反向解析：
1. 用 `angle` 和 `ships` 反推 `target_id`（在候选行星中找角度最接近的）
2. 或用 `src/engine/interception.py:aim_at()` 重算

### 8.3 内存考虑

- 单个 JSON 文件: ~3-15 MB
- 2631 个文件总计: ~8.5 GB (未压缩)
- 建议**流式读取**或用 `ijson` 逐元素解析

---

## 9. 常用选手名

从 manifest 和 episode 数据中高频出现的选手：

| 选手名 | 出现频率 |
|--------|---------|
| bowwowforeach | 极高 |
| Vadasz | 极高 |
| HY2017 | 极高 |
| kovi | 高 |
| 其他 | 中等 |

> 这些选手的策略可作为对手建模的参考。

---

## 10. 与项目内部数据格式的映射

| Kaggle 原生 | 项目内部 |
|-------------|---------|
| `observation` (dict) | `GameState` (via `parse_observation()`) |
| `action = [source, angle, ships]` | `(target_id, ship_ratio)` (via `aim_at()` + `DecisionRow`) |
| `planets[i] = [id, owner, x, y, r, ships, prod]` | `PlanetState` dataclass |
| `fleets[i] = [id, owner, x, y, angle, from, ships]` | `FleetState` dataclass |
| `comets[i] = {path_index, paths, planet_ids}` | `comets` list + `comet_ids` set |
| `rewards = [±1, ±1, ±1, ±1]` | `score` via `compute_step_reward()` |
