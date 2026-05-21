"""v4 Agent 行为克隆 —— 收集 v4+ML+topk1 前 45 回合演示数据，训练策略网络模仿。

用法:
  python scripts/bc_v4.py                           # 收集数据 + 训练
  python scripts/bc_v4.py --collect-only            # 仅收集数据
  python scripts/bc_v4.py --train-only              # 仅训练（使用已有数据）
  python scripts/bc_v4.py --demo-games 500 --epochs 100
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

import argparse
import importlib.util
import math
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from kaggle_environments import make

from src.world.observation import parse_observation
from src.features.builder import build_decision_matrix
from src.policy.model import PolicyNetwork


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "artifacts" / "bc_v4_data.pkl"
CHECKPOINT_PATH = ROOT / "artifacts" / "bc_v4_policy.pt"
V4_PATH = ROOT / "other's_work" / "train-submit-v4-ml-validator-topk2-tutorial.py"

MAX_STEPS = 45  # 每局收集前 45 回合
CANDIDATE_COUNT = 20
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════════
# v4 Agent 模块加载
# ═══════════════════════════════════════════════════════════════════

def _load_v4_agent():
    """加载 v4+ML+topk1 agent 模块。"""
    spec = importlib.util.spec_from_file_location("v4_agent", str(V4_PATH))
    v4 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v4)
    has_ml = v4._W is not None
    if has_ml:
        print(f"v4 agent 加载成功 (含 ML 验证器权重)")
    else:
        print(f"v4 agent 加载成功 (无 ML 验证器，仅 topk1)")
    return v4


# ═══════════════════════════════════════════════════════════════════
# 数据收集
# ═══════════════════════════════════════════════════════════════════

def collect_demonstrations(v4, n_games: int = 200, max_steps: int = MAX_STEPS):
    """运行 v4 vs Random 对局，收集 (state, v4_actions) 演示数据。

    Returns:
        list of (GameState, list[list]) — 每步的解析后状态和 v4 动作
    """
    raw_data = []
    total_steps_collected = 0

    for game_idx in range(n_games):
        env = make("orbit_wars", debug=True)
        trainer = env.train([None, "random"])
        obs = trainer.reset()

        for step in range(max_steps):
            v4_actions = v4.agent(obs) or []

            state = parse_observation(obs, episode_steps=500)
            raw_data.append((state, v4_actions))
            total_steps_collected += 1

            obs, reward, done, info = trainer.step([v4_actions])
            if done:
                break

        if (game_idx + 1) % 50 == 0:
            print(f"  已收集 {game_idx + 1}/{n_games} 局 ({total_steps_collected} 步)")

    print(f"数据收集完成: {n_games} 局, {total_steps_collected} 步")
    return raw_data


# ═══════════════════════════════════════════════════════════════════
# 动作映射：v4 动作 → 策略网络动作空间
# ═══════════════════════════════════════════════════════════════════

def map_demonstrations(v4, raw_data, candidate_count=CANDIDATE_COUNT):
    """将 v4 原始动作映射为策略网络的训练样本。

    每一条演示数据 (state, v4_actions) → 多个训练样本 (per-source-planet)。

    Returns:
        list of dicts: {self_feat, cand_feat, global_feat, mask, target_idx, ship_ratio}
    """
    examples = []
    skipped_no_target = 0
    skipped_no_candidate = 0

    for state, v4_actions in raw_data:
        rows = build_decision_matrix(state, candidate_count=candidate_count)
        if not rows:
            continue

        source_groups = {}
        for row in rows:
            source_groups.setdefault(row.source_id, []).append(row)

        v4_by_source = {}
        for action in v4_actions:
            src_id, angle, ships = int(action[0]), float(action[1]), int(action[2])
            # 用 v4 的射线追踪恢复目标行星
            src_planet = _find_planet(state, src_id)
            if src_planet is None:
                continue
            src_xy = (src_planet.x, src_planet.y)
            tgt_id = v4._find_target_ray_h(
                src_xy, angle,
                [[p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production]
                 for p in state.planets]
            )
            if tgt_id >= 0:
                v4_by_source[src_id] = (tgt_id, ships)

        for source_id, srows in source_groups.items():
            v4_info = v4_by_source.get(source_id)
            if v4_info is None:
                # v4 未从此行星派兵 → target = no-op
                target_idx = 0
                ship_ratio = 0.0
            else:
                tgt_id, v4_ships = v4_info

                # 在候选列表中找 v4 的目标行星
                target_idx = 0
                for i, row in enumerate(srows):
                    if row.candidate_id == tgt_id and row.mask:
                        target_idx = i
                        break

                if target_idx == 0 and tgt_id != -1:
                    skipped_no_candidate += 1
                    continue

                available = max(1, srows[0].action_info.get("available",
                                  int(_find_planet(state, source_id).ships)))
                ship_ratio = min(1.0, v4_ships / available)

            self_t = torch.tensor(srows[0].self_feat, dtype=torch.float32)
            cand_t = torch.tensor(
                np.stack([r.cand_feat for r in srows]), dtype=torch.float32
            )
            global_t = torch.tensor(srows[0].global_feat, dtype=torch.float32)
            mask_t = torch.tensor([r.mask for r in srows], dtype=torch.bool)

            examples.append({
                "self_feat": self_t,
                "cand_feat": cand_t,
                "global_feat": global_t,
                "mask": mask_t,
                "target_idx": target_idx,
                "ship_ratio": ship_ratio,
            })

    print(f"映射完成: {len(examples)} 样本"
          f"  (跳过: {skipped_no_candidate} 目标不在候选列表)"
          f"  (无目标: {skipped_no_target})")
    return examples


def _find_planet(state, planet_id: int):
    for p in state.planets:
        if p.id == planet_id:
            return p
    return None


# ═══════════════════════════════════════════════════════════════════
# BC Dataset & Training
# ═══════════════════════════════════════════════════════════════════

class BCDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_variable_k(batch):
    max_k = max(e["cand_feat"].shape[0] for e in batch)
    D = batch[0]["cand_feat"].shape[1]

    B = len(batch)
    self_feat = torch.zeros(B, 21)
    cand_feat = torch.zeros(B, max_k, D)
    global_feat = torch.zeros(B, 16)
    mask = torch.zeros(B, max_k, dtype=torch.bool)
    target_idx = torch.zeros(B, dtype=torch.long)
    ship_ratio = torch.zeros(B)

    for i, e in enumerate(batch):
        K = e["cand_feat"].shape[0]
        self_feat[i] = e["self_feat"]
        cand_feat[i, :K] = e["cand_feat"]
        global_feat[i] = e["global_feat"]
        mask[i, :K] = e["mask"]
        target_idx[i] = e["target_idx"]
        ship_ratio[i] = e["ship_ratio"]

    return self_feat, cand_feat, global_feat, mask, target_idx, ship_ratio


def bc_train(policy, examples, epochs=50, lr=0.001, val_split=0.1):
    n = len(examples)
    n_val = int(n * val_split)
    indices = torch.randperm(n).tolist()
    train_examples = [examples[i] for i in indices[n_val:]]
    val_examples = [examples[i] for i in indices[:n_val]]

    train_ds = BCDataset(train_examples)
    val_ds = BCDataset(val_examples)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_variable_k)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_variable_k)

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"BC 训练: {len(train_examples)} 训练样本, "
          f"{len(val_examples)} 验证样本, {epochs} epochs")

    for epoch in range(epochs):
        policy.train()
        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0

        for self_feat, cand_feat, global_feat, mask, target_idx, ship_ratio in train_loader:
            self_feat = self_feat.to(DEVICE)
            cand_feat = cand_feat.to(DEVICE)
            global_feat = global_feat.to(DEVICE)
            mask = mask.to(DEVICE)
            target_idx = target_idx.to(DEVICE)
            ship_ratio = ship_ratio.to(DEVICE)

            out = policy.forward_batch(self_feat, cand_feat, global_feat, mask=mask)

            target_loss = F.cross_entropy(out["target_logits"], target_idx)

            alpha = out["ship_alpha"].clamp(min=1.01)
            beta = out["ship_beta"].clamp(min=1.01)
            beta_dist = torch.distributions.Beta(alpha, beta)
            ship_loss = -beta_dist.log_prob(ship_ratio.clamp(1e-6, 1 - 1e-6)).mean()

            value_loss = 0.5 * (out["value"] ** 2).mean()

            loss = target_loss + 0.5 * ship_loss + 0.1 * value_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            pred = out["target_logits"].argmax(dim=-1)
            total_acc += (pred == target_idx).float().mean().item()
            n_batches += 1

        avg_train_loss = total_loss / max(1, n_batches)
        avg_train_acc = total_acc / max(1, n_batches)

        policy.eval()
        val_loss = 0.0
        val_acc = 0.0
        n_val_b = 0
        with torch.no_grad():
            for self_feat, cand_feat, global_feat, mask, target_idx, ship_ratio in val_loader:
                self_feat = self_feat.to(DEVICE)
                cand_feat = cand_feat.to(DEVICE)
                global_feat = global_feat.to(DEVICE)
                mask = mask.to(DEVICE)
                target_idx = target_idx.to(DEVICE)
                ship_ratio = ship_ratio.to(DEVICE)

                out = policy.forward_batch(self_feat, cand_feat, global_feat, mask=mask)

                target_loss = F.cross_entropy(out["target_logits"], target_idx)
                alpha = out["ship_alpha"].clamp(min=1.01)
                beta = out["ship_beta"].clamp(min=1.01)
                beta_dist = torch.distributions.Beta(alpha, beta)
                ship_loss = -beta_dist.log_prob(ship_ratio.clamp(1e-6, 1 - 1e-6)).mean()
                value_loss = 0.5 * (out["value"] ** 2).mean()
                loss = target_loss + 0.5 * ship_loss + 0.1 * value_loss

                val_loss += loss.item()
                pred = out["target_logits"].argmax(dim=-1)
                val_acc += (pred == target_idx).float().mean().item()
                n_val_b += 1

        avg_val_loss = val_loss / max(1, n_val_b)
        avg_val_acc = val_acc / max(1, n_val_b)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc"].append(avg_train_acc)
        history["val_acc"].append(avg_val_acc)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [{epoch+1:3d}/{epochs}] "
                  f"train_loss={avg_train_loss:.4f} train_acc={avg_train_acc:.3f} "
                  f"val_loss={avg_val_loss:.4f} val_acc={avg_val_acc:.3f}")

    return history


# ═══════════════════════════════════════════════════════════════════
# 验证
# ═══════════════════════════════════════════════════════════════════

def validate_bc_policy(policy, n_games=5):
    """用 BC 训练后的策略跑几局 vs Random，报告结果。"""
    from src.env.wrapper import OrbitWarsEnv

    policy.eval()
    env = OrbitWarsEnv(opponent="random", candidate_count=CANDIDATE_COUNT,
                       episode_steps=500)
    wins = 0
    total_captures_45 = 0

    for i in range(n_games):
        state, raw_obs = env.reset()
        done = False
        captures_by_45 = 0

        while not done:
            decisions, _ = env.collect_decisions(policy, state, deterministic=True)
            if decisions:
                next_state, _, _, done, _ = env.step(decisions, state, comet_warmup=0.0)
            else:
                next_state, _, _, done, _ = env.step([], state, comet_warmup=0.0)

            if state.step <= 45:
                for p_prev, p_curr in zip(state.planets, next_state.planets):
                    if p_prev.owner != 0 and p_curr.owner == 0:
                        captures_by_45 += 1

            state = next_state

        my_ships = sum(p.ships for p in state.planets if p.owner == 0)
        my_ships += sum(f.ships for f in state.fleets if f.owner == 0)
        enemy_ships = sum(p.ships for p in state.planets if p.owner == 1)
        enemy_ships += sum(f.ships for f in state.fleets if f.owner == 1)
        if my_ships > enemy_ships:
            wins += 1
        total_captures_45 += captures_by_45
        print(f"  第{i+1}局: {'胜' if my_ships > enemy_ships else '负'} "
              f"(我 {my_ships} vs 敌 {enemy_ships}), "
              f"前45回合占领 {captures_by_45} 颗星")

    print(f"\n验证结果: {wins}/{n_games} 胜, "
          f"平均前45回合占领 {total_captures_45/n_games:.1f} 颗星")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="v4 Agent 行为克隆")
    parser.add_argument("--demo-games", type=int, default=500,
                        help="收集数据局数")
    parser.add_argument("--epochs", type=int, default=50,
                        help="BC 训练轮数")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="BC 学习率")
    parser.add_argument("--resume", type=str, default=None,
                        help="从已有 checkpoint 继续训练 (如 artifacts/bc_sniper_policy.pt)")
    parser.add_argument("--collect-only", action="store_true",
                        help="仅收集数据，不训练")
    parser.add_argument("--train-only", action="store_true",
                        help="仅训练，使用已有数据")
    parser.add_argument("--validate", action="store_true",
                        help="训练后用策略跑几局验证")
    args = parser.parse_args()

    print(f"设备: {DEVICE}")
    artifacts_dir = ROOT / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    # ── 加载 v4 agent ──
    v4 = _load_v4_agent()

    # ── 数据收集 ──
    if not args.train_only:
        print(f"\n{'='*60}")
        print(f"收集 v4 Agent 演示数据 ({args.demo_games} 局, 前 {MAX_STEPS} 回合)")
        print(f"{'='*60}")
        t0 = time.time()
        raw_data = collect_demonstrations(v4, n_games=args.demo_games, max_steps=MAX_STEPS)
        print(f"耗时: {time.time() - t0:.0f}s")

        print("\n映射到策略动作空间...")
        examples = map_demonstrations(v4, raw_data)
        print(f"训练样本数: {len(examples)}")

        with open(DATA_PATH, "wb") as f:
            pickle.dump(examples, f)
        print(f"数据已保存到 {DATA_PATH}")
    else:
        print(f"\n从 {DATA_PATH} 加载数据...")
        with open(DATA_PATH, "rb") as f:
            examples = pickle.load(f)
        print(f"加载了 {len(examples)} 个训练样本")

    if args.collect_only:
        print("仅收集模式，跳过训练。")
        return

    # ── BC 训练 ──
    print(f"\n{'='*60}")
    print(f"行为克隆训练 ({args.epochs} epochs)")
    print(f"{'='*60}")
    policy = PolicyNetwork(hidden=256).to(DEVICE)
    print(f"策略参数量: {sum(p.numel() for p in policy.parameters()):,}")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=DEVICE)
        policy.load_state_dict(ckpt["policy_state_dict"])
        prev_acc = ckpt.get("history", {}).get("val_acc", [0])[-1]
        print(f"从 {args.resume} 加载权重 (先前 val_acc={prev_acc:.3f})")

    history = bc_train(policy, examples, epochs=args.epochs, lr=args.lr)

    torch.save({
        "policy_state_dict": policy.state_dict(),
        "history": history,
    }, CHECKPOINT_PATH)
    print(f"\n模型已保存到 {CHECKPOINT_PATH}")

    print(f"\n最终 train_acc={history['train_acc'][-1]:.3f}, "
          f"val_acc={history['val_acc'][-1]:.3f}")

    # ── 验证 ──
    if args.validate:
        print(f"\n{'='*60}")
        print("验证 BC 策略 (vs Random, 前 45 回合)")
        print(f"{'='*60}")
        validate_bc_policy(policy, n_games=5)


if __name__ == "__main__":
    main()
