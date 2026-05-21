"""Orbit Wars PPO 训练入口。

用法：
  python src/train.py                          # 使用默认配置
  python src/train.py --config configs/default.yaml  # 指定配置文件
  python src/train.py --resume checkpoint_100.pt    # 从 checkpoint 恢复
"""

import sys
import io
import os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import torch

from src.policy.model import PolicyNetwork
from src.ppo.trainer import Trainer
from src.opponents import SniperOpponent, HeuristicOpponent, OpponentPool, LB1200Opponent, V4HybridOpponent, SearchOpponent


DEFAULT_CONFIG = {
    "ppo": {
        "rollout_steps": 128,
        "total_updates": 3000,
        "epochs": 4,
        "minibatch_size": 512,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_coef": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "lr": 0.0003,
        "max_grad_norm": 0.5,
    },
    "model": {
        "hidden_size": 256,
        "candidate_count": 20,
    },
    "env": {
        "candidate_count": 20,
        "episode_steps": 500,
    },
    "opponent": {
        "type": "random",
    },
    "search": {
        "alpha": 0.05,
        "alpha_decay": 0.999,
        "alpha_min": 0.005,
    },
    "terminal": {
        "early_terminal_scale": 0.0,
    },
}


def _build_single_opponent(opp_type: str):
    """构建单个对手对象（不含 pool/staged 等复合类型）。"""
    if opp_type == "random":
        return "random"
    elif opp_type == "sniper":
        return SniperOpponent()
    elif opp_type == "heuristic":
        return HeuristicOpponent()
    elif opp_type == "lb1200":
        return LB1200Opponent()
    elif opp_type == "v4_hybrid":
        return V4HybridOpponent()
    elif opp_type == "search":
        return SearchOpponent()
    else:
        print(f"警告: 未知对手类型 '{opp_type}'，使用默认 random 对手")
        return "random"


def _build_opponent(opponent_cfg: dict):
    """根据配置构建对手（字符串、OpponentPool 或 staged 初始对手）。

    staged 类型返回第一个阶段的对手对象，trainer 内部管理阶段切换。
    """
    opp_type = opponent_cfg.get("type", "random")

    if opp_type == "pool":
        pool = OpponentPool()
        for entry in opponent_cfg.get("pool", []):
            name = entry["name"]
            weight = float(entry["weight"])
            sub_type = entry.get("type", "random")
            pool.add(_build_single_opponent(sub_type), weight, name)
        if len(pool) == 0:
            print("警告: 对手池为空，使用默认 random 对手")
            return "random"
        return pool

    if opp_type == "staged":
        stages = opponent_cfg.get("stages", [])
        if not stages:
            print("警告: staged 对手无阶段，使用默认 random 对手")
            return "random"
        first_type = stages[0].get("type", "random")
        return _build_single_opponent(first_type)

    return _build_single_opponent(opp_type)


def main():
    parser = argparse.ArgumentParser(description="Orbit Wars PPO Training")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML 配置文件路径")
    parser.add_argument("--resume", type=str, default=None,
                        help="从 checkpoint 恢复训练")
    parser.add_argument("--updates", type=int, default=None,
                        help="覆盖总更新次数")
    parser.add_argument("--lr", type=float, default=None,
                        help="覆盖学习率")
    args = parser.parse_args()

    # 加载配置
    config = DEFAULT_CONFIG.copy()
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            file_config = yaml.safe_load(f)
            _deep_update(config, file_config)

    # 命令行覆盖
    if args.updates:
        config["ppo"]["total_updates"] = args.updates
    if args.lr:
        config["ppo"]["lr"] = args.lr

    print("配置:")
    for section, items in config.items():
        print(f"  [{section}]")
        for k, v in items.items():
            print(f"    {k}: {v}")

    # 初始化策略网络
    policy = PolicyNetwork(
        hidden=config["model"]["hidden_size"],
    )
    print(f"\n策略参数量: {sum(p.numel() for p in policy.parameters()):,}")

    # 恢复 checkpoint
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        policy.load_state_dict(ckpt["policy_state_dict"])
        ckpt_update = ckpt.get('update', 'N/A')
        print(f"从 {args.resume} 恢复 (update {ckpt_update})")

    # 构建对手
    opponent_cfg = config.get("opponent", {"type": "random"})
    opponent = _build_opponent(opponent_cfg)
    print(f"对手: {opponent}")

    # 训练
    trainer = Trainer(policy, config, opponent=opponent, opponent_cfg=opponent_cfg)
    stats = trainer.train()


def _deep_update(base, override):
    """递归更新嵌套字典。"""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


if __name__ == "__main__":
    main()
