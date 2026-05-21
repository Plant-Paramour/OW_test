"""PPO 更新 —— 混合动作（离散 target + 连续 ship_ratio）clipped objective。

支持离散 Categorical target + 连续 Beta ship_ratio 的联合 log_prob 和 ratio 计算。
"""

import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Beta


def ppo_update(policy, buffer, optimizer, *,
               clip_coef: float = 0.2,
               ent_coef: float = 0.01,
               vf_coef: float = 0.5,
               max_grad_norm: float = 0.5,
               epochs: int = 4,
               minibatch_size: int = 512):
    """执行一次 PPO 更新（多 epoch，多 minibatch）。

    Args:
        policy:     PolicyNetwork 实例
        buffer:     RolloutBuffer（已填充 + 已计算 GAE）
        optimizer:  torch.optim 实例
        clip_coef:  PPO clip 范围 [1−ε, 1+ε]
        ent_coef:   熵正则系数
        vf_coef:    价值 loss 系数
        max_grad_norm: 梯度裁剪阈值
        epochs:     每个 batch 的训练轮数
        minibatch_size: mini-batch 大小

    Returns:
        dict 包含各 loss 项的均值（用于日志）
    """
    stats = {"policy_loss": 0.0, "value_loss": 0.0,
             "target_entropy": 0.0, "ship_entropy": 0.0,
             "total_loss": 0.0, "n_updates": 0}

    for epoch in range(epochs):
        for batch in buffer.sample(minibatch_size):
            out = policy.forward_batch(
                batch["self_feats"], batch["cand_feats"],
                batch["global_feats"], batch["masks"],
            )

            # ── 离散 target log_prob ──
            target_logits = out["target_logits"].clone()
            target_logits[~batch["masks"]] = float('-inf')
            target_dist = Categorical(logits=target_logits)
            new_target_lp = target_dist.log_prob(batch["target_idxs"])

            # ── 连续 ship_ratio log_prob ──
            beta_dist = Beta(out["ship_alpha"], out["ship_beta"])
            new_ship_lp = beta_dist.log_prob(batch["ship_ratios"])

            # ── 联合 ratio ──
            new_lp = new_target_lp + new_ship_lp
            old_lp = batch["target_log_probs"] + batch["ship_log_probs"]
            ratio = (new_lp - old_lp).exp()

            # ── PPO clip loss ──
            adv = batch["advantages"]
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            # ── Value loss ──
            value_loss = 0.5 * (batch["returns"] - out["value"]).pow(2).mean()

            # ── 混合熵 ──
            target_ent = target_dist.entropy().mean()
            ship_ent = beta_dist.entropy().mean()

            # ── 总 loss ──
            loss = policy_loss + vf_coef * value_loss - ent_coef * (target_ent + ship_ent)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            stats["policy_loss"] += policy_loss.item()
            stats["value_loss"] += value_loss.item()
            stats["target_entropy"] += target_ent.item()
            stats["ship_entropy"] += ship_ent.item()
            stats["total_loss"] += loss.item()
            stats["n_updates"] += 1

    n = max(1, stats["n_updates"])
    for k in ["policy_loss", "value_loss", "target_entropy",
              "ship_entropy", "total_loss"]:
        stats[k] /= n

    return stats
