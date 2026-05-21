"""混合动作空间输出头 —— 离散目标选择 + 连续舰船比例。

架构: per-candidate 共享权重 TargetHead + 全局 Beta 分布 ship_ratio。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Beta

MIN_FLEET = 1  # 游戏引擎仅要求 ships > 0，无硬性最低舰队数


class TargetHead(nn.Module):
    """Per-candidate 目标选择头: hidden_dim → 1。

    对每个候选的 joint embedding 独立输出一个 logit，
    所有候选共享同一组权重（排列不变性）。
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, joint):
        """Args:
            joint: (K, hidden_dim) K 个候选的联合嵌入

        Returns:
            target_logits: (K,) 每个候选的未归一化 logit
        """
        return self.fc(joint).squeeze(-1)


class ShipAlphaHead(nn.Module):
    """Beta 分布 α 参数头: hidden_dim → 1。

    softplus + 1.0 确保 α > 1，避免 Beta(α,β) 在边界处的 U 型分布。
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, value_input):
        """Args:
            value_input: (hidden_dim,) 价值联合表示

        Returns:
            alpha: scalar, > 1.0
        """
        return F.softplus(self.fc(value_input)).squeeze(-1) + 1.0


class ShipBetaHead(nn.Module):
    """Beta 分布 β 参数头: hidden_dim → 1。

    softplus + 1.0 确保 β > 1。
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, value_input):
        return F.softplus(self.fc(value_input)).squeeze(-1) + 1.0


def sample_action(target_logits, ship_alpha, ship_beta, mask=None,
                  deterministic=False):
    """从策略分布中采样 (target_index, ship_ratio)。

    Args:
        target_logits: (K,) 每个候选的未归一化 logit
        ship_alpha:   scalar Beta 分布 α 参数
        ship_beta:    scalar Beta 分布 β 参数
        mask:         (K,) bool 张量, True=有效, False=屏蔽
        deterministic: True 时取 argmax / mean

    Returns:
        (target_idx, ship_ratio, target_log_prob, ship_log_prob)
    """
    if mask is not None:
        target_logits = target_logits.clone()
        target_logits[~mask] = float('-inf')
        if not mask.any():
            device = target_logits.device
            beta_dist = Beta(ship_alpha, ship_beta)
            ship_ratio = beta_dist.mean.clamp(1e-6, 1.0 - 1e-6)
            ship_log_prob = beta_dist.log_prob(ship_ratio)
            return (torch.zeros(1, dtype=torch.long, device=device).squeeze(),
                    ship_ratio, torch.tensor(0.0, device=device), ship_log_prob)

    target_dist = Categorical(logits=target_logits)
    if deterministic:
        target_idx = target_dist.probs.argmax()
    else:
        target_idx = target_dist.sample()

    beta_dist = Beta(ship_alpha, ship_beta)
    if deterministic:
        ship_ratio = beta_dist.mean
    else:
        ship_ratio = beta_dist.sample()
    ship_ratio = ship_ratio.clamp(1e-6, 1.0 - 1e-6)

    target_log_prob = target_dist.log_prob(target_idx)
    ship_log_prob = beta_dist.log_prob(ship_ratio)

    return target_idx, ship_ratio, target_log_prob, ship_log_prob


def compute_ships_to_send(available: int, ship_ratio: float,
                          min_fleet: int = MIN_FLEET) -> int:
    """将 ship_ratio 转换为实际舰船数。

    Args:
        available:  源行星可用舰船数（已扣除 keep_needed）
        ship_ratio: [0, 1] 比例
        min_fleet:  最低派兵门槛

    Returns:
        实际发送的舰船数，小于 min_fleet 时返回 0（隐式 no-op）
    """
    raw = int(available * ship_ratio + 1e-9)
    if raw < min_fleet:
        return 0
    return min(raw, available)
