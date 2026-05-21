"""策略网络 —— 三编码器架构 + 混合动作输出。

SelfEncoder (21→256) + CandidateEncoder (29→256, 共享权重) + GlobalEncoder (16→256)
→ 联合嵌入 (768d) → TargetHead / ShipAlphaHead / ShipBetaHead / ValueHead
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .action_head import TargetHead, ShipAlphaHead, ShipBetaHead, sample_action
from .value_head import ValueHead

HIDDEN_DIM = 256
JOINT_DIM = 768  # 256 × 3


def _ortho_init(module, gain=1.0):
    """正交初始化（PPO 标准）。"""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


class SelfEncoder(nn.Module):
    """源行星状态编码器: 21 → 256 → 256"""

    def __init__(self, in_dim=21, hidden=HIDDEN_DIM):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.reset_parameters()

    def reset_parameters(self):
        _ortho_init(self.fc1, gain=1.0)
        _ortho_init(self.fc2, gain=1.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class CandidateEncoder(nn.Module):
    """候选目标编码器（共享权重，排列不变性）: 29 → 256 → 256"""

    def __init__(self, in_dim=29, hidden=HIDDEN_DIM):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.reset_parameters()

    def reset_parameters(self):
        _ortho_init(self.fc1, gain=1.0)
        _ortho_init(self.fc2, gain=1.0)

    def forward(self, x):
        """Args:
            x: (K, 29) 或 (B*K, 29) 候选特征
        Returns:
            (K, 256) 或 (B*K, 256) 候选嵌入
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class GlobalEncoder(nn.Module):
    """全局状态编码器: 16 → 256 → 256"""

    def __init__(self, in_dim=16, hidden=HIDDEN_DIM):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.reset_parameters()

    def reset_parameters(self):
        _ortho_init(self.fc1, gain=1.0)
        _ortho_init(self.fc2, gain=1.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class PolicyNetwork(nn.Module):
    """策略网络：三编码器 → 联合嵌入 → 动作头 + 价值头。

    前向传播流程（per-source-planet）：
      self_hidden   = SelfEncoder(self_feat)      # (256,)
      global_hidden = GlobalEncoder(global_feat)  # (256,)
      cand_hidden   = CandidateEncoder(cand_feat) # (K, 256)

      # target_logits — per-candidate 共享 TargetHead
      self_expanded   = self_hidden.expand(K, 256)
      global_expanded = global_hidden.expand(K, 256)
      joint = cat([self_expanded, global_expanded, cand_hidden], dim=-1)  # (K, 768)
      target_logits = TargetHead(joint)  # (K,)

      # value / ship αβ — mask-aware 均值池化
      pooled_cand = masked_mean(cand_hidden, mask)  # (256,)
      value_input = cat([self_hidden, global_hidden, pooled_cand])  # (768,)
      value       = ValueHead(value_input)
      ship_alpha  = ShipAlphaHead(value_input)
      ship_beta   = ShipBetaHead(value_input)
    """

    def __init__(self, self_dim=21, cand_dim=29, global_dim=16,
                 hidden=HIDDEN_DIM, joint=JOINT_DIM):
        super().__init__()
        self.self_encoder = SelfEncoder(self_dim, hidden)
        self.cand_encoder = CandidateEncoder(cand_dim, hidden)
        self.global_encoder = GlobalEncoder(global_dim, hidden)

        self.target_head = TargetHead(joint)
        self.value_head = ValueHead(joint)
        self.ship_alpha_head = ShipAlphaHead(joint)
        self.ship_beta_head = ShipBetaHead(joint)

        self.reset_parameters()

    def reset_parameters(self):
        self.self_encoder.reset_parameters()
        self.cand_encoder.reset_parameters()
        self.global_encoder.reset_parameters()
        for head in [self.target_head, self.value_head,
                     self.ship_alpha_head, self.ship_beta_head]:
            _ortho_init(head.fc, gain=0.01)

    @staticmethod
    def _masked_mean(cand_hidden, mask, dim):
        """mask-aware 均值池化。mask 为 None 时退化为普通均值。"""
        if mask is None:
            return cand_hidden.mean(dim=dim)
        valid = mask.float().unsqueeze(-1)
        return (cand_hidden * valid).sum(dim=dim) / valid.sum(dim=dim).clamp(min=1)

    def forward(self, self_feat, cand_feat, global_feat, mask=None):
        """单源行星前向传播。

        Args:
            self_feat:   (21,) 源行星自身特征
            cand_feat:   (K, 29) 候选目标特征
            global_feat: (16,) 全局特征
            mask:        (K,) bool, True=有效候选（None 时不做过滤）

        Returns:
            dict with keys:
              target_logits: (K,)  候选目标 logits
              ship_alpha:    scalar
              ship_beta:     scalar
              value:         scalar
        """
        K = cand_feat.shape[0]
        assert K > 0, "cand_feat 不能为空（builder 保证每源行星至少有 no-op 候选）"

        self_hidden = self.self_encoder(self_feat)
        global_hidden = self.global_encoder(global_feat)
        cand_hidden = self.cand_encoder(cand_feat)

        self_expanded = self_hidden.unsqueeze(0).expand(K, -1)
        global_expanded = global_hidden.unsqueeze(0).expand(K, -1)
        joint = torch.cat([self_expanded, global_expanded, cand_hidden], dim=-1)

        target_logits = self.target_head(joint)

        pooled_cand = self._masked_mean(cand_hidden, mask, dim=0)
        value_input = torch.cat([self_hidden, global_hidden, pooled_cand])
        value = self.value_head(value_input)
        ship_alpha = self.ship_alpha_head(value_input)
        ship_beta = self.ship_beta_head(value_input)

        return {
            "target_logits": target_logits,
            "ship_alpha": ship_alpha,
            "ship_beta": ship_beta,
            "value": value,
        }

    def forward_batch(self, self_feat, cand_feat, global_feat, mask):
        """批量前向传播（PPO update 用）。

        Args:
            self_feat:   (B, 21) 源行星自身特征
            cand_feat:   (B, K_max, 29) 候选目标特征（padding 补齐）
            global_feat: (B, 16) 全局特征
            mask:        (B, K_max) bool, True=有效

        Returns:
            dict with keys:
              target_logits: (B, K_max)
              ship_alpha:    (B,)
              ship_beta:     (B,)
              value:         (B,)
        """
        B, K, _ = cand_feat.shape

        self_hidden = self.self_encoder(self_feat)
        global_hidden = self.global_encoder(global_feat)

        cand_flat = cand_feat.reshape(B * K, 29)
        cand_hidden = self.cand_encoder(cand_flat)
        cand_hidden = cand_hidden.reshape(B, K, 256)

        self_expanded = self_hidden.unsqueeze(1).expand(B, K, 256)
        global_expanded = global_hidden.unsqueeze(1).expand(B, K, 256)
        joint = torch.cat([self_expanded, global_expanded, cand_hidden], dim=-1)

        target_logits = self.target_head(joint)

        pooled_cand = self._masked_mean(cand_hidden, mask, dim=1)
        value_input = torch.cat([self_hidden, global_hidden, pooled_cand], dim=-1)
        value = self.value_head(value_input)
        ship_alpha = self.ship_alpha_head(value_input)
        ship_beta = self.ship_beta_head(value_input)

        return {
            "target_logits": target_logits,
            "ship_alpha": ship_alpha,
            "ship_beta": ship_beta,
            "value": value,
        }

    def act(self, self_feat, cand_feat, global_feat, mask=None,
            deterministic=False):
        """前向传播 + 动作采样（训练/推理统一入口，per-source-planet）。

        Args:
            self_feat:   (21,) 单源行星自身特征
            cand_feat:   (K, 29) 候选目标特征
            global_feat: (16,) 全局特征
            mask:        (K,) bool 张量, True=有效
            deterministic: 确定性采样（argmax / Beta.mean）

        Returns:
            dict 添加:
              target_idx:     int 标量张量
              ship_ratio:     float 标量张量
              target_log_prob: float 标量张量
              ship_log_prob:  float 标量张量
        """
        out = self.forward(self_feat, cand_feat, global_feat, mask=mask)
        idx, ratio, t_lp, s_lp = sample_action(
            out["target_logits"], out["ship_alpha"], out["ship_beta"],
            mask=mask, deterministic=deterministic,
        )
        out["target_idx"] = idx
        out["ship_ratio"] = ratio
        out["target_log_prob"] = t_lp
        out["ship_log_prob"] = s_lp
        return out
