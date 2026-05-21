"""状态价值估计头 —— 从联合表示预测期望折现回报。"""

import torch.nn as nn


class ValueHead(nn.Module):
    """价值估计: hidden_dim → 1。

    输入 value_input = concat(self_hidden, global_hidden, pooled_cand)
    对应当前源行星的全局态势。
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, value_input):
        return self.fc(value_input).squeeze(-1)
