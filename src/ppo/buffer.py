"""RolloutBuffer —— 存储 transitions + GAE 计算 + minibatch 采样。

每颗源行星的决策作为独立 transition，共享所在 step 的 team reward。
GAE 在 step 级别计算（V_team bootstrap），再分配到每个 transition：
  R_i = R_step_t（同一 step 的所有 transition 共享相同的 return）
  A_i = R_step_t − V_i （衡量"该行星的预期 vs 实际结果"）
"""

import torch


class RolloutBuffer:
    """固定容量 rollout buffer，预分配张量以避免动态分配开销。

    使用方式：
      for each game step:
        for each source planet:
          buffer.store(...)
        buffer.end_step()   # ← 标记 step 结束
      buffer.compute_gae(gamma, gae_lambda)
      for batch in buffer.sample(batch_size):
        ...
      buffer.clear()
    """

    def __init__(self, capacity: int, K_max: int = 20):
        self.capacity = capacity
        self.K_max = K_max

        self.self_feats = torch.zeros(capacity, 21)
        self.cand_feats = torch.zeros(capacity, K_max, 29)
        self.global_feats = torch.zeros(capacity, 16)
        self.masks = torch.zeros(capacity, K_max, dtype=torch.bool)
        self.target_idxs = torch.zeros(capacity, dtype=torch.long)
        self.ship_ratios = torch.zeros(capacity)
        self.target_log_probs = torch.zeros(capacity)
        self.ship_log_probs = torch.zeros(capacity)
        self.values = torch.zeros(capacity)
        self.rewards = torch.zeros(capacity)
        self.dones = torch.zeros(capacity, dtype=torch.bool)

        self.returns = torch.zeros(capacity)
        self.advantages = torch.zeros(capacity)

        self.step_boundaries = []  # [(start_idx, count), ...]

        self.ptr = 0
        self.size = 0
        self._step_start = 0

    def _pad_cand(self, cand_feat, mask):
        """将变长候选特征 pad 到 K_max。"""
        K = cand_feat.shape[0]
        if K >= self.K_max:
            return cand_feat[:self.K_max], mask[:self.K_max]
        padded = torch.zeros(self.K_max, 29)
        padded[:K] = cand_feat[:K]
        pad_mask = torch.zeros(self.K_max, dtype=torch.bool)
        pad_mask[:K] = mask[:K]
        return padded, pad_mask

    def store(self, self_feat, cand_feat, global_feat, mask,
              target_idx, ship_ratio, target_log_prob, ship_log_prob, value,
              reward, done):
        """存储一个 transition。cand_feat/mask 自动 pad 到 K_max。"""
        idx = self.ptr

        cand_pad, mask_pad = self._pad_cand(cand_feat, mask)

        self.self_feats[idx] = self_feat
        self.cand_feats[idx] = cand_pad
        self.global_feats[idx] = global_feat
        self.masks[idx] = mask_pad
        self.target_idxs[idx] = target_idx
        self.ship_ratios[idx] = ship_ratio
        self.target_log_probs[idx] = target_log_prob
        self.ship_log_probs[idx] = ship_log_prob
        self.values[idx] = value
        self.rewards[idx] = reward
        self.dones[idx] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def end_step(self):
        """标记一个 game step 结束。同一 step 的所有 transition 共享 reward。"""
        count = self.size - self._step_start
        if count > 0:
            self.step_boundaries.append((self._step_start, count))
            self._step_start = self.size

    def compute_gae(self, gamma: float, gae_lambda: float,
                    bootstrap_value: float = 0.0):
        """计算 GAE —— step 级别 V_team bootstrap，分配到 transition 级别。

        Step 级别:
          V_team(s) = mean(V_i(s))  所有源行星的 value 均值
          δ_s = r_s + γ · V_team(s+1) · (1−done_s) − V_team(s)
          GAE_s = δ_s + γλ · GAE_{s+1} · (1−done_s)
          R_s = GAE_s + V_team(s)

        Transition 级别:
          R_i = R_s  （共享 return）
          A_i = R_s − V_i  （独立 advantage）

        Args:
            bootstrap_value: 末尾 step 之后的状态价值 V(s_{T+1})。
                             若 rollout 以 done 结束则自动用 0。
        """
        if not self.step_boundaries:
            return

        n_steps = len(self.step_boundaries)

        V_teams = torch.zeros(n_steps)
        step_rewards = torch.zeros(n_steps)
        step_dones = torch.zeros(n_steps, dtype=torch.bool)

        for s, (start, count) in enumerate(self.step_boundaries):
            V_teams[s] = self.values[start:start + count].mean()
            step_rewards[s] = self.rewards[start]
            step_dones[s] = self.dones[start]

        step_gae = torch.zeros(n_steps)
        step_returns = torch.zeros(n_steps)
        gae = 0.0

        for s in reversed(range(n_steps)):
            if step_dones[s]:
                gae = 0.0
                step_returns[s] = step_rewards[s]
            else:
                next_v = V_teams[s + 1] if s + 1 < n_steps else bootstrap_value
                # 若下一步是终局，V(terminal)=0，不使用网络估值
                if s + 1 < n_steps and step_dones[s + 1]:
                    next_v = 0.0
                delta = step_rewards[s] + gamma * next_v - V_teams[s]
                gae = delta + gamma * gae_lambda * gae
                step_returns[s] = V_teams[s] + gae
            step_gae[s] = gae

        for s, (start, count) in enumerate(self.step_boundaries):
            R_s = step_returns[s]
            for i in range(start, start + count):
                self.returns[i] = R_s
                self.advantages[i] = R_s - self.values[i]

        adv = self.advantages[:self.size]
        self.advantages[:self.size] = (adv - adv.mean()) / (adv.std() + 1e-8)

    def sample(self, batch_size: int):
        """随机采样 minibatch。"""
        indices = torch.randperm(self.size)
        for start in range(0, self.size, batch_size):
            batch_idx = indices[start:start + batch_size]
            yield {
                "self_feats": self.self_feats[batch_idx],
                "cand_feats": self.cand_feats[batch_idx],
                "global_feats": self.global_feats[batch_idx],
                "masks": self.masks[batch_idx],
                "target_idxs": self.target_idxs[batch_idx],
                "ship_ratios": self.ship_ratios[batch_idx],
                "target_log_probs": self.target_log_probs[batch_idx],
                "ship_log_probs": self.ship_log_probs[batch_idx],
                "values": self.values[batch_idx],
                "returns": self.returns[batch_idx],
                "advantages": self.advantages[batch_idx],
            }

    def clear(self):
        """清空 buffer（重置指针和计数，不重新分配内存）。"""
        self.ptr = 0
        self.size = 0
        self._step_start = 0
        self.step_boundaries.clear()
