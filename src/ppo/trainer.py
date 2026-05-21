"""Trainer —— 串联 rollout 收集 + GAE + PPO 更新循环。

支持可选的动作搜索 Logit Bias：搜索算法评估每个候选的占领价值，
作为偏置加入策略 logits，引导训练初期的探索方向。
"""

import time
import torch
import numpy as np

from .buffer import RolloutBuffer
from .update import ppo_update
from ..env.wrapper import OrbitWarsEnv
from ..env.reward import get_comet_warmup
from ..opponents.base import OpponentLike


class Trainer:
    """PPO 训练器。

    使用单环境（保证 GAE 时序正确性），未来可扩展为多环境。
    可选集成动作搜索为策略提供 Logit Bias（search_alpha > 0）。
    """

    def __init__(self, policy, config: dict, opponent: OpponentLike = "random",
                 opponent_cfg: dict | None = None):
        self.policy = policy
        self.config = config
        self.opponent = opponent
        self.opponent_cfg = opponent_cfg or {}

        ppo_cfg = config["ppo"]
        env_cfg = config["env"]

        self.rollout_steps = ppo_cfg["rollout_steps"]
        self.total_updates = ppo_cfg["total_updates"]
        self.gamma = ppo_cfg["gamma"]
        self.gae_lambda = ppo_cfg["gae_lambda"]
        self.epochs = ppo_cfg["epochs"]
        self.minibatch_size = ppo_cfg["minibatch_size"]
        self.clip_coef = ppo_cfg["clip_coef"]
        self.ent_coef = ppo_cfg["ent_coef"]
        self.vf_coef = ppo_cfg["vf_coef"]
        self.lr = ppo_cfg["lr"]
        self.max_grad_norm = ppo_cfg["max_grad_norm"]
        self.candidate_count = env_cfg.get("candidate_count", 20)
        self.K_max = self.candidate_count + 1
        self.episode_steps = env_cfg.get("episode_steps", 500)

        # ── 搜索集成配置 ──
        search_cfg = config.get("search", {})
        self.search_alpha = float(search_cfg.get("alpha", 0.0))
        self.search_alpha_decay = float(search_cfg.get("alpha_decay", 1.0))
        self.search_alpha_min = float(search_cfg.get("alpha_min", 0.0))

        # ── 终局配置 (Phase 0 早期发育) ──
        terminal_cfg = config.get("terminal", {})
        self.early_terminal_scale = float(terminal_cfg.get("early_terminal_scale", 0.0))

        # ── 分阶段对手配置 ──
        self._setup_staged_opponents()

        self.optimizer = torch.optim.Adam(policy.parameters(), lr=self.lr)

        capacity = self.rollout_steps * 50
        self.buffer = RolloutBuffer(capacity=capacity, K_max=self.K_max)

        self.env = OrbitWarsEnv(
            opponent=self.current_opponent,
            candidate_count=self.candidate_count,
            episode_steps=self.episode_steps,
        )

        self.stats_history = []
        self.update_count = 0

    def _compute_terminal_value(self, state):
        """计算 rollout 末尾的状态价值（GAE bootstrap 用）。"""
        from ..features.builder import build_decision_matrix

        rows = build_decision_matrix(state, candidate_count=self.candidate_count)
        if not rows:
            return 0.0

        source_groups = {}
        for row in rows:
            source_groups.setdefault(row.source_id, []).append(row)

        values = []
        with torch.no_grad():
            for srows in source_groups.values():
                self_t = torch.tensor(srows[0].self_feat, dtype=torch.float32)
                cand_t = torch.tensor(
                    np.stack([r.cand_feat for r in srows]), dtype=torch.float32
                )
                global_t = torch.tensor(srows[0].global_feat, dtype=torch.float32)
                mask_t = torch.tensor([r.mask for r in srows])
                out = self.policy.forward(self_t, cand_t, global_t, mask=mask_t)
                values.append(out["value"].item())

        return float(np.mean(values)) if values else 0.0

    # ── 分阶段对手 ──

    def _setup_staged_opponents(self):
        """解析 staged 对手配置，预构建所有阶段对手。"""
        cfg = self.opponent_cfg
        self.is_staged = cfg.get("type") == "staged"
        self.stage_opponents = []
        self.current_stage_idx = 0
        self.current_opponent = self.opponent
        self.stage_eval_interval = 50
        self.stage_eval_games = 100
        self.stage_win_threshold = 0.90

        if not self.is_staged:
            return

        from ..opponents import (SniperOpponent, HeuristicOpponent,
                                  LB1200Opponent, V4HybridOpponent,
                                  SearchOpponent)

        def _build_one(opp_type):
            if opp_type == "random":
                return "random"
            if opp_type == "sniper":
                return SniperOpponent()
            if opp_type == "heuristic":
                return HeuristicOpponent()
            if opp_type == "lb1200":
                return LB1200Opponent()
            if opp_type == "v4_hybrid":
                return V4HybridOpponent()
            if opp_type == "search":
                return SearchOpponent()
            print(f"警告: 未知对手类型 '{opp_type}'，使用 random")
            return "random"

        stages = cfg.get("stages", [])
        for entry in stages:
            opp = _build_one(entry.get("type", "random"))
            self.stage_opponents.append(opp)

        self.stage_eval_interval = int(cfg.get("eval_interval", 50))
        self.stage_eval_games = int(cfg.get("eval_games", 100))
        self.stage_win_threshold = float(cfg.get("win_threshold", 0.90))

        self.current_stage_idx = 0
        self.current_opponent = self.stage_opponents[0]

        names = " → ".join(str(o) for o in self.stage_opponents)
        print(f"  分阶段对手: {names}")
        print(f"  评估: 每 {self.stage_eval_interval} updates "
              f"跑 {self.stage_eval_games} 局, 胜率 ≥ {self.stage_win_threshold:.0%} 晋级")

    def _eval_win_rate(self):
        """评估当前策略对当前阶段对手的胜率。

        胜 = 50 回合结束时 Φ > 0（我方终局资产 > 敌方）。
        使用确定性策略，关闭搜索引导。
        使用 env.run() 确保与 verify_search_vs_sniper.py 一致的评估语义。
        """
        from kaggle_environments import make
        from ..world.observation import parse_observation
        from ..features.builder import build_decision_matrix
        from ..policy.action_head import sample_action, compute_ships_to_send
        from ..engine.interception import aim_at
        from ..env.reward import state_potential
        import numpy as np

        def _find_p(planets, pid):
            for p in planets:
                if p.id == pid:
                    return p
            return None

        policy = self.policy
        candidate_count = self.candidate_count
        episode_steps = self.episode_steps
        opponent = self.current_opponent

        def _agent(obs, _cfg):
            state = parse_observation(obs, episode_steps=episode_steps)
            rows = build_decision_matrix(state, candidate_count=candidate_count)
            if not rows:
                return []
            source_groups = {}
            for row in rows:
                source_groups.setdefault(row.source_id, []).append(row)
            actions = []
            for source_id, srows in source_groups.items():
                self_t = torch.tensor(srows[0].self_feat, dtype=torch.float32)
                cand_t = torch.tensor(
                    np.stack([r.cand_feat for r in srows]), dtype=torch.float32)
                global_t = torch.tensor(srows[0].global_feat, dtype=torch.float32)
                mask_t = torch.tensor([r.mask for r in srows])
                available = srows[0].action_info.get("available", 0)
                with torch.no_grad():
                    out = policy.forward(self_t, cand_t, global_t, mask=mask_t)
                target_logits = out["target_logits"].clone()
                idx, ratio, _, _ = sample_action(
                    target_logits, out["ship_alpha"], out["ship_beta"],
                    mask=mask_t, deterministic=True)
                target_idx = idx.item()
                if target_idx >= len(srows):
                    continue
                target_row = srows[target_idx]
                if target_row.candidate_id == -1:
                    continue
                ships = compute_ships_to_send(available, ratio.item())
                if ships == 0:
                    continue
                src = _find_p(state.planets, source_id)
                tgt = _find_p(state.planets, target_row.candidate_id)
                if src is None or tgt is None:
                    continue
                aim = aim_at(src, tgt, ships, state.initial_by_id,
                            state.angular_velocity, state.comets, state.comet_ids)
                if aim is None:
                    continue
                actions.append([source_id, aim[0], ships])
            return actions

        wins = 0
        for _ in range(self.stage_eval_games):
            env = make("orbit_wars", debug=True,
                       configuration={"episodeSteps": episode_steps})
            env.run([_agent, opponent])
            final_obs = env.steps[-1][0].observation
            final_state = parse_observation(final_obs, player_override=0)
            phi = state_potential(final_state, player=0)
            if phi > 0:
                wins += 1

        return wins / self.stage_eval_games

    def _advance_stage(self):
        """切换到下一个阶段对手。"""
        self.current_stage_idx += 1
        self.current_opponent = self.stage_opponents[self.current_stage_idx]
        self.env = OrbitWarsEnv(
            opponent=self.current_opponent,
            candidate_count=self.candidate_count,
            episode_steps=self.episode_steps,
        )
        print(f"\n  >>> 晋级! 切换对手 → {self.current_opponent}"
              f" (阶段 {self.current_stage_idx + 1}/{len(self.stage_opponents)})")

    # ── 训练主循环 ──

    def train(self, log_interval: int = 10, checkpoint_interval: int = 100):
        """执行完整训练循环。

        Args:
            log_interval: 每 N 次 update 打印一次日志
            checkpoint_interval: 每 N 次 update 保存一次 checkpoint
        """
        use_search = self.search_alpha > 0
        print(f"开始训练: {self.total_updates} updates, "
              f"rollout_steps={self.rollout_steps}, lr={self.lr}")
        if use_search:
            print(f"  搜索集成: alpha={self.search_alpha:.4f}, "
                  f"decay={self.search_alpha_decay:.4f}, min={self.search_alpha_min:.4f}")
        else:
            print(f"  搜索集成: 关闭 (alpha=0)")
        t_start = time.time()

        comet_utilization_ema = 0.0

        for update_idx in range(1, self.total_updates + 1):
            self.update_count += 1

            # ── 1. Rollout 收集 ──
            state, _ = self.env.reset()
            episode_reward = 0.0
            episode_steps = 0
            attacked_from_comet = False
            warmup = get_comet_warmup(comet_utilization_ema)

            for step in range(self.rollout_steps):
                decisions, transitions = self.env.collect_decisions(
                    self.policy, state, search_alpha=self.search_alpha,
                )

                if not attacked_from_comet:
                    for src_id, tgt_id, _ships, _angle in decisions:
                        if src_id in state.comet_ids:
                            for p in state.planets:
                                if p.id == tgt_id and p.owner not in (-1, self.env.player_id):
                                    attacked_from_comet = True
                                    break
                        if attacked_from_comet:
                            break

                if decisions:
                    next_state, _, reward, done, _ = self.env.step(
                        decisions, state, comet_warmup=warmup,
                        early_terminal_scale=self.early_terminal_scale)
                else:
                    next_state, _, reward, done, _ = self.env.step(
                        [], state, comet_warmup=warmup,
                        early_terminal_scale=self.early_terminal_scale)

                for t in transitions:
                    self.buffer.store(**t, reward=reward, done=done)
                self.buffer.end_step()

                episode_reward += reward
                episode_steps += 1

                if done:
                    comet_utilization_ema = (0.99 * comet_utilization_ema
                                            + 0.01 * float(attacked_from_comet))
                    attacked_from_comet = False
                    warmup = get_comet_warmup(comet_utilization_ema)
                    state, _ = self.env.reset()
                else:
                    state = next_state

            # ── 2. GAE ──
            terminal_v = self._compute_terminal_value(state)
            self.buffer.compute_gae(self.gamma, self.gae_lambda,
                                    bootstrap_value=terminal_v)

            # ── 3. PPO 更新 ──
            stats = ppo_update(
                self.policy, self.buffer, self.optimizer,
                clip_coef=self.clip_coef, ent_coef=self.ent_coef,
                vf_coef=self.vf_coef, max_grad_norm=self.max_grad_norm,
                epochs=self.epochs, minibatch_size=self.minibatch_size,
            )
            stats["episode_reward"] = episode_reward
            stats["episode_steps"] = episode_steps
            self.stats_history.append(stats)

            self.buffer.clear()

            # ── 4. 搜索 alpha 退火 ──
            if use_search:
                self.search_alpha = max(
                    self.search_alpha_min,
                    self.search_alpha * self.search_alpha_decay,
                )

            # ── 4.5. 分阶段对手评估 ──
            if self.is_staged and update_idx % self.stage_eval_interval == 0:
                wr = self._eval_win_rate()
                stage_name = str(self.current_opponent)
                print(f"\n  [评估] update={update_idx} 阶段={self.current_stage_idx + 1}"
                      f" 对手={stage_name} 胜率={wr:.1%} ({int(wr * self.stage_eval_games)}"
                      f"/{self.stage_eval_games})")
                if (wr >= self.stage_win_threshold
                        and self.current_stage_idx + 1 < len(self.stage_opponents)):
                    self._advance_stage()

            # ── 5. 日志 ──
            if update_idx % log_interval == 0:
                elapsed = time.time() - t_start
                recent = self.stats_history[-log_interval:]
                avg_policy_loss = np.mean([s["policy_loss"] for s in recent])
                avg_value_loss = np.mean([s["value_loss"] for s in recent])
                avg_entropy = np.mean([s["target_entropy"] + s["ship_entropy"]
                                       for s in recent])
                avg_reward = np.mean([s["episode_reward"] for s in recent])
                extras = ""
                if use_search:
                    extras = f" search_α={self.search_alpha:.4f}"
                print(f"[{update_idx:5d}/{self.total_updates}] "
                      f"policy_loss={avg_policy_loss:.4f} "
                      f"value_loss={avg_value_loss:.4f} "
                      f"entropy={avg_entropy:.4f} "
                      f"reward={avg_reward:.2f}"
                      f"{extras} "
                      f"elapsed={elapsed:.0f}s")

            # ── 6. Checkpoint ──
            if update_idx % checkpoint_interval == 0:
                path = f"checkpoint_{update_idx}.pt"
                torch.save({
                    "update": update_idx,
                    "policy_state_dict": self.policy.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "stats_history": self.stats_history,
                }, path)
                print(f"  checkpoint → {path}")

        print(f"训练完成! 总耗时 {time.time() - t_start:.0f}s")
        return self.stats_history
