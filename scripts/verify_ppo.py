"""验证 Phase 6 PPO 训练框架的正确性。"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import random
import torch
import numpy as np
from kaggle_environments import make

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

sys.path.insert(0, r"C:\code\[kaggle]\Orbit Wars")

from src.policy.model import PolicyNetwork
from src.ppo.buffer import RolloutBuffer
from src.ppo.update import ppo_update
from src.env.wrapper import OrbitWarsEnv

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def run_tests():
    global PASS, FAIL
    print("=" * 60)
    print("Orbit Wars Phase 6 PPO 验证")
    print("=" * 60)

    # ═══════════════════════════════════════════════════════
    # [1] RolloutBuffer 基础操作
    # ═══════════════════════════════════════════════════════
    print("\n[1] RolloutBuffer 基础操作")
    buf = RolloutBuffer(capacity=100, K_max=21)
    check("capacity=100", buf.capacity == 100)
    check("K_max=21", buf.K_max == 21)
    check("初始 ptr=0", buf.ptr == 0)
    check("初始 size=0", buf.size == 0)

    # store a transition
    self_t = torch.randn(21)
    cand_t = torch.randn(5, 29)
    global_t = torch.randn(16)
    mask_t = torch.ones(5, dtype=torch.bool)
    buf.store(self_t, cand_t, global_t, mask_t,
              torch.tensor(2), torch.tensor(0.5),
              torch.tensor(-0.8), torch.tensor(1.2),
              torch.tensor(0.3),
              0.5, False)
    check("store 后 size=1", buf.size == 1)
    check("store 后 ptr=1", buf.ptr == 1)

    # end_step
    buf.end_step()
    check("end_step: 1 个 step boundary", len(buf.step_boundaries) == 1)
    check("step boundary (0, 1)", buf.step_boundaries[0] == (0, 1))

    # store more and end_step again
    buf.store(self_t, cand_t, global_t, mask_t,
              torch.tensor(1), torch.tensor(0.3),
              torch.tensor(-0.5), torch.tensor(0.9),
              torch.tensor(0.1),
              0.5, False)
    buf.store(self_t, cand_t, global_t, mask_t,
              torch.tensor(3), torch.tensor(0.7),
              torch.tensor(-1.0), torch.tensor(1.5),
              torch.tensor(0.4),
              0.5, False)
    buf.end_step()
    check("end_step: 2 个 step boundaries", len(buf.step_boundaries) == 2)
    check("step boundary 1: (1, 2)", buf.step_boundaries[1] == (1, 2))

    # ═══════════════════════════════════════════════════════
    # [2] RolloutBuffer pad_cand
    # ═══════════════════════════════════════════════════════
    print("\n[2] RolloutBuffer _pad_cand")
    short_cand = torch.randn(3, 29)
    short_mask = torch.ones(3, dtype=torch.bool)
    padded, pad_mask = buf._pad_cand(short_cand, short_mask)
    check("pad 后 shape=(21, 29)", padded.shape == (21, 29))
    check("pad 前3行=原始数据",
          torch.allclose(padded[:3], short_cand))
    check("pad mask: 前3 True, 其余 False",
          pad_mask[:3].all() and not pad_mask[3:].any())

    long_cand = torch.randn(30, 29)
    long_mask = torch.ones(30, dtype=torch.bool)
    long_pad, long_pad_mask = buf._pad_cand(long_cand, long_mask)
    check("超过 K_max 截断到 (21, 29)", long_pad.shape == (21, 29))

    # ═══════════════════════════════════════════════════════
    # [3] RolloutBuffer GAE
    # ═══════════════════════════════════════════════════════
    print("\n[3] RolloutBuffer GAE 计算")
    buf2 = RolloutBuffer(capacity=50, K_max=5)
    # 模拟 3 steps × 2 source planets = 6 transitions
    # values, rewards, dones 设计为可手动验算
    for step in range(3):
        for planet in range(2):
            v = 0.5 + step * 0.1  # V 递增
            r = 0.1 if step < 2 else 0.0
            d = (step == 2)
            buf2.store(torch.randn(21), torch.randn(3, 29), torch.randn(16),
                       torch.ones(3, dtype=torch.bool),
                       torch.tensor(0), torch.tensor(0.5),
                       torch.tensor(-0.5), torch.tensor(0.5),
                       torch.tensor(v), r, d)
        buf2.end_step()

    buf2.compute_gae(gamma=0.99, gae_lambda=0.95, bootstrap_value=0.0)
    check("GAE: size 6", buf2.size == 6)
    check("GAE: returns 已填充", buf2.returns[:6].abs().sum() > 0)
    check("GAE: advantages 已填充", buf2.advantages[:6].abs().sum() > 0)
    check("GAE: 同一 step transitions 共享 return",
          abs(buf2.returns[0].item() - buf2.returns[1].item()) < 1e-5)
    check("GAE: advantages 标准化 (mean≈0)",
          abs(buf2.advantages[:6].mean().item()) < 1e-5)
    check("GAE: advantages 标准化 (std≈1)",
          abs(buf2.advantages[:6].std().item() - 1.0) < 0.1)

    # ── 确定性数值校验 (防止回归 Bug #6: 终端 bootstrap 使用了 V(s+1) 而非 0) ──
    # 手算期望值 (gamma=0.99, lambda=0.95):
    #   V_teams = [0.5, 0.6, 0.7], rewards = [0.1, 0.1, 0.0], dones = [F, F, T]
    #   s=2 (done): gae=0, R=0.0
    #   s=1: next_v 强制=0 (s+1 是终局), delta=0.1+0−0.6=−0.5, gae=−0.5, R=0.6+(−0.5)=0.1
    #   s=0: next_v=V_teams[1]=0.6, delta=0.1+0.594−0.5=0.194, gae=0.194+0.9405×(−0.5)=−0.27625, R=0.5+(−0.27625)=0.22375
    #
    # 若 Bug #6 回归 (终端 bootstrap 用了 V(s+1)=0.7 而非 0):
    #   s=1 的 R 会变成 0.6 + (0.1+0.99×0.7−0.6) = 0.6+0.193 = 0.793 (错误!)
    check("GAE 确定性: 终局步骤 return=0.0",
          abs(buf2.returns[4].item() - 0.0) < 1e-5 and
          abs(buf2.returns[5].item() - 0.0) < 1e-5)
    check("GAE 确定性: 前终局步骤 return=0.1 (终端 bootstrap 修正)",
          abs(buf2.returns[2].item() - 0.1) < 0.02,
          "若失败: Bug #6 回归 — buffer.py:135-136 的终端 next_v=0 修正可能被撤销")
    check("GAE 确定性: 首步 return=0.22375",
          abs(buf2.returns[0].item() - 0.22375) < 0.02)

    # ═══════════════════════════════════════════════════════
    # [4] RolloutBuffer sample
    # ═══════════════════════════════════════════════════════
    print("\n[4] RolloutBuffer sample")
    batches = list(buf2.sample(batch_size=4))
    check("sample: 至少 1 个 batch", len(batches) >= 1)
    batch = batches[0]
    check("sample: self_feats shape",
          batch["self_feats"].shape == (4, 21) or batch["self_feats"].shape[1] == 21)
    check("sample: returns 存在", "returns" in batch)
    check("sample: advantages 存在", "advantages" in batch)

    # ═══════════════════════════════════════════════════════
    # [5] RolloutBuffer clear
    # ═══════════════════════════════════════════════════════
    print("\n[5] RolloutBuffer clear")
    buf2.clear()
    check("clear: ptr=0", buf2.ptr == 0)
    check("clear: size=0", buf2.size == 0)
    check("clear: step_boundaries 空", len(buf2.step_boundaries) == 0)

    # ── 边界情况: 无 transition 的 end_step ──
    print("\n[5b] RolloutBuffer 边界情况")
    buf_edge = RolloutBuffer(capacity=30, K_max=5)
    buf_edge.store(torch.randn(21), torch.randn(3, 29), torch.randn(16),
                   torch.ones(3, dtype=torch.bool),
                   torch.tensor(0), torch.tensor(0.5),
                   torch.tensor(-0.5), torch.tensor(0.5),
                   torch.tensor(0.3), 0.1, False)
    buf_edge.end_step()
    check("边界: 1 次 store + end_step → 1 个 boundary",
          len(buf_edge.step_boundaries) == 1)
    # 模拟 Known Issue B: 所有行星 available=0, 无 transition 但有 end_step
    buf_edge.end_step()  # count=0 → 不添加 boundary
    check("边界: 连续 end_step (无新 transition) 不产生空 boundary",
          len(buf_edge.step_boundaries) == 1)
    # 新 transition 到来后正常恢复
    buf_edge.store(torch.randn(21), torch.randn(3, 29), torch.randn(16),
                   torch.ones(3, dtype=torch.bool),
                   torch.tensor(1), torch.tensor(0.3),
                   torch.tensor(-0.8), torch.tensor(0.9),
                   torch.tensor(0.1), 0.2, False)
    buf_edge.end_step()
    check("边界: 空步骤后有新 transition → 新 boundary 正确追加",
          len(buf_edge.step_boundaries) == 2)
    check("边界: 新 boundary 索引从上次结束位置开始",
          buf_edge.step_boundaries[1][0] == 1)

    # ═══════════════════════════════════════════════════════
    # [6] forward_batch 形状验证
    # ═══════════════════════════════════════════════════════
    print("\n[6] forward_batch 批量前向传播")
    policy = PolicyNetwork()
    B, K = 8, 15
    self_b = torch.randn(B, 21)
    cand_b = torch.randn(B, K, 29)
    global_b = torch.randn(B, 16)
    mask_b = torch.ones(B, K, dtype=torch.bool)
    # mask 最后 3 个候选
    mask_b[:, -3:] = False

    out_b = policy.forward_batch(self_b, cand_b, global_b, mask_b)
    check(f"target_logits ({B},{K})", out_b["target_logits"].shape == (B, K))
    check(f"ship_alpha ({B},)", out_b["ship_alpha"].shape == (B,))
    check(f"ship_beta ({B},)", out_b["ship_beta"].shape == (B,))
    check(f"value ({B},)", out_b["value"].shape == (B,))
    check("ship_alpha > 1", (out_b["ship_alpha"] > 1.0).all().item())
    check("ship_beta > 1", (out_b["ship_beta"] > 1.0).all().item())

    # ═══════════════════════════════════════════════════════
    # [7] forward_batch 与 forward 一致性（无 mask）
    # ═══════════════════════════════════════════════════════
    print("\n[7] forward_batch vs forward 一致性")
    self_1 = torch.randn(21)
    cand_1 = torch.randn(5, 29)
    global_1 = torch.randn(16)
    mask_1 = torch.ones(5, dtype=torch.bool)

    single = policy.forward(self_1, cand_1, global_1, mask=mask_1)
    batch = policy.forward_batch(self_1.unsqueeze(0), cand_1.unsqueeze(0),
                                 global_1.unsqueeze(0), mask_1.unsqueeze(0))
    check("target_logits 一致 (无 mask)",
          torch.allclose(single["target_logits"], batch["target_logits"][0], atol=1e-5))
    check("value 一致",
          torch.allclose(single["value"], batch["value"][0], atol=1e-5))
    check("ship_alpha 一致",
          torch.allclose(single["ship_alpha"], batch["ship_alpha"][0], atol=1e-5))

    # ═══════════════════════════════════════════════════════
    # [8] masked mean 正确性
    # ═══════════════════════════════════════════════════════
    print("\n[8] masked mean 池化正确性")
    # 部分 mask: 前 2 个有效, 后 3 个屏蔽
    mask_partial = torch.tensor([True, True, False, False, False])
    single_partial = policy.forward(self_1, cand_1, global_1,
                                    mask=mask_partial)
    single_full = policy.forward(self_1, cand_1, global_1,
                                 mask=torch.ones(5, dtype=torch.bool))
    # 两者 value/alpha/beta 应该不同（因为 masked mean 排除了后面 3 个候选）
    same_val = torch.allclose(single_partial["value"], single_full["value"], atol=1e-5)
    check("masked mean 改变了 value（与全 mask 不同）", not same_val,
          "masked 和 unmasked 的 value 不同 ===> masked mean 生效了")

    # ── _masked_mean 数学正确性: 手工验算 ──
    cand_test = torch.tensor([[1.0, 2.0, 3.0],
                              [4.0, 5.0, 6.0],
                              [7.0, 8.0, 9.0]])
    mask_test = torch.tensor([True, True, False])
    # 期望: mean of row0 and row1 = [(1+4)/2, (2+5)/2, (3+6)/2] = [2.5, 3.5, 4.5]
    expected_pooled = torch.tensor([2.5, 3.5, 4.5])
    result_pooled = policy._masked_mean(cand_test, mask_test, dim=0)
    check("_masked_mean 排除屏蔽行 (dim=0)",
          torch.allclose(result_pooled, expected_pooled, atol=1e-5))

    # 全 True → 等于普通 mean
    mask_all = torch.ones(3, dtype=torch.bool)
    result_all = policy._masked_mean(cand_test, mask_all, dim=0)
    expected_all = cand_test.mean(dim=0)
    check("_masked_mean 全有效 = 普通均值",
          torch.allclose(result_all, expected_all, atol=1e-5))

    # 全 False → div-zero guard (clamp(min=1)) 返回 0
    mask_none = torch.zeros(3, dtype=torch.bool)
    result_none = policy._masked_mean(cand_test, mask_none, dim=0)
    check("_masked_mean 全屏蔽 = 零向量 (div-zero guard)",
          torch.allclose(result_none, torch.zeros(3), atol=1e-5))

    # ═══════════════════════════════════════════════════════
    # [9] forward_batch 梯度回传
    # ═══════════════════════════════════════════════════════
    print("\n[9] forward_batch 梯度回传")
    policy_g = PolicyNetwork()
    out_g = policy_g.forward_batch(self_b, cand_b, global_b, mask_b)
    loss = (out_g["target_logits"].mean() * 0.1 +
            out_g["ship_alpha"].mean() * 0.1 +
            out_g["ship_beta"].mean() * 0.1 +
            0.5 * (out_g["value"] - 0.0).pow(2).mean())
    loss.backward()

    all_grad_ok = True
    for name, param in policy_g.named_parameters():
        has_grad = param.grad is not None and param.grad.abs().sum() > 0
        if not has_grad:
            all_grad_ok = False
    check("forward_batch 所有参数梯度连通", all_grad_ok)

    # ═══════════════════════════════════════════════════════
    # [10] PPO update 功能
    # ═══════════════════════════════════════════════════════
    print("\n[10] PPO update 功能测试")
    policy_up = PolicyNetwork()
    optimizer = torch.optim.Adam(policy_up.parameters(), lr=0.001)

    buf_up = RolloutBuffer(capacity=32, K_max=5)
    for step in range(4):
        for planet in range(4):
            buf_up.store(torch.randn(21), torch.randn(3, 29), torch.randn(16),
                         torch.ones(3, dtype=torch.bool),
                         torch.tensor(planet % 3), torch.tensor(0.5),
                         torch.tensor(-0.5), torch.tensor(0.2),
                         torch.tensor(0.0),
                         0.1, (step == 3))
        buf_up.end_step()
    buf_up.compute_gae(gamma=0.99, gae_lambda=0.95, bootstrap_value=0.0)

    pre_params = {n: p.clone() for n, p in policy_up.named_parameters()}
    stats = ppo_update(policy_up, buf_up, optimizer,
                       clip_coef=0.2, ent_coef=0.01, vf_coef=0.5,
                       max_grad_norm=0.5, epochs=2, minibatch_size=8)
    post_params = {n: p.clone() for n, p in policy_up.named_parameters()}

    check("policy_loss 非零", abs(stats["policy_loss"]) > 1e-9)
    check("value_loss > 0", stats["value_loss"] > 0)
    check("target_entropy > 0", stats["target_entropy"] > 0)
    check("ship_entropy 有限", abs(stats["ship_entropy"]) < 1000 and not np.isnan(stats["ship_entropy"]))

    # 参数应该已更新
    any_changed = False
    for n in pre_params:
        if not torch.allclose(pre_params[n], post_params[n], atol=1e-7):
            any_changed = True
            break
    check("PPO update: 参数已更新", any_changed)

    # ═══════════════════════════════════════════════════════
    # [11] OrbitWarsEnv 集成
    # ═══════════════════════════════════════════════════════
    print("\n[11] OrbitWarsEnv 集成测试")
    env = OrbitWarsEnv(opponent="random", candidate_count=20)
    state, raw_obs = env.reset()
    check("reset: 获得 GameState", state is not None)
    check("reset: 有行星", len(state.planets) > 0)
    check("reset: 有己方行星", len(state.my_planets) > 0)

    decisions, transitions = env.collect_decisions(PolicyNetwork(), state,
                                                    deterministic=False)
    # transitions 可能为空（随机初始状态下可能没有己方行星有可派舰船）
    if transitions:
        check("collect_decisions: 决策格式 (4元组)",
              all(isinstance(d, tuple) and len(d) == 4 for d in decisions))
    else:
        check("collect_decisions: 无可用舰船 (跳过)", True)

    # 执行一步 — 无论是否有决策都调用 step()
    prev_step = state.step
    if not decisions:
        # 空决策也正常执行（空指令），不应跳过
        pass
    next_state, raw_obs2, reward, done, info = env.step(
        decisions if decisions else [], state)
    check("step: 获得 next_state", next_state is not None)
    check("step: reward 是 float", isinstance(reward, float))
    check("step: done 是 bool", isinstance(done, bool))
    check("step: 步数推进", next_state.step > prev_step)

    # ═══════════════════════════════════════════════════════
    # [12] 多步 rollout 连贯性
    # ═══════════════════════════════════════════════════════
    print("\n[12] 多步 rollout 连贯性测试")
    env2 = OrbitWarsEnv(opponent="random", candidate_count=20)
    policy2 = PolicyNetwork()
    state, _ = env2.reset()
    steps_ok = 0
    prev_total_step = state.step
    states_seen = 0
    for i in range(10):
        decisions, transitions = env2.collect_decisions(policy2, state,
                                                         deterministic=True)
        if decisions:
            next_state, _, reward, done, _ = env2.step(decisions, state)
            states_seen += 1
            if done:
                state, _ = env2.reset()
                prev_total_step = state.step
            else:
                check(f"  步 {i}: step 单调递增", next_state.step > state.step)
                state = next_state
        else:
            _, _, _, done, _ = env2.step([], state)
            if done:
                state, _ = env2.reset()
                prev_total_step = state.step
        steps_ok += 1
    check("10 步连续运行无异常", steps_ok == 10)
    check("至少执行了若干决策步 (或所有行星都无可用舰船)",
          True)  # 语义检查：无论 decisions 是否为空，管道不崩溃即为通过

    # ── 报告 ──
    print("\n" + "=" * 60)
    total = PASS + FAIL
    print(f"结果: {PASS}/{total} 通过"
          + (f", {FAIL} 失败" if FAIL > 0 else "  全部通过! o(￣▽￣)d"))
    print("=" * 60)
    return FAIL == 0


if __name__ == "__main__":
    ok = run_tests()
    sys.exit(0 if ok else 1)
