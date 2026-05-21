"""验证第五阶段策略网络模块的正确性。"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import random
import torch
import torch.nn as nn
import numpy as np
from kaggle_environments import make

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

sys.path.insert(0, r"C:\code\[kaggle]\Orbit Wars")

from src.policy.model import (
    SelfEncoder, CandidateEncoder, GlobalEncoder, PolicyNetwork,
    HIDDEN_DIM, JOINT_DIM,
)
from src.policy.action_head import sample_action, compute_ships_to_send
from src.world.observation import parse_observation
from src.features.builder import build_decision_matrix

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
    print("Orbit Wars 策略网络验证")
    print("=" * 60)

    # ═══════════════════════════════════════════════════════
    # [1] 模块实例化
    # ═══════════════════════════════════════════════════════
    print("\n[1] 模块实例化")
    se = SelfEncoder()
    ce = CandidateEncoder()
    ge = GlobalEncoder()
    policy = PolicyNetwork()

    check("SelfEncoder 实例化", se is not None)
    check("CandidateEncoder 实例化", ce is not None)
    check("GlobalEncoder 实例化", ge is not None)
    check("PolicyNetwork 实例化", policy is not None)

    # ═══════════════════════════════════════════════════════
    # [2] Encoder forward pass 测试
    # ═══════════════════════════════════════════════════════
    print("\n[2] Encoder forward pass")

    self_in = torch.randn(21)
    self_out = se(self_in)
    check(f"SelfEncoder: (21,) → ({HIDDEN_DIM},)",
          self_out.shape == (HIDDEN_DIM,),
          f"actual: {self_out.shape}")

    K = 8
    cand_in = torch.randn(K, 29)
    cand_out = ce(cand_in)
    check(f"CandidateEncoder: ({K},29) → ({K},{HIDDEN_DIM})",
          cand_out.shape == (K, HIDDEN_DIM),
          f"actual: {cand_out.shape}")

    global_in = torch.randn(16)
    global_out = ge(global_in)
    check(f"GlobalEncoder: (16,) → ({HIDDEN_DIM},)",
          global_out.shape == (HIDDEN_DIM,),
          f"actual: {global_out.shape}")

    # ═══════════════════════════════════════════════════════
    # [3] PolicyNetwork forward pass
    # ═══════════════════════════════════════════════════════
    print("\n[3] PolicyNetwork forward pass")
    out = policy.forward(self_in, cand_in, global_in)

    check(f"target_logits  shape = ({K},)",
          out["target_logits"].shape == (K,),
          f"actual: {out['target_logits'].shape}")
    check("ship_alpha 是标量",
          out["ship_alpha"].dim() == 0,
          f"actual dim={out['ship_alpha'].dim()}")
    check("ship_beta 是标量",
          out["ship_beta"].dim() == 0,
          f"actual dim={out['ship_beta'].dim()}")
    check("value 是标量",
          out["value"].dim() == 0)
    check("ship_alpha > 1.0",
          out["ship_alpha"].item() > 1.0,
          f"alpha={out['ship_alpha'].item():.4f}")
    check("ship_beta > 1.0",
          out["ship_beta"].item() > 1.0,
          f"beta={out['ship_beta'].item():.4f}")
    check("ship_alpha 有限",
          torch.isfinite(out["ship_alpha"]).item())
    check("ship_beta 有限",
          torch.isfinite(out["ship_beta"]).item())

    # ── 变长 K 测试 ──
    for test_k in [1, 4, 12, 20]:
        cand_k = torch.randn(test_k, 29)
        out_k = policy.forward(self_in, cand_k, global_in)
        check(f"K={test_k}: target_logits shape ({test_k},)",
              out_k["target_logits"].shape == (test_k,),
              f"actual: {out_k['target_logits'].shape}")
        check(f"K={test_k}: ship_alpha > 1", out_k["ship_alpha"].item() > 1.0)

    # ═══════════════════════════════════════════════════════
    # [4] 参数数量 < 1M
    # ═══════════════════════════════════════════════════════
    print("\n[4] 参数量")
    total = sum(p.numel() for p in policy.parameters())
    check(f"参数量 {total:,} < 1,000,000", total < 1_000_000,
          f"实际: {total:,}")
    print(f"  实际参数量: {total:,}")

    # ═══════════════════════════════════════════════════════
    # [5] 梯度回传
    # ═══════════════════════════════════════════════════════
    print("\n[5] 梯度回传")
    policy_g = PolicyNetwork()
    out_g = policy_g.forward(self_in, cand_in, global_in)

    # 模拟 PPO loss —— 四个 head 都参与 loss
    loss = (out_g["target_logits"].mean() * 0.1 +
            out_g["ship_alpha"].squeeze() * 0.1 +
            out_g["ship_beta"].squeeze() * 0.1 +
            0.5 * (out_g["value"] - 0.0).pow(2))
    loss.backward()

    encoder_names = ["self_encoder", "cand_encoder", "global_encoder"]
    head_names = ["target_head", "value_head", "ship_alpha_head", "ship_beta_head"]

    all_ok = True
    for prefix in encoder_names:
        enc = getattr(policy_g, prefix)
        for name, param in enc.named_parameters():
            has_grad = param.grad is not None
            nonzero = has_grad and param.grad.abs().sum() > 0
            ok = has_grad and nonzero
            check(f"{prefix}.{name} 梯度正常", ok,
                  f"has_grad={has_grad}, nonzero={nonzero}")
            if not ok:
                all_ok = False

    for prefix in head_names:
        head = getattr(policy_g, prefix)
        for name, param in head.named_parameters():
            has_grad = param.grad is not None
            nonzero = has_grad and param.grad.abs().sum() > 0
            ok = has_grad and nonzero
            check(f"{prefix}.{name} 梯度正常", ok,
                  f"has_grad={has_grad}, nonzero={nonzero}")
            if not ok:
                all_ok = False

    check("所有组件梯度连通", all_ok)

    # ═══════════════════════════════════════════════════════
    # [6] 正交初始化验证
    # ═══════════════════════════════════════════════════════
    print("\n[6] 正交初始化")
    policy_init = PolicyNetwork()
    for encoder_name in ["self_encoder", "cand_encoder", "global_encoder"]:
        enc = getattr(policy_init, encoder_name)
        for layer_name in ["fc1", "fc2"]:
            w = getattr(enc, layer_name).weight
            m, n = w.shape
            # 对非方阵：若 m > n, 列正交 → W^T@W ≈ I；若 m < n, 行正交 → W@W^T ≈ I
            if m > n:
                gram = w.T @ w
                size = n
            else:
                gram = w @ w.T
                size = m
            diag_ok = torch.allclose(torch.diag(gram), torch.ones(size),
                                     atol=0.3)
            off_diag = gram - torch.diag(torch.diag(gram))
            off_ok = off_diag.abs().mean() < 0.1
            check(f"{encoder_name}.{layer_name}({m}×{n}) 近似正交",
                  diag_ok and off_ok,
                  f"diag_ok={diag_ok}, off_mean={off_diag.abs().mean():.4f}")

    # ═══════════════════════════════════════════════════════
    # [7] Candidate 排列不变性
    # ═══════════════════════════════════════════════════════
    print("\n[7] Candidate 排列不变性")
    policy_inv = PolicyNetwork()
    perm = torch.randperm(K)
    cand_perm = cand_in[perm]

    out_orig = policy_inv.forward(self_in, cand_in, global_in)
    out_perm = policy_inv.forward(self_in, cand_perm, global_in)

    # target_logits 应按相同排列重排后一致
    logits_match = torch.allclose(out_orig["target_logits"][perm],
                                  out_perm["target_logits"], atol=1e-5)
    check("target_logits 排列等变性", logits_match)

    # ship_alpha/beta/value 应不变（均值池化）
    alpha_match = torch.allclose(out_orig["ship_alpha"], out_perm["ship_alpha"])
    beta_match = torch.allclose(out_orig["ship_beta"], out_perm["ship_beta"])
    value_match = torch.allclose(out_orig["value"], out_perm["value"])
    check("ship_alpha 排列不变", alpha_match)
    check("ship_beta 排列不变", beta_match)
    check("value 排列不变", value_match)

    # ═══════════════════════════════════════════════════════
    # [8] PolicyNetwork.act() 集成
    # ═══════════════════════════════════════════════════════
    print("\n[8] PolicyNetwork.act() 采样集成")
    mask_all_true = torch.ones(K, dtype=torch.bool)

    for mode, det in [("随机", False), ("确定性", True)]:
        act = policy.act(self_in, cand_in, global_in, mask=mask_all_true,
                         deterministic=det)
        check(f"{mode}: target_idx 在有效范围",
              0 <= act["target_idx"].item() < K,
              f"idx={act['target_idx'].item()}")
        check(f"{mode}: ship_ratio ∈ (0,1)",
              0.0 < act["ship_ratio"].item() < 1.0,
              f"ratio={act['ship_ratio'].item():.6f}")
        check(f"{mode}: target_log_prob 有限",
              torch.isfinite(act["target_log_prob"]).item())
        check(f"{mode}: ship_log_prob 有限",
              torch.isfinite(act["ship_log_prob"]).item())

    # ── Mask 测试 ──
    print("  [8a] Mask 测试")
    mask_partial = torch.zeros(K, dtype=torch.bool)
    mask_partial[:3] = True
    act_m = policy.act(self_in, cand_in, global_in, mask=mask_partial,
                       deterministic=False)
    check("mask: target_idx < 3", act_m["target_idx"].item() < 3,
          f"idx={act_m['target_idx'].item()}")

    det_m = policy.act(self_in, cand_in, global_in, mask=mask_partial,
                       deterministic=True)
    check("mask 确定性: target_idx < 3", det_m["target_idx"].item() < 3)

    # ── 全 mask (仅 no-op 有效) ──
    all_false = torch.zeros(K, dtype=torch.bool)
    try:
        policy.act(self_in, cand_in, global_in, mask=all_false,
                   deterministic=False)
        check("全 mask 采样无异常", True)
    except Exception as e:
        check("全 mask 采样无异常", False, str(e)[:80])

    # ═══════════════════════════════════════════════════════
    # [9] 端到端：真实特征 → PolicyNetwork → sample_action
    # ═══════════════════════════════════════════════════════
    print("\n[9] 端到端集成测试（真实游戏特征 → PolicyNetwork → 动作）")
    env = make("orbit_wars", debug=True)
    env.run(["random", "random"])
    obs = env.steps[-1][0].observation
    state = parse_observation(obs)

    policy_e2e = PolicyNetwork()
    rows = build_decision_matrix(state, candidate_count=12)
    check(f"决策矩阵行数 > 0 (实际: {len(rows)})", len(rows) > 0)

    # 按源行星分组测试
    current_source = None
    source_rows = []
    sampled_sources = 0
    for row in rows:
        if row.source_id != current_source:
            if source_rows:
                self_t = torch.tensor(source_rows[0].self_feat, dtype=torch.float32)
                cand_t = torch.tensor(
                    np.stack([r.cand_feat for r in source_rows]), dtype=torch.float32
                )
                global_t = torch.tensor(source_rows[0].global_feat, dtype=torch.float32)
                mask_t = torch.tensor([r.mask for r in source_rows])

                try:
                    act = policy_e2e.act(self_t, cand_t, global_t,
                                         mask=mask_t, deterministic=False)
                    check(f"src={current_source}: idx 有效",
                          0 <= act["target_idx"].item() < len(source_rows))
                    check(f"src={current_source}: ratio ∈ (0,1)",
                          0.0 < act["ship_ratio"].item() < 1.0)
                    sampled_sources += 1
                except Exception as e:
                    check(f"src={current_source}: act 正常",
                          False, str(e)[:80])

            current_source = row.source_id
            source_rows = [row]
        else:
            source_rows.append(row)

    # 最后一组
    if source_rows:
        self_t = torch.tensor(source_rows[0].self_feat, dtype=torch.float32)
        cand_t = torch.tensor(
            np.stack([r.cand_feat for r in source_rows]), dtype=torch.float32
        )
        global_t = torch.tensor(source_rows[0].global_feat, dtype=torch.float32)
        mask_t = torch.tensor([r.mask for r in source_rows])
        act = policy_e2e.act(self_t, cand_t, global_t, mask=mask_t,
                             deterministic=False)
        check(f"src={current_source}: idx 有效",
              0 <= act["target_idx"].item() < len(source_rows))
        sampled_sources += 1

    check(f"所有源行星采样成功 (共 {sampled_sources} 颗)",
          sampled_sources > 0)

    # ═══════════════════════════════════════════════════════
    # [10] 确定性与随机采样行为差异
    # ═══════════════════════════════════════════════════════
    print("\n[10] 确定性与随机采样")
    torch.manual_seed(123)
    det_out = policy.act(self_in, cand_in, global_in, deterministic=True)

    torch.manual_seed(123)
    sto_out = policy.act(self_in, cand_in, global_in, deterministic=False)

    # 确定性应选 argmax
    det_probs = torch.softmax(det_out["target_logits"], dim=-1)
    expected_idx = det_probs.argmax()
    check("确定性: target_idx = argmax",
          det_out["target_idx"].item() == expected_idx.item())

    # 确定性: ship_ratio = Beta.mean = α/(α+β)
    expected_mean = out["ship_alpha"] / (out["ship_alpha"] + out["ship_beta"])
    check("确定性: ship_ratio ≈ Beta.mean",
          abs(det_out["ship_ratio"].item() - expected_mean.item()) < 1e-5,
          f"ratio={det_out['ship_ratio'].item():.6f}, mean={expected_mean.item():.6f}")

    # 确定性 log_prob 有限
    check("确定性: target_log_prob 有限",
          torch.isfinite(det_out["target_log_prob"]).item())

    # ═══════════════════════════════════════════════════════
    # [11] 多个源行星顺序处理一致性（确定性）
    # ═══════════════════════════════════════════════════════
    print("\n[11] 多源行星顺序一致性")
    # 造两组不同的 (K, cand) 确保独立处理
    K1, K2 = 5, 7
    self_a = torch.randn(21)
    self_b = torch.randn(21)
    cand_a = torch.randn(K1, 29)
    cand_b = torch.randn(K2, 29)
    glob_a = torch.randn(16)
    glob_b = torch.randn(16)

    policy_seq = PolicyNetwork()

    # A → B 顺序
    with torch.no_grad():
        out_a1 = policy_seq.forward(self_a, cand_a, glob_a)
        out_b1 = policy_seq.forward(self_b, cand_b, glob_b)

    # B → A 顺序 (新实例)
    policy_seq2 = PolicyNetwork()
    policy_seq2.load_state_dict(policy_seq.state_dict())
    with torch.no_grad():
        out_b2 = policy_seq2.forward(self_b, cand_b, glob_b)
        out_a2 = policy_seq2.forward(self_a, cand_a, glob_a)

    check("顺序无关: A 结果一致",
          torch.allclose(out_a1["target_logits"], out_a2["target_logits"]))
    check("顺序无关: B 结果一致",
          torch.allclose(out_b1["target_logits"], out_b2["target_logits"]))

    # ═══════════════════════════════════════════════════════
    # [12] 训练模式下 log_prob 非零且有限
    # ═══════════════════════════════════════════════════════
    print("\n[12] 训练模式 log_prob")
    torch.manual_seed(99)
    train_act = policy.act(self_in, cand_in, global_in,
                           mask=mask_all_true, deterministic=False)
    t_lp = train_act["target_log_prob"]
    s_lp = train_act["ship_log_prob"]

    # log_prob 不应为 0（除非极端情况概率=1）
    check("target_log_prob 非零", abs(t_lp.item()) > 1e-9,
          f"t_lp={t_lp.item():.6f}")
    check("ship_log_prob 非零", abs(s_lp.item()) > 1e-9,
          f"s_lp={s_lp.item():.6f}")
    check("total_log_prob 有限", torch.isfinite(t_lp + s_lp).item())

    # ═══════════════════════════════════════════════════════
    # [13] compute_ships_to_send 单元测试
    # ═══════════════════════════════════════════════════════
    print("\n[13] compute_ships_to_send 舰船数转换")
    check("100 × 0.50 = 50", compute_ships_to_send(100, 0.50) == 50)
    check("100 × 1.00 = 100", compute_ships_to_send(100, 1.00) == 100)
    check("100 × 0.00 = 0", compute_ships_to_send(100, 0.00) == 0)
    check("20 × 0.50 = 10 < 12 → 0", compute_ships_to_send(20, 0.50) == 0,
          str(compute_ships_to_send(20, 0.50)))
    check("24 × 0.50 = 12 ≥ 12 → 12", compute_ships_to_send(24, 0.50) == 12)
    check("30 × 0.40 = 12 ≥ 12 → 12", compute_ships_to_send(30, 0.40) == 12)
    check("11 × 1.00 = 11 < 12 → 0 (边界)", compute_ships_to_send(11, 1.00) == 0,
          str(compute_ships_to_send(11, 1.00)))
    check("13 × 1.00 = 13 ≥ 12 → 13 (边界)", compute_ships_to_send(13, 1.00) == 13)
    check("min(raw, available) 上限", compute_ships_to_send(50, 2.00) == 50)
    check("float 精度: 100 × 0.999 + 1e-9", compute_ships_to_send(100, 0.999) == 99)

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
