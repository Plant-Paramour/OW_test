"""验证第三阶段动作空间模块的正确性。"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import random
import torch
import torch.nn as nn
import numpy as np
from kaggle_environments import make

# Fixed seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

sys.path.insert(0, r"C:\code\[kaggle]\Orbit Wars")

from src.policy.action_head import (
    TargetHead, ShipAlphaHead, ShipBetaHead,
    sample_action, compute_ships_to_send, MIN_FLEET,
)
from src.policy.value_head import ValueHead
from src.world.observation import parse_observation
from src.features.builder import build_decision_matrix, INVALID_CANDIDATE_VECTOR

HIDDEN_DIM = 768
PASS = 0
FAIL = 0


class MockEncoder(nn.Module):
    """模拟联合编码器 —— 将 66d 特征向量投影到 768d 联合嵌入空间。

    替代随机嵌入，使端到端测试真正验证 特征→策略→动作 的数据流。
    """
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(66, HIDDEN_DIM)

    def forward(self, self_feat, cand_feat, global_feat):
        combined = torch.cat([self_feat, cand_feat, global_feat], dim=-1)
        return self.proj(combined)


def _test_source_sampling(source_rows, target_head, alpha_head, beta_head,
                          mock_encoder, tag="特征"):
    """对一组同源行星的候选行进行特征→嵌入→采样→ships 计算全流程测试。

    对随机采样和确定性采样分别验证:
      - target_idx 在有效范围
      - ship_ratio ∈ (0, 1)
      - log_prob 有限
      - 选中行的 mask 一致性
      - ships 计算结果合法
    """
    K_src = len(source_rows)
    self_feats = torch.tensor(
        np.stack([r.self_feat for r in source_rows]), dtype=torch.float32
    )
    cand_feats = torch.tensor(
        np.stack([r.cand_feat for r in source_rows]), dtype=torch.float32
    )
    global_feats = torch.tensor(
        np.stack([r.global_feat for r in source_rows]), dtype=torch.float32
    )
    joint = mock_encoder(self_feats, cand_feats, global_feats)
    val_input = joint.mean(dim=0)

    logits = target_head(joint)
    alpha = alpha_head(val_input)
    beta = beta_head(val_input)
    mask = torch.tensor([r.mask for r in source_rows])

    for mode, deterministic in [("随机", False), ("确定性", True)]:
        idx, ratio, t_lp, s_lp = sample_action(
            logits, alpha, beta, mask=mask, deterministic=deterministic
        )
        check(f"{tag} {mode}: target_idx 在有效范围",
              0 <= idx.item() < K_src, f"idx={idx.item()}, K={K_src}")
        check(f"{tag} {mode}: ship_ratio ∈ (0,1)",
              0.0 < ratio.item() < 1.0, f"ratio={ratio.item():.6f}")
        check(f"{tag} {mode}: target_log_prob 有限",
              torch.isfinite(t_lp).item(),
              f"t_lp={t_lp.item():.6f}")
        check(f"{tag} {mode}: ship_log_prob 有限",
              torch.isfinite(s_lp).item(),
              f"s_lp={s_lp.item():.6f}")

        chosen = source_rows[idx.item()]
        check(f"{tag} {mode}: 选中行 mask=True", chosen.mask,
              f"source={chosen.source_id}, cand={chosen.candidate_id}")

        if chosen.candidate_id != -1 and chosen.action_info:
            available = chosen.action_info.get("available", 0)
            ships = compute_ships_to_send(available, ratio.item())
            check(f"{tag} {mode}: ships ∈ [0, available]",
                  0 <= ships <= available,
                  f"ships={ships}, available={available}")


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
    print("Orbit Wars 动作空间验证")
    print("=" * 60)

    # ── 实例化模块 ──
    print("\n[1] 模块实例化")
    target_head = TargetHead(HIDDEN_DIM)
    alpha_head = ShipAlphaHead(HIDDEN_DIM)
    beta_head = ShipBetaHead(HIDDEN_DIM)
    value_head = ValueHead(HIDDEN_DIM)
    check("TargetHead 实例化", target_head is not None)
    check("ShipAlphaHead 实例化", alpha_head is not None)
    check("ShipBetaHead 实例化", beta_head is not None)
    check("ValueHead 实例化", value_head is not None)

    # ── 模拟 K 个候选的 forward ──
    print("\n[2] Forward pass 测试")
    K = 10
    joint = torch.randn(K, HIDDEN_DIM)        # K 个候选的联合嵌入
    value_input = torch.randn(HIDDEN_DIM)      # 价值输入

    logits = target_head(joint)
    alpha = alpha_head(value_input)
    beta_val = beta_head(value_input)
    value = value_head(value_input)

    check(f"target_logits.shape == ({K},)", logits.shape == (K,),
          f"实际: {logits.shape}")
    check("alpha 是标量", alpha.dim() == 0)
    check("beta 是标量", beta_val.dim() == 0)
    check("value 是标量", value.dim() == 0)
    check("alpha > 1.0", alpha.item() > 1.0, f"alpha={alpha.item():.4f}")
    check("beta > 1.0", beta_val.item() > 1.0, f"beta={beta_val.item():.4f}")

    # ── 采样测试 ──
    print("\n[3] 采样测试")
    for mode_name, deterministic in [("随机", False), ("确定性", True)]:
        idx, ratio, t_lp, s_lp = sample_action(
            logits, alpha, beta_val, deterministic=deterministic
        )
        check(f"{mode_name}: target_idx 在有效范围",
              0 <= idx.item() < K, f"idx={idx.item()}")
        check(f"{mode_name}: ship_ratio ∈ (0,1)",
              0.0 < ratio.item() < 1.0, f"ratio={ratio.item():.6f}")
        check(f"{mode_name}: target_log_prob 有限",
              torch.isfinite(t_lp).item())
        check(f"{mode_name}: ship_log_prob 有限",
              torch.isfinite(s_lp).item())

    # ── Mask 测试 ──
    print("\n[4] Mask 测试")
    mask = torch.ones(K, dtype=torch.bool)
    mask[5:] = False  # 只保留前 5 个候选
    idx_m, ratio_m, t_lp_m, s_lp_m = sample_action(
        logits, alpha, beta_val, mask=mask, deterministic=False
    )
    check("mask 后 target_idx < 5", idx_m.item() < 5,
          f"idx={idx_m.item()} (应该只选前5个)")
    idx_d, _, _, _ = sample_action(
        logits, alpha, beta_val, mask=mask, deterministic=True
    )
    check("mask 后确定性选前5中最大", idx_d.item() < 5)

    # 全 mask（只 no-op 有效）
    all_masked = torch.zeros(K, dtype=torch.bool)
    try:
        sample_action(logits, alpha, beta_val, mask=all_masked, deterministic=False)
        check("全 mask 时采样无异常", True)
    except Exception as e:
        check("全 mask 时采样无异常", False, str(e)[:80])

    # ── 舰船数计算 ──
    print("\n[5] compute_ships_to_send 测试")
    check("available=100, ratio=0.5 → 50",
          compute_ships_to_send(100, 0.5) == 50)
    check("available=100, ratio=0.05 → 5 (MIN_FLEET=1 允许小额舰队)",
          compute_ships_to_send(100, 0.05) == 5)
    check("available=10, ratio=0.3 → 3 (MIN_FLEET=1 允许小额舰队)",
          compute_ships_to_send(10, 0.3) == 3)
    check("available=0, ratio=1.0 → 0",
          compute_ships_to_send(0, 1.0) == 0)
    check("available=200, ratio=1.0 → 200",
          compute_ships_to_send(200, 1.0) == 200)
    check("available=50, ratio=0.3 → 15",
          compute_ships_to_send(50, 0.3) == 15)

    # ── 端到端：真实游戏状态 → 决策矩阵 → 动作采样模拟 ──
    print("\n[6] 端到端测试（真实游戏状态 + MockEncoder）")
    env = make("orbit_wars", debug=True)
    env.run(["random", "random"])
    obs = env.steps[-1][0].observation
    state = parse_observation(obs)

    mock_encoder = MockEncoder()
    rows = build_decision_matrix(state, candidate_count=8)
    check(f"决策矩阵行数 > 0 (实际: {len(rows)})", len(rows) > 0)

    # 检查每组源行星的第一行是否为 no-op
    seen_sources = set()
    noop_count = 0
    for row in rows:
        if row.source_id not in seen_sources:
            seen_sources.add(row.source_id)
            if row.candidate_id == -1:
                noop_count += 1
    check(f"每个源行星第一行是 no-op ({noop_count}/{len(seen_sources)})",
          noop_count == len(seen_sources),
          f"noop: {noop_count}, sources: {len(seen_sources)}")

    # 检查 no-op 行的 mask
    for row in rows:
        if row.candidate_id == -1:
            check(f"no-op 行 mask=True (src={row.source_id})", row.mask)
            check("no-op action_info.available 存在",
                  "available" in row.action_info)
            break

    # 用 MockEncoder 将真实特征向量编码为联合嵌入，验证特征→动作数据流
    print("  [6a] 特征→MockEncoder→联合嵌入→采样")
    current_source = None
    source_rows = []
    for row in rows:
        if row.source_id != current_source:
            if source_rows:
                _test_source_sampling(
                    source_rows, target_head, alpha_head, beta_head,
                    mock_encoder, "真实特征"
                )
            current_source = row.source_id
            source_rows = [row]
        else:
            source_rows.append(row)
    if source_rows:
        _test_source_sampling(
            source_rows, target_head, alpha_head, beta_head,
            mock_encoder, "真实特征"
        )

    # ── 梯度回传测试 ──
    print("\n[7] 梯度回传测试")
    target_head_g = TargetHead(768)
    alpha_head_g = ShipAlphaHead(768)
    beta_head_g = ShipBetaHead(768)
    value_head_g = ValueHead(768)

    K_g = 8
    joint_g = torch.randn(K_g, 768, requires_grad=False)
    val_in_g = torch.randn(768, requires_grad=False)
    mask_g = torch.ones(K_g, dtype=torch.bool)

    logits_g = target_head_g(joint_g)
    alpha_g = alpha_head_g(val_in_g)
    beta_g = beta_head_g(val_in_g)
    value_g = value_head_g(val_in_g)

    idx_g, ratio_g, t_lp_g, s_lp_g = sample_action(
        logits_g, alpha_g, beta_g, mask=mask_g, deterministic=False
    )

    # 构造模拟 PPO loss 并回传
    loss = -(t_lp_g + s_lp_g) + 0.5 * (value_g - 0.0).pow(2)
    loss.backward()

    grads_ok = []
    for name, module in [("TargetHead", target_head_g),
                          ("ShipAlphaHead", alpha_head_g),
                          ("ShipBetaHead", beta_head_g),
                          ("ValueHead", value_head_g)]:
        has_grad = all(p.grad is not None for p in module.parameters())
        grad_nonzero = any(p.grad.abs().sum() > 0 for p in module.parameters())
        ok = has_grad and grad_nonzero
        grads_ok.append(ok)
        check(f"{name} 梯度正常回传", ok,
              f"has_grad={has_grad}, nonzero={grad_nonzero}")
    check("所有模块梯度连通", all(grads_ok))

    # ── 批次推理测试 ──
    print("\n[8] 批次推理测试")
    B = 4
    K_b = 10
    joint_batch = torch.randn(B, K_b, 768)
    val_batch = torch.randn(B, 768)
    mask_batch = torch.ones(B, K_b, dtype=torch.bool)

    all_logits, all_alphas, all_betas = [], [], []
    for b in range(B):
        all_logits.append(target_head(joint_batch[b]))
        all_alphas.append(alpha_head(val_batch[b]))
        all_betas.append(beta_head(val_batch[b]))

    logits_stacked = torch.stack(all_logits)
    alphas_stacked = torch.stack(all_alphas)
    betas_stacked = torch.stack(all_betas)

    check(f"批次 logits shape = ({B},{K_b})",
          logits_stacked.shape == (B, K_b),
          f"实际: {logits_stacked.shape}")
    check(f"批次 alpha shape = ({B},)",
          alphas_stacked.shape == (B,),
          f"实际: {alphas_stacked.shape}")
    check(f"批次 beta shape = ({B},)",
          betas_stacked.shape == (B,),
          f"实际: {betas_stacked.shape}")
    check("批次 alpha > 1.0", (alphas_stacked > 1.0).all().item())
    check("批次 beta > 1.0", (betas_stacked > 1.0).all().item())

    for b in range(B):
        idx_b, ratio_b, t_lp_b, s_lp_b = sample_action(
            logits_stacked[b], alphas_stacked[b], betas_stacked[b],
            mask=mask_batch[b], deterministic=False
        )
        check(f"batch[{b}]: target_idx 有效", 0 <= idx_b.item() < K_b)
        check(f"batch[{b}]: ship_ratio ∈ (0,1)", 0.0 < ratio_b.item() < 1.0)

    # 模拟：为每颗源行星单独采样（实际推理流程，使用 MockEncoder）
    print("\n[9] 逐行星采样模拟（MockEncoder + 真实特征）")
    mock_encoder2 = MockEncoder()
    all_sampled = 0
    current_source_9 = None
    source_rows_9 = []
    for row in rows:
        if row.source_id != current_source_9:
            if source_rows_9:
                _test_source_sampling(
                    source_rows_9, target_head, alpha_head, beta_head,
                    mock_encoder2, "逐行星"
                )
                all_sampled += 1
            current_source_9 = row.source_id
            source_rows_9 = [row]
        else:
            source_rows_9.append(row)

    if source_rows_9:
        _test_source_sampling(
            source_rows_9, target_head, alpha_head, beta_head,
            mock_encoder2, "逐行星"
        )
        all_sampled += 1
    check(f"逐行星采样完成 (共 {all_sampled} 颗源行星)", all_sampled > 0)

    # ── 报告 ──
    print("\n" + "=" * 60)
    total = PASS + FAIL
    print(f"结果: {PASS}/{total} 通过"
          + (f", {FAIL} 失败" if FAIL > 0 else "  全部通过! (￣▽￣)b"))
    print("=" * 60)
    return FAIL == 0


if __name__ == "__main__":
    ok = run_tests()
    sys.exit(0 if ok else 1)
