# SPDX-License-Identifier: Apache-2.0
"""All-expert decode MoE of Qwen4-Exp with FP8 row-scaled expert weights, on nkilib's ``moe_tkg``.

``vllm_neuron.functional.moe_tkg`` only forwards bf16 / MX expert weights, so the FP8 decode path calls
the nkilib kernel itself (``nkilib.core.moe.moe_tkg.moe_tkg`` as shipped with Neuron SDK 2.33, unmodified)
in its FP8 ROW format: ``expert_gate_up_weights`` [E_L, H, 2, I] and ``expert_down_weights`` [E_L, I, H]
hold trn2 FP8 e4m3 codes (torch.float8_e4m3fn, |q| <= 240) with one fp32 scale per output channel,
``expert_gate_up_weights_scale`` [E_L, 2, I] and ``expert_down_weights_scale`` [E_L, H]. The kernel
multiplies the bf16 activations by the FP8 weights and dequantizes after each matmul.

All-expert mode: every local expert of this rank runs for every token; ``rank_id`` [1, 1] selects the
experts [rank * E_L, rank * E_L + E_L) of the dense [T, E] affinities, and each expert's output is scaled
by its affinity (POST_SCALE; a token not routed to the expert has affinity 0). In this mode the kernel
copies the affinities straight to SBUF (any T <= 128), so the row count needs no padding to the LNC grid.

Launch with ``wrap_nki(moe_tkg_all_expert_fp8)[lnc](...)``; returns [T, H] in the dtype of ``hidden_input``.
"""

from nkilib.core.moe.moe_tkg.moe_tkg import moe_tkg
from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode


def moe_tkg_all_expert_fp8(hidden_input, expert_gate_up_weights, expert_down_weights, expert_affinities,
                           expert_index, rank_id, expert_gate_up_weights_scale, expert_down_weights_scale):
    return moe_tkg(
        hidden_input=hidden_input,
        expert_gate_up_weights=expert_gate_up_weights,
        expert_down_weights=expert_down_weights,
        expert_affinities=expert_affinities,
        expert_index=expert_index,
        is_all_expert=True,
        rank_id=rank_id,
        expert_gate_up_weights_scale=expert_gate_up_weights_scale,
        expert_down_weights_scale=expert_down_weights_scale,
        expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
        activation_fn=ActFnType.SiLU,
    )
