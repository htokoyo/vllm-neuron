# SPDX-License-Identifier: Apache-2.0
"""Prefill MoE (moe_cte shard_on_block, bf16) that does not compute the empty blocks.

nkilib 2.33's ``bwmm_shard_on_block`` runs a fully static loop over the worst-case block count
N = ceil((T*top_k - (E_L - 1)) / B) + E_L - 1 (55 at T=512, TP32: 16 local experts, top-10), and
every block does the full gate-up / down matmuls, empty or not. At TP32 about 16 of the 55 carry
tokens; the kernel is Tensor-engine bound (71 ms per 512-token chunk, device prefill profile).
``skip_weight`` only skips DMAs. The ``moe_cte_dyn`` copy adds ``n_static_blocks`` +
``conditions``: the first ``n_static_blocks`` blocks stay in the original static loop, the rest
run in one dynamic loop whose trip count is the number of non-empty blocks (from ``conditions``,
computed on device), so the cost scales with the routed blocks.
Checked against nkilib's kernel in the NKI simulator (2026-11-05).
"""

from __future__ import annotations

from nkilib.core.moe.moe_cte.moe_cte_utils import BlockShardStrategy, SkipMode
from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode

from .moe_cte_dyn.bwmm_shard_on_block import bwmm_shard_on_block


def _moe_cte_dyn_kernel(hidden_states, expert_affinities_masked, gate_up_proj_weight, down_proj_weight,
                        token_position_to_id, block_to_expert, conditions, block_size: int,
                        n_static_blocks: int, skip_token: bool = True, skip_weight: bool = False,
                        gate_up_proj_scale=None, down_proj_scale=None):
    return bwmm_shard_on_block(
        hidden_states=hidden_states,
        expert_affinities_masked=expert_affinities_masked,
        gate_up_proj_weight=gate_up_proj_weight,
        down_proj_weight=down_proj_weight,
        block_size=block_size,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        activation_function=ActFnType.SiLU,
        skip_dma=SkipMode(skip_token, skip_weight),
        is_tensor_update_accumulating=True,
        expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
        block_sharding_strategy=BlockShardStrategy.PING_PONG,
        conditions=conditions,
        n_static_blocks=n_static_blocks,
        gate_up_proj_scale=gate_up_proj_scale,
        down_proj_scale=down_proj_scale,
    )


def default_static_blocks(num_local_experts: int) -> int:
    """Static blocks before the dynamic loop: one per local expert plus one (the kernel lowers it
    to the parity of N). Uniform top-10 routing at TP32 fills about one block per local expert."""
    return num_local_experts + 1


def moe_cte_dynamic(x, masked, gate_up, down, token_position_to_id, block_to_expert, conditions,
                    block_size: int, n_static_blocks: int, skip_token: bool = True,
                    skip_weight: bool = False, gate_up_proj_scale=None, down_proj_scale=None):
    """SiLU / POST_SCALE / accumulating shard_on_block; returns [T, H] like ``NF.moe_cte``.

    ``x`` [T, H]; ``masked`` [T*E_L, 1]; ``gate_up`` [E_L, H, 2, I]; ``down`` [E_L, I, H];
    ``token_position_to_id`` [N*B] int32; ``block_to_expert`` [N] int32; ``conditions`` [N] int32.
    ``n_static_blocks`` < 0 runs the original fully static kernel (all N blocks).
    FP8 prototype: ``gate_up`` / ``down`` FP8 (torch.float8_e4m3fn holding trn2-e4m3 values, |q| <= 240)
    with row scales ``gate_up_proj_scale`` [E_L, 1, 2*I] fp32 and ``down_proj_scale`` [E_L, 1, H] fp32
    -> double_row FP8 matmuls with per-token dynamic activation scales (moe_cte_dyn docstring).
    """
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from vllm_neuron.utils.hardware_config import get_logical_nc_config

    out = wrap_nki(_moe_cte_dyn_kernel)[get_logical_nc_config()](
        hidden_states=x,
        expert_affinities_masked=masked,
        gate_up_proj_weight=gate_up,
        down_proj_weight=down,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        conditions=conditions,
        block_size=block_size,
        n_static_blocks=int(n_static_blocks),
        skip_token=bool(skip_token),
        skip_weight=bool(skip_weight),
        **({} if gate_up_proj_scale is None else
           dict(gate_up_proj_scale=gate_up_proj_scale, down_proj_scale=down_proj_scale)),
    )
    return out[:, 0, : x.shape[-1]]
