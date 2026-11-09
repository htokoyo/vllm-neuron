# SPDX-License-Identifier: Apache-2.0
"""Tensor / expert parallel layout of Qwen4-Exp over one vLLM TP group of ``W`` ranks.

Attention and Gated DeltaNet are split over an *attention group* of ``A`` ranks
(their head counts allow at most 8): the whole TP group, or -- when the TP group is
larger -- the EP-TP sub-groups of ``vllm_neuron.parallel`` (``--enable-expert-parallel``
with ``ep_degree = W / A``), e.g. TP=32, ep_degree=4 -> four groups of 8 that compute
the same attention. Everything else (experts, vocab rows) is split over all ``W``.

| Part | Sharding | Combine |
|---|---|---|
| Gated DeltaNet | key heads ``Hk/W``, value heads ``Hv/W`` (fused q|k|v rows per part) | ``out_proj`` row-parallel, all-reduce |
| Full attention | query heads ``Hq/W`` (query/gate pairs), the KV head(s) those heads read | ``o_proj`` row-parallel, all-reduce |
| Routed experts | expert parallel, ``E/W`` experts per rank, intermediate unsplit | all-reduce |
| Embedding, PLE n-gram table | rows (vocab) | masked lookup, all-reduce |
| Shared expert | intermediate ``I/W`` (gate/up rows, down columns), when ``W`` divides ``I`` | joins the routed all-reduce |
| LM head | rows (vocab) | all-gather (or vocab-sharded sampling) |
| Hyper-connections, router, shared-expert gate, PLE projections, norms, final mixer | replicated | -- |

A shared expert that ``W`` does not divide stays replicated and is added after the
routed all-reduce, so it is counted once. Per-rank offsets are taken from the group's ``rank_in_group_tensor``
(a graph input on device), not from a Python int, so every rank traces the same
graph.

:func:`shard_pieces` gives the checkpoint slices that make up one rank's parameter;
the loader reads only those (``safetensors`` ``get_slice``).
"""

from dataclasses import dataclass

import torch

from .config import Qwen4ExpConfig


@dataclass(frozen=True)
class TPInfo:
    size: int = 1
    rank: int = 0
    attn: bool = False  # True: the attention group; False: the whole TP group

    @classmethod
    def current(cls, enabled: bool, attn: bool = False) -> "TPInfo":
        """The runner's group (whole TP group, or the attention group), else one rank."""
        if not enabled:
            return cls(attn=attn)
        g = _group(attn)
        return cls(g.world_size, g.rank_in_group, attn)


def _group(attn: bool):
    from vllm.distributed.parallel_state import get_tp_group

    tp = get_tp_group()
    if attn:
        from vllm_neuron.parallel import neuron_parallel_state as nps

        if nps._NEURON_EP is not None and nps.get_neuron_ep_tp_group().world_size < tp.world_size:
            return nps.get_neuron_ep_tp_group()
    return tp


def tp_group(tp: "TPInfo | None" = None):
    return _group(bool(tp and tp.attn))


def all_reduce(x: torch.Tensor, tp: TPInfo) -> torch.Tensor:
    return x if tp.size == 1 else tp_group(tp).all_reduce(x)


def all_gather_last(x: torch.Tensor, tp: TPInfo) -> torch.Tensor:
    return x if tp.size == 1 else tp_group(tp).all_gather(x, dim=-1)


def rank_tensor(tp: TPInfo, device) -> torch.Tensor:
    """This rank's index as an int32 ``[1]`` tensor (a graph input, not a constant)."""
    if tp.size == 1:
        return torch.zeros(1, dtype=torch.int32, device=device)
    return tp_group(tp).rank_in_group_tensor.reshape(1).to(device=device, dtype=torch.int32)


def sharded_lookup(table: torch.Tensor, ids: torch.Tensor, tp: TPInfo) -> torch.Tensor:
    """Embedding lookup in a row-sharded table: local rows, zeros elsewhere, all-reduce."""
    if tp.size == 1:
        return torch.nn.functional.embedding(ids.long(), table)
    rows = table.shape[0]
    local = ids.long() - rank_tensor(tp, ids.device).long() * rows
    mine = (local >= 0) & (local < rows)
    out = torch.nn.functional.embedding(torch.where(mine, local, torch.zeros_like(local)), table)
    out = torch.where(mine.unsqueeze(-1), out, torch.zeros_like(out))
    return all_reduce(out, tp)


def validate(config: Qwen4ExpConfig, tp: TPInfo, ngram_rows: list[int], attn: TPInfo) -> None:
    w, a = tp.size, attn.size
    c = config
    checks = {
        "linear_num_key_heads": c.linear_num_key_heads % a == 0,
        "linear_num_value_heads": c.linear_num_value_heads % a == 0,
        "num_attention_heads": c.num_attention_heads % a == 0,
        "num_key_value_heads": (a % c.num_key_value_heads == 0
                                and (c.num_attention_heads // c.num_key_value_heads)
                                % (c.num_attention_heads // a) == 0)
                               or c.num_key_value_heads % a == 0,
        "num_experts": c.num_experts % w == 0,
        "vocab_size": c.vocab_size % w == 0,
        "ngram rows": all(r % w == 0 for r in ngram_rows),
    }
    bad = [k for k, ok in checks.items() if not ok]
    if bad:
        raise ValueError(f"Qwen4-Exp cannot be split over {w} ranks (attention {a}): {bad} not divisible")


def kv_heads_per_rank(config: Qwen4ExpConfig, tp: TPInfo) -> int:
    return max(1, config.num_key_value_heads // tp.size)


def _rows(lo: int, hi: int, ndim: int, dim: int = 0) -> tuple:
    s = [slice(None)] * ndim
    s[dim] = slice(lo, hi)
    return tuple(s)


def shard_pieces(name: str, shape: tuple[int, ...], config: Qwen4ExpConfig,
                 tp: TPInfo, attn: TPInfo | None = None) -> list[tuple]:
    """Slices of the full (HF-layout) checkpoint tensor ``name`` that form this rank's
    parameter, concatenated along dim 0. One full slice means replicated. Attention and
    GDN tensors follow ``attn`` (default: ``tp``), the rest ``tp``."""
    attn = attn or tp
    n = len(shape)
    full = [tuple(slice(None) for _ in shape)]
    c = config
    is_attn = ".linear_attn." in name or (".self_attn." in name and ".indexer." not in name)
    w, r = (attn.size, attn.rank) if is_attn else (tp.size, tp.rank)
    if w == 1:
        return full
    if ".linear_attn." in name:
        kd = c.linear_num_key_heads * c.linear_key_head_dim
        vd = c.linear_num_value_heads * c.linear_value_head_dim
        kl, vl, hl = kd // w, vd // w, c.linear_num_value_heads // w
        leaf = name.rsplit(".linear_attn.", 1)[1]
        if leaf in ("in_proj_qkv.weight", "conv1d.weight"):
            return [_rows(r * kl, (r + 1) * kl, n), _rows(kd + r * kl, kd + (r + 1) * kl, n),
                    _rows(2 * kd + r * vl, 2 * kd + (r + 1) * vl, n)]
        if leaf == "in_proj_z.weight":
            return [_rows(r * vl, (r + 1) * vl, n)]
        if leaf in ("in_proj_a.weight", "in_proj_b.weight", "A_log", "dt_bias"):
            return [_rows(r * hl, (r + 1) * hl, n)]
        if leaf == "out_proj.weight":
            return [_rows(r * vl, (r + 1) * vl, n, dim=1)]
        return full
    if ".self_attn." in name and ".indexer." not in name:
        hd, nh, nkv = c.head_dim, c.num_attention_heads, c.num_key_value_heads
        ql = nh // w
        leaf = name.rsplit(".self_attn.", 1)[1]
        if leaf == "q_proj.weight":
            return [_rows(r * ql * 2 * hd, (r + 1) * ql * 2 * hd, n)]
        if leaf in ("k_proj.weight", "v_proj.weight"):
            if w >= nkv:
                h = (r * ql) // (nh // nkv)
                return [_rows(h * hd, (h + 1) * hd, n)]
            kl = nkv // w
            return [_rows(r * kl * hd, (r + 1) * kl * hd, n)]
        if leaf == "o_proj.weight":
            return [_rows(r * ql * hd, (r + 1) * ql * hd, n, dim=1)]
        return full
    if name.endswith((".mlp.experts.gate_up_proj", ".mlp.experts.down_proj")):
        el = shape[0] // w
        return [_rows(r * el, (r + 1) * el, n)]
    if name in ("model.embed_tokens.weight", "lm_head.weight") or name.endswith(
        ".ple_embedding.ngram_embedding.weight"
    ):
        rl = shape[0] // w
        return [_rows(r * rl, (r + 1) * rl, n)]
    if ".mlp.shared_expert." in name and c.shared_expert_tp:
        il = c.shared_expert_intermediate_size // w
        if name.endswith(("gate_proj.weight", "up_proj.weight")):
            return [_rows(r * il, (r + 1) * il, n)]
        if name.endswith("down_proj.weight"):
            return [_rows(r * il, (r + 1) * il, n, dim=1)]
    return full


def local_expert_range(num_experts: int, tp: TPInfo) -> tuple[int, int]:
    el = num_experts // tp.size
    return tp.rank * el, (tp.rank + 1) * el


def e4m3fn_bytes_to(b: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Exact value of FP8 e4m3fn codes stored as ``uint8`` bytes, in ``dtype`` (bf16 or wider).

    Integer ops and one int16 -> bf16 bitcast, no FP8 dtype and no subnormal operand: the magnitude bits
    ``eeee mmm`` are moved into a bf16 with the exponent rebased by 120 (normal codes are then exact); a
    subnormal code (``eeee = 0``, value ``m * 2**-9``) is built as ``2**-6 * (1 + m/8)`` and ``2**-6`` taken
    off (exact). Every code |q| <= 448 decodes like ``torch.float8_e4m3fn`` -> bf16 (the NaN codes 0x7F / 0xFF
    give +-480; the loader refuses them). Unlike a trn2 e4m3 read, codes 256..448 stay finite."""
    x = b.to(torch.int32)
    mag = x & 0x7F
    sub = (mag >> 3) == 0
    bits = (mag << 4) + torch.where(sub, torch.full_like(mag, 121 << 7), torch.full_like(mag, 120 << 7))
    v = bits.to(torch.int16).view(torch.bfloat16)
    v = torch.where(sub, v - 2.0 ** -6, v)
    v = torch.where((x & 0x80) != 0, -v, v)
    return v.to(dtype)


def sharded_lookup_dequant(table: torch.Tensor, scale: torch.Tensor, ids: torch.Tensor, tp: TPInfo) -> torch.Tensor:
    """:func:`sharded_lookup` of an FP8 row-sharded table held as its e4m3fn bytes (``uint8``), with one
    per-tensor ``scale``.

    The stored bytes are gathered as they are and decoded after the gather
    (:func:`e4m3fn_bytes_to` ``(rows, scale.dtype) * scale``, the values and the op the BF16 path applies to
    the whole table at load), before the mask and the all-reduce: equal to the lookup in the dequantized
    table bit for bit."""
    def deq(rows):
        return e4m3fn_bytes_to(rows, scale.dtype) * scale

    if tp.size == 1:
        return deq(torch.nn.functional.embedding(ids.long(), table))
    rows = table.shape[0]
    local = ids.long() - rank_tensor(tp, ids.device).long() * rows
    mine = (local >= 0) & (local < rows)
    out = deq(torch.nn.functional.embedding(torch.where(mine, local, torch.zeros_like(local)), table))
    out = torch.where(mine.unsqueeze(-1), out, torch.zeros_like(out))
    return all_reduce(out, tp)


# ── Device fix (2026-11-06): no bitcast in the decode ────────────────────────────────────────────────────────
# On trn2 (1 core, neuron_native / Lite, SDK 2.33) the int16 -> bf16 ``.view`` in e4m3fn_bytes_to came out as a
# VALUE conversion (int bits 17072 -> 17024.0 instead of 88.0): every decoded element was wrong (655,360 of
# 655,360), so the FP8 PLE table produced garbage. Every integer op before it was bit-exact, and the
# same ``.view`` was exact when returned directly or multiplied by 1.0; it broke once its result fed
# ``torch.where(sub, v - 2**-6, v)`` in the same graph (device bisection, 11-06). This decode uses no bitcast: for
# e = |q| >> 3, m = |q| & 7 the value is base * 2**(max(e, 1) - 10) with base = m (e = 0, subnormal) or 8 + m,
# the power of two built exactly from the 4 bits of e (products of 2, 4, 16, 256), all exact in fp32 (base has
# <= 4 significant bits), so the result equals e4m3fn_bytes_to bit for bit for all 256 bytes. Rebinding the
# module global switches sharded_lookup_dequant (it looks the name up at call time); no line above moves.
e4m3fn_bytes_to_bitcast = e4m3fn_bytes_to


def e4m3fn_bytes_to_arith(b: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Exact value of e4m3fn codes stored as ``uint8`` (same values as :func:`e4m3fn_bytes_to_bitcast`)
    from integer ops, int -> fp32 conversions of values <= 15 and fp32 products of powers of two."""
    x = b.to(torch.int32)
    e = (x >> 3) & 0xF
    m = x & 0x7
    base = torch.where(e == 0, m, m + 8).to(torch.float32)
    k = torch.where(e == 0, torch.ones_like(e), e)  # exponent of 2**(k - 10), k in 1..15
    p = torch.full_like(base, 2.0 ** -10)
    for bit, f in ((1, 2.0), (2, 4.0), (4, 16.0), (8, 256.0)):
        p = p * torch.where((k & bit) != 0, torch.full_like(base, f), torch.ones_like(base))
    # sign as a factor of -1 / +1 (a select of -v gave +0.0 for the code 0x80 on device; the product keeps -0.0)
    v = base * p * (1 - 2 * ((x >> 7) & 1)).to(torch.float32)
    return v.to(dtype)


e4m3fn_bytes_to = e4m3fn_bytes_to_arith
