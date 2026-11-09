# SPDX-License-Identifier: Apache-2.0
"""Qwen4-Exp building blocks (single rank, PyTorch).

These modules define the math of the model and are the CPU path. Parameter names
match the HF checkpoint below ``model.language_model``, so a checkpoint state dict
loads without renaming. Per-request recurrent state is passed in and returned
explicitly (``[B, ...]`` tensors), so each block can be checked against the HF
reference on its own before it is wired to the runner's state cache.

Attribution: the model math (and parts of its PyTorch expression) follows the Hugging Face
Transformers 5.17 reference, ``transformers/models/qwen4_exp/modeling_qwen4_exp.py`` (Copyright 2026
The Qwen Team and The HuggingFace Inc. team, Apache License 2.0).

Layouts: activations are ``[B, T, ...]``. Prefill is one request (``B=1``) and
decode is ``T=1``.

The Gated DeltaNet core calls the PyTorch references of the nkilib GDN kernels
(``gdn_cte`` / ``gdn_conv1d_*``), so the interface here is the one the device
kernels take.
"""

import math
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn

from .config import FULL_ATTENTION, Qwen4ExpConfig
from .parallel import all_reduce, kv_heads_per_rank, rank_tensor, sharded_lookup

def window_after(x: torch.Tensor, num_valid: torch.Tensor | None, length: int) -> torch.Tensor:
    """The ``length`` columns of ``x`` ``[B, C, length + T]`` ending at the last valid input.

    ``x`` is ``cat(old_window, new_inputs)``. Inputs are right-padded, so with
    ``num_valid`` ``[B]`` valid inputs per row the new window starts at
    ``num_valid``; 0 keeps the old window. ``None`` means every input is valid.
    """
    if num_valid is None:
        return x[..., -length:]
    idx = num_valid.long().view(-1, 1, 1) + torch.arange(length, device=x.device).view(1, 1, -1)
    return x.gather(-1, idx.expand(x.shape[0], x.shape[1], length))


# ─────────────────────────────────────────────────────────────────────────────
# Norms
# ─────────────────────────────────────────────────────────────────────────────


class Qwen4ExpRMSNorm(nn.Module):
    """RMSNorm with a ``(1 + weight)`` scale, optionally per ``group_size`` slice."""

    def __init__(self, dim: int, eps: float, group_size: int | None = None):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps
        self.group_size = group_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.float()
        if self.group_size is not None:
            h = h.unflatten(-1, (-1, self.group_size))
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.group_size is not None:
            h = h.flatten(-2)
        return (h * (1.0 + self.weight.float())).type_as(x)


class Qwen4ExpRMSNormGated(nn.Module):
    """``weight * norm(x) * act(gate)``; the weight is used as is (not ``1 + w``)."""

    def __init__(self, dim: int, eps: float, activation: str):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.activation = activation
        self.act = torch.sigmoid if activation == "sigmoid" else F.silu

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        h = self.weight * h.to(dtype)
        return (h * self.act(gate.float())).to(dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Hyper-connections
# ─────────────────────────────────────────────────────────────────────────────


class Qwen4ExpGatedResidual(nn.Module):
    """Read gate over the ``hc_count`` residual streams, plus per-branch write gates.

    ``forward`` returns ``(mixed, streams, inject)`` where ``mixed`` ``[..., H]`` is
    the block input, ``streams`` is the unchanged residual and ``inject``
    ``[..., hc]`` scales the block output into each stream (see :func:`inject`).
    With ``use_combine=False`` (the final mixer) only ``mixed`` is returned.
    """

    def __init__(self, config: Qwen4ExpConfig, use_combine: bool = True):
        super().__init__()
        self.hc = config.hc_count
        self.hidden = config.hidden_size
        hc_hidden = config.hc_hidden_size
        self.hc_norm = Qwen4ExpRMSNorm(hc_hidden, config.rms_norm_eps, group_size=self.hidden)
        self.input_mix_weight_down = nn.Linear(hc_hidden, config.hc_lowrank, bias=False)
        self.input_mix_weight_up = nn.Linear(config.hc_lowrank, hc_hidden, bias=False)
        self.block_inject_weight = (
            nn.Linear(hc_hidden, self.hc, bias=False) if use_combine else None
        )

    def forward(self, streams: torch.Tensor):
        normed = self.hc_norm(streams)
        read = F.silu(self.input_mix_weight_down(normed) / self.hc)
        read = torch.sigmoid(self.input_mix_weight_up(read)).unflatten(-1, (self.hc, self.hidden))
        mixed = (read * normed.unflatten(-1, (self.hc, self.hidden))).mean(dim=-2)
        if self.block_inject_weight is None:
            return mixed
        inject = 2 * torch.sigmoid(self.block_inject_weight(normed) / self.hc)
        return mixed, streams, inject


def inject(streams: torch.Tensor, block_out: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """``streams + block_out`` written into every stream with its own scale."""
    return streams + (block_out.unsqueeze(-2) * weights.unsqueeze(-1)).flatten(-2)


# ─────────────────────────────────────────────────────────────────────────────
# Per-layer n-gram embedding
# ─────────────────────────────────────────────────────────────────────────────


class Qwen4ExpNGramEmbedding(nn.Module):
    """The hashed n-gram table. Row ids come from the host (:mod:`.ngram`)."""

    def __init__(self, config: Qwen4ExpConfig, padded_vocab_size: int):
        super().__init__()
        self.tp = config.tp
        self.ngram_embedding = nn.Embedding(
            padded_vocab_size // self.tp.size, config.ple_embed_dim // config.ngram_heads
        )

    def forward(self, ngram_ids: torch.Tensor) -> torch.Tensor:
        """``[..., num_heads]`` int ids -> ``[..., ple_embed_dim]`` (rows sharded over TP)."""
        return sharded_lookup(self.ngram_embedding.weight, ngram_ids, self.tp).flatten(-2)


class Qwen4ExpPLELayer(nn.Module):
    """Gate the n-gram value into every stream, then add a dilated short conv.

    State: ``conv_state`` ``[B, hc*H, (kernel-1)*ngram_size]`` holds the last
    normalized gated values (zeros at the start of a request).
    """

    def __init__(self, config: Qwen4ExpConfig, padded_vocab_size: int):
        super().__init__()
        self.hc = config.hc_count
        self.hidden = config.hidden_size
        hc_hidden = config.hc_hidden_size
        e = config.ple_embed_dim
        eps = config.rms_norm_eps
        self.dilation = config.ngram_size
        self.state_len = (config.ple_conv_kernel_size - 1) * self.dilation
        self.ple_embedding = Qwen4ExpNGramEmbedding(config, padded_vocab_size)
        self.key_proj = nn.Linear(e, hc_hidden, bias=False)
        self.value_proj = nn.Linear(e, self.hidden, bias=False)
        self.norm_key = Qwen4ExpRMSNorm(hc_hidden, eps, group_size=self.hidden)
        self.norm_query = Qwen4ExpRMSNorm(hc_hidden, eps, group_size=self.hidden)
        self.norm_conv = Qwen4ExpRMSNorm(hc_hidden, eps, group_size=self.hidden)
        self.conv1d = nn.Conv1d(
            hc_hidden,
            hc_hidden,
            kernel_size=config.ple_conv_kernel_size,
            groups=hc_hidden,
            dilation=self.dilation,
            bias=False,
        )

    def init_state(self, batch: int, dtype: torch.dtype, device=None) -> torch.Tensor:
        return torch.zeros(batch, self.conv1d.in_channels, self.state_len, dtype=dtype, device=device)

    def forward(
        self,
        streams: torch.Tensor,
        ngram_ids: torch.Tensor,
        conv_state: torch.Tensor,
        num_valid: torch.Tensor | None = None,
    ):
        emb = self.ple_embedding(ngram_ids).to(streams.dtype)
        key = self.norm_key(self.key_proj(emb)).unflatten(-1, (self.hc, self.hidden))
        value = self.value_proj(emb)
        query = self.norm_query(streams).unflatten(-1, (self.hc, self.hidden))
        gate = (key * query).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gated = (torch.sigmoid(gate) * value.unsqueeze(-2)).flatten(-2)
        gated_normed = self.norm_conv(gated)

        seq_len = gated.shape[1]
        x = torch.cat([conv_state, gated_normed.transpose(1, 2)], dim=-1)  # [B, C, state+T]
        new_state = window_after(x, num_valid, self.state_len)
        conv = F.silu(self.conv1d(x)).transpose(1, 2)  # [B, T, C]
        assert conv.shape[1] == seq_len
        return gated + conv, new_state


# ─────────────────────────────────────────────────────────────────────────────
# Gated full attention (dense)
# ─────────────────────────────────────────────────────────────────────────────


def rope_cos_sin(config: Qwen4ExpConfig, positions: torch.Tensor, dtype: torch.dtype):
    """Partial RoPE tables ``[..., rotary_dim]`` for text positions.

    Interleaved M-RoPE with identical T/H/W positions (text-only input) is plain 1-D RoPE.
    """
    dim = config.rotary_dim
    inv_freq = 1.0 / (
        config.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    )
    freqs = positions.float()[..., None] * inv_freq.to(positions.device)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_partial_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate the first ``rotary_dim`` channels of ``x`` ``[B, T, heads, D]``."""
    r = cos.shape[-1]
    cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
    x_rot, x_pass = x[..., :r], x[..., r:]
    return torch.cat([x_rot * cos + _rotate_half(x_rot) * sin, x_pass], dim=-1)


def apply_partial_rope_no_concat(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """:func:`apply_partial_rope` with the rotated / pass-through channels joined by pad + select instead of cat.

    Used by the QSA attention (max_model_len above ``dense_attention_exact_len``): at max_model_len 81,920 the target
    prefill graphs failed in the neuronx-cc 2.28.7631 backend with NCC_IINAR001 "ISA validation failed: Copy ...
    start_addr_active_channels" on a partition-free transpose of the attention K RoPE concatenate ([B, T, 1, 256] = 64
    rotated + 192 pass-through channels). The rotated channels zero-padded to full width, selected against x itself by
    channel index, give the same values bit for bit (torch.where only selects) with no concatenate and no slice / pad of
    the pass-through part at channel offset r. The dense attention keeps :func:`apply_partial_rope`.
    """
    r, d = cos.shape[-1], x.shape[-1]
    if r >= d:
        return apply_partial_rope(x, cos, sin)
    cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
    x_rot = x[..., :r]
    rot = x_rot * cos + _rotate_half(x_rot) * sin
    first = torch.arange(d, device=x.device) < r
    return torch.where(first, F.pad(rot, (0, d - r)), x)  # pass-through channels straight from x (no slice at offset r)


class Qwen4ExpAttention(nn.Module):
    """Gated GQA attention with QK-norm and partial RoPE, run dense.

    Exact w.r.t. QSA while the context is at most ``config.dense_attention_exact_len``.
    The indexer weights are loaded but unused in this mode.

    KV state: ``(k, v)`` each ``[B, kv_heads, S_past, head_dim]``.
    """

    def __init__(self, config: Qwen4ExpConfig):
        super().__init__()
        self.config = config
        self.tp = config.attn_tp
        self.num_heads = config.num_attention_heads // self.tp.size  # per rank
        self.num_kv_heads = kv_heads_per_rank(config, self.tp)
        self.head_dim = config.head_dim
        h = config.hidden_size
        self.q_proj = nn.Linear(h, self.num_heads * self.head_dim * 2, bias=False)
        self.k_proj = nn.Linear(h, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(h, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, h, bias=False)
        self.q_norm = Qwen4ExpRMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = Qwen4ExpRMSNorm(self.head_dim, config.rms_norm_eps)
        if config.indexer_n_heads:
            self.indexer = Qwen4ExpQSAIndexerWeights(config)
        # Plain causal attention when there is no indexer, or when no request can be
        # longer than the QSA budget (then QSA selects every block: dense is exact and
        # the indexer and its cache are skipped).
        self.dense = not config.indexer_n_heads or (
            config.max_model_len is not None
            and config.max_model_len <= config.dense_attention_exact_len
        )

    def _qkv(self, x: torch.Tensor, positions: torch.Tensor):
        b, t, _ = x.shape
        q, gate = self.q_proj(x).view(b, t, self.num_heads, 2 * self.head_dim).chunk(2, dim=-1)
        # Normalise q / k as contiguous [rows, head_dim]: on the strided [B, T, H, D] view neuronx-cc
        # issued one Scalar + one Vector instruction per (token, head) row on 4 partitions (12 ms per
        # 512-token prefill chunk across the 12 full-attention layers, 2026-11-05 profile).
        d = self.head_dim
        q = self.q_norm(q.contiguous().view(-1, d)).view(b, t, self.num_heads, d)
        k = self.k_norm(self.k_proj(x).reshape(-1, d)).view(b, t, self.num_kv_heads, d)
        v = self.v_proj(x).view(b, t, self.num_kv_heads, self.head_dim)
        cos, sin = rope_cos_sin(self.config, positions, x.dtype)
        rope = apply_partial_rope if self.dense else apply_partial_rope_no_concat
        return rope(q, cos, sin), rope(k, cos, sin), v, gate.reshape(b, t, -1)

    def _qsa_mask_paged(self, x, positions, meta, rows, col):
        """QSA mask over the paged raw indexer keys (key-only cache, one key per token).

        Same read-after-write rule as the KV: the current tokens' keys are placed into
        the gathered window directly (``rows`` / ``col`` as for K/V)."""
        b, t, _ = x.shape
        iq, ik = self.indexer.project(x, positions)
        cache = self.idx_cache  # [blocks, 1, block, D]
        bs = cache.shape[2]
        slots = meta["slot_mapping"].long()
        tok = ik.reshape(b * t, 1, -1).to(cache.dtype)
        kv_write(cache, kv_rows(slots, cache), tok)
        table = meta["block_table_tensor"][:b].long()
        win = cache[table].flatten(1, 3)[..., : ik.shape[-1]]  # [B, S, D]
        s_ctx = win.shape[1]
        win = torch.cat([win, win.new_zeros(b, 1, win.shape[-1])], dim=1)
        win[rows, col.clamp(max=s_ctx)] = tok.view(b, t, -1)
        return self.indexer.select(iq, win[:, :s_ctx].to(ik.dtype), positions)

    def forward_paged(self, x: torch.Tensor, positions: torch.Tensor, meta: dict,
                      num_valid: torch.Tensor | None = None, idx_meta: dict | None = None):
        """Attention over the runner's paged KV cache (``[blocks, kv_heads, block, D]``).
        Writes the new K/V at ``slot_mapping`` (padding goes to the null block) and
        attends over every block of each row's block table, masked to key positions
        ``<=`` the query position (plain ops: head_dim 256 > the NKI kernels' 128).
        The current tokens' K/V are placed into the gathered window directly instead
        of being read back from the cache: on neuronx-cc a cache write is not
        guaranteed visible to a gather in the same graph (recorded on the Qwen3.6
        port). Padding tokens (past ``num_valid``) go to a discarded extra column.
        One-token decode takes :func:`_paged_decode` (no index_put into the window).
        """
        if self.dense or idx_meta is not None:  # dense / QSA forms without index_put (end of module)
            return (_paged_decode if x.shape[1] == 1 else _paged_prefill)(self, x, positions, meta, num_valid, idx_meta)
        b, t, _ = x.shape
        q, k, v, gate = self._qkv(x, positions)
        bs = self.k_cache.shape[2]
        slots = meta["slot_mapping"].long()
        k_tok = k.reshape(b * t, self.num_kv_heads, self.head_dim).to(self.k_cache.dtype)
        v_tok = v.reshape(b * t, self.num_kv_heads, self.head_dim).to(self.v_cache.dtype)
        kv_rows_ = kv_rows(slots, self.k_cache)
        kv_write(self.k_cache, kv_rows_, k_tok)
        kv_write(self.v_cache, kv_rows_, v_tok)

        table = meta["block_table_tensor"][:b].long()  # [B, max_blocks]
        kc = self.k_cache[table].permute(0, 2, 1, 3, 4).flatten(2, 3)  # [B, KVH, S, D]
        vc = self.v_cache[table].permute(0, 2, 1, 3, 4).flatten(2, 3)
        s_ctx = kc.shape[2]
        col = positions.long()
        if num_valid is not None:
            pad = torch.arange(t, device=x.device).view(1, t) >= num_valid.view(b, 1)
            col = torch.where(pad, torch.full_like(col, s_ctx), col)
        rows = torch.arange(b, device=x.device).view(b, 1).expand(b, t)
        extra = kc.new_zeros(b, self.num_kv_heads, 1, self.head_dim)
        kc, vc = torch.cat([kc, extra], dim=2), torch.cat([vc, extra], dim=2)
        kc[rows, :, col] = k_tok.view(b, t, self.num_kv_heads, self.head_dim)
        vc[rows, :, col] = v_tok.view(b, t, self.num_kv_heads, self.head_dim)
        kc, vc = kc[:, :, :s_ctx], vc[:, :, :s_ctx]
        kv_pos = torch.arange(s_ctx, device=x.device)
        # Positions past a row's last query were never written (block-table padding, the
        # rest of the open block): device HBM there is not zeroed and may hold NaN, and a
        # masked key still meets its value as 0 * NaN in the softmax-weighted sum.
        written = kv_pos.view(1, 1, -1, 1) <= positions.amax(dim=1).view(b, 1, 1, 1)
        kc = torch.where(written, kc, torch.zeros_like(kc))
        vc = torch.where(written, vc, torch.zeros_like(vc))
        mask = kv_pos.view(1, 1, 1, -1) <= positions.view(b, 1, t, 1)
        if not self.dense:
            mask = mask & self._qsa_mask_paged(x, positions, idx_meta, rows, col)[:, None]
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), kc.to(q.dtype), vc.to(q.dtype),
            attn_mask=mask, scale=self.head_dim**-0.5, enable_gqa=True,
        )
        out = out.transpose(1, 2).reshape(b, t, -1) * torch.sigmoid(gate)
        return all_reduce(self.o_proj(out), self.tp)

    def forward(self, x: torch.Tensor, positions: torch.Tensor, kv: tuple | None):
        """Reference path. KV state ``(k, v, raw indexer keys [B, S_past, D])``."""
        b, t, _ = x.shape
        q, k, v, gate = self._qkv(x, positions)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ik = None
        if not self.dense:
            iq, ik = self.indexer.project(x, positions)
        if kv is not None:
            k = torch.cat([kv[0], k], dim=2)
            v = torch.cat([kv[1], v], dim=2)
            if ik is not None:
                ik = torch.cat([kv[2], ik], dim=1)
        past = k.shape[2] - t
        mask = torch.ones(t, k.shape[2], dtype=torch.bool, device=x.device).tril(diagonal=past)
        if ik is not None:
            mask = mask & self.indexer.select(iq, ik, positions)[:, None]
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, scale=self.head_dim**-0.5, enable_gqa=True
        )
        out = out.transpose(1, 2).reshape(b, t, -1) * torch.sigmoid(gate)
        return all_reduce(self.o_proj(out), self.tp), (k, v, ik)


class Qwen4ExpQSAIndexerWeights(nn.Module):
    """QSA indexer: scores 4-token key blocks per query and selects the top blocks.

    ``project`` gives the per-token indexer query ``[B, T, Hi, D]`` (normed, RoPE at the
    query position) and the raw key ``[B, T, D]`` that is cached (one per token).
    ``select`` turns a window of raw keys at absolute positions ``0..S-1`` into the
    boolean token mask ``[B, T, S]`` of QSA: the top ``budget / ratio`` complete
    blocks plus the open block's tokens up to the query (the reference semantics:
    a block is the mean of ``ratio`` raw keys, normed, RoPE at its first position;
    score = sum over heads of relu(q . k) / sqrt(D)). With at most ``budget / ratio``
    complete blocks every block is selected, i.e. dense causal attention.
    """

    def __init__(self, config: Qwen4ExpConfig):
        super().__init__()
        d = config.indexer_head_dim
        self.config = config
        self.n_heads = config.indexer_n_heads
        self.head_dim = d
        self.ratio = config.indexer_compress_ratio
        self.block_topk = config.indexer_budget // config.indexer_compress_ratio
        self.index_qk_proj = nn.Linear(
            config.hidden_size, (config.indexer_n_heads + config.indexer_kv_heads) * d, bias=False
        )
        self.q_layernorm = Qwen4ExpRMSNorm(d, config.rms_norm_eps)
        self.k_layernorm = Qwen4ExpRMSNorm(d, config.rms_norm_eps)

    def project(self, x: torch.Tensor, positions: torch.Tensor):
        b, t, _ = x.shape
        q, k = self.index_qk_proj(x).split([self.n_heads * self.head_dim, self.head_dim], dim=-1)
        q = self.q_layernorm(q.view(b, t, self.n_heads, self.head_dim))
        cos, sin = rope_cos_sin(self.config, positions, x.dtype)
        return apply_partial_rope_no_concat(q, cos, sin), k

    def select(self, q: torch.Tensor, raw_k: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        b, t = positions.shape
        s = raw_k.shape[1]
        r = self.ratio
        nb = s // r
        pooled = raw_k[:, : nb * r].reshape(b, nb, r, self.head_dim).float().mean(dim=2).to(raw_k.dtype)
        pooled = self.k_layernorm(pooled)
        start = (torch.arange(nb, device=raw_k.device) * r).view(1, nb).expand(b, nb)
        cos, sin = rope_cos_sin(self.config, start, raw_k.dtype)
        pooled = apply_partial_rope_no_concat(pooled.unsqueeze(2), cos, sin).squeeze(2)  # [B, nb, D]
        scores = torch.einsum("bthd,bjd->btjh", q.float(), pooled.float()).relu().sum(-1)
        scores = scores / self.head_dim**0.5  # [B, T, nb]
        complete = (positions.long() + 1) // r  # complete blocks visible to each query
        valid = torch.arange(nb, device=raw_k.device).view(1, 1, nb) < complete.unsqueeze(-1)
        scores = torch.where(valid, scores, torch.full_like(scores, float("-inf")))
        top = torch.topk(scores, min(self.block_topk, nb), dim=-1).indices
        chosen = torch.zeros_like(valid).scatter(-1, top, True) & valid
        tok = torch.arange(s, device=raw_k.device)
        in_block = chosen.gather(-1, (tok // r).clamp(max=nb - 1).view(1, 1, s).expand(b, t, s))
        tail_start = (complete * r).unsqueeze(-1)
        before_tail = tok.view(1, 1, s) < tail_start
        tail = (tok.view(1, 1, s) >= tail_start) & (tok.view(1, 1, s) <= positions.long().unsqueeze(-1))
        return (in_block & before_tail) | tail


# ─────────────────────────────────────────────────────────────────────────────
# Gated DeltaNet
# ─────────────────────────────────────────────────────────────────────────────


# Sequence length granularity of the gdn_cte (Schur) prefill kernel.
GDN_CTE_CHUNK = 128


def _kernels_available(x: torch.Tensor) -> bool:
    """NKI kernels run on device, or on CPU under the NKI simulator."""
    from vllm_neuron.utils.neuron_utils import can_run_kernel

    return can_run_kernel(x)


def _nki(kernel):
    """``kernel`` wrapped for torch, launched on the LNC grid.

    The grid must equal the compiler's LNC (NCC_EVRF066), also for kernels that do not
    split work across the grid (gdn_conv1d_*): both cores then compute the same output.
    """
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from vllm_neuron.utils.hardware_config import get_logical_nc_config

    return wrap_nki(kernel)[get_logical_nc_config()]


@lru_cache(maxsize=1)
def _gdn_mask_pack() -> torch.Tensor:
    from nkilib.experimental.gdn import build_mask_pack

    return torch.as_tensor(build_mask_pack())


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


class Qwen4ExpGatedDeltaNet(nn.Module):
    """Gated DeltaNet linear attention.

    State: ``conv_state`` ``[B, conv_dim, kernel]`` (last raw q|k|v inputs) and
    ``recurrent_state`` ``[B, Hv, Dk, Dv]`` fp32.
    """

    def __init__(self, config: Qwen4ExpConfig):
        super().__init__()
        self.tp = config.attn_tp
        self.num_k_heads = config.linear_num_key_heads // self.tp.size  # per rank
        self.num_v_heads = config.linear_num_value_heads // self.tp.size
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.kernel = config.linear_conv_kernel_dim
        h = config.hidden_size
        self.in_proj_qkv = nn.Linear(h, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(h, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(h, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(h, self.num_v_heads, bias=False)
        self.conv1d = nn.Conv1d(
            self.conv_dim, self.conv_dim, self.kernel, groups=self.conv_dim, bias=False
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads))
        self.norm = Qwen4ExpRMSNormGated(
            self.head_v_dim, config.rms_norm_eps, config.output_gate_type
        )
        self.out_proj = nn.Linear(self.value_dim, h, bias=False)

    def init_state(self, batch: int, dtype: torch.dtype, device=None):
        conv = torch.zeros(batch, self.conv_dim, self.kernel, dtype=dtype, device=device)
        rec = torch.zeros(
            batch, self.num_v_heads, self.head_k_dim, self.head_v_dim,
            dtype=torch.float32, device=device,
        )
        return conv, rec

    def _conv(self, mixed: torch.Tensor, conv_state: torch.Tensor, num_valid: torch.Tensor | None):
        """Causal depthwise conv + silu over ``[B, C, T]`` continuing from ``conv_state``."""
        weight = self.conv1d.weight.squeeze(1)
        x = torch.cat([conv_state, mixed], dim=-1)  # [B, C, K + T]
        if mixed.shape[-1] == 1:
            # One-token decode: the output is the window of the last K inputs dotted with the
            # weights, and that window is the new state. Elementwise, without the prefill kernel
            # and the window gather (8.6 + 3.3 ms per TP32 step at B=8, 2026-11-03 profile).
            win = x[..., 1:]
            out = F.silu((win.float() * weight.float()).sum(-1, keepdim=True)).to(mixed.dtype)
            if num_valid is not None:  # a row with no valid input keeps its window
                win = torch.where(num_valid.view(-1, 1, 1) > 0, win, conv_state)
            return out, win
        # The prefill conv starts from zero padding; continuing a request prepends the
        # last K-1 inputs and drops the corresponding outputs. The new window is taken
        # here (it must end at the last valid input), not from the kernel.
        if _kernels_available(x):
            from nkilib.experimental.gdn import gdn_conv1d_prefill

            out, _ = _nki(gdn_conv1d_prefill)(mixed=x[..., 1:], conv_weight=weight.float())
        else:
            from nkilib.experimental.gdn.gdn_conv1d_torch import gdn_conv1d_prefill_torch_ref

            out, _ = gdn_conv1d_prefill_torch_ref(x[..., 1:], weight)
        return out[..., self.kernel - 1 :], window_after(x, num_valid, self.kernel)

    def _decode_kernel(self, x, mixed, recurrent_state):
        """One-token step with the fused nkilib-derived ``gdn_tkg`` (l2norm, beta, g,
        recurrence and the gated RMSNorm in one kernel)."""
        from vllm_neuron._kernels.qwen4_exp import gdn_tkg

        b = x.shape[0]
        rep = self.num_v_heads // self.num_k_heads
        q, k, v = mixed[..., 0].split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
        # GQA-expand q/k to value heads (value head h reads key head h // rep).
        q = q.view(b, self.num_k_heads, self.head_k_dim).repeat_interleave(rep, dim=1)
        k = k.view(b, self.num_k_heads, self.head_k_dim).repeat_interleave(rep, dim=1)
        bh = b * self.num_v_heads
        out, state = _nki(gdn_tkg)(
            q=q.reshape(bh, -1).to(x.dtype),
            k=k.reshape(bh, -1).to(x.dtype),
            v=v.reshape(bh, -1).to(x.dtype),
            b=self.in_proj_b(x).float().reshape(bh),
            a=self.in_proj_a(x).float().reshape(bh),
            A_log=self.A_log.float(),
            dt_bias=self.dt_bias.float(),
            z=self.in_proj_z(x).reshape(bh, -1).to(x.dtype),
            norm_weight=self.norm.weight.float(),
            state_in=recurrent_state.reshape(bh, self.head_k_dim, self.head_v_dim).float(),
            gate_activation=self.norm.activation,
        )
        out = out.to(x.dtype).reshape(b, 1, self.value_dim)
        return all_reduce(self.out_proj(out), self.tp), state.reshape(recurrent_state.shape)

    def forward(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
        num_valid: torch.Tensor | None = None,
    ):
        """``num_valid`` ``[B]``: inputs past it (right padding) leave the state untouched."""
        b, t, _ = x.shape
        mixed, conv_state = self._conv(self.in_proj_qkv(x).transpose(1, 2), conv_state, num_valid)
        if t == 1 and _kernels_available(x):
            # Decode rows are never padding within a row (padding rows use a dummy slot).
            out, recurrent_state = self._decode_kernel(x, mixed, recurrent_state)
            return out, conv_state, recurrent_state
        q, k, v = mixed.transpose(1, 2).split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = _l2norm(q.reshape(b, t, self.num_k_heads, self.head_k_dim).float())
        k = _l2norm(k.reshape(b, t, self.num_k_heads, self.head_k_dim).float())
        v = v.reshape(b, t, self.num_v_heads, self.head_v_dim); q, k, v = _zero_padding(num_valid, q, k, v)

        beta = torch.sigmoid(self.in_proj_b(x).float()).transpose(1, 2)  # [B, Hv, T]
        g_log = -self.A_log.float().exp() * F.softplus(self.in_proj_a(x).float() + self.dt_bias)
        if num_valid is not None:
            # decay 1 and no update on padding: the recurrent state passes through.
            pad = torch.arange(t, device=x.device).view(1, t) >= num_valid.view(b, 1)
            beta = beta.masked_fill(pad.view(b, 1, t), 0.0)
            g_log = g_log.masked_fill(pad.view(b, t, 1), 0.0)
        cte_args = dict(
            query=q.transpose(1, 2).to(x.dtype),
            key=k.transpose(1, 2).to(x.dtype),
            value=v.transpose(1, 2).to(x.dtype),
            g_log=g_log.transpose(1, 2).contiguous(),
            beta=beta.contiguous(),
            scale=self.head_k_dim**-0.5,
            mask_pack=_gdn_mask_pack().to(x.device),
            recurrent_state=recurrent_state.float(),
        )
        if t % GDN_CTE_CHUNK == 0 and _kernels_available(x):
            from nkilib.experimental.gdn import gdn_cte

            out, recurrent_state = _nki(gdn_cte)(**cte_args)
        else:
            from nkilib.experimental.gdn.gdn_cte_torch import gdn_cte_torch_ref

            out, recurrent_state = gdn_cte_torch_ref(**cte_args)
        out = out.to(x.dtype).transpose(1, 2)  # [B, T, Hv, Dv]
        z = self.in_proj_z(x).reshape(b, t, self.num_v_heads, self.head_v_dim)
        out = self.norm(out, z).reshape(b, t, self.value_dim)
        return all_reduce(self.out_proj(out), self.tp), conv_state, recurrent_state


# ─────────────────────────────────────────────────────────────────────────────
# MoE
# ─────────────────────────────────────────────────────────────────────────────


class Qwen4ExpRouter(nn.Module):
    def __init__(self, config: Qwen4ExpConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.weight = nn.Parameter(torch.zeros(config.num_experts, config.hidden_size))

    def forward(self, x: torch.Tensor):
        logits = F.linear(x, self.weight)
        probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
        weights, experts = torch.topk(probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        return weights.to(logits.dtype), experts


class Qwen4ExpExperts(nn.Module):
    """Expert weights as 3-D tensors.

    HF layout (``kernel_layout=False``): ``gate_up_proj`` ``[E, 2I, H]`` (gate rows
    first), ``down_proj`` ``[E, H, I]``. Kernel layout (``kernel_layout=True``, what
    the nkilib MoE kernels read): ``gate_up_proj`` ``[E, H, 2I]`` (gate columns first,
    viewed as ``[E, H, 2, I]``) and ``down_proj`` ``[E, I, H]`` -- the last two dims
    transposed, done once at load so no step transposes the experts.

    ``E`` is this rank's ``num_experts / W`` experts with the full intermediate.

    ``fp8=True`` (kernel layout only): the weights are FP8 (``torch.float8_e4m3fn`` holding trn2
    e4m3 values, ``|q| <= 240``) with per-output-channel fp32 scales ``gate_up_scale``
    ``[E, 1, 2I]`` (gate columns then up columns) and ``down_scale`` ``[E, 1, H]``:
    ``W = q * scale`` (see ``fp8_row_quantize`` at the end of this module).
    """

    def __init__(self, config: Qwen4ExpConfig, kernel_layout: bool = False, fp8: bool = False):
        super().__init__()
        e, h, i = config.num_experts // config.tp.size, config.hidden_size, config.moe_intermediate_size
        self.kernel_layout = kernel_layout
        self.fp8 = fp8
        if fp8:
            if not kernel_layout:
                raise ValueError("FP8 experts exist in the kernel layout (expert parallel) only")
            f8 = torch.float8_e4m3fn
            self.gate_up_proj = nn.Parameter(torch.empty(e, h, 2 * i, dtype=f8), requires_grad=False)
            self.down_proj = nn.Parameter(torch.empty(e, i, h, dtype=f8), requires_grad=False)
            self.gate_up_scale = nn.Parameter(torch.empty(e, 1, 2 * i, dtype=torch.float32), requires_grad=False)
            self.down_scale = nn.Parameter(torch.empty(e, 1, h, dtype=torch.float32), requires_grad=False)
        elif kernel_layout:
            self.gate_up_proj = nn.Parameter(torch.empty(e, h, 2 * i))
            self.down_proj = nn.Parameter(torch.empty(e, i, h))
        else:
            self.gate_up_proj = nn.Parameter(torch.empty(e, 2 * i, h))
            self.down_proj = nn.Parameter(torch.empty(e, h, i))

    def forward(self, x: torch.Tensor, weights: torch.Tensor, experts: torch.Tensor):
        """Reference (data-dependent loop over the selected experts; not traceable)."""
        gate_up, down = self.gate_up_proj, self.down_proj
        if self.fp8:
            gate_up, down = fp8_row_dequant(gate_up, self.gate_up_scale, x.dtype), fp8_row_dequant(
                down, self.down_scale, x.dtype)
        gate_up = gate_up.transpose(-1, -2) if self.kernel_layout else gate_up
        down = down.transpose(-1, -2) if self.kernel_layout else down
        out = torch.zeros_like(x)
        for e in torch.unique(experts).tolist():
            tok, slot = torch.where(experts == e)
            gate, up = F.linear(x[tok], gate_up[e]).chunk(2, dim=-1)
            y = F.linear(F.silu(gate) * up, down[e]) * weights[tok, slot, None]
            out.index_add_(0, tok, y.to(out.dtype))
        return out


class Qwen4ExpMLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# Prefill tokens per block of the blockwise MoE kernel.
MOE_CTE_BLOCK_SIZE = 128


def _moe_decode_gather(x, gate_up, down, weights, experts, base):
    """Static-shape decode MoE in PyTorch: gather each token's top-k expert weights.

    ``gate_up`` ``[E_L, H, 2, I]``, ``down`` ``[E_L, I, H]`` (kernel layout, this rank's
    experts ``[base, base + E_L)``); ``weights`` / ``experts`` ``[T, K]`` global. Experts
    of other ranks contribute zero here. Memory is ``T*K`` expert matrices (CPU runs).
    """
    local = experts.long() - base.long()
    mine = (local >= 0) & (local < gate_up.shape[0])
    local = torch.where(mine, local, torch.zeros_like(local))
    gu = gate_up[local]  # [T, K, H, 2, I]
    h = torch.einsum("th,tkhgi->tkgi", x, gu)
    act = F.silu(h[:, :, 0]) * h[:, :, 1]  # [T, K, I]
    y = torch.einsum("tki,tkih->tkh", act, down[local])
    w = torch.where(mine, weights, torch.zeros_like(weights))
    return (y * w.unsqueeze(-1)).sum(dim=1).to(x.dtype)


class Qwen4ExpSparseMoe(nn.Module):
    """Softmax top-k MoE + sigmoid-gated shared expert.

    ``use_kernels=False`` runs the PyTorch reference (used by the equivalence
    harnesses). ``use_kernels=True`` (the runner path) routes through
    ``vllm_neuron.functional``: router, then ``moe_cte`` for prefill and ``moe_tkg``
    for decode -- static shapes, NKI kernels on device, PyTorch fallbacks on CPU.
    """

    def __init__(self, config: Qwen4ExpConfig, use_kernels: bool = False):
        super().__init__()
        self.tp = config.tp
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.use_kernels = use_kernels
        self.gate = Qwen4ExpRouter(config)
        self.experts = Qwen4ExpExperts(config, kernel_layout=use_kernels, fp8=use_kernels and config.expert_fp8)
        self.shard_shared = config.shared_expert_tp
        self.shared_expert = Qwen4ExpMLP(
            config.hidden_size,
            config.shared_expert_intermediate_size // (config.tp_size if self.shard_shared else 1))
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, x: torch.Tensor, is_decode: bool = False) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        shared = torch.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
        if self.use_kernels:
            # Expert parallel: each rank adds its local experts. A split shared expert
            # (intermediate columns per rank) joins the same reduction; a replicated one
            # is added once after it.
            routed = self._routed_kernels(x, is_decode)
            if self.shard_shared:
                out = all_reduce(routed + shared, self.tp)
            else:
                out = all_reduce(routed, self.tp) + shared
        else:
            weights, experts = self.gate(x)
            out = self.experts(x, weights, experts) + shared
        return out.reshape(shape)

    def _routed_kernels(self, x: torch.Tensor, is_decode: bool) -> torch.Tensor:
        import vllm_neuron.functional as NF
        from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode
        from vllm.distributed.parallel_state import get_tp_group

        from vllm_neuron.functional.moe.router import RouterComputationOrder

        # softmax over all experts -> top-k -> L1 renorm -> scatter to [T, E]
        affinities32 = NF.router(
            hidden_states=x,
            router_weights=self.gate.weight.T,
            top_k=self.top_k,
            activation="softmax",
            computation_dtype=torch.float32,
            router_computation_order=RouterComputationOrder.PRENORM_LINEAR_ACT_TOPK_RENORM_SCATTER,
        )
        affinities = affinities32.to(x.dtype)
        e, h, two_i = self.experts.gate_up_proj.shape  # e = local experts
        gate_up = self.experts.gate_up_proj.view(e, h, 2, two_i // 2)
        down = self.experts.down_proj
        rank = rank_tensor(self.tp, x.device)
        fp8 = self.experts.fp8  # FP8 weights + row scales (Qwen4ExpExperts); routed by _routed_fp8
        if fp8:
            return _routed_fp8(self, x, affinities, rank, is_decode)
        if is_decode:
            weights, experts = torch.topk(affinities, self.top_k, dim=-1)
            experts = experts.to(torch.int32)  # the kernel has no int64 index support
            from vllm_neuron.utils.neuron_utils import can_run_kernel

            if not can_run_kernel(x):
                # moe_tkg has no PyTorch fallback (kernel or NKI simulator only).
                return _moe_decode_gather(x, gate_up, down, weights, experts, rank * e)
            # With EP the kernel processes this rank's experts for every token
            # (all-expert mode, rank_id selects the expert block).
            all_experts = self.tp.size > 1 or x.shape[0] * self.top_k >= self.num_experts
            return NF.moe_tkg(
                hidden_input=x,
                expert_gate_up_weights=gate_up,
                expert_down_weights=down,
                # selective mode scales by the affinity as an fp32 scalar operand
                expert_affinities=affinities if all_experts else affinities32,
                expert_index=experts,
                is_all_expert=all_experts,
                rank_id=rank.reshape(1, 1) if all_experts else None,
                expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
                activation_fn=ActFnType.SiLU,
            )
        if self.tp.size > 1:
            local = torch.arange(e, device=x.device, dtype=torch.int32) + rank * e
            affinities = NF.get_local_expert_affinities(affinities, local)
        from nkilib.core.moe.moe_cte.moe_cte import MoECTEImplementation

        masked, token_position_to_id, block_to_expert, conditions = NF.build_blockwise_mapping(
            expert_affinities=affinities,
            num_local_experts=e,
            num_experts_per_token=self.top_k,
            block_size=MOE_CTE_BLOCK_SIZE,
            moe_group=get_tp_group(),
            tp_degree=1,
        )
        from vllm_neuron.utils.neuron_utils import can_run_kernel

        if can_run_kernel(x):  # skip the compute of empty blocks (_kernels/qwen4_exp/moe_cte_dynamic.py)
            from vllm_neuron._kernels.qwen4_exp.moe_cte_dynamic import default_static_blocks, moe_cte_dynamic

            return moe_cte_dynamic(
                x, masked, gate_up, down, token_position_to_id.to(dtype=torch.int32),
                block_to_expert.to(dtype=torch.int32), conditions.to(dtype=torch.int32),
                MOE_CTE_BLOCK_SIZE, default_static_blocks(e), skip_token=True, skip_weight=False)
        return NF.moe_cte(  # CPU without the NKI simulator
            implementation=MoECTEImplementation.shard_on_block,
            conditions=conditions,
            hidden_states=x,
            expert_affinities_masked=masked,
            gate_up_proj_weight=gate_up,
            down_proj_weight=down,
            activation_function=ActFnType.SiLU,
            block_size=MOE_CTE_BLOCK_SIZE,
            token_position_to_id=token_position_to_id.to(dtype=torch.int32),
            block_to_expert=block_to_expert.to(dtype=torch.int32),
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            skip_token=True,
            skip_weight=False,
            is_tensor_update_accumulating=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Decoder layer
# ─────────────────────────────────────────────────────────────────────────────


class Qwen4ExpDecoderLayer(nn.Module):
    """``[PLE] -> attn/GDN -> MoE``, each wrapped in a gated residual.

    ``state`` is a dict holding this layer's per-request state: ``kv`` for
    full attention (or ``paged``, the layer's runner metadata, to use the paged
    KV cache), ``conv`` / ``rec`` for GDN and ``ple_conv`` when the layer owns a
    PLE module. The updated dict is returned.
    """

    def __init__(self, config: Qwen4ExpConfig, layer_idx: int, ngram_vocab_size: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == FULL_ATTENTION:
            self.self_attn = Qwen4ExpAttention(config)
        else:
            self.linear_attn = Qwen4ExpGatedDeltaNet(config)
        self.mlp = Qwen4ExpSparseMoe(config, use_kernels=config.neuron_config is not None)
        self.ple = (
            Qwen4ExpPLELayer(config, ngram_vocab_size)
            if layer_idx in config.ple_layer_indices
            else None
        )
        self.attn_hyper_connection = Qwen4ExpGatedResidual(config)
        self.mlp_hyper_connection = Qwen4ExpGatedResidual(config)

    def init_state(self, batch: int, dtype: torch.dtype, device=None) -> dict:
        state = {}
        if self.layer_type == FULL_ATTENTION:
            state["kv"] = None
        else:
            state["conv"], state["rec"] = self.linear_attn.init_state(batch, dtype, device)
        if self.ple is not None:
            state["ple_conv"] = self.ple.init_state(batch, dtype, device)
        return state

    def forward(self, streams, positions, ngram_ids, state: dict, num_valid=None, is_decode=False):
        state = dict(state)
        if self.ple is not None:
            ple_out, state["ple_conv"] = self.ple(streams, ngram_ids, state["ple_conv"], num_valid)
            streams = streams + ple_out

        x, streams, w = self.attn_hyper_connection(streams)
        if self.layer_type == FULL_ATTENTION:
            if "paged" in state:
                x = self.self_attn.forward_paged(x, positions, state["paged"], num_valid,
                                                 state.get("paged_idx"))
            else:
                x, state["kv"] = self.self_attn(x, positions, state["kv"])
        else:
            x, state["conv"], state["rec"] = self.linear_attn(
                x, state["conv"], state["rec"], num_valid
            )
        streams = inject(streams, x, w)

        x, streams, w = self.mlp_hyper_connection(streams)
        streams = inject(streams, self.mlp(x, is_decode), w)
        return streams, state


def _zero_padding(num_valid, *xs):
    """Zero the right-padding tokens of ``[B, T, ...]`` tensors (``None``: no padding).

    The GDN prefill's beta = 0 / g = 0 on padding stops the state update only for finite
    inputs: ``k * (v - k^T S) * 0`` is NaN when the padding's q / k / v are, and padding
    hidden states on device are not guaranteed finite. Defined at the end of the module
    so the call site keeps every other line (and so the decode graphs' NEFF keys) as is.
    """
    if num_valid is None:
        return xs
    b, t = xs[0].shape[:2]
    pad = torch.arange(t, device=xs[0].device).view(1, t) >= num_valid.view(b, 1)
    return tuple(torch.where(pad.view(b, t, *([1] * (x.dim() - 2))), torch.zeros_like(x), x) for x in xs)


def _paged_decode(attn: Qwen4ExpAttention, x, positions, meta, num_valid=None, idx_meta=None):
    """One-token decode of :meth:`Qwen4ExpAttention.forward_paged` (dense attention).

    The window is gathered with one ``index_select`` over the flattened block table;
    the row attends to its past positions (``<`` the current one, zeroed past that:
    never-written HBM may hold NaN) plus the current token's K/V as one extra column,
    and the cache is written last. The general path's index_put of the current
    tokens into the gathered window ran 45.6 ms per layer at B=2 on device against
    0.79 ms for this form (1.40 vs 0.86 ms at B=1; device microbenchmark).
    The strict mask makes the result independent of whether the gather sees this
    step's cache write.
    """
    b = x.shape[0]
    q, k, v, gate = attn._qkv(x, positions)
    kvh, d, bs = attn.num_kv_heads, attn.head_dim, attn.k_cache.shape[2]
    slots = meta["slot_mapping"].long()
    k_tok = k.reshape(b, kvh, d).to(attn.k_cache.dtype)
    v_tok = v.reshape(b, kvh, d).to(attn.v_cache.dtype)
    # Written before the window read (see kv_write): the window masks positions >= the current one.
    rows_kv = kv_rows(slots, attn.k_cache)
    kv_write(attn.k_cache, rows_kv, k_tok)
    kv_write(attn.v_cache, rows_kv, v_tok)
    # Unused block-table entries may be negative (-1); they are masked, so read block 0.
    table = meta["block_table_tensor"][:b].long().clamp(min=0)
    nblk = table.shape[1]

    def window(cache):
        w = cache.index_select(0, table.flatten()).view(b, nblk, kvh, bs, d)
        return w.permute(0, 2, 1, 3, 4).reshape(b, kvh, nblk * bs, d)

    kc, vc = window(attn.k_cache), window(attn.v_cache)
    s_ctx = kc.shape[2]
    past = torch.arange(s_ctx, device=x.device).view(1, 1, s_ctx, 1) < positions.view(b, 1, 1, 1)
    kc = torch.cat([torch.where(past, kc, torch.zeros_like(kc)), k_tok.view(b, kvh, 1, d)], dim=2)
    vc = torch.cat([torch.where(past, vc, torch.zeros_like(vc)), v_tok.view(b, kvh, 1, d)], dim=2)
    mask = torch.cat([past.view(b, 1, 1, s_ctx), torch.ones(b, 1, 1, 1, dtype=torch.bool, device=x.device)], -1)
    if not attn.dense:  # QSA: the selected blocks and the tail (a completed block holding the
        # current token is attended only if it is selected, as in the reference)
        cur = positions.view(b, 1).long()
        sel = _qsa_token_mask(attn, x, positions, idx_meta, cur, None)
        mask = mask & torch.cat([sel[:, :, :s_ctx], sel.gather(-1, cur.view(b, 1, 1))], -1)[:, None]
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), kc.to(q.dtype), vc.to(q.dtype), attn_mask=mask,
        scale=attn.head_dim**-0.5, enable_gqa=True,
    )
    out = out.transpose(1, 2).reshape(b, 1, -1) * torch.sigmoid(gate)
    return all_reduce(attn.o_proj(out), attn.tp)


def _paged_prefill(attn: Qwen4ExpAttention, x, positions, meta, num_valid=None, idx_meta=None):
    """Chunked prefill of :meth:`Qwen4ExpAttention.forward_paged` (dense attention).

    Like :func:`_paged_decode`: the row attends to the positions before this chunk
    (gathered from the cache with one ``index_select``, zeroed from the chunk start
    on) plus this chunk's own K/V appended, causally and up to ``num_valid``; the
    cache is written last. No index_put into the gathered window, and the result does
    not depend on whether the gather sees this step's cache writes. Assumes each row's
    valid tokens are consecutive positions starting at ``positions[:, 0]``.
    """
    b, t, _ = x.shape
    q, k, v, gate = attn._qkv(x, positions)
    kvh, d, bs = attn.num_kv_heads, attn.head_dim, attn.k_cache.shape[2]
    slots = meta["slot_mapping"].long()
    k_new = k.transpose(1, 2).to(attn.k_cache.dtype)  # [B, KVH, T, D]
    v_new = v.transpose(1, 2).to(attn.v_cache.dtype)
    # Written before the window read (see kv_write): the window keeps only positions < start.
    rows_kv = kv_rows(slots, attn.k_cache)
    kv_write(attn.k_cache, rows_kv, k.reshape(b * t, kvh, d))
    kv_write(attn.v_cache, rows_kv, v.reshape(b * t, kvh, d))
    # Unused block-table entries may be negative (-1); they are masked, so read block 0.
    table = meta["block_table_tensor"][:b].long().clamp(min=0)
    nblk = table.shape[1]

    def window(cache):
        w = cache.index_select(0, table.flatten()).view(b, nblk, kvh, bs, d)
        return w.permute(0, 2, 1, 3, 4).reshape(b, kvh, nblk * bs, d)

    kc, vc = window(attn.k_cache), window(attn.v_cache)
    s_ctx = kc.shape[2]
    start = positions[:, :1].long()  # [B, 1]
    past = torch.arange(s_ctx, device=x.device).view(1, s_ctx) < start  # [B, S]
    kc = torch.cat([torch.where(past.view(b, 1, s_ctx, 1), kc, torch.zeros_like(kc)), k_new], dim=2)
    vc = torch.cat([torch.where(past.view(b, 1, s_ctx, 1), vc, torch.zeros_like(vc)), v_new], dim=2)
    i = torch.arange(t, device=x.device)
    own = i.view(1, 1, t) <= i.view(1, t, 1)  # causal within the chunk
    if num_valid is not None:
        own = own & (i.view(1, 1, t) < num_valid.view(b, 1, 1))  # padding keys never attended
    if not attn.dense:  # QSA mask over positions -> past columns and this chunk's columns
        sel = _qsa_token_mask(attn, x, positions, idx_meta, start, num_valid)
        cols = (start + i.view(1, t)).clamp(max=s_ctx - 1)  # [B, T] position of each new key
        own = own & sel.gather(-1, cols.view(b, 1, t).expand(b, t, t))
        past = past.view(b, 1, s_ctx) & sel[:, :, :s_ctx]
    mask = torch.cat([past.view(b, -1, s_ctx).expand(b, t, s_ctx), own.expand(b, t, t)], dim=-1)
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), kc.to(q.dtype), vc.to(q.dtype), attn_mask=mask[:, None],
        scale=attn.head_dim**-0.5, enable_gqa=True,
    )
    out = out.transpose(1, 2).reshape(b, t, -1) * torch.sigmoid(gate)
    return all_reduce(attn.o_proj(out), attn.tp)


def _qsa_token_mask(attn: Qwen4ExpAttention, x, positions, idx_meta, start, num_valid):
    """QSA token mask ``[B, T, S]`` (over key positions) from the paged raw indexer keys.

    This step's raw indexer keys are written to the cache FIRST, as kv_write does for K/V (a cache write that does not
    depend on an earlier window gather of the same cache got three whole-cache copies per write from neuronx-cc; at
    long context the indexer cache is 12 layers x tokens x 256 B per rank). The window of raw keys then keeps the
    cache's keys before ``start`` (each row's first position this step), this step's keys at their positions (gathered
    from the projection, not index_put), and zeros elsewhere (never-written HBM may hold NaN), so the result does not
    depend on the order of write and read.
    """
    b, t, _ = x.shape
    iq, ik = attn.indexer.project(x, positions)
    cache = attn.idx_cache  # [blocks, 1, block, D]
    bs, dc = cache.shape[2], ik.shape[-1]
    slots = idx_meta["slot_mapping"].long()
    kv_write(cache, kv_rows(slots, cache), ik.reshape(b * t, 1, -1))
    table = idx_meta["block_table_tensor"][:b].long().clamp(min=0)
    win = cache.index_select(0, table.flatten()).view(b, table.shape[1] * bs, -1)[..., :dc]
    s_ctx = win.shape[1]
    rel = torch.arange(s_ctx, device=x.device).view(1, s_ctx) - start  # [B, S]
    n_new = num_valid.view(b, 1) if num_valid is not None else t
    new = ik.to(win.dtype).gather(1, rel.clamp(0, t - 1).unsqueeze(-1).expand(b, s_ctx, dc))
    win = torch.where((rel < 0).unsqueeze(-1), win,
                      torch.where(((rel >= 0) & (rel < n_new)).unsqueeze(-1), new, torch.zeros_like(win)))
    return attn.indexer.select(iq, win.to(ik.dtype), positions)


def kv_rows(slot_mapping: torch.Tensor, cache: torch.Tensor) -> torch.Tensor:
    """Row ids into ``cache.view(-1, D)`` for every (token, kv head), token-major; ``cache`` is
    ``[num_blocks, KVH, block_size, D]``. PAD_SLOT_ID (-1) and out-of-range slots go to slot 0 of the
    null block 0 (``cache[blk, :, off]`` sent -1 to the last row of the last block by wrap-around)."""
    nb, kvh, bs, _ = cache.shape
    s = slot_mapping.long()
    s = torch.where((s < 0) | (s >= nb * bs), torch.zeros_like(s), s)
    if kvh == 1:
        return s
    h = torch.arange(kvh, device=s.device)
    return (((s // bs).view(-1, 1) * kvh + h) * bs + (s % bs).view(-1, 1)).reshape(-1)


def kv_write(cache: torch.Tensor, rows: torch.Tensor, new: torch.Tensor) -> None:
    """In-place ``cache[blk, h, off] = new[token, h]`` as a windowed row scatter (``new`` ``[T, KVH, D]``).

    ``cache[blk, :, off] = new`` lowered to an element-wise scatter whose coordinates the tensorizer
    could not prove contiguous inside neuronx-cc's reused layer modules: 2-byte indirect DMAs from the
    second prefill module on (about 16 ms per layer per 512-token chunk, 2026-11-05 profile). A row
    scatter into ``view(-1, D)`` keeps 512-byte rows in every layer. Writing before the window gather of
    the same cache (the callers mask the positions written this step) also removes the three whole-cache
    copies per write that neuronx-cc inserted when the write was independent of an earlier read
    (CPU neuronx-cc on a synthetic repro, 2026-11-05)."""
    d = cache.shape[-1]
    cache.view(-1, d)[rows] = new.reshape(-1, d).to(cache.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Speculative decoding: the target's verify step
# ─────────────────────────────────────────────────────────────────────────────


def verify_decoder_layer(layer: Qwen4ExpDecoderLayer, streams, positions, ngram_ids, state: dict,
                         is_decode: bool = True):
    """:meth:`Qwen4ExpDecoderLayer.forward` for ``[B, 1 + k]`` verify tokens per row.

    Outputs are those of every token (each row's tokens are consecutive). The recurrent
    states are not committed here: the GDN recurrence runs one token at a time (the
    one-token decode path) and the state after each token is kept, and for the short convs
    (GDN, PLE) the window ending at each token. Returns ``(streams, state, candidates)``
    where ``candidates[key]`` is ``[B, 1 + k, ...]``; :func:`select_verified_state` picks
    the one after the last accepted token. Full-attention K/V are written for every token;
    rejected positions are overwritten later and never read (the mask is ``< position``).
    """
    state = dict(state)
    cand = {}
    if layer.ple is not None:
        ple_out, cand["ple_conv"] = _ple_verify(layer.ple, streams, ngram_ids, state["ple_conv"])
        streams = streams + ple_out
    x, streams, w = layer.attn_hyper_connection(streams)
    if layer.layer_type == FULL_ATTENTION:
        if "paged" in state:
            x = layer.self_attn.forward_paged(x, positions, state["paged"], None, state.get("paged_idx"))
        else:
            x, state["kv"] = layer.self_attn(x, positions, state["kv"])
    else:  # the 1 + k tokens in one gdn_tkg_multi call (in batch chunks above _GDN_VERIFY_MULTI_MAX_ROWS rows)
        x = _gdn_verify_multi(layer.linear_attn, x, state, cand)
    streams = inject(streams, x, w)
    x, streams, w = layer.mlp_hyper_connection(streams)
    streams = inject(streams, layer.mlp(x, is_decode), w)
    return streams, state, cand


def _ple_verify(ple: Qwen4ExpPLELayer, streams, ngram_ids, conv_state):
    """:meth:`Qwen4ExpPLELayer.forward` that returns the conv window ending at every token
    (``[B, T, C, state_len]``) instead of the last one."""
    emb = ple.ple_embedding(ngram_ids).to(streams.dtype)
    key = ple.norm_key(ple.key_proj(emb)).unflatten(-1, (ple.hc, ple.hidden))
    value = ple.value_proj(emb)
    query = ple.norm_query(streams).unflatten(-1, (ple.hc, ple.hidden))
    gate = (key * query).sum(dim=-1, keepdim=True) / math.sqrt(ple.hidden)
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    gated = (torch.sigmoid(gate) * value.unsqueeze(-2)).flatten(-2)
    x = torch.cat([conv_state, ple.norm_conv(gated).transpose(1, 2)], dim=-1)  # [B, C, L + T]
    t, n = gated.shape[1], ple.state_len
    windows = torch.stack([x[..., j + 1 : j + 1 + n] for j in range(t)], dim=1)
    return gated + F.silu(ple.conv1d(x)).transpose(1, 2), windows


def select_verified_state(candidates: dict, count: torch.Tensor) -> dict:
    """Pick, per row, the candidate state after token ``count - 1`` (``count`` ``[B]`` =
    accepted drafts + 1, i.e. the verify tokens whose state is kept)."""
    # One whole-row index_select over the flattened [B * (1 + k)] candidates: b contiguous row copies.
    # (A gather with the index expanded to the state's shape lowers to one indirect read per element:
    # B * (1 + k) * Hv * Dk * Dv for the recurrent state.)
    # Small states (the GDN conv window, [C, K] bf16 per candidate) select with a chain of
    # torch.where over the 1 + k candidates instead: their index_select lowered to one 2-byte-element
    # indirect DMA per channel run once B >= 2 (1,024 per layer, 53 ms per TP32 verify step at B=2,
    # 2026-11-05 profile), while B=1 was one DMA. where copies values exactly (no arithmetic).
    out = {}
    for key, c in candidates.items():
        b, n = c.shape[0], c.shape[1]
        sel = (count.long() - 1).clamp(min=0, max=n - 1)
        if c[0, 0].numel() <= _SELECT_WHERE_MAX_NUMEL:
            pick = sel.view(b, *([1] * (c.dim() - 2)))
            o = c[:, 0]
            for j in range(1, n):
                o = torch.where(pick == j, c[:, j], o)
            out[key] = o
            continue
        rows = torch.arange(b, device=c.device) * n + sel
        out[key] = c.reshape(b * n, *c.shape[2:]).index_select(0, rows)
    return out


_SELECT_WHERE_MAX_NUMEL = 16384  # GDN conv (1280 x 4 at TP32) uses where; rec and PLE conv keep index_select


# Prefill MoE (BF16 and FP8, on device): the moe_cte_dyn copy of nkilib's blockwise kernel runs the first
# local experts + 1 blocks (lowered by the kernel to the parity of N) in the static loop and the rest in a dynamic loop
# that stops after the last block with a token (_kernels/qwen4_exp/moe_cte_dynamic.py). The static kernel computes
# all N = 55 blocks at T=512 (about 16 carry tokens). Bit-identical to nkilib in the NKI simulator. CPU (no kernel)
# runs the NF.moe_cte fallback.


# MTP verify: the GDN recurrence of the 1 + k tokens runs in ONE gdn_tkg_multi call (recurrent state kept in SBUF
# between the tokens, activation tables loaded once: 4 ACT_TABLE_LOADs per call instead of 9 per gdn_tkg call), and
# the in-projections and the short conv run once over the tokens. Bit-identical to 1 + k gdn_tkg calls in the NKI
# simulator.
def _gdn_verify_multi_unchunked(la: Qwen4ExpGatedDeltaNet, x: torch.Tensor, state: dict, cand: dict) -> torch.Tensor:
    """GDN mixer of :func:`verify_decoder_layer` for ``x`` ``[B, T, H]`` (T = 1 + k).

    Returns the projected, all-reduced output ``[B, T, H]`` and sets ``cand["conv"]``
    ``[B, T, C, kernel]`` (the conv window ending at each token) and ``cand["rec"]``
    ``[B, T, Hv, Dk, Dv]`` fp32 (the recurrent state after each token), as the per-token loop does.
    """
    b, t, _ = x.shape
    mixed = la.in_proj_qkv(x).transpose(1, 2)  # [B, C, T]
    weight = la.conv1d.weight.squeeze(1)
    xc = torch.cat([state["conv"], mixed], dim=-1)  # [B, C, kernel + T]
    # The one-token decode conv of Qwen4ExpGatedDeltaNet._conv, for every token's window.
    wins = torch.stack([xc[..., j + 1 : j + 1 + la.kernel] for j in range(t)], dim=1)  # [B, T, C, kernel]
    conv = F.silu((wins.float() * weight.float()).sum(-1)).to(mixed.dtype)  # [B, T, C]
    q, k, v = conv.split([la.key_dim, la.key_dim, la.value_dim], dim=-1)
    rep = la.num_v_heads // la.num_k_heads
    # GQA-expand q/k to value heads (value head h reads key head h // rep), as _decode_kernel.
    q = q.reshape(b, t, la.num_k_heads, la.head_k_dim).repeat_interleave(rep, dim=2)
    k = k.reshape(b, t, la.num_k_heads, la.head_k_dim).repeat_interleave(rep, dim=2)
    args = dict(
        q=q.to(x.dtype).contiguous(),
        k=k.to(x.dtype).contiguous(),
        v=v.reshape(b, t, la.num_v_heads, la.head_v_dim).to(x.dtype).contiguous(),
        b=la.in_proj_b(x).float().contiguous(),
        a=la.in_proj_a(x).float().contiguous(),
        A_log=la.A_log.float(),
        dt_bias=la.dt_bias.float(),
        z=la.in_proj_z(x).reshape(b, t, la.num_v_heads, la.head_v_dim).to(x.dtype).contiguous(),
        norm_weight=la.norm.weight.float(),
        state_in=state["rec"].float().contiguous(),
        gate_activation=la.norm.activation,
    )
    if _kernels_available(x):
        from vllm_neuron._kernels.qwen4_exp.gdn_tkg_multi import gdn_tkg_multi

        out, recs = _nki(gdn_tkg_multi)(**args)
    else:  # CPU without the simulator: the kernel's torch reference (fp32 recurrence)
        from vllm_neuron._kernels.qwen4_exp.gdn_tkg_multi_torch import gdn_tkg_multi_torch

        out, recs = gdn_tkg_multi_torch(**args)
    cand["conv"], cand["rec"] = wins, recs
    out = out.to(x.dtype).reshape(b, t, la.value_dim)
    return all_reduce(la.out_proj(out), la.tp)


# Fused decode MoE block: the decode MoE of up to _MOE_BLOCK_FUSED_MAX_T
# rows (4: the B=1 MTP verify step, 1 + k tokens, and the MTP draft's own MoE) runs as ONE kernel per
# layer before the all-reduce (_kernels/qwen4_exp/moe_block_tkg.py): router + top-k + renorm + this rank's
# routed experts (a dynamic loop over the routed local experts only, the next expert's weights prefetched) + the
# sharded shared expert and its sigmoid gate. Replaces five launches per layer (torch router, cast + top-k,
# routed-expert moe_tkg, shared expert, shared gate). For the same affinities the routed sum is bit-identical to
# nkilib's all-expert moe_tkg in the NKI simulator; the router (fp32 dot in another summation order) and the shared
# expert (fp32 until the end instead of bf16 per linear) agree with the current path within bf16 tolerance
# (2026-11-06). Needs the router / shared-expert weights in the kernel layout:
# three extra buffers per MoE layer filled after load_state_dict (+2.9 MB per layer per rank). Installed by
# patching Qwen4ExpSparseMoe below; larger decode batches, prefill and CPU (no kernel) take the original forward.
_MOE_BLOCK_FUSED_MAX_T = 4


def _moe_block_fill(moe: "Qwen4ExpSparseMoe", *_args) -> None:
    """load_state_dict post-hook: the kernel-layout copies of the router and shared-expert weights."""
    w = moe.gate.weight
    if w.is_meta:  # not loaded (meta-device construction, partial loads)
        return
    with torch.no_grad():
        moe._mbf_router_wt = w.t().contiguous()  # [H, E]
        se = moe.shared_expert
        moe._mbf_shared_gu = torch.cat(  # [H, 2 * I_s + 1]: gate | up | shared-expert gate
            [se.gate_proj.weight.t(), se.up_proj.weight.t(), moe.shared_expert_gate.weight.t()], dim=1).contiguous()
        moe._mbf_shared_down = se.down_proj.weight.t().contiguous()  # [I_s, H]


def _moe_block_init(self, config, use_kernels=False):
    _moe_block_orig_init(self, config, use_kernels)
    if not use_kernels:
        return
    w = self.gate.weight
    e, h = w.shape
    i_s = self.shared_expert.down_proj.weight.shape[1]
    # non-persistent: not in state_dict / named_parameters (the loaders' completeness checks), moved by .to()
    self.register_buffer("_mbf_router_wt", torch.empty(h, e, dtype=w.dtype, device=w.device), persistent=False)
    self.register_buffer("_mbf_shared_gu", torch.empty(h, 2 * i_s + 1, dtype=w.dtype, device=w.device), persistent=False)
    self.register_buffer("_mbf_shared_down", torch.empty(i_s, h, dtype=w.dtype, device=w.device), persistent=False)
    self.register_load_state_dict_post_hook(_moe_block_fill)


def _moe_block_forward(self, x: torch.Tensor, is_decode: bool = False) -> torch.Tensor:
    shape = x.shape
    t = x.numel() // shape[-1]
    if not (self.use_kernels and is_decode and 0 < t <= _MOE_BLOCK_FUSED_MAX_T and self.tp.size > 1
            and _kernels_available(x)):
        return _moe_block_orig_forward(self, x, is_decode)
    from vllm_neuron._kernels.qwen4_exp.moe_block_tkg import moe_block_tkg

    x2 = x.reshape(t, shape[-1])
    e, h, two_i = self.experts.gate_up_proj.shape
    rank = rank_tensor(self.tp, x.device).reshape(1, 1)
    # The shared expert joins the kernel only when it is split over the TP group (its partial sums then
    # belong to the same all-reduce); a replicated one is added after it, as in the original forward.
    out = _nki(moe_block_tkg)(
        x=x2, router_wt=self._mbf_router_wt, gate_up=self.experts.gate_up_proj.view(e, h, 2, two_i // 2),
        down=self.experts.down_proj, shared_gu=self._mbf_shared_gu, shared_down=self._mbf_shared_down,
        rank_id=rank, top_k=self.top_k, with_shared=bool(self.shard_shared))
    out = all_reduce(out, self.tp)
    if not self.shard_shared:
        out = out + torch.sigmoid(self.shared_expert_gate(x2)) * self.shared_expert(x2)
    return out.reshape(shape)


_moe_block_orig_init = Qwen4ExpSparseMoe.__init__
_moe_block_orig_forward = Qwen4ExpSparseMoe.forward
Qwen4ExpSparseMoe.__init__ = _moe_block_init
Qwen4ExpSparseMoe.forward = _moe_block_forward


# ── FP8 routed experts (FP8 checkpoint; Qwen4ExpConfig.expert_fp8) ───────────────────────────
# Weights: trn2 FP8 e4m3 (IEEE-style, max 240) stored as torch.float8_e4m3fn (Lite passes the bits
# through; codes |q| < 256 are the same value in both formats), one fp32 scale per output channel,
# requantized at load from the checkpoint's 128x128 block scales (model_bf16._load_per_expert).
# Kernels: prefill moe_cte_dyn FP8 x FP8 double_row with per-token activation scales; decode
# nkilib's all-expert moe_tkg in its FP8 ROW format (bf16 activations x FP8 weights, dequant after
# the matmul; _kernels/qwen4_exp/moe_tkg_fp8.py).
# Without a kernel (CPU mode) the weights are dequantized (W = q * s, rounded to the activation
# dtype) and the BF16 fallbacks run. ⚠ NKI edits do not change the NEFF key: new cache tag.
FP8_E4M3_TRN2_MAX = 240.0


def fp8_row_quantize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``w`` ``[out, in]`` (fp32) -> (``q`` float8_e4m3fn ``[out, in]`` with ``|q| <= 240``,
    fp32 scale ``[out]``) such that ``w ~= q * scale[:, None]``. All-zero rows get scale 1."""
    w = w.float()
    amax = w.abs().amax(dim=1)
    scale = torch.where(amax > 0, amax / FP8_E4M3_TRN2_MAX, torch.ones_like(amax))
    q = (w / scale[:, None]).clamp(-FP8_E4M3_TRN2_MAX, FP8_E4M3_TRN2_MAX).to(torch.float8_e4m3fn)
    return q, scale


def fp8_row_dequant(q: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``q`` (kernel layout, last dim = output channel; ``gate_up`` may be viewed ``[E, H, 2, I]``)
    times its row scale ``[E, 1, C]``, in ``dtype``."""
    s = scale.view(*scale.shape[:-1], *q.shape[scale.dim() - 1:]) if q.dim() > scale.dim() else scale
    return (q.to(torch.float32) * s).to(dtype)


def _routed_fp8(moe: "Qwen4ExpSparseMoe", x: torch.Tensor, affinities: torch.Tensor, rank: torch.Tensor,
                is_decode: bool) -> torch.Tensor:
    """``Qwen4ExpSparseMoe._routed_kernels`` for FP8 experts: this rank's routed-expert sum ``[T, H]``."""
    import vllm_neuron.functional as NF
    from vllm.distributed.parallel_state import get_tp_group

    from vllm_neuron.utils.neuron_utils import can_run_kernel

    ex = moe.experts
    e, h, two_i = ex.gate_up_proj.shape
    i = two_i // 2
    gate_up = ex.gate_up_proj.view(e, h, 2, i)
    down = ex.down_proj
    kernel = can_run_kernel(x)
    if is_decode:
        weights, experts = torch.topk(affinities, moe.top_k, dim=-1)
        experts = experts.to(torch.int32)
        if not kernel:
            return _moe_decode_gather(x, fp8_row_dequant(gate_up, ex.gate_up_scale, x.dtype),
                                      fp8_row_dequant(down, ex.down_scale, x.dtype), weights, experts, rank * e)
        from vllm_neuron._kernels.qwen4_exp.moe_tkg_fp8 import moe_tkg_all_expert_fp8

        # Always all-expert mode (rank_id selects this rank's block; TP=1 is rank 0 of all experts).
        return _nki(moe_tkg_all_expert_fp8)(
            hidden_input=x, expert_gate_up_weights=gate_up, expert_down_weights=down, expert_affinities=affinities,
            expert_index=experts, rank_id=rank.reshape(1, 1),
            expert_gate_up_weights_scale=ex.gate_up_scale.view(e, 2, i), expert_down_weights_scale=ex.down_scale.view(e, h))
    if moe.tp.size > 1:
        local = torch.arange(e, device=x.device, dtype=torch.int32) + rank * e
        affinities = NF.get_local_expert_affinities(affinities, local)
    masked, token_position_to_id, block_to_expert, conditions = NF.build_blockwise_mapping(
        expert_affinities=affinities,
        num_local_experts=e,
        num_experts_per_token=moe.top_k,
        block_size=MOE_CTE_BLOCK_SIZE,
        moe_group=get_tp_group(),
        tp_degree=1,
    )
    if kernel:
        from vllm_neuron._kernels.qwen4_exp.moe_cte_dynamic import default_static_blocks, moe_cte_dynamic

        # The FP8 path exists in the moe_cte_dyn copy only (skip_weight is not supported there).
        return moe_cte_dynamic(
            x, masked, gate_up, down, token_position_to_id.to(dtype=torch.int32),
            block_to_expert.to(dtype=torch.int32), conditions.to(dtype=torch.int32),
            MOE_CTE_BLOCK_SIZE, default_static_blocks(e), skip_token=True, skip_weight=False,
            gate_up_proj_scale=ex.gate_up_scale, down_proj_scale=ex.down_scale)
    from nkilib.core.moe.moe_cte.moe_cte import MoECTEImplementation
    from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode

    return NF.moe_cte(
        implementation=MoECTEImplementation.shard_on_block,
        conditions=conditions,
        hidden_states=x,
        expert_affinities_masked=masked,
        gate_up_proj_weight=fp8_row_dequant(gate_up, ex.gate_up_scale, x.dtype),
        down_proj_weight=fp8_row_dequant(down, ex.down_scale, x.dtype),
        activation_function=ActFnType.SiLU,
        block_size=MOE_CTE_BLOCK_SIZE,
        token_position_to_id=token_position_to_id.to(dtype=torch.int32),
        block_to_expert=block_to_expert.to(dtype=torch.int32),
        expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
        skip_token=True,
        skip_weight=False,
        is_tensor_update_accumulating=True,
    )


# ── FP8 PLE n-gram table (FP8 checkpoint; a BF16 checkpoint keeps the BF16 table) ─────────────────────────
# The table stays FP8 on the device (1 B per element instead of 2: 3.20 -> 1.60 GB per rank at TP32), held as the
# checkpoint's e4m3fn bytes in a uint8 parameter, with its single per-tensor scale as a 0-d non-persistent buffer
# ``ngram_scale`` in the model dtype (filled by the loader, model_bf16). The lookup gathers byte rows and decodes
# them after the gather (parallel.sharded_lookup_dequant: exact e4m3fn value x scale, the values and the op
# model_bf16._ple_rows applies to the whole table of a BF16 PLE), so the output is bit-identical to the
# dequantized-table path. Not an FP8 dtype on purpose: the real table has codes 256..448 (73 of 4.4e9 in 11
# scanned shards, ~10 per shard; 2026-11-06 scan), which a trn2 e4m3 read turns into inf and FP8
# copies may rewrite (bytes 0x78..0x7F / 0xF8..0xFF); bytes are neither converted nor normalized. Selected at
# construction from the checkpoint's quantization (Qwen4ExpConfig.fp8_block_size); a BF16 checkpoint keeps the
# BF16 table.


def _ple_fp8_init(self, config: Qwen4ExpConfig, padded_vocab_size: int):
    _ple_fp8_orig_init(self, config, padded_vocab_size)
    if config.fp8_block_size is None:  # BF16 checkpoint: BF16 table
        return
    w = self.ngram_embedding.weight
    self.ngram_embedding.weight = nn.Parameter(  # e4m3fn bytes
        torch.empty(w.shape, dtype=torch.uint8, device=w.device), requires_grad=False)
    # NaN until the loader sets it, so a table loaded without its scale cannot pass silently.
    self.register_buffer("ngram_scale", torch.full((), float("nan"), dtype=config.torch_dtype, device=w.device),
                         persistent=False)


def _ple_fp8_forward(self, ngram_ids: torch.Tensor) -> torch.Tensor:
    if "ngram_scale" not in self._buffers:
        return _ple_fp8_orig_forward(self, ngram_ids)
    from .parallel import sharded_lookup_dequant

    # .to(dtype) / .float() of the module (the references do that) leave the uint8 table alone.
    return sharded_lookup_dequant(self.ngram_embedding.weight, self.ngram_scale, ngram_ids, self.tp).flatten(-2)


_ple_fp8_orig_init = Qwen4ExpNGramEmbedding.__init__
_ple_fp8_orig_forward = Qwen4ExpNGramEmbedding.forward


Qwen4ExpNGramEmbedding.__init__ = _ple_fp8_init
Qwen4ExpNGramEmbedding.forward = _ple_fp8_forward


# Fused hyper-connection read on prefill: for more than hc_fused.TKG_MAX_N (32) rows the hyper-connection read
# (Qwen4ExpGatedResidual.forward: grouped hc_norm, low-rank read gate, mean over the streams, inject weights) runs as
# ONE NKI kernel, hc_cte (_kernels/qwen4_exp/hc_fused.py; tokens split over the pcores, 512-token blocks), instead of
# the XLA ops on fp32 [T, hc * H]. Decode / MTP verify (<= 32 rows), CPU (no kernel) and non-bf16 runs keep the torch
# forward on the original weights. Same math and the same bf16 rounding points as the torch path (NKI simulator vs
# this forward on CPU, random and real weights: 99.6-99.9 % bit-identical at T = 33..2048, every difference <= 1 bf16
# ulp of max |out|). The kernel reads Wd^T [D, R] / Wup^T [R, D] (+ tiny norm / inject copies), built once by a
# load_state_dict post-hook; the original weights are kept for the torch forward (+13.1 MB per hyper-connection per
# rank, ~100 of them with MTP). Installed by patching Qwen4ExpGatedResidual below.
_HC_KL_KEYS = ("wd_t", "wup_t", "w1_cte", "inj_cte")


def _hc_fill(mod: "Qwen4ExpGatedResidual", *_args) -> None:
    """load_state_dict post-hook: the kernel-layout weights."""
    wd = mod.input_mix_weight_down.weight
    if wd.is_meta or wd.numel() == 0:  # not loaded (meta construction, partial loads)
        return
    from vllm_neuron._kernels.qwen4_exp.hc_fused import kernel_layout

    inj = mod.block_inject_weight.weight if mod.block_inject_weight is not None else None
    with torch.no_grad():
        kl = kernel_layout(mod.hc_norm.weight, wd, mod.input_mix_weight_up.weight, inj, mod.hc)
        for k in _HC_KL_KEYS:
            setattr(mod, "_hcf_" + k, kl[k].to(wd.dtype))
    mod._hcf_ready = True


def _hc_init(self, config, use_combine=True):
    _hc_orig_init(self, config, use_combine)
    self._hcf_ready = False
    w = self.input_mix_weight_down.weight
    r, d = w.shape
    hc, s_cte = self.hc, self.hidden // 128
    shapes = dict(wd_t=(d, r), wup_t=(r, d), w1_cte=(128, hc, s_cte), inj_cte=(128, hc, s_cte, hc))
    for k, shape in shapes.items():
        # non-persistent: not in state_dict / named_parameters (the loaders' completeness checks), moved by .to();
        # all in the weights' dtype (w1 = the raw hc_norm weight, the kernels add 1 in fp32), so a later
        # model-wide .to(dtype) cannot round anything the torch path keeps in fp32.
        self.register_buffer("_hcf_" + k, torch.empty(shape, dtype=w.dtype, device=w.device), persistent=False)
    self.register_load_state_dict_post_hook(_hc_fill)


def _hc_forward(self, streams: torch.Tensor):
    shape = streams.shape
    d = shape[-1]
    n = streams.numel() // d
    from vllm_neuron._kernels.qwen4_exp.hc_fused import TKG_MAX_N, hc_cte

    if not (self._hcf_ready and n > TKG_MAX_N and streams.dtype == torch.bfloat16 and _kernels_available(streams)):
        return _hc_orig_forward(self, streams)
    x2 = streams.reshape(n, d)
    mixed, w = _nki(hc_cte)(x=x2, wd_t=self._hcf_wd_t, wup_t=self._hcf_wup_t, w1=self._hcf_w1_cte,
                            inj=self._hcf_inj_cte, eps=self.hc_norm.eps, hc=self.hc)
    mixed = mixed.reshape(*shape[:-1], self.hidden)
    if self.block_inject_weight is None:
        return mixed
    return mixed, streams, w.reshape(*shape[:-1], self.hc)


_hc_orig_init = Qwen4ExpGatedResidual.__init__
_hc_orig_forward = Qwen4ExpGatedResidual.forward
Qwen4ExpGatedResidual.__init__ = _hc_init
Qwen4ExpGatedResidual.forward = _hc_forward


# Fused decode MoE block with FP8 routed experts (FP8 checkpoint): FP8 MoE layers run moe_block_tkg_fp8
# (_kernels/qwen4_exp/moe_block_tkg.py, end of module) -- the fused router + top-k + routed experts + shared expert with
# the FP8 e4m3 codes and their per-output-channel fp32 scales, applied where nkilib moe_tkg ROW applies them (gate/up: at
# the PSUM drain before the LNC sum; down: at the PSUM drain, into bf16). For the same affinities the routed sum is
# bit-identical to nkilib's all-expert moe_tkg FP8 ROW (_routed_fp8) in the NKI simulator. Buffers: the BF16 path's
# router / shared-expert copies (_moe_block_init) + the gate/up scales transposed to [E_L, 128, 2, I / 128] (+80 KB per
# layer per rank at I = 640); the down scales are read in place. Selected per layer from the checkpoint
# (Qwen4ExpConfig.expert_fp8): BF16 layers take _moe_block_init. An FP8 intermediate that is not a multiple of 128 keeps
# the original FP8 module (moe_block_tkg_fp8 needs 128-row tiles; the BF16 kernel would multiply the e4m3fn codes
# unscaled).
import types as _types  # noqa: E402


def _moe_block_fill_fp8(moe: "Qwen4ExpSparseMoe", *_args) -> None:
    """load_state_dict post-hook: gate/up row scales [E_L, 1, 2I] -> [E_L, 128, 2, I / 128] ([e, i0, g, i1])."""
    s = moe.experts.gate_up_scale
    if s.is_meta:
        return
    e, _, two_i = s.shape
    with torch.no_grad():
        moe._mbf_gu_scale_t = s.view(e, 2, two_i // 256, 128).permute(0, 3, 1, 2).contiguous()


def _moe_block_init_fp8(self, config, use_kernels=False):
    if not (use_kernels and config.expert_fp8):  # BF16 layers (and the reference path)
        _moe_block_init(self, config, use_kernels)
        return
    if config.moe_intermediate_size % 128:  # the original FP8 module, forward bound per instance
        _moe_block_orig_init(self, config, use_kernels)
        self.forward = _types.MethodType(_moe_block_orig_forward, self)
        return
    _moe_block_init(self, config, use_kernels)  # router / shared-expert kernel-layout buffers + their post-hook
    s = self.experts.gate_up_scale
    e, _, two_i = s.shape
    self.register_buffer("_mbf_gu_scale_t", torch.empty(e, 128, 2, two_i // 256, dtype=torch.float32, device=s.device),
                         persistent=False)
    self.register_load_state_dict_post_hook(_moe_block_fill_fp8)
    self.forward = _types.MethodType(_moe_block_forward_fp8, self)


def _moe_block_forward_fp8(self, x: torch.Tensor, is_decode: bool = False) -> torch.Tensor:
    shape = x.shape
    t = x.numel() // shape[-1]
    if not (self.use_kernels and is_decode and 0 < t <= _MOE_BLOCK_FUSED_MAX_T and self.tp.size > 1
            and _kernels_available(x)):
        return _moe_block_orig_forward(self, x, is_decode)
    from vllm_neuron._kernels.qwen4_exp.moe_block_tkg import moe_block_tkg_fp8

    ex = self.experts
    x2 = x.reshape(t, shape[-1])
    e, h, two_i = ex.gate_up_proj.shape
    rank = rank_tensor(self.tp, x.device).reshape(1, 1)
    out = _nki(moe_block_tkg_fp8)(
        x=x2, router_wt=self._mbf_router_wt, gate_up=ex.gate_up_proj.view(e, h, 2, two_i // 2), down=ex.down_proj,
        gate_up_scale_t=self._mbf_gu_scale_t, down_scale=ex.down_scale.view(e, h), shared_gu=self._mbf_shared_gu,
        shared_down=self._mbf_shared_down, rank_id=rank, top_k=self.top_k, with_shared=bool(self.shard_shared))
    out = all_reduce(out, self.tp)
    if not self.shard_shared:
        out = out + torch.sigmoid(self.shared_expert_gate(x2)) * self.shared_expert(x2)
    return out.reshape(shape)


Qwen4ExpSparseMoe.__init__ = _moe_block_init_fp8


# Fused decode attention projections: for up to _ATTN_TKG_FUSED_MAX_N rows (B * T; 32: decode and MTP verify) the full-attention projections run as two NKI kernels (_kernels/qwen4_exp/attn_tkg.py):
# attn_qkv_tkg = q|k|v projection + q/k RMSNorm + partial rope in one launch, attn_out_tkg = o_proj. They read
# pre-transposed [in, out] copies of the weights with the rows stationary; the XLA lowering transposed every
# 128 x 128 weight tile on the PE each step (~3,100 of ~4,200 static instructions of a T=3 verify mixer, CPU
# neuronx-cc, 2026-11-06). The copies are built by a load_state_dict post-hook
# (+14.4 MB per full-attention layer per rank at attention group 8; the originals stay for prefill). Needs one KV
# head per rank and head_dim 256 (else the torch path; also CPU without the NKI simulator). Installed by patching
# Qwen4ExpAttention below.
_ATTN_TKG_FUSED_MAX_N = 32  # <= attn_tkg.MAX_N


def _atf_fill(mod: "Qwen4ExpAttention", *_args) -> None:
    """load_state_dict post-hook: [q; k; v]^T and o_proj^T."""
    w = mod.q_proj.weight
    if w.is_meta or w.numel() == 0:
        return
    from vllm_neuron._kernels.qwen4_exp.attn_tkg import kernel_layout

    with torch.no_grad():
        kl = kernel_layout(mod.q_proj.weight, mod.k_proj.weight, mod.v_proj.weight, mod.o_proj.weight)
        mod._atf_wqkv_t = kl["wqkv_t"].to(w.dtype)
        mod._atf_wo_t = kl["wo_t"].to(w.dtype)
    mod._atf_ready = True


def _atf_init(self, config):
    _atf_orig_init(self, config)
    self._atf_ready = False
    w = self.q_proj.weight
    h, d = config.hidden_size, self.head_dim
    nq = self.num_heads * d
    self.register_buffer("_atf_wqkv_t", torch.empty(h, 2 * nq + 2 * self.num_kv_heads * d, dtype=w.dtype,
                                                    device=w.device), persistent=False)
    self.register_buffer("_atf_wo_t", torch.empty(nq, h, dtype=w.dtype, device=w.device), persistent=False)
    self.register_load_state_dict_post_hook(_atf_fill)
    self.o_proj.forward = _types.MethodType(_atf_o_proj_forward(self), self.o_proj)


def _atf_ok(attn: "Qwen4ExpAttention", x2: torch.Tensor) -> bool:
    from vllm_neuron._kernels.qwen4_exp.attn_tkg import MAX_N

    n = x2.shape[0]
    return bool(attn._atf_ready and 0 < n <= min(_ATTN_TKG_FUSED_MAX_N, MAX_N) and x2.dtype == torch.bfloat16
                and attn.num_kv_heads == 1 and attn.head_dim == 256 and _kernels_available(x2))


def _atf_qkv(self, x: torch.Tensor, positions: torch.Tensor):
    b, t, _ = x.shape
    x2 = x.reshape(b * t, x.shape[-1])
    if not _atf_ok(self, x2):
        return _atf_orig_qkv(self, x, positions)
    from vllm_neuron._kernels.qwen4_exp.attn_tkg import attn_qkv_tkg

    d = self.head_dim
    cos, sin = rope_cos_sin(self.config, positions, x.dtype)
    r = cos.shape[-1]
    out = _nki(attn_qkv_tkg)(x=x2, w_t=self._atf_wqkv_t, cos=cos.reshape(b * t, r), sin=sin.reshape(b * t, r),
                              q_w=self.q_norm.weight, k_w=self.k_norm.weight, eps=self.q_norm.eps, d=d)
    nq = self.num_heads * 2 * d
    qg = out[:, :nq].reshape(b, t, self.num_heads, 2 * d)
    k = out[:, nq : nq + d].reshape(b, t, 1, d)
    v = out[:, nq + d :].reshape(b, t, 1, d)
    return qg[..., :d], k, v, qg[..., d:].reshape(b, t, -1)


def _atf_o_proj_forward(attn: "Qwen4ExpAttention"):
    def forward(lin, inp: torch.Tensor) -> torch.Tensor:
        x2 = inp.reshape(-1, inp.shape[-1])
        if not _atf_ok(attn, x2):
            return F.linear(inp, lin.weight)
        from vllm_neuron._kernels.qwen4_exp.attn_tkg import attn_out_tkg

        return _nki(attn_out_tkg)(a=x2, wo_t=attn._atf_wo_t).reshape(*inp.shape[:-1], attn._atf_wo_t.shape[1])

    return forward


_atf_orig_init = Qwen4ExpAttention.__init__
_atf_orig_qkv = Qwen4ExpAttention._qkv
Qwen4ExpAttention.__init__ = _atf_init
Qwen4ExpAttention._qkv = _atf_qkv


# [in, out] weight copies: decode / verify projections read a copy of the weight stored [in, out]
# (x @ W_t) instead of F.linear on the [out, in] parameter. With the [out, in] layout neuronx-cc transposes every
# 128 x 128 tile of the weight on the PE in every step (CPU neuronx-cc counts of one full-attention verify mixer,
# 2026-11-06: 1,701 PE transposes on graph-input weights, 73 % of its instructions in the projections; [in, out]
# removed 1,660 of them). Covers the GDN in-projections and out_proj, the hyper-connection read / inject linears, the MoE
# router + shared expert + its gate and the PLE key / value projections (this module) and lm_head (model_bf16).
# Rows <= _WEIGHTS_IN_OUT_MAX_T (32: decode, MTP verify, MTP draft) take the copy;
# prefill keeps F.linear on the original, so the prefill graphs are unchanged. The copies are non-persistent
# buffers filled by a load_state_dict post-hook (not in state_dict / named_parameters; moved by .to()):
# +14.5 MB per GDN layer per rank at TP32 / attention group 8; hyper-connections +13.1 MB each (97 of them; the
# fused hyper-connection kernel keeps its own copies), MoE +2.9 MB per layer, PLE +65.5 MB (one layer). Installed by
# patching __init__ below.
_WEIGHTS_IN_OUT_MAX_T = 32


def _wio_fill(lin: nn.Linear, *_args) -> None:
    """load_state_dict post-hook of a wrapped Linear: the [in, out] copy of its weight."""
    w = lin.weight
    if w.is_meta or w.numel() == 0:
        return
    with torch.no_grad():
        wt = w.t().contiguous()
        if wt.data_ptr() == w.data_ptr():  # [1, in] (or [out, 1]): .t() is already contiguous, a view of w
            wt = wt.clone()
        lin._wio_t = wt
    lin._wio_ready = True


def _wio_forward(lin: nn.Linear, inp: torch.Tensor) -> torch.Tensor:
    rows = inp.numel() // inp.shape[-1]
    if lin._wio_ready and 0 < rows <= _WEIGHTS_IN_OUT_MAX_T:
        return torch.matmul(inp, lin._wio_t)
    return F.linear(inp, lin.weight, lin.bias)


def wio_wrap(lin: nn.Linear) -> nn.Linear:
    """Give ``lin`` an [in, out] weight copy for decode / verify rows (bias-free Linears only)."""
    assert lin.bias is None
    w = lin.weight
    lin._wio_ready = False
    lin.register_buffer("_wio_t", torch.empty(w.shape[1], w.shape[0], dtype=w.dtype, device=w.device),
                        persistent=False)
    lin.register_load_state_dict_post_hook(_wio_fill)
    lin.forward = _types.MethodType(_wio_forward, lin)
    return lin


def _wio_gdn_init(self, config):
    _wio_gdn_orig_init(self, config)
    for lin in (self.in_proj_qkv, self.in_proj_z, self.in_proj_b, self.in_proj_a, self.out_proj):
        wio_wrap(lin)


def _wio_hc_init(self, config, use_combine=True):
    _wio_hc_orig_init(self, config, use_combine)
    for lin in (self.input_mix_weight_down, self.input_mix_weight_up, self.block_inject_weight):
        if lin is not None:
            wio_wrap(lin)


def _wio_ple_init(self, config, padded_vocab_size):
    _wio_ple_orig_init(self, config, padded_vocab_size)
    wio_wrap(self.key_proj)
    wio_wrap(self.value_proj)


_WIO_ROUTER = None  # the Qwen4ExpRouter whose [H, E] copy NF.router reads, while _routed_kernels runs


def _wio_moe_init(self, config, use_kernels=False):
    _wio_moe_orig_init(self, config, use_kernels)
    for lin in (self.shared_expert.gate_proj, self.shared_expert.up_proj, self.shared_expert.down_proj,
                self.shared_expert_gate):
        wio_wrap(lin)
    g = self.gate  # not a Linear: only its weight copy; NF.router gets it through _wio_nf_router
    g._wio_ready = False
    g.register_buffer("_wio_t", torch.empty(g.weight.shape[1], g.weight.shape[0], dtype=g.weight.dtype,
                                            device=g.weight.device), persistent=False)
    g.register_load_state_dict_post_hook(_wio_fill)


def _wio_routed_kernels(self, x, is_decode):
    global _WIO_ROUTER
    _WIO_ROUTER = self.gate if self.gate._wio_ready and 0 < x.shape[0] <= _WEIGHTS_IN_OUT_MAX_T else None
    try:
        return _wio_orig_routed_kernels(self, x, is_decode)
    finally:
        _WIO_ROUTER = None


def _wio_nf_router(*args, **kwargs):
    if _WIO_ROUTER is not None:  # router_weights is gate.weight.T ([H, E]); the copy holds the same values
        kwargs["router_weights"] = _WIO_ROUTER._wio_t
    return _wio_orig_nf_router(*args, **kwargs)


_wio_gdn_orig_init = Qwen4ExpGatedDeltaNet.__init__
_wio_hc_orig_init = Qwen4ExpGatedResidual.__init__
_wio_ple_orig_init = Qwen4ExpPLELayer.__init__
_wio_moe_orig_init = Qwen4ExpSparseMoe.__init__
_wio_orig_routed_kernels = Qwen4ExpSparseMoe._routed_kernels
Qwen4ExpGatedDeltaNet.__init__ = _wio_gdn_init
Qwen4ExpGatedResidual.__init__ = _wio_hc_init
Qwen4ExpPLELayer.__init__ = _wio_ple_init
import vllm_neuron.functional as _wio_nf  # noqa: E402

_wio_orig_nf_router = _wio_nf.router
_wio_nf.router = _wio_nf_router
Qwen4ExpSparseMoe.__init__ = _wio_moe_init
Qwen4ExpSparseMoe._routed_kernels = _wio_routed_kernels
# GDN prefill mask pack on the device: the gdn_cte (Schur) mask pack [10, 128, 128] fp32 is a device buffer of each
# GDN layer (a graph input like a weight; the fp32 bits held as int32) instead of being rebuilt by device instructions
# in every prefill.
# ``_gdn_mask_pack`` is lru_cache'd, but dynamo ignores the cache wrapper ("Dynamo detected a call to a
# functools.lru_cache-wrapped function ... directly traces the wrapped function") and traces nkilib's numpy
# ``build_mask_pack`` as torch ops: 148 FX ops per GDN layer (127 slice writes at gdn_cte_utils.py:76); AOT CSE keeps
# one copy, ~1,760 StableHLO ops (pad / compare / and / select per slice write), which neuronx-cc runs on device in
# the pre-layer segment on pcore 0 only: 6,334 instructions, 2.9 ms union (device prefill profile), the likely source
# of the 15,845 32-channel STREAM_TRANSPOSEs next to them (2.6 ms; no source location), before the embed all-reduce.
# The pack is built once on the host (the same array, so outputs are bit-identical), filled into a non-persistent
# buffer after load_state_dict (meta construction) or at construction (real device), and moved by .to(). The
# original forward picks it up through the module global ``_gdn_mask_pack`` (rebound below to read a slot the
# patched forward sets).
# (torch.compiler.assume_constant_result is not enough: dynamo makes the result a CPU buffer of the compiled graph,
# which AOT lifts to a graph INPUT that lives on the host.) +655,360 B of HBM per GDN layer per rank.
_gdn_mask_pack_orig = _gdn_mask_pack


def _gdn_mask_pack_host() -> torch.Tensor:
    """The pack on the host CPU, never from the lru_cache: model construction runs under ``torch.device("meta")``,
    so the first ``_gdn_mask_pack()`` call there cached a META tensor (``torch.as_tensor`` follows the default device)
    and the fill failed with "Cannot copy out of meta tensor" (device run, 11-06). ``from_numpy`` is always CPU."""
    from nkilib.experimental.gdn import build_mask_pack

    return torch.from_numpy(build_mask_pack())
_GDN_HOIST_SLOT: list = [None]  # this layer's device pack while its forward runs


def _gdn_hoist_fill(mod: "Qwen4ExpGatedDeltaNet", *_args) -> None:
    """load_state_dict post-hook (and real-device construction): the host-built pack into the buffer."""
    with torch.no_grad():
        mod._gdn_mask_pack_dev = _gdn_mask_pack_host().to(torch.float32).to(mod.A_log.device).clone()


def _gdn_hoist_init(self, config: Qwen4ExpConfig):
    _gdn_hoist_orig_init(self, config)
    host = _gdn_mask_pack_host()
    # non-persistent: not in state_dict / named_parameters (the loaders' completeness checks), moved by .to().
    # fp32, NOT int32 bits: an in-graph int32 -> fp32 bitcast_convert of the graph input made neuronx-cc insert an IO
    # transpose with src int32 / dst fp32 and abort (NCC_IDDT901, every prefill graph, device runs of 11-06).
    # The runner does not cast the whole model (fp32 FP8 scales stay fp32 on device). Was: int32 so that a model-wide
    # .to(dtype) (harnesses cast whole modules to bf16) cannot
    # round it: -1e9 in bf16 is not the value the kernel / reference validate. On the meta device it stays meta until
    # the post-hook fills it (an unfilled one fails loudly in .to(device)).
    self.register_buffer("_gdn_mask_pack_dev", torch.empty(host.shape, dtype=torch.float32, device=self.A_log.device),
                         persistent=False)
    if not self.A_log.is_meta:
        _gdn_hoist_fill(self)
    self.register_load_state_dict_post_hook(_gdn_hoist_fill)


def _gdn_hoist_forward(self, *args, **kwargs):
    _GDN_HOIST_SLOT[0] = self._gdn_mask_pack_dev
    out = _gdn_hoist_orig_forward(self, *args, **kwargs)
    _GDN_HOIST_SLOT[0] = None
    return out


def _gdn_mask_pack_hoisted() -> torch.Tensor:
    pack = _GDN_HOIST_SLOT[0]
    if pack is None:  # called outside a patched forward
        return _gdn_mask_pack_orig()
    return pack  # fp32 already (no in-graph bitcast; see _gdn_hoist_init)


_gdn_hoist_orig_init = Qwen4ExpGatedDeltaNet.__init__
_gdn_hoist_orig_forward = Qwen4ExpGatedDeltaNet.forward
Qwen4ExpGatedDeltaNet.__init__ = _gdn_hoist_init
Qwen4ExpGatedDeltaNet.forward = _gdn_hoist_forward
_gdn_mask_pack = _gdn_mask_pack_hoisted


# _GDN_VERIFY_MULTI_MAX_ROWS: gdn_tkg_multi keeps every (token, head) block of
# V / K on SBUF partition 0 (2,048 B x 3 * B * T per core at TP32 / attention group 8 / LNC 2): at 32 rows that is one whole 192 KiB partition and,
# with the 48 static FP8 MoE kernels of the same verify graph, walrus aborts in insert_spill_code (sb_spill.cpp:448; MTP k=3 B=8 and k=2 B=16 on
# 11-06, bisected with CPU compiles). Above MAX_ROWS rows the verify mixer runs over batch chunks of a power-of-two
# size with chunk * T <= MAX_ROWS (each chunk: projections, gdn_tkg_multi, out_proj, all-reduce), so every graph with B * T <= MAX_ROWS (B=1,
# k=2 up to B=8, k=3 up to B=4) runs in one call.
_GDN_VERIFY_MULTI_MAX_ROWS = 24


def _gdn_verify_multi(la, x, state, cand):
    b, t = x.shape[0], x.shape[1]
    if b * t <= _GDN_VERIFY_MULTI_MAX_ROWS:
        return _gdn_verify_multi_unchunked(la, x, state, cand)
    bc = 1 << (max(1, _GDN_VERIFY_MULTI_MAX_ROWS // t).bit_length() - 1)
    outs, convs, recs = [], [], []
    for b0 in range(0, b, bc):
        sub = {"conv": state["conv"][b0 : b0 + bc], "rec": state["rec"][b0 : b0 + bc]}
        c = {}
        outs.append(_gdn_verify_multi_unchunked(la, x[b0 : b0 + bc], sub, c))
        convs.append(c["conv"])
        recs.append(c["rec"])
    cand["conv"], cand["rec"] = torch.cat(convs), torch.cat(recs)
    return torch.cat(outs)


# ─────────────────────────────────────────────────────────────────────────────
# QSA forms with the block-level mask (max_model_len > 2051)
# ─────────────────────────────────────────────────────────────────────────────
# The QSA forms at windows > 2,051 without the [B, T, S] token-mask construction.
# A query at p sees c = (p + 1) // ratio complete blocks; every top-k block it keeps is complete (j < c, so all its tokens
# are <= p) and the tail is the open block j == c up to p, so the reference mask is exactly
#     sel[t, s] = blk[t, s // ratio] & (s <= p_t),   blk = (j == c).scatter(top, True) & (j <= c)   ([B, T, nb] ops)
# and the past columns (s < start <= p_t) need one broadcast & of blk with s < start over [B, T, nb, ratio]. This step's
# indexer keys enter the raw-key window by a row index_select (one where) instead of the [S, D] element gather. Scores and
# top-k are the same lines as Qwen4ExpQSAIndexerWeights.select, so the selection and the SDPA inputs are bit-identical.
# Graphs with T <= _QSA_SPARSE_MAX_T (4) (one-token decode / MTP draft, MTP
# verify) instead attend only to the top-k blocks and the open block gathered from the cache (4-token rows, positions
# before this step) plus this step's own keys: the same key set and the same softmax over it, no S-wide K / V window;
# not bit-identical (another summation order). Rebinds the module globals used by forward_paged; the dense attention
# (max_model_len <= 2051) takes the reference forms.
_QSA_SPARSE_MAX_T = 4
_paged_decode_qsa_ref, _paged_prefill_qsa_ref = _paged_decode, _paged_prefill


def _qsa_cheap_window(attn: Qwen4ExpAttention, x, positions, idx_meta, start, num_valid):
    """The raw indexer-key window of :func:`_qsa_token_mask` (same values; the cache written first); returns ``(iq, window)``."""
    b, t, _ = x.shape
    iq, ik = attn.indexer.project(x, positions)
    cache = attn.idx_cache  # [blocks, 1, block, D]
    bs, dc = cache.shape[2], ik.shape[-1]
    slots = idx_meta["slot_mapping"].long()
    kv_write(cache, kv_rows(slots, cache), ik.reshape(b * t, 1, -1))
    table = idx_meta["block_table_tensor"][:b].long().clamp(min=0)
    win = cache.index_select(0, table.flatten()).view(b, table.shape[1] * bs, -1)[..., :dc]
    s_ctx = win.shape[1]
    rel = torch.arange(s_ctx, device=x.device).view(1, s_ctx) - start  # [B, S]
    n_new = num_valid.view(b, 1) if num_valid is not None else t
    src = torch.cat([ik.to(win.dtype).reshape(b * t, dc), win.new_zeros(1, dc)], dim=0)  # row b * t = 0
    row = rel.clamp(0, t - 1) + t * torch.arange(b, device=x.device).view(b, 1)
    row = torch.where((rel >= 0) & (rel < n_new), row, torch.full_like(row, b * t))
    new = src.index_select(0, row.flatten()).view(b, s_ctx, dc)
    win = torch.where((rel < 0).unsqueeze(-1), win, new)
    return iq, win.to(ik.dtype)


def _qsa_cheap_scores(ix: Qwen4ExpQSAIndexerWeights, q, raw_k, positions):
    """The block scores of :meth:`Qwen4ExpQSAIndexerWeights.select` (its lines): ``(scores [B, T, nb], complete [B, T], nb)``."""
    b, t = positions.shape
    s = raw_k.shape[1]
    r = ix.ratio
    nb = s // r
    pooled = raw_k[:, : nb * r].reshape(b, nb, r, ix.head_dim).float().mean(dim=2).to(raw_k.dtype)
    pooled = ix.k_layernorm(pooled)
    start = (torch.arange(nb, device=raw_k.device) * r).view(1, nb).expand(b, nb)
    cos, sin = rope_cos_sin(ix.config, start, raw_k.dtype)
    pooled = apply_partial_rope_no_concat(pooled.unsqueeze(2), cos, sin).squeeze(2)  # [B, nb, D]
    scores = torch.einsum("bthd,bjd->btjh", q.float(), pooled.float()).relu().sum(-1)
    scores = scores / ix.head_dim**0.5  # [B, T, nb]
    complete = (positions.long() + 1) // r
    valid = torch.arange(nb, device=raw_k.device).view(1, 1, nb) < complete.unsqueeze(-1)
    return torch.where(valid, scores, float("-inf")), complete, nb


def _qsa_cheap_blocks(scores, complete, k):
    """``[B, T, nb]``: selected complete blocks or the open block (``j == complete``)."""
    nb = scores.shape[-1]
    j = torch.arange(nb, device=scores.device).view(1, 1, nb)
    c = complete.unsqueeze(-1)
    top = torch.topk(scores, k, dim=-1).indices
    return (j == c).scatter(-1, top, True) & (j <= c)


def _qsa_sparse_fits(attn: Qwen4ExpAttention, meta: dict, t: int) -> bool:
    ix, bs = attn.indexer, attn.k_cache.shape[2]
    s = meta["block_table_tensor"].shape[1] * bs
    return 0 < t <= _QSA_SPARSE_MAX_T and bs % ix.ratio == 0 and (min(ix.block_topk, s // ix.ratio) + 1) * ix.ratio < s


def _qsa_sparse(attn: Qwen4ExpAttention, x, positions, meta, num_valid, idx_meta, decode: bool):
    """QSA over the gathered top-k blocks + the open block (cache, positions < this step's first) + this step's keys."""
    b, t, _ = x.shape
    q, k, v, gate = attn._qkv(x, positions)
    kvh, d, bs = attn.num_kv_heads, attn.head_dim, attn.k_cache.shape[2]
    ix = attn.indexer
    r = ix.ratio
    slots = meta["slot_mapping"].long()
    k_new = k.transpose(1, 2).to(attn.k_cache.dtype)  # [B, KVH, T, D]
    v_new = v.transpose(1, 2).to(attn.v_cache.dtype)
    # Written before the gather (see kv_write): the gathered keys keep only positions < start.
    rows_kv = kv_rows(slots, attn.k_cache)
    kv_write(attn.k_cache, rows_kv, k.reshape(b * t, kvh, d))
    kv_write(attn.v_cache, rows_kv, v.reshape(b * t, kvh, d))
    pos = positions.long()
    start = pos[:, :1]  # [B, 1] (decode: the current position, strict past as in _paged_decode)
    nv = None if decode else num_valid
    iq, win = _qsa_cheap_window(attn, x, positions, idx_meta, start, nv)
    scores, complete, nb = _qsa_cheap_scores(ix, iq, win, positions)
    kk = min(ix.block_topk, nb)
    top = torch.topk(scores, kk, dim=-1).indices  # [B, T, kk]
    c = complete.unsqueeze(-1)  # [B, T, 1]
    blocks = torch.cat([top, c.clamp(max=nb - 1)], dim=-1)  # [B, T, kk + 1]: the top-k, then the open block
    tokp = blocks.unsqueeze(-1) * r + torch.arange(r, device=x.device)  # [B, T, kk + 1, r] key positions
    keep = torch.cat([(top < c).unsqueeze(-1).expand(b, t, kk, r),
                      (tokp[:, :, kk:] >= (c * r).unsqueeze(-1)) & (tokp[:, :, kk:] <= pos.view(b, t, 1, 1))], dim=2)
    keep = keep & (tokp < start.view(b, 1, 1, 1))
    # 4-token rows of the cache viewed as [-1, ratio * D] (kv heads inside a block are ratio-aligned runs of tokens).
    per = bs // r
    table = meta["block_table_tensor"][:b].long().clamp(min=0)
    s_ctx = table.shape[1] * bs
    pblk = table.gather(1, (blocks // per).view(b, -1)).view(b, t, 1, kk + 1)
    h = torch.arange(kvh, device=x.device).view(1, 1, kvh, 1)
    rows = ((pblk * kvh + h) * per + (blocks % per).view(b, t, 1, kk + 1)).flatten()
    n = (kk + 1) * r
    keep4 = keep.view(b * t, 1, n, 1)

    def gather(cache):  # positions not kept (never written, this step's, padding blocks) may hold NaN: zeroed
        g = cache.view(-1, r * d).index_select(0, rows).view(b * t, kvh, n, d)
        return torch.where(keep4, g, 0.0)

    i = torch.arange(t, device=x.device)
    own = i.view(1, 1, t) <= i.view(1, t, 1)  # causal within the step
    if nv is not None:
        own = own & (i.view(1, 1, t) < nv.view(b, 1, 1))
    cols = (start + i.view(1, t)).clamp(max=s_ctx - 1)  # [B, T] position of each new key
    cb = (cols // r).view(b, 1, 1, t)
    in_top = ((top.unsqueeze(-1) == cb) & (top < c).unsqueeze(-1)).any(dim=2)  # [B, T, T]
    own = own & (in_top | ((cb.view(b, 1, t) == c) & (cols.view(b, 1, t) <= pos.view(b, t, 1))))

    def own_rows(new):  # [B, KVH, T, D] -> [B * T, KVH, T, D] (every query sees the whole step; 3-D expand)
        return new.reshape(b, 1, kvh * t * d).expand(b, t, kvh * t * d).reshape(b * t, kvh, t, d)

    kc = torch.cat([gather(attn.k_cache), own_rows(k_new)], dim=2)
    vc = torch.cat([gather(attn.v_cache), own_rows(v_new)], dim=2)
    mask = torch.cat([keep.view(b * t, 1, 1, n), own.expand(b, t, t).reshape(b * t, 1, 1, t)], dim=-1)
    out = F.scaled_dot_product_attention(
        q.reshape(b * t, 1, attn.num_heads, d).transpose(1, 2), kc.to(q.dtype), vc.to(q.dtype), attn_mask=mask,
        scale=attn.head_dim**-0.5, enable_gqa=True,
    )  # [B * T, Hq, 1, D]
    out = out.transpose(1, 2).reshape(b, t, -1) * torch.sigmoid(gate)
    return all_reduce(attn.o_proj(out), attn.tp)


def _paged_decode_cheap(attn: Qwen4ExpAttention, x, positions, meta, num_valid=None, idx_meta=None):
    """:func:`_paged_decode` (QSA) with the block-level mask (bit-identical), or :func:`_qsa_sparse`."""
    if attn.dense or idx_meta is None:
        return _paged_decode_qsa_ref(attn, x, positions, meta, num_valid, idx_meta)
    if _qsa_sparse_fits(attn, meta, 1):
        return _qsa_sparse(attn, x, positions, meta, None, idx_meta, True)
    b = x.shape[0]
    q, k, v, gate = attn._qkv(x, positions)
    kvh, d, bs = attn.num_kv_heads, attn.head_dim, attn.k_cache.shape[2]
    slots = meta["slot_mapping"].long()
    k_tok = k.reshape(b, kvh, d).to(attn.k_cache.dtype)
    v_tok = v.reshape(b, kvh, d).to(attn.v_cache.dtype)
    rows_kv = kv_rows(slots, attn.k_cache)
    kv_write(attn.k_cache, rows_kv, k_tok)
    kv_write(attn.v_cache, rows_kv, v_tok)
    table = meta["block_table_tensor"][:b].long().clamp(min=0)
    nblk = table.shape[1]

    def window(cache):
        w = cache.index_select(0, table.flatten()).view(b, nblk, kvh, bs, d)
        return w.permute(0, 2, 1, 3, 4).reshape(b, kvh, nblk * bs, d)

    kc, vc = window(attn.k_cache), window(attn.v_cache)
    s_ctx = kc.shape[2]
    past = torch.arange(s_ctx, device=x.device).view(1, 1, s_ctx, 1) < positions.view(b, 1, 1, 1)
    kc = torch.cat([torch.where(past, kc, 0.0), k_tok.view(b, kvh, 1, d)], dim=2)
    vc = torch.cat([torch.where(past, vc, 0.0), v_tok.view(b, kvh, 1, d)], dim=2)
    ix = attn.indexer
    r = ix.ratio
    cur = positions.view(b, 1).long()
    iq, win = _qsa_cheap_window(attn, x, positions, idx_meta, cur, None)
    scores, complete, nb = _qsa_cheap_scores(ix, iq, win, positions)
    blk = _qsa_cheap_blocks(scores, complete, min(ix.block_topk, nb))  # [B, 1, nb]
    # past columns: blk & (s < p) (s < p implies s <= p); the current token: blk at its block (p <= p)
    mask = torch.cat([(blk.view(b, nb, 1) & past.view(b, nb, r)).view(b, 1, 1, s_ctx),
                      blk.gather(-1, (cur // r).view(b, 1, 1)).view(b, 1, 1, 1)], -1)
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), kc.to(q.dtype), vc.to(q.dtype), attn_mask=mask,
        scale=attn.head_dim**-0.5, enable_gqa=True,
    )
    out = out.transpose(1, 2).reshape(b, 1, -1) * torch.sigmoid(gate)
    return all_reduce(attn.o_proj(out), attn.tp)


def _paged_prefill_cheap(attn: Qwen4ExpAttention, x, positions, meta, num_valid=None, idx_meta=None):
    """:func:`_paged_prefill` (QSA) with the block-level mask (bit-identical for every query at a position >= the
    row's first position this step, i.e. every valid one), or :func:`_qsa_sparse` for T <= QSA_SPARSE_MAX_T."""
    if attn.dense or idx_meta is None:
        return _paged_prefill_qsa_ref(attn, x, positions, meta, num_valid, idx_meta)
    if _qsa_sparse_fits(attn, meta, x.shape[1]):
        return _qsa_sparse(attn, x, positions, meta, num_valid, idx_meta, False)
    b, t, _ = x.shape
    q, k, v, gate = attn._qkv(x, positions)
    kvh, d, bs = attn.num_kv_heads, attn.head_dim, attn.k_cache.shape[2]
    slots = meta["slot_mapping"].long()
    k_new = k.transpose(1, 2).to(attn.k_cache.dtype)  # [B, KVH, T, D]
    v_new = v.transpose(1, 2).to(attn.v_cache.dtype)
    rows_kv = kv_rows(slots, attn.k_cache)
    kv_write(attn.k_cache, rows_kv, k.reshape(b * t, kvh, d))
    kv_write(attn.v_cache, rows_kv, v.reshape(b * t, kvh, d))
    table = meta["block_table_tensor"][:b].long().clamp(min=0)
    nblk = table.shape[1]

    def window(cache):
        w = cache.index_select(0, table.flatten()).view(b, nblk, kvh, bs, d)
        return w.permute(0, 2, 1, 3, 4).reshape(b, kvh, nblk * bs, d)

    kc, vc = window(attn.k_cache), window(attn.v_cache)
    s_ctx = kc.shape[2]
    start = positions[:, :1].long()  # [B, 1]
    past = torch.arange(s_ctx, device=x.device).view(1, s_ctx) < start  # [B, S]
    kc = torch.cat([torch.where(past.view(b, 1, s_ctx, 1), kc, 0.0), k_new], dim=2)
    vc = torch.cat([torch.where(past.view(b, 1, s_ctx, 1), vc, 0.0), v_new], dim=2)
    i = torch.arange(t, device=x.device)
    own = i.view(1, 1, t) <= i.view(1, t, 1)  # causal within the chunk
    if num_valid is not None:
        own = own & (i.view(1, 1, t) < num_valid.view(b, 1, 1))  # padding keys never attended
    ix = attn.indexer
    r = ix.ratio
    iq, win = _qsa_cheap_window(attn, x, positions, idx_meta, start, num_valid)
    scores, complete, nb = _qsa_cheap_scores(ix, iq, win, positions)
    blk = _qsa_cheap_blocks(scores, complete, min(ix.block_topk, nb))  # [B, T, nb]
    cols = (start + i.view(1, t)).clamp(max=s_ctx - 1)  # [B, T] position of each new key
    own = own & blk.gather(-1, (cols // r).view(b, 1, t).expand(b, t, t)) & (cols.view(b, 1, t) <= positions.long().view(b, t, 1))
    past = (blk.view(b, t, nb, 1) & past.view(b, 1, nb, r)).view(b, t, s_ctx)
    mask = torch.cat([past, own.expand(b, t, t)], dim=-1)
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), kc.to(q.dtype), vc.to(q.dtype), attn_mask=mask[:, None],
        scale=attn.head_dim**-0.5, enable_gqa=True,
    )
    out = out.transpose(1, 2).reshape(b, t, -1) * torch.sigmoid(gate)
    return all_reduce(attn.o_proj(out), attn.tp)


_paged_decode, _paged_prefill = _paged_decode_cheap, _paged_prefill_cheap


# ─────────────────────────────────────────────────────────────────────────────
# QSA_TOPK_ROW_CHUNK (2026-11-07)
# ─────────────────────────────────────────────────────────────────────────────
# _QSA_TOPK_ROW_CHUNK = 128: at max_model_len > 65,536 the QSA block
# top-k runs over nb = S / 4 > 16,384 blocks, and neuronx-cc 2.28.7631 (SDK 2.33) fails on AwsNeuronTopK (what
# torch.topk lowers to) with ISGV902 "Assertion failed: False" in SimplifyNeuronTensor (IntegerSetAnalysis.build_aff)
# whenever the operand has more than 128 rows, the last dim is > 16,384 and k > 8 (minimal repro: f32[512, 16,385], k 512;
# 128 rows, 16,384 wide or k 8 compile). That is every prefill / MTP-draft-prefill graph with T >= 129 at 81,920. Each
# row's top-k is independent of the other rows, so the top-k over chunks of <= C rows, concatenated, is the same
# result (indices bit-identical). Applied only where the shape needs it (rows > C and width > 16,384); every other graph
# calls the original code (same lines, same HLO). Covers Qwen4ExpQSAIndexerWeights.select (the QSA forms) and
# _qsa_cheap_blocks; _qsa_sparse's top-k has B * T <= 4 * _QSA_SPARSE_MAX_T rows.
_QSA_TOPK_ROW_CHUNK = 128
_QSA_TOPK_MAX_WIDTH = 16384  # widest AwsNeuronTopK operand that compiled with > 128 rows (16,385 failed)


def _topk_needs_row_chunks(rows: int, width: int) -> bool:
    return rows > _QSA_TOPK_ROW_CHUNK and width > _QSA_TOPK_MAX_WIDTH


def _topk_indices_row_chunked(scores: torch.Tensor, k: int) -> torch.Tensor:
    """``torch.topk(scores, k, dim=-1).indices`` computed over chunks of <= QSA_TOPK_ROW_CHUNK rows."""
    lead, n = scores.shape[:-1], scores.shape[-1]
    x = scores.reshape(-1, n)
    c = _QSA_TOPK_ROW_CHUNK
    parts = [torch.topk(x[r:r + c], k, dim=-1).indices for r in range(0, x.shape[0], c)]
    return torch.cat(parts, dim=0).view(*lead, k)


_qsa_select_orig = Qwen4ExpQSAIndexerWeights.select


def _qsa_select_row_chunked(self, q: torch.Tensor, raw_k: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """:meth:`Qwen4ExpQSAIndexerWeights.select` (its lines) with the block top-k over row chunks."""
    b, t = positions.shape
    s = raw_k.shape[1]
    r = self.ratio
    nb = s // r
    pooled = raw_k[:, : nb * r].reshape(b, nb, r, self.head_dim).float().mean(dim=2).to(raw_k.dtype)
    pooled = self.k_layernorm(pooled)
    start = (torch.arange(nb, device=raw_k.device) * r).view(1, nb).expand(b, nb)
    cos, sin = rope_cos_sin(self.config, start, raw_k.dtype)
    pooled = apply_partial_rope_no_concat(pooled.unsqueeze(2), cos, sin).squeeze(2)  # [B, nb, D]
    scores = torch.einsum("bthd,bjd->btjh", q.float(), pooled.float()).relu().sum(-1)
    scores = scores / self.head_dim**0.5  # [B, T, nb]
    complete = (positions.long() + 1) // r  # complete blocks visible to each query
    valid = torch.arange(nb, device=raw_k.device).view(1, 1, nb) < complete.unsqueeze(-1)
    scores = torch.where(valid, scores, torch.full_like(scores, float("-inf")))
    top = _topk_indices_row_chunked(scores, min(self.block_topk, nb))
    chosen = torch.zeros_like(valid).scatter(-1, top, True) & valid
    tok = torch.arange(s, device=raw_k.device)
    in_block = chosen.gather(-1, (tok // r).clamp(max=nb - 1).view(1, 1, s).expand(b, t, s))
    tail_start = (complete * r).unsqueeze(-1)
    before_tail = tok.view(1, 1, s) < tail_start
    tail = (tok.view(1, 1, s) >= tail_start) & (tok.view(1, 1, s) <= positions.long().unsqueeze(-1))
    return (in_block & before_tail) | tail


def _qsa_select_dispatch(self, q: torch.Tensor, raw_k: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    if _topk_needs_row_chunks(positions.numel(), raw_k.shape[1] // self.ratio):
        return _qsa_select_row_chunked(self, q, raw_k, positions)
    return _qsa_select_orig(self, q, raw_k, positions)


_qsa_cheap_blocks_orig = _qsa_cheap_blocks


def _qsa_cheap_blocks_row_chunked(scores, complete, k):
    """:func:`_qsa_cheap_blocks` with the top-k over row chunks."""
    nb = scores.shape[-1]
    j = torch.arange(nb, device=scores.device).view(1, 1, nb)
    c = complete.unsqueeze(-1)
    top = _topk_indices_row_chunked(scores, k)
    return (j == c).scatter(-1, top, True) & (j <= c)


def _qsa_cheap_blocks_dispatch(scores, complete, k):
    if _topk_needs_row_chunks(scores[..., 0].numel(), scores.shape[-1]):
        return _qsa_cheap_blocks_row_chunked(scores, complete, k)
    return _qsa_cheap_blocks_orig(scores, complete, k)


Qwen4ExpQSAIndexerWeights.select = _qsa_select_dispatch
_qsa_cheap_blocks = _qsa_cheap_blocks_dispatch


# ─────────────────────────────────────────────────────────────────────────────
# DECODE_ROW_CHUNK (2026-11-07, MTP acceptance 0 at decode buckets >= 16)
# ─────────────────────────────────────────────────────────────────────────────
# _DECODE_ROW_CHUNK = C = 32: the row-wise parts of a graph whose row count n (tokens =
# B * T of a decode / MTP verify / MTP draft step) is above C and not a multiple of 128 run in chunks of C rows: the
# hyper-connection reads (Qwen4ExpGatedResidual, final mixer included), the MoE (router, experts, shared expert and their
# all-reduce, per chunk), the attention q|k|v / o projections, the [in, out] weight-copy linears (GDN, LM head, PLE, router,
# shared expert) and the sampler's NKI top-k. On device the MTP verify graphs with 48 / 64 / 96 rows (k=2 B=16 / B=32, k=3
# B=16) accepted 0 % of the drafts while 3 .. 32 rows (B <= 8) and 128 rows (k=3 B=32) were correct; above 32 rows these
# parts leave the decode shapes (hc_cte instead of the torch hyper-connection, F.linear on the original weights instead of
# the copies, torch attention projections, moe_tkg / top-k at > 32 rows) that were only ever run at multiples of 128 rows
# (prefill buckets), and the CPU path and the NKI simulator are correct at those shapes, so which part fails on device is
# not known. With C = 32 every such call runs at a row count that decoded correctly on device (32, or the remainder: 16 at
# 48 rows). For finite inputs the rows of these parts are independent, so the chunked result equals the unchunked one up
# to GEMM-shape rounding (a NaN row does reach other rows inside some decode kernels: see model_bf16._pool_write).
# Prefill graphs (multiples of 128 tokens) and decode graphs with <= C rows are not chunked; installed by rebinding below
# (wio_wrap / _atf_init / the FP8 MoE inits read the rebound names when the model is built).
_DECODE_ROW_CHUNK = 32


def _drc_rows(n: int) -> bool:
    return 0 < _DECODE_ROW_CHUNK < n and n % 128 != 0


def _drc_spans(n: int):
    return [(i, min(n, i + _DECODE_ROW_CHUNK)) for i in range(0, n, _DECODE_ROW_CHUNK)]


def _drc_hc_forward(fn):
    def forward(self, streams: torch.Tensor):
        d = streams.shape[-1]
        n = streams.numel() // d
        if not _drc_rows(n):
            return fn(self, streams)
        flat = streams.reshape(n, d)
        parts = [fn(self, flat[a:b]) for a, b in _drc_spans(n)]
        if self.block_inject_weight is None:
            return torch.cat(parts).reshape(*streams.shape[:-1], -1)
        mixed = torch.cat([p[0] for p in parts]).reshape(*streams.shape[:-1], -1)
        w = torch.cat([p[2] for p in parts]).reshape(*streams.shape[:-1], -1)
        return mixed, streams, w

    return forward


def _drc_moe_forward(fn):
    def forward(self, x: torch.Tensor, is_decode: bool = False) -> torch.Tensor:
        n = x.numel() // x.shape[-1]
        if not (is_decode and _drc_rows(n)):
            return fn(self, x, is_decode)
        flat = x.reshape(n, x.shape[-1])
        return torch.cat([fn(self, flat[a:b], is_decode) for a, b in _drc_spans(n)]).reshape(x.shape)

    return forward


def _drc_qkv(fn):
    def qkv(self, x: torch.Tensor, positions: torch.Tensor):
        b, t = x.shape[0], x.shape[1]
        if not _drc_rows(b * t) or t > _DECODE_ROW_CHUNK:
            return fn(self, x, positions)
        bc = max(1, _DECODE_ROW_CHUNK // t)
        parts = [fn(self, x[i : i + bc], positions[i : i + bc]) for i in range(0, b, bc)]
        return tuple(torch.cat([p[j] for p in parts]) for j in range(len(parts[0])))

    return qkv


def _drc_linear(forward_fn):
    def forward(lin, inp: torch.Tensor) -> torch.Tensor:
        n = inp.numel() // inp.shape[-1]
        if not _drc_rows(n):
            return forward_fn(lin, inp)
        flat = inp.reshape(n, inp.shape[-1])
        out = torch.cat([forward_fn(lin, flat[a:b]) for a, b in _drc_spans(n)])
        return out.reshape(*inp.shape[:-1], out.shape[-1])

    return forward


def _drc_atf_o_proj_forward(make):
    def build(attn):
        return _drc_linear(make(attn))

    return build


def _drc_topk_nki(fn):
    def topk_nki(tensor, k, dim, method):
        if dim < 0:
            dim += tensor.ndim
        n = tensor.numel() // tensor.shape[-1]
        if dim != tensor.ndim - 1 or not _drc_rows(n):
            return fn(tensor, k, dim, method)
        flat = tensor.reshape(n, tensor.shape[-1])
        parts = [fn(flat[a:b], k, 1, method) for a, b in _drc_spans(n)]
        values = torch.cat([p[0] for p in parts]).reshape(*tensor.shape[:-1], k)
        indices = torch.cat([p[1] for p in parts]).reshape(*tensor.shape[:-1], k)
        return values, indices

    return topk_nki


import vllm_neuron.functional.topk as _drc_topk  # noqa: E402

Qwen4ExpGatedResidual.forward = _drc_hc_forward(Qwen4ExpGatedResidual.forward)
Qwen4ExpSparseMoe.forward = _drc_moe_forward(Qwen4ExpSparseMoe.forward)
_moe_block_forward_fp8 = _drc_moe_forward(_moe_block_forward_fp8)  # instance forward of the fused FP8 MoE init
_moe_block_orig_forward = _drc_moe_forward(_moe_block_orig_forward)  # the FP8 init's instance forward (no-op inside chunks)
Qwen4ExpAttention._qkv = _drc_qkv(Qwen4ExpAttention._qkv)
_atf_o_proj_forward = _drc_atf_o_proj_forward(_atf_o_proj_forward)  # used by _atf_init
_wio_forward = _drc_linear(_wio_forward)  # used by wio_wrap
_drc_topk._topk_nki = _drc_topk_nki(_drc_topk._topk_nki)


# ─────────────────────────────────────────────────────────────────────────────
# QSA sparse prefill (max_model_len > 2051)
# ─────────────────────────────────────────────────────────────────────────────
# QSA prefill (and verify, when _QSA_SPARSE_MAX_T does not take it) without any
# [T, S]-sized tensor. The QSA forms above build, per prefill chunk, the [T, S] token mask, the [Hq, T, S + T] SDPA scores over
# the S-wide K / V window and the [T, S / ratio, Hi] indexer scores; at max_model_len 81,920 that did not fit the device (T=4,096:
# 8.94 GB scratchpad per core and the decode graphs then fail to load; T=2,048: spill-DMA descriptor ring full). This form runs the
# chunk in query tiles of _QSA_SPF_TILE rows (128, also the AwsNeuronTopK row limit at widths > 16,384). Per tile:
#   indexer   block scores [B, Tt, nb] against the pooled block keys (pooled once per chunk; select's lines, the same values), top-k
#             over the tile's rows, blk = selected complete blocks or the open block ([B, Tt, nb] bool);
#   past      keys before the chunk start: the top-k blocks + the open block of each query gathered as 4-token 2 KB rows of the
#             paged cache (as _qsa_sparse); n = (k + 1) * ratio keys per query, no S-wide window;
#   own       this chunk's keys 0 .. the tile's last query (causal), dense, masked by blk at their block (or the open block up to p);
#   softmax   one exact normalisation over every key set (common max, summed exponentials, fp32 statistics).
# The attended key set is the reference's; outputs are not bit-identical (summation order). Rebinds the module global used by
# forward_paged; the dense attention (max_model_len <= 2051) takes the reference form.
_QSA_SPF_TILE = 128
_paged_prefill_spf_prev = _paged_prefill_cheap


def _qsa_spf_fits(attn: Qwen4ExpAttention, meta: dict, t: int) -> bool:
    ix, bs = attn.indexer, attn.k_cache.shape[2]
    s = meta["block_table_tensor"].shape[1] * bs
    if t <= _QSA_SPARSE_MAX_T:  # _qsa_sparse (via the previous form)
        return False
    return bs % ix.ratio == 0 and (min(ix.block_topk, s // ix.ratio) + 1) * ix.ratio < s


def _qsa_spf_pooled(ix: Qwen4ExpQSAIndexerWeights, raw_k: torch.Tensor) -> torch.Tensor:
    """The pooled block keys of :meth:`Qwen4ExpQSAIndexerWeights.select` (its lines), as fp32 ``[B, nb, D]``."""
    b, s = raw_k.shape[:2]
    r = ix.ratio
    nb = s // r
    pooled = raw_k[:, : nb * r].reshape(b, nb, r, ix.head_dim).float().mean(dim=2).to(raw_k.dtype)
    pooled = ix.k_layernorm(pooled)
    start = (torch.arange(nb, device=raw_k.device) * r).view(1, nb).expand(b, nb)
    cos, sin = rope_cos_sin(ix.config, start, raw_k.dtype)
    pooled = apply_partial_rope_no_concat(pooled.unsqueeze(2), cos, sin).squeeze(2)  # [B, nb, D]
    return pooled.float()


def _qsa_spf_scores(ix: Qwen4ExpQSAIndexerWeights, q: torch.Tensor, pooled: torch.Tensor, positions: torch.Tensor):
    """Block scores of one query tile (select's lines): ``(scores [B, Tt, nb], -inf at blocks not complete; complete [B, Tt])``."""
    nb = pooled.shape[1]
    scores = torch.einsum("bthd,bjd->btjh", q.float(), pooled).relu().sum(-1)
    scores = scores / ix.head_dim**0.5  # [B, Tt, nb]
    complete = (positions.long() + 1) // ix.ratio
    valid = torch.arange(nb, device=q.device).view(1, 1, nb) < complete.unsqueeze(-1)
    return torch.where(valid, scores, float("-inf")), complete


def _qsa_spf_attend(q, rows, shared, scale):
    """Softmax attention of one query tile over several key sets under one normalisation (exact; fp32 statistics).

    q ``[B, Tt, Hq, D]``. ``rows``: per-query sets ``(k, v, keep)``, k / v ``[B * Tt, KVH, n, D]``, keep ``[B * Tt, 1, 1, n]``.
    ``shared``: sets seen by every query of the tile ``(k, v, mask)``, k / v ``[B, KVH, w, D]``, mask ``[B, Tt, w]``. Masked scores
    are replaced by -inf with torch.where (k there may hold NaN; v must be finite or zero). Both products run in q's dtype
    (as the reference's SDPA over ``kc / vc .to(q.dtype)``), the softmax statistics in fp32. Query head h reads KV head
    h // (Hq / KVH) (SDPA enable_gqa). A query with no attended key gets 0. Returns ``[B, Tt, Hq, D]`` in q's dtype."""
    b, tt, hq, d = q.shape
    kvh = (rows or shared)[0][0].shape[1]
    g = hq // kvh
    sd = torch.promote_types(q.dtype, torch.float32)  # statistics dtype (fp32; fp64 for an fp64 q)
    neg = float("-inf")
    qr = q.reshape(b * tt, kvh, g, d)
    qs = q.permute(0, 2, 1, 3).reshape(b * kvh, g * tt, d)
    sr, ss, m = [], [], None
    for k_, v_, keep in rows:
        s_ = torch.matmul(qr, k_.to(q.dtype).transpose(-1, -2)).to(sd) * scale  # [B*Tt, KVH, G, n]
        s_ = torch.where(keep, s_, neg)
        sr.append((s_, v_))
        mx = s_.amax(dim=-1).view(b, tt, hq).permute(0, 2, 1)  # [B, Hq, Tt]
        m = mx if m is None else torch.maximum(m, mx)
    for k_, v_, mask in shared:
        w = k_.shape[2]
        s_ = torch.matmul(qs, k_.to(q.dtype).reshape(b * kvh, w, d).transpose(-1, -2)).to(sd) * scale  # [B*KVH, G*Tt, w]
        s_ = s_.view(b * kvh, g, tt, w)
        mm = mask.view(b, 1, tt, w).expand(b, kvh, tt, w).reshape(b * kvh, 1, tt, w)
        s_ = torch.where(mm, s_, neg)
        ss.append((s_, v_))
        mx = s_.amax(dim=-1).view(b, hq, tt)
        m = mx if m is None else torch.maximum(m, mx)
    m = torch.where(torch.isfinite(m), m, 0.0)  # [B, Hq, Tt]
    mr, ms = m.permute(0, 2, 1).reshape(b * tt, kvh, g, 1), m.reshape(b * kvh, g, tt, 1)
    z, o = None, None
    for s_, v_ in sr:
        e = torch.exp(s_ - mr)
        zi = e.sum(dim=-1).view(b, tt, hq)
        oi = torch.matmul(e.to(q.dtype), v_.to(q.dtype)).to(sd).view(b, tt, hq, d)
        z, o = (zi, oi) if z is None else (z + zi, o + oi)
    for s_, v_ in ss:
        w = s_.shape[-1]
        e = torch.exp(s_ - ms)
        zi = e.sum(dim=-1).view(b, hq, tt).permute(0, 2, 1)
        oi = torch.matmul(e.view(b * kvh, g * tt, w).to(q.dtype), v_.to(q.dtype).reshape(b * kvh, w, d)).to(sd)
        oi = oi.view(b, hq, tt, d).permute(0, 2, 1, 3)
        z, o = (zi, oi) if z is None else (z + zi, o + oi)
    z = torch.where(z > 0, z, 1.0).unsqueeze(-1)  # [B, Tt, Hq, 1]
    return (o / z).to(q.dtype)


def _qsa_sparse_prefill(attn: Qwen4ExpAttention, x, positions, meta, num_valid=None, idx_meta=None):
    """QSA prefill chunk over query tiles: the selected past keys (top-k blocks + open block, positions < the chunk start) and
    this chunk's own keys (causal, selected), one softmax. Same key set per valid query as :func:`_paged_prefill` (QSA)."""
    b, t, _ = x.shape
    q, k, v, gate = attn._qkv(x, positions)
    kvh, d, bs = attn.num_kv_heads, attn.head_dim, attn.k_cache.shape[2]
    ix = attn.indexer
    r = ix.ratio
    slots = meta["slot_mapping"].long()
    # Written before the gather (see kv_write): the gathered keys keep only positions < start.
    rows_kv = kv_rows(slots, attn.k_cache)
    kv_write(attn.k_cache, rows_kv, k.reshape(b * t, kvh, d))
    kv_write(attn.v_cache, rows_kv, v.reshape(b * t, kvh, d))
    pos = positions.long()
    start = pos[:, :1]  # [B, 1]
    iq, win = _qsa_cheap_window(attn, x, positions, idx_meta, start, num_valid)
    pooled = _qsa_spf_pooled(ix, win)  # [B, nb, Di] fp32, shared by every tile
    nb = pooled.shape[1]
    kk = min(ix.block_topk, nb)
    n = (kk + 1) * r
    per = bs // r
    table = meta["block_table_tensor"][:b].long().clamp(min=0)
    s_ctx = table.shape[1] * bs
    i = torch.arange(t, device=x.device)
    cols = (start + i.view(1, t)).clamp(max=s_ctx - 1)  # [B, T] position of each new key
    k_new = k.transpose(1, 2).to(attn.k_cache.dtype)  # [B, KVH, T, D]
    v_new = v.transpose(1, 2).to(attn.v_cache.dtype)
    colv = None
    if num_valid is not None:  # padding keys never attended; zeroed (padding hidden states may be NaN on device)
        colv = i.view(1, t) < num_valid.view(b, 1)  # [B, T]
        k_new = torch.where(colv.view(b, 1, t, 1), k_new, 0.0)
        v_new = torch.where(colv.view(b, 1, t, 1), v_new, 0.0)
    h = torch.arange(kvh, device=x.device).view(1, 1, kvh, 1)
    kflat, vflat = attn.k_cache.view(-1, r * d), attn.v_cache.view(-1, r * d)
    jn = torch.arange(nb, device=x.device).view(1, 1, nb)
    tile = _QSA_SPF_TILE
    outs = []
    for t0 in range(0, t, tile):
        t1 = min(t, t0 + tile)
        tt = t1 - t0
        pt = pos[:, t0:t1]  # [B, Tt]
        scores, complete = _qsa_spf_scores(ix, iq[:, t0:t1], pooled, pt)
        if _topk_needs_row_chunks(b * tt, nb):
            top = _topk_indices_row_chunked(scores, kk)
        else:
            top = torch.topk(scores, kk, dim=-1).indices  # [B, Tt, kk]
        c = complete.unsqueeze(-1)  # [B, Tt, 1]
        blk = (jn == c).scatter(-1, top, True) & (jn <= c)  # [B, Tt, nb]: selected complete blocks or the open block
        # the top-k blocks, then the open block; kept if complete-and-selected / open up to p, and before the chunk
        blocks = torch.cat([top, c.clamp(max=nb - 1)], dim=-1)  # [B, Tt, kk + 1]
        tokp = blocks.unsqueeze(-1) * r + torch.arange(r, device=x.device)  # [B, Tt, kk + 1, r]
        keep = torch.cat([(top < c).unsqueeze(-1).expand(b, tt, kk, r),
                          (tokp[:, :, kk:] >= (c * r).unsqueeze(-1)) & (tokp[:, :, kk:] <= pt.view(b, tt, 1, 1))], dim=2)
        keep = keep & (tokp < start.view(b, 1, 1, 1))
        pblk = table.gather(1, (blocks // per).view(b, -1)).view(b, tt, 1, kk + 1)
        rows = ((pblk * kvh + h) * per + (blocks % per).view(b, tt, 1, kk + 1)).flatten()
        # Rows not kept (never written, this chunk's positions, padding blocks) may hold NaN: V is zeroed (0 * NaN in the
        # weighted sum) and so is K (masking the scores alone is not enough: a NaN key can reach the score before the select).
        kg = torch.where(keep.view(b * tt, 1, n, 1), kflat.index_select(0, rows).view(b * tt, kvh, n, d), 0.0)
        vg = torch.where(keep.view(b * tt, 1, n, 1), vflat.index_select(0, rows).view(b * tt, kvh, n, d), 0.0)
        rows_sets, shared = [(kg, vg, keep.view(b * tt, 1, 1, n))], []
        # own: this chunk's keys up to the tile's last query (causal), selected at their block or open block up to p
        cw = cols[:, :t1]
        own = i[:t1].view(1, 1, t1) <= i[t0:t1].view(1, tt, 1)
        own = own & blk.gather(-1, (cw // r).view(b, 1, t1).expand(b, tt, t1)) & (cw.view(b, 1, t1) <= pt.view(b, tt, 1))
        if colv is not None:
            own = own & colv[:, :t1].view(b, 1, t1)
        shared.append((k_new[:, :, :t1], v_new[:, :, :t1], own))
        outs.append(_qsa_spf_attend(q[:, t0:t1], rows_sets, shared, attn.head_dim**-0.5))
    out = torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]  # [B, T, Hq, D]
    if colv is not None:
        out = torch.where(colv.view(b, t, 1, 1), out, 0.0)
    out = out.reshape(b, t, -1) * torch.sigmoid(gate)
    return all_reduce(attn.o_proj(out), attn.tp)


def _paged_prefill_spf(attn: Qwen4ExpAttention, x, positions, meta, num_valid=None, idx_meta=None):
    if attn.dense or idx_meta is None or not _qsa_spf_fits(attn, meta, x.shape[1]):
        return _paged_prefill_spf_prev(attn, x, positions, meta, num_valid, idx_meta)
    return _qsa_sparse_prefill(attn, x, positions, meta, num_valid, idx_meta)


_paged_prefill = _paged_prefill_spf
