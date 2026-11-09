# SPDX-License-Identifier: Apache-2.0
"""Qwen4-Exp gated full attention, decode / MTP verify (few rows): the two projection ends in NKI.

Per rank (attention group 8: ``nh`` = 3 query heads, one KV head, ``d`` = 256, rotary 64, ``H`` = 2560) the
XLA lowering of ``q_proj`` / ``k_proj`` / ``v_proj`` / ``o_proj`` for a handful of rows transposes every
128 x 128 tile of the ``[out, in]`` weights on the PE before using it (``PfTransposeInstructionsForIo``)
and keeps the weights stationary: ≈ 3,100 of the ≈ 4,200 static instructions of one T=3 verify mixer
(CPU neuronx-cc, 2026-11-06). These kernels read pre-transposed
``[in, out]`` weights (built once at load) with the ROWS stationary and the weights moving (512 columns
per matmul), the columns split over the LNC programs (no exchange):

* ``attn_qkv_tkg(x, w_t, cos, sin, q_w, k_w)``: ``x`` [N, H] bf16, ``w_t`` [H, nh * 2d + 2d] = [q_proj; k_proj;
  v_proj]^T (columns: per head ``q_h | gate_h``, then ``k | v``), ``cos`` / ``sin`` [N, r] bf16 (``rope_cos_sin``),
  ``q_w`` / ``k_w`` [d] (q_norm / k_norm weights; ``1 + w`` is formed in fp32). Returns ``out`` [N, nh * 2d + 2d]
  bf16 in the projection's column order with every ``q_h`` and ``k`` RMS-normed and partially roped, ``gate_h`` /
  ``v`` raw. Rounding as the torch path: bf16 projection output, fp32 norm, one bf16 rounding after ``* (1 + w)``,
  bf16 rope products and sum.
* ``attn_out_tkg(a, wo_t, g=None)``: ``a`` [N, nh * d] bf16 (attention output), ``wo_t`` [nh * d, H] = o_proj^T,
  optional gate ``g`` [N, nh * d]. Returns ``a @ wo_t`` (or ``(a * sigmoid(g)) @ wo_t``: bf16 sigmoid, bf16
  product) [N, H] bf16, fp32 accumulation.

Differences to the CPU torch path: matmul summation order, the hardware rsqrt / sigmoid.
"""

from __future__ import annotations

import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.kernel_assert import kernel_assert
from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info

_P = 128
_CW = 512  # columns per matmul (one PSUM bank of fp32)
MAX_N = 128


def _my_chunks(n_chunks, n_prg, prg):
    per = (n_chunks + n_prg - 1) // n_prg
    lo = min(n_chunks, prg * per)
    return lo, min(n_chunks, lo + per)


def _load_rows_t(dst, src, N, K):
    """dst [128, K / 128, N] <- src [N, K]^T: contiguous row load ([N, K], one run of 2 K bytes per row), then one PE
    transpose per 128 columns (identity as the moving operand; exact for bf16). An element-strided transposing DMA
    would move 2-byte elements (K * N of them)."""
    KT = K // _P
    rows = nl.ndarray((N, K), dtype=src.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=rows, src=src)
    io = nl.ndarray((N, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=io, pattern=[[-1, N]], offset=0, channel_multiplier=1)  # p - i
    ident = nl.ndarray((N, N), dtype=src.dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=ident, data=io, op0=nl.equal, operand0=0.0)
    G = max(1, _CW // N)  # transposes per PSUM bank
    for g0 in range(0, KT, G):
        g = min(G, KT - g0)
        ps = nl.ndarray((_P, G, N), dtype=nl.float32, buffer=nl.psum)
        for j in range(g):
            kt = g0 + j
            nisa.nc_matmul(dst=ps[:, j, :], stationary=rows[:, kt * _P : (kt + 1) * _P], moving=ident,
                           accumulate=False)
        nisa.tensor_copy(dst=dst[:, g0 : g0 + g, :], src=ps[:, 0:g, :])


def _load_w_cols(dst, w, K, M, c0, C):
    """dst [128, K / 128, C] <- w [K, M][:, c0 : c0 + C] (rows of C contiguous elements)."""
    nisa.dma_copy(dst=dst, src=w.ap(pattern=[[M, _P], [_P * M, K // _P], [1, C]], offset=c0))


@nki.jit
def attn_qkv_tkg(x, w_t, cos, sin, q_w, k_w, eps: float = 1e-6, d: int = 256):
    N, H = x.shape
    M = w_t.shape[1]
    r = cos.shape[1]
    half = r // 2
    _, n_prg, prg = get_verified_program_sharding_info("attn_qkv_tkg", (0, 1), 2)
    kernel_assert(N <= MAX_N, f"attn_qkv_tkg: N={N} > {MAX_N}")
    kernel_assert(H % _P == 0 and 2 * d == _CW, "attn_qkv_tkg: H % 128 and 2 d == 512")
    kernel_assert(M % _CW == 0, "attn_qkv_tkg: columns must be heads of q | gate plus k | v")
    kernel_assert(r % 2 == 0 and r <= d, "attn_qkv_tkg: rotary dim")
    KT = H // _P
    n_ch = M // _CW  # nh query chunks (q_h | gate_h) + one k | v chunk
    lo, hi = _my_chunks(n_ch, n_prg, prg)
    out = nl.ndarray((N, M), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    if hi <= lo:
        return out
    C = (hi - lo) * _CW

    x_sb = nl.ndarray((_P, KT, N), dtype=x.dtype, buffer=nl.sbuf)
    _load_rows_t(x_sb, x, N, H)
    w_sb = nl.ndarray((_P, KT, C), dtype=w_t.dtype, buffer=nl.sbuf)
    _load_w_cols(w_sb, w_t, H, M, lo * _CW, C)
    cos_sb = nl.ndarray((N, r), dtype=nl.bfloat16, buffer=nl.sbuf)
    sin_sb = nl.ndarray((N, r), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=cos_sb, src=cos)
    nisa.dma_copy(dst=sin_sb, src=sin)
    nw = nl.ndarray((N, 2, d), dtype=nl.float32, buffer=nl.sbuf)  # 1 + w (q, k) on every row
    for i, w in enumerate((q_w, k_w)):
        raw = nl.ndarray((N, d), dtype=w.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=raw, src=w.ap(pattern=[[0, N], [1, d]], offset=0))
        nisa.tensor_scalar(dst=nw[:, i, :], data=raw, op0=nl.add, operand0=1.0)
    eps_sb = nl.ndarray((N, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(eps_sb, eps)

    for ci in range(hi - lo):
        ch = lo + ci
        ps = nl.ndarray((N, _CW), dtype=nl.float32, buffer=nl.psum)
        for kt in range(KT):
            nisa.nc_matmul(dst=ps, stationary=x_sb[:, kt, :], moving=w_sb[:, kt, ci * _CW : (ci + 1) * _CW],
                           accumulate=(kt > 0))
        y = nl.ndarray((N, _CW), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=y, src=ps)  # bf16 linear output, as torch
        # first half: q_h (ch < n_ch - 1) or k (last chunk) -> RMS norm * (1 + w), then partial rope
        wi = 1 if ch == n_ch - 1 else 0
        sq = nl.ndarray((N, d), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=sq, op=nl.square, data=y[:, 0:d])
        ss = nl.ndarray((N, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=ss, op=nl.add, data=sq, axis=(1,))
        rstd = nl.ndarray((N, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=rstd, op=nl.rsqrt, data=ss, scale=1.0 / d, bias=eps_sb)
        h = nl.ndarray((N, d), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=h, data1=y[:, 0:d], data2=rstd.broadcast(1, d), op=nl.multiply)
        qn = nl.ndarray((N, d), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=qn, data1=h, data2=nw[:, wi, :], op=nl.multiply)
        # rope on [0, r): out = x * cos + rotate_half(x) * sin, rotate_half(x) = (-x[half:], x[:half])
        a = nl.ndarray((N, r), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=a, data1=qn[:, 0:r], data2=cos_sb, op=nl.multiply)
        b = nl.ndarray((N, r), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=b[:, 0:half], data1=qn[:, half:r], data2=sin_sb[:, 0:half], op=nl.multiply)
        nisa.tensor_tensor(dst=b[:, half:r], data1=qn[:, 0:half], data2=sin_sb[:, half:r], op=nl.multiply)
        nisa.tensor_tensor(dst=y[:, 0:half], data1=a[:, 0:half], data2=b[:, 0:half], op=nl.subtract)
        nisa.tensor_tensor(dst=y[:, half:r], data1=a[:, half:r], data2=b[:, half:r], op=nl.add)
        nisa.tensor_copy(dst=y[:, r:d], src=qn[:, r:d])
        nisa.dma_copy(dst=out[:, ch * _CW : (ch + 1) * _CW], src=y)
    return out


@nki.jit
def attn_out_tkg(a, wo_t, g=None):
    N, K = a.shape
    M = wo_t.shape[1]
    _, n_prg, prg = get_verified_program_sharding_info("attn_out_tkg", (0, 1), 2)
    kernel_assert(N <= MAX_N, f"attn_out_tkg: N={N} > {MAX_N}")
    kernel_assert(K % _P == 0 and M % _CW == 0, "attn_out_tkg: K % 128, M % 512")
    KT = K // _P
    lo, hi = _my_chunks(M // _CW, n_prg, prg)
    out = nl.ndarray((N, M), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    if hi <= lo:
        return out
    C = (hi - lo) * _CW
    a_sb = nl.ndarray((_P, KT, N), dtype=a.dtype, buffer=nl.sbuf)
    _load_rows_t(a_sb, a, N, K)
    w_sb = nl.ndarray((_P, KT, C), dtype=wo_t.dtype, buffer=nl.sbuf)
    _load_w_cols(w_sb, wo_t, K, M, lo * _CW, C)
    y = a_sb
    if g is not None:
        g_sb = nl.ndarray((_P, KT, N), dtype=g.dtype, buffer=nl.sbuf)
        _load_rows_t(g_sb, g, N, K)
        sg = nl.ndarray((_P, KT, N), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.activation(dst=sg, op=nl.sigmoid, data=g_sb)  # bf16 sigmoid, as torch
        y = nl.ndarray((_P, KT, N), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=y, data1=a_sb, data2=sg, op=nl.multiply)
    for ci in range(hi - lo):
        ps = nl.ndarray((N, _CW), dtype=nl.float32, buffer=nl.psum)
        for kt in range(KT):
            nisa.nc_matmul(dst=ps, stationary=y[:, kt, :], moving=w_sb[:, kt, ci * _CW : (ci + 1) * _CW],
                           accumulate=(kt > 0))
        o = nl.ndarray((N, _CW), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=o, src=ps)
        nisa.dma_copy(dst=out[:, (lo + ci) * _CW : (lo + ci + 1) * _CW], src=o)
    return out


def kernel_layout(q_w, k_w, v_w, o_w):
    """``w_t`` = [q_proj; k_proj; v_proj]^T [H, nh * 2d + 2d] and ``wo_t`` = o_proj^T [nh * d, H] (torch)."""
    import torch

    return dict(wqkv_t=torch.cat([q_w, k_w, v_w], 0).t().contiguous(), wo_t=o_w.t().contiguous())
