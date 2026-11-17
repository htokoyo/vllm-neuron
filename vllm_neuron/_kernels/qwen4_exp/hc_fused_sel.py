# SPDX-License-Identifier: Apache-2.0
"""``hc_cte_sel``: :func:`hc_fused.hc_cte` on rows chosen at run time (the hyper-connection token split).

See the comment above the kernel. Shapes, layouts and rounding are hc_fused's (its module docstring).
"""

from __future__ import annotations

import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.kernel_assert import kernel_assert
from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info

from .hc_fused import _P, _ceil_div, _identity, _r_tiles

# ─────────────────────────────────────────────────────────────────────────────
# hc_cte on rows chosen at run time (the hyper-connection token split, model/qwen4_exp/layers.py _hcs_forward)
# ─────────────────────────────────────────────────────────────────────────────
# (A new file, so that hc_fused.py and the graphs that call hc_cte are unchanged.) With the token split each rank of the
# attention group computes the read for its own rows of the step. Selecting them in XLA (index_select of a [m, D] block by the rank tensor) lowered to a
# gather of 640 dynamic DMAs + ~2,600 GpSimd instructions per call at T = 2,048 (CPU compile, 2026-11-15); here the
# kernel reads them itself: the same tile loads as hc_cte, each 64-row granule from a row id held in SBUF (the
# dynamic-select DMA of moe_block_tkg_pf). Everything after the loads is hc_cte's code, line for line, so the
# outputs equal hc_cte's on the same rows bit for bit.
_SEL_G = 64


@nki.jit
def hc_cte_sel(x, gid, wd_t, wup_t, w1, inj, eps: float = 1e-6, hc: int = 4, blk: int = 512, round_up: bool = True):
    """hc_cte on rows chosen at run time. ``x`` [G, 64, D]: the streams in granules of _SEL_G = 64 rows; ``gid`` [1, K]
    int32: K granule ids (each in [0, G)). Computes the N = 64 * K rows x[gid[0]], x[gid[1]], ... (in that order) with
    exactly hc_cte's instructions; only the token-tile loads differ (one dynamic-offset DMA per granule, the id read
    from SBUF). N must be a multiple of 64 * (programs), so that every program's rows, block and tile starts are whole
    granules. Returns ``(mixed [N, H], inject [N, hc])``. Other inputs as hc_cte."""
    G, GR, D = x.shape
    K = gid.shape[1]
    N = K * _SEL_G
    R = wup_t.shape[0]
    H = D // hc
    _, n_prg, prg = get_verified_program_sharding_info("hc_cte_sel", (0, 1), 2)
    kernel_assert(H % _P == 0, "hc_cte: H % 128")
    kernel_assert(GR == _SEL_G and tuple(gid.shape) == (1, K), "hc_cte_sel: x [G, 64, D], gid [1, K]")
    kernel_assert(N % (_SEL_G * n_prg) == 0, "hc_cte_sel: rows per program not whole granules")
    S = H // _P
    kernel_assert(tuple(w1.shape) == (_P, hc, S), f"hc_cte: w1 {w1.shape}")
    kernel_assert(tuple(inj.shape) == (_P, hc, S, hc), "hc_cte: inj layout")
    kernel_assert(blk % _P == 0 and blk <= 512, "hc_cte: blk")
    rts = _r_tiles(R)
    NRT = len(rts)
    RW = R + hc  # down | inject columns of the token-stationary down matmul

    n_per = _ceil_div(N, n_prg)
    t_lo = min(N, prg * n_per)
    t_hi = min(N, t_lo + n_per)

    mixed = nl.ndarray((N, H), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    w_out = nl.ndarray((N, hc), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    ident = nl.ndarray((_P, _P), dtype=nl.bfloat16, buffer=nl.sbuf)
    _identity(ident, _P)
    ident_f = nl.ndarray((_P, _P), dtype=nl.float32, buffer=nl.sbuf)
    _identity(ident_f, _P)
    quarter = nl.ndarray((_P, _P), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=quarter, data=ident, op0=nl.multiply, operand0=1.0 / hc)
    ones = nl.ndarray((_P, _P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(ones, 1.0)
    eps_sb = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(eps_sb, eps)
    w1_raw = nl.ndarray((_P, hc, S), dtype=w1.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=w1_raw, src=w1)
    w1_sb = nl.ndarray((_P, hc, S), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=w1_sb, data=w1_raw, op0=nl.add, operand0=1.0)  # 1 + w in fp32
    inj_sb = nl.ndarray((_P, hc, S, hc), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=inj_sb, src=inj)
    gid_sb = nl.ndarray((1, K), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=gid_sb, src=gid)

    for b0 in range(t_lo, t_hi, blk):
        nb = min(blk, t_hi - b0)
        tiles = [(t, b0 + t * _P, min(_P, nb - t * _P)) for t in range(_ceil_div(nb, _P))]

        # ---- per token tile: sum of squares (token layout), rstd broadcast, x^T, normed^T ---------
        nrm = nl.ndarray((_P, hc, S, blk), dtype=nl.bfloat16, buffer=nl.sbuf)  # becomes read * normed
        for t, r0, nt in tiles:
            x_sb = nl.ndarray((_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            for k in range(nt // _SEL_G):  # r0 and nt are whole granules (kernel_assert above)
                g_ap = gid_sb.ap(pattern=[[K, 1], [1, 1]], offset=r0 // _SEL_G + k)
                nisa.dma_copy(dst=x_sb[k * _SEL_G : (k + 1) * _SEL_G, :], src=x.select(dim=0, index=g_ap),
                              dge_mode=nisa.dge_mode.hwdge)
            ss = nl.ndarray((_P, hc), dtype=nl.float32, buffer=nl.sbuf)
            sq_junk = nl.ndarray((_P, H), dtype=nl.bfloat16, buffer=nl.sbuf)
            for g in range(hc):
                nisa.activation_reduce(dst=sq_junk[0:nt, :], op=nl.square, data=x_sb[0:nt, g * H : (g + 1) * H],
                                       reduce_op=nl.add, reduce_res=ss[0:nt, g : g + 1])
            rs = nl.ndarray((_P, hc), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=rs[0:nt, :], op=nl.rsqrt, data=ss[0:nt, :], scale=1.0 / H, bias=eps_sb[0:nt, :])
            # rstd_b[m, g, n] = rs[n, g] for every partition m: ones^T @ diag(rs[:, g]) (fp32, exact)
            dg = nl.ndarray((_P, hc, _P), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=dg[0:nt, :, 0:nt], data1=ident_f[0:nt, 0:nt].expand_dim(1).broadcast(1, hc),
                               data2=rs[0:nt, :].expand_dim(2).broadcast(2, nt), op=nl.multiply)
            rb_ps = nl.ndarray((_P, hc, _P), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=rb_ps[:, :, 0:nt], stationary=ones[0:nt, :], moving=dg[0:nt, :, 0:nt],
                           accumulate=False)
            rstd_b = nl.ndarray((_P, hc, _P), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=rstd_b[:, :, 0:nt], src=rb_ps[:, :, 0:nt])

            xv = x_sb.reshape((_P, hc, _P, S))  # [n, g, p, j]
            for g in range(hc):
                for j0 in range(0, S, 4):
                    xT_ps = nl.ndarray((_P, 4, _P), dtype=nl.float32, buffer=nl.psum)
                    for jj in range(4):
                        nisa.nc_matmul(dst=xT_ps[:, jj, 0:nt], stationary=xv[0:nt, g, :, j0 + jj],
                                       moving=ident[0:nt, 0:nt], accumulate=False)
                    for jj in range(4):
                        nisa.scalar_tensor_tensor(
                            dst=nrm[:, g, j0 + jj, t * _P : t * _P + nt], data=xT_ps[:, jj, 0:nt],
                            op0=nl.multiply, operand0=w1_sb[:, g, j0 + jj : j0 + jj + 1],
                            op1=nl.multiply, operand1=rstd_b[:, g, 0:nt])

        # ---- down | inject (token stationary, weights moving), per token tile ------------------------
        # moving = [Wd^T rows | inject^T rows] of one slice in one SBUF tile: one matmul per (slice, tile)
        d_ps = [nl.ndarray((_P, RW), dtype=nl.float32, buffer=nl.psum) for _ in tiles]
        for g in range(hc):
            wd_sb = nl.ndarray((_P, S, RW), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=wd_sb[:, :, 0:R], src=wd_t.ap(pattern=[[S * R, _P], [R, S], [1, R]], offset=g * H * R))
            nisa.tensor_copy(dst=wd_sb[:, :, R:RW], src=inj_sb[:, g])
            for j in range(S):
                s = g * S + j
                for t, r0, nt in tiles:
                    nisa.nc_matmul(dst=d_ps[t][0:nt, :], stationary=nrm[:, g, j, t * _P : t * _P + nt],
                                   moving=wd_sb[:, j, :], accumulate=(s > 0))

        # ---- h = silu(bf16(d) / hc) -> h^T; inject = 2 sigmoid(bf16(logit) / hc) -> HBM ---------------
        hT = nl.ndarray((_P, NRT, blk), dtype=nl.bfloat16, buffer=nl.sbuf)
        for t, r0, nt in tiles:
            d_bf = nl.ndarray((_P, RW), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=d_bf[0:nt, :], src=d_ps[t][0:nt, :])
            h_tok = nl.ndarray((_P, R), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.activation(dst=h_tok[0:nt, :], op=nl.silu, data=d_bf[0:nt, 0:R], scale=1.0 / hc)
            wsig = nl.ndarray((_P, hc), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=wsig[0:nt, :], op=nl.sigmoid, data=d_bf[0:nt, R:RW], scale=1.0 / hc)
            w_bf = nl.ndarray((_P, hc), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=w_bf[0:nt, :], data=wsig[0:nt, :], op0=nl.multiply, operand0=2.0)
            nisa.dma_copy(dst=w_out[r0 : r0 + nt, :], src=w_bf[0:nt, :])
            hT_ps = nl.ndarray((_P, NRT, _P), dtype=nl.float32, buffer=nl.psum)
            for rt, rows in rts:
                nisa.nc_matmul(dst=hT_ps[0:rows, rt, 0:nt], stationary=h_tok[0:nt, rt * _P : rt * _P + rows],
                               moving=ident[0:nt, 0:nt], accumulate=False)
            for rt, rows in rts:
                nisa.tensor_copy(dst=hT[0:rows, rt, t * _P : t * _P + nt], src=hT_ps[0:rows, rt, 0:nt])

        # ---- up (weights stationary, stride-S feature columns) -> read -> read * normed, in place -------
        for g in range(hc):
            wup_sb = nl.ndarray((_P, NRT, H), dtype=nl.bfloat16, buffer=nl.sbuf)
            for rt, rows in rts:
                nisa.dma_copy(dst=wup_sb[0:rows, rt, :], src=wup_t[rt * _P : rt * _P + rows, g * H : (g + 1) * H])
            wv = wup_sb.reshape((_P, NRT, _P, S))
            for j in range(S):
                up_ps = nl.ndarray((_P, blk), dtype=nl.float32, buffer=nl.psum)
                for rt, rows in rts:
                    nisa.nc_matmul(dst=up_ps[:, 0:nb], stationary=wv[0:rows, rt, :, j], moving=hT[0:rows, rt, 0:nb],
                                   accumulate=(rt > 0))
                rd = nl.ndarray((_P, blk), dtype=nl.bfloat16, buffer=nl.sbuf)
                if round_up:  # bf16 up output before the sigmoid, as torch (one more PSUM pass)
                    nisa.tensor_copy(dst=rd[:, 0:nb], src=up_ps[:, 0:nb])
                    nisa.activation(dst=rd[:, 0:nb], op=nl.sigmoid, data=rd[:, 0:nb])
                else:
                    nisa.activation(dst=rd[:, 0:nb], op=nl.sigmoid, data=up_ps[:, 0:nb])
                # read * normed on GpSimd (SBUF operands): keeps the Vector engine for the PSUM-side work
                nisa.tensor_tensor(dst=nrm[:, g, j, 0:nb], data1=rd[:, 0:nb], data2=nrm[:, g, j, 0:nb], op=nl.multiply,
                                   engine=nisa.engine.gpsimd)

        # ---- mixed[n, S p + j] = sum_g prod / hc, token layout, per token tile ------------------------
        for t, r0, nt in tiles:
            mixed_sb = nl.ndarray((_P, H), dtype=nl.bfloat16, buffer=nl.sbuf)
            mv = mixed_sb.reshape((_P, _P, S))
            for j in range(S):
                # one PSUM tile per accumulation group (see hc_tkg's up: groups sharing a tile lose terms on device)
                mix_ps = nl.ndarray((_P, _P), dtype=nl.float32, buffer=nl.psum)
                for g in range(hc):
                    nisa.nc_matmul(dst=mix_ps[0:nt, :], stationary=nrm[:, g, j, t * _P : t * _P + nt],
                                   moving=quarter, accumulate=(g > 0))
                nisa.tensor_copy(dst=mv[0:nt, :, j], src=mix_ps[0:nt, :])
            nisa.dma_copy(dst=mixed[r0 : r0 + nt, :], src=mixed_sb[0:nt, :])
    return mixed, w_out
