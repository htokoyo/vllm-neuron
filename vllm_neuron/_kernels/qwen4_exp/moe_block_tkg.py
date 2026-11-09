# SPDX-License-Identifier: Apache-2.0
"""Qwen4-Exp decode MoE block in ONE kernel: router -> top-k -> routed local experts -> shared expert.

The decode MoE of a TP32 / EP32 rank (B=1 MTP verify: T = 1 + k <= 4 rows) used to be five
launches per layer: the router (torch: fp32 linear, softmax, top-k, renorm, scatter), the bf16
cast and a second top-k for the expert ids (torch), a routed-expert ``moe_tkg`` launch, and
the sharded shared expert with its sigmoid gate (torch, three linears). This kernel does all of
it before the all-reduce; the caller only all-reduces the result.

Per program (LNC=2: H is split in two halves, ``h_off = prg * H / 2``; both programs run the same
routing, bit-identically):

1. Router: ``logits [T, E]`` = x[:, half] @ Wr[half, :] (bf16 x bf16 products, fp32 PSUM, i.e. an
   exact fp32 dot up to the summation order), one ``sendrecv`` of the partial logits, fp32 add.
   Top-k on the logits (softmax is monotonic): ``max8`` (+ ``nc_match_replace8`` + ``max8`` for
   k > 8). The renormalised affinity of a selected expert is
   ``exp(l_e - l_max) / sum_topk exp(l_j - l_max)`` (the softmax denominator cancels), so no
   softmax over all E experts is computed. The rank's E_L logits are picked with a one-hot over
   the E / E_L rank blocks (``rank_id`` is a runtime input, the NEFF is shared by all ranks).
2. Routing of the local experts: affinities [T, E_L] (bf16, cast once) -> nonzero indicator ->
   per-expert token count (one matmul with a ones column) -> ``nonzero_with_count`` -> the
   ascending ids of the routed local experts and their count ``n``.
3. Shared expert (``with_shared``; intermediate ``I_s`` columns of this rank, the sigmoid gate as
   one more column of the gate/up weight): gate/up/gate-logit matmuls on the H half, one
   ``sendrecv``, SiLU(gate) * up, down projection, times sigmoid(gate logit).
4. Routed experts: three dynamic loops with weight prefetch. The weights of one expert live in two
   full-size SBUF buffers (gate/up [128, H1/2, 2, I], down [128, I/128, H/2]); L0 loads the first
   routed expert (0 or 1 iteration), L1 (n - 1 iterations) computes the current expert and, once
   its gate/up (down) matmuls have read a buffer, loads the NEXT routed expert's gate/up (down)
   weights into it, L2 (0 or 1 iteration) computes the last expert. The arithmetic of one expert
   (and the accumulation over experts in ascending id, one bf16 add each) is that of nkilib's
   all-expert ``moe_tkg``, so for the same affinities the routed sum is bit-identical to it
   (NKI simulator); gate and up share one ``sendrecv``.
5. out[T, H half] = routed (bf16) + shared (bf16), written once.

Inputs (HBM): ``x`` [T, H] bf16; ``router_wt`` [H, E] bf16 (the router weight transposed once at
load); ``gate_up`` [E_L, H, 2, I] and ``down`` [E_L, I, H] (kernel layout, this rank's experts);
``shared_gu`` [H, 2 * I_s + 1] (gate columns, up columns, shared-expert-gate column) and
``shared_down`` [I_s, H] bf16 (this rank's shared-expert columns, transposed once at load);
``rank_id`` [1, 1] int32 (local experts are [rank * E_L, rank * E_L + E_L)).
Returns ``out`` [T, H] bf16 (partial: all-reduce outside); with ``return_affinities`` also the
local affinities [n_programs * T, E_L] bf16 (one copy per program).
"""

from __future__ import annotations

import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.interleave_copy import interleave_copy
from nkilib.core.utils.kernel_assert import kernel_assert
from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info

_P = 128
_PAD_ID = -1  # nonzero_with_count padding value of the routed expert ids
_NEG = -3.0e38  # replaces the first 8 maxima before the second max8


def _bcast_row(dst, src_row):
    """dst [128, F] <- src_row [1, F] (partition 0) on every partition (4 stream shuffles)."""
    nisa.tensor_copy(src=src_row, dst=dst[0:1, :])
    for pg in range(4):
        nisa.nc_stream_shuffle(dst=dst[nl.ds(32 * pg, 32), :], src=dst[0:1, :], shuffle_mask=[0] * 32)


@nki.jit
def moe_block_tkg(
    x,
    router_wt,
    gate_up,
    down,
    shared_gu,
    shared_down,
    rank_id,
    top_k: int = 10,
    with_shared: bool = True,
    return_affinities: bool = False,
    gu_chunks: int = 5,
):
    """See the module docstring. ``gu_chunks``: DMAs per expert gate/up weight load (H1/2 must divide)."""
    _, n_prgs, prg = get_verified_program_sharding_info("moe_block_tkg", (0, 1))
    T, H = x.shape
    E = router_wt.shape[1]
    E_L, _, _, I = gate_up.shape
    P = _P
    H1 = H // P
    kernel_assert(H % P == 0 and H1 % n_prgs == 0, f"H={H} must split into 128-row tiles per program")
    H1s = H1 // n_prgs  # 128-row tiles of H per program
    Hs = H1s * P
    h_off = prg * Hs
    NI = I // P
    kernel_assert(I % P == 0, f"I={I} must be a multiple of 128")
    kernel_assert(1 <= T <= 32, f"T={T}: 1..32 rows (PSUM tiles hold T columns per H tile)")
    kernel_assert(E <= 512 and E % E_L == 0, f"E={E} must be <= 512 (one fp32 PSUM bank) and a multiple of E_L={E_L}")
    kernel_assert(1 <= top_k <= 16, f"top_k={top_k}: 1..16 (two max8)")
    kernel_assert(H1s % gu_chunks == 0, f"gu_chunks={gu_chunks} must divide H1/n_prgs={H1s}")
    R = E // E_L  # expert blocks (EP ranks)
    other = 1 - prg
    dt = x.dtype

    # ---- 0. Loads that do not depend on anything (issued first) --------------------------------
    # x[t, h_off + p * H1s + j] -> x_sb[p, t, j]: the [H0, T, H1_shard] layout of nkilib's moe_tkg.
    x_sb = nl.ndarray((P, T, H1s), dtype=dt, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_sb, src=x.ap(pattern=[[H1s, P], [H, T], [1, H1s]], offset=h_off))
    wr_sb = nl.ndarray((P, H1s, E), dtype=router_wt.dtype, buffer=nl.sbuf)
    half = H1s // 2 if H1s >= 2 else H1s
    for c0, c1 in ((0, half), (half, H1s)) if half < H1s else ((0, H1s),):
        nisa.dma_copy(
            dst=wr_sb[:, c0:c1, :],
            src=router_wt.ap(pattern=[[H1s * E, P], [E, c1 - c0], [1, E]], offset=(h_off + c0) * E),
        )
    rank_i = nl.ndarray((T, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=rank_i, src=rank_id.ap(pattern=[[0, T], [1, 1]], offset=0))
    if with_shared:
        I_s = shared_down.shape[0]
        C = shared_gu.shape[1]
        kernel_assert(C == 2 * I_s + 1 and I_s <= P, f"shared_gu [H, {C}] must be [H, 2 * I_s + 1], I_s={I_s} <= 128")
        sgu_sb = nl.ndarray((P, H1s, C), dtype=shared_gu.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=sgu_sb, src=shared_gu.ap(pattern=[[H1s * C, P], [C, H1s], [1, C]], offset=h_off * C))
        sd_sb = nl.ndarray((I_s, Hs), dtype=shared_down.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=sd_sb, src=shared_down[0:I_s, h_off : h_off + Hs])

    # Constants (no dependencies; scheduled early)
    iota_r = nl.ndarray((T, R), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=iota_r, pattern=[[1, R]], offset=0, channel_multiplier=0)
    ones_t = nl.ndarray((T, 1), dtype=dt, buffer=nl.sbuf)
    nisa.memset(ones_t, 1.0)
    routed_sb = nl.ndarray((P, E_L), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(routed_sb, 0.0)
    acc = nl.ndarray((P, H1s, T), dtype=dt, buffer=nl.sbuf)
    nisa.memset(acc, 0)

    # ---- 1. Router logits (H half per program, partial sums exchanged) -------------------------
    lg_ps = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.psum)
    for h1 in range(H1s):
        nisa.nc_matmul(dst=lg_ps, stationary=x_sb[:, :, h1], moving=wr_sb[:, h1, :], accumulate=(h1 > 0))
    lg = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=lg, src=lg_ps)
    if n_prgs > 1:
        lg_rv = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.sbuf)
        nisa.sendrecv(src=lg, dst=lg_rv, send_to_rank=other, recv_from_rank=other, pipe_id=0)
        nisa.tensor_tensor(dst=lg, data1=lg, data2=lg_rv, op=nl.add)  # p0 + p1 == p1 + p0 (bitwise)

    # ---- 2. Top-k and the local affinities -----------------------------------------------------
    top = nl.ndarray((T, 16), dtype=nl.float32, buffer=nl.sbuf)
    nisa.max8(dst=top[:, 0:8], src=lg)
    if top_k > 8:
        lg_m = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.sbuf)
        nisa.nc_match_replace8(dst=lg_m, data=lg, vals=top[:, 0:8], imm=_NEG)
        nisa.max8(dst=top[:, 8:16], src=lg_m)
    thr = top[:, top_k - 1 : top_k]  # k-th largest logit
    neg_max = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=neg_max, data=top[:, 0:1], op0=nl.multiply, operand0=-1.0)
    ex_top = nl.ndarray((T, top_k), dtype=nl.float32, buffer=nl.sbuf)
    den = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation_reduce(dst=ex_top, op=nl.exp, data=top[:, 0:top_k], reduce_op=nl.add, reduce_res=den, bias=neg_max)
    # This rank's logits: lg[t, r * E_L + e] for r == rank (one-hot over the R blocks, exact).
    rank_f = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=rank_f, src=rank_i)
    onehot = nl.ndarray((T, R), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=onehot, data=iota_r, op0=nl.equal, operand0=rank_f)
    picked = nl.ndarray((T, E_L, R), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=picked,
        data1=lg.reshape_dim(dim=1, shape=(R, E_L)).permute(dims=(0, 2, 1)),
        data2=onehot.expand_dim(dim=1).broadcast(dim=1, size=E_L),
        op=nl.multiply,
    )
    lg_loc = nl.ndarray((T, E_L), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=lg_loc, op=nl.add, data=picked, axis=(2,))
    sel = nl.ndarray((T, E_L), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=sel, data=lg_loc, op0=nl.greater_equal, operand0=thr)
    ex_loc = nl.ndarray((T, E_L), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=ex_loc, op=nl.exp, data=lg_loc, bias=neg_max)
    # renormalised (times 1 / den: divide is rejected as a tensor_scalar / scalar_tensor_tensor op0 by the
    # verifier / ISA validation, NCC_IINAR001, although the simulator accepts it), masked, cast to bf16 once.
    rden = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=rden, data=den)
    aff = nl.ndarray((T, E_L), dtype=dt, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=aff, data=ex_loc, op0=nl.multiply, operand0=rden, op1=nl.multiply, operand1=sel)

    # Per-program copy in HBM: the routed loop reads its expert's row with a dynamic offset.
    aff_hbm = nl.ndarray((n_prgs * T, E_L), dtype=dt, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=aff_hbm[nl.ds(prg * T, T), 0:E_L], src=aff)

    # ---- 3. Routed local expert ids (ascending), their count ------------------------------------
    nz = nl.ndarray((T, E_L), dtype=dt, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=nz, data=aff, op0=nl.not_equal, operand0=0.0)  # routed iff affinity != 0
    cnt_ps = nl.ndarray((1, E_L), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=cnt_ps, stationary=ones_t, moving=nz, accumulate=False)
    nisa.tensor_copy(dst=routed_sb[0:1, 0:E_L], src=cnt_ps)
    ids_sb = nl.ndarray((P, E_L + 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.nonzero_with_count(src=routed_sb, dst=ids_sb, padding_val=_PAD_ID)
    n_first_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)  # min(n, 1)
    nisa.tensor_scalar(dst=n_first_sb, data=ids_sb[0:1, E_L : E_L + 1], op0=nl.minimum, operand0=1)
    n_rest_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)  # max(n - 1, 0)
    nisa.tensor_scalar(dst=n_rest_sb, data=ids_sb[0:1, E_L : E_L + 1], op0=nl.add, operand0=-1,
                       op1=nl.maximum, operand1=0)
    n_first_reg = nisa.register_alloc()
    nisa.register_load(src=n_first_sb, dst=n_first_reg)
    n_rest_reg = nisa.register_alloc()
    nisa.register_load(src=n_rest_sb, dst=n_rest_reg)
    cur_ids = nl.ndarray((1, E_L + 1), dtype=nl.int32, buffer=nl.sbuf)  # [0, 0] = current expert
    nisa.tensor_copy(src=ids_sb[0:1, :], dst=cur_ids)
    shift_tmp = nl.ndarray((1, E_L), dtype=nl.int32, buffer=nl.sbuf)

    # ---- 4. Shared expert ----------------------------------------------------------------------
    if with_shared:
        sh_ps = nl.ndarray((I_s, 3, T), dtype=nl.float32, buffer=nl.psum)
        for h1 in range(H1s):
            # accumulate=None (inferred per region, as nkilib's down projection): the three
            # accumulation groups share one PSUM tile and are interleaved over h1.
            nisa.nc_matmul(dst=sh_ps[:, 0, :], stationary=sgu_sb[:, h1, 0:I_s], moving=x_sb[:, :, h1])
            nisa.nc_matmul(dst=sh_ps[:, 1, :], stationary=sgu_sb[:, h1, I_s : 2 * I_s], moving=x_sb[:, :, h1])
            nisa.nc_matmul(dst=sh_ps[0:1, 2, :], stationary=sgu_sb[:, h1, 2 * I_s : C], moving=x_sb[:, :, h1])
        sh = nl.ndarray((I_s, 3, T), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(sh[:, 2, :], 0.0)  # rows 1.. of the gate-logit slot are sent but unused
        nisa.tensor_copy(dst=sh[:, 0:2, :], src=sh_ps[:, 0:2, :])
        nisa.tensor_copy(dst=sh[0:1, 2, :], src=sh_ps[0:1, 2, :])
        if n_prgs > 1:
            sh_rv = nl.ndarray((I_s, 3, T), dtype=nl.float32, buffer=nl.sbuf)
            nisa.sendrecv(src=sh, dst=sh_rv, send_to_rank=other, recv_from_rank=other, pipe_id=0)
            nisa.tensor_tensor(dst=sh, data1=sh, data2=sh_rv, op=nl.add)
        # sigmoid(g) = 1 / (1 + exp(-g)): the exp table is already loaded for the router.
        sg = nl.ndarray((1, T), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=sg, op=nl.exp, data=sh[0:1, 2, :], scale=-1.0)
        nisa.tensor_scalar(dst=sg, data=sg, op0=nl.add, operand0=1.0)
        nisa.reciprocal(dst=sg, data=sg)
        sg_bc = nl.ndarray((P, T), dtype=nl.float32, buffer=nl.sbuf)
        _bcast_row(sg_bc, sg)
        nisa.activation(dst=sh[:, 0, :], op=nl.silu, data=sh[:, 0, :], scale=1.0)
        act_s = nl.ndarray((I_s, T), dtype=dt, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=act_s, data1=sh[:, 0, :], data2=sh[:, 1, :], op=nl.multiply)
        sd_ps = nl.ndarray((P, H1s, T), dtype=nl.float32, buffer=nl.psum)
        sd_v = sd_sb.reshape_dim(dim=1, shape=(P, H1s))  # [I_s, h0, h1]: h = h0 * H1s + h1
        for h1 in range(H1s):
            nisa.nc_matmul(dst=sd_ps[:, h1, :], stationary=sd_v.select(dim=2, index=h1), moving=act_s, accumulate=False)
        sh_out = nl.ndarray((P, H1s, T), dtype=dt, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=sh_out, data1=sd_ps, data2=sg_bc.expand_dim(dim=1).broadcast(dim=1, size=H1s), op=nl.multiply)

    # ---- 5. Routed experts: L0 load first, L1 compute + prefetch next, L2 compute last ---------
    wgu = nl.ndarray((P, H1s, 2, I), dtype=gate_up.dtype, buffer=nl.sbuf)  # loop-carried weights
    wd = nl.ndarray((P, NI, Hs), dtype=down.dtype, buffer=nl.sbuf)
    gu_sb = nl.ndarray((P, 2, NI, T), dtype=nl.float32, buffer=nl.sbuf)
    gu_rv = nl.ndarray((P, 2, NI, T), dtype=nl.float32, buffer=nl.sbuf)
    act = nl.ndarray((P, NI, T), dtype=dt, buffer=nl.sbuf)
    d_bf = nl.ndarray((P, H1s, T), dtype=dt, buffer=nl.sbuf)
    aff_row = nl.ndarray((1, T), dtype=dt, buffer=nl.sbuf)
    aff_bc = nl.ndarray((P, T), dtype=dt, buffer=nl.sbuf)
    cs = H1s // gu_chunks

    def eid(pos):
        return cur_ids.ap(pattern=[[E_L + 1, 1], [1, 1]], offset=pos)

    def load_gate_up(e_ap):
        src = gate_up.select(dim=0, index=e_ap).slice(dim=0, start=h_off, end=h_off + Hs).reshape_dim(dim=0, shape=(P, H1s))
        for c in range(gu_chunks):
            nisa.dma_copy(dst=wgu[:, c * cs : (c + 1) * cs], src=src.slice(dim=1, start=c * cs, end=(c + 1) * cs),
                          dge_mode=nisa.dge_mode.hwdge)

    def load_down(e_ap):
        src = down.select(dim=0, index=e_ap).slice(dim=1, start=h_off, end=h_off + Hs)
        for i1 in range(NI):
            nisa.dma_copy(dst=wd[:, i1, :], src=src.slice(dim=0, start=i1 * P, end=(i1 + 1) * P),
                          dge_mode=nisa.dge_mode.hwdge)

    def expert(prefetch: bool):
        e_ap = eid(0)
        # this expert's affinities aff[0:T, e] -> [1, T] -> every partition
        nisa.dma_copy(dst=aff_row,
                      src=aff_hbm.ap(pattern=[[1, 1], [E_L, T]], offset=prg * T * E_L, scalar_offset=e_ap, indirect_dim=1))
        _bcast_row(aff_bc, aff_row)
        # gate / up: [I0, T] tiles, fp32 PSUM, summed over this program's H tiles (ascending h1)
        gu_ps = nl.ndarray((P, 2, NI, T), dtype=nl.float32, buffer=nl.psum)
        for h1 in range(H1s):
            for g in range(2):
                for i1 in range(NI):
                    nisa.nc_matmul(
                        dst=gu_ps[:, g, i1, :],
                        stationary=wgu.select(dim=1, index=h1).select(dim=1, index=g).slice(dim=1, start=i1 * P, end=(i1 + 1) * P),
                        moving=x_sb[:, :, h1],
                    )  # accumulate inferred per region (10 groups interleaved over h1)
        if prefetch:
            load_gate_up(eid(1))  # WAR on wgu: after the matmuls above
        nisa.tensor_copy(dst=gu_sb, src=gu_ps)
        if n_prgs > 1:
            nisa.sendrecv(src=gu_sb, dst=gu_rv, send_to_rank=other, recv_from_rank=other, pipe_id=0)
            nisa.tensor_tensor(dst=gu_sb, data1=gu_sb, data2=gu_rv, op=nl.add)
        nisa.activation(dst=gu_sb[:, 0], op=nl.silu, data=gu_sb[:, 0], scale=1.0)
        nisa.tensor_tensor(dst=act, data1=gu_sb[:, 0], data2=gu_sb[:, 1], op=nl.multiply)
        # down: [H0, T] per h1 (h = h0 * H1s + h1), summed over the I tiles (ascending i1)
        d_ps = nl.ndarray((P, H1s, T), dtype=nl.float32, buffer=nl.psum)
        for i1 in range(NI):
            wv = wd.select(dim=1, index=i1).reshape_dim(dim=1, shape=(P, H1s))
            for h1 in range(H1s):
                nisa.nc_matmul(dst=d_ps[:, h1, :], stationary=wv.select(dim=2, index=h1), moving=act[:, i1, :])
        if prefetch:
            load_down(eid(1))
        interleave_copy(index=0, dst=d_bf.flatten_dims(start_dim=1, end_dim=2), src=d_ps.flatten_dims(start_dim=1, end_dim=2))
        nisa.tensor_tensor(dst=d_bf, data1=d_bf, data2=aff_bc.expand_dim(dim=1).broadcast(dim=1, size=H1s), op=nl.multiply)
        nisa.tensor_tensor(dst=acc, data1=d_bf, data2=acc, op=nl.add)

    def shift_ids():
        nisa.tensor_copy(src=cur_ids[0:1, 1 : E_L + 1], dst=shift_tmp[0:1, 0:E_L])
        nisa.tensor_copy(src=shift_tmp[0:1, 0:E_L], dst=cur_ids[0:1, 0:E_L])

    def _l0(_):
        load_gate_up(eid(0))
        load_down(eid(0))

    def _l1(_):
        expert(prefetch=True)
        shift_ids()

    def _l2(_):
        expert(prefetch=False)

    nl.fori_loop(0, n_first_reg, _l0)
    nl.fori_loop(0, n_rest_reg, _l1)
    nl.fori_loop(0, n_first_reg, _l2)

    # ---- 6. out = routed + shared, [P, T, H1s] so each (h0, t) row of H1s elements is contiguous --
    out_sb = nl.ndarray((P, T, H1s), dtype=dt, buffer=nl.sbuf)
    out_v = out_sb.permute(dims=(0, 2, 1))
    if with_shared:
        nisa.tensor_tensor(dst=out_v, data1=acc, data2=sh_out, op=nl.add)
    else:
        nisa.tensor_copy(dst=out_v, src=acc)
    out = nl.ndarray((T, H), dtype=dt, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out.ap(pattern=[[H1s, P], [H, T], [1, H1s]], offset=h_off), src=out_sb)
    if return_affinities:
        return out, aff_hbm
    return out


# ── FP8 routed experts (FP8 checkpoint) ────────────────────────────────────────────────────────
# A separate kernel so that moe_block_tkg above (the BF16 kernel) keeps its exact source. Same
# steps 0-6; only the routed-expert weights differ: ``gate_up`` / ``down`` hold trn2 FP8 e4m3
# codes (torch.float8_e4m3fn, |q| <= 240) with one fp32 scale per output channel (W = q * s),
# the FP8 ROW format of nkilib's moe_tkg (_routed_fp8). The scales are applied exactly where
# nkilib's moe_tkg ROW path applies them, so for the same affinities the routed sum is that of
# nkilib's all-expert moe_tkg FP8 ROW in the NKI simulator:
#   gate / up: per program, PSUM (fp32, H half) * scale[g, i] -> fp32 SBUF, THEN the sendrecv sum
#              (p0 * s + p1 * s, as nkilib: dequant at the PSUM drain, before _post_process_gate_up);
#   down:      PSUM (fp32) * scale[h] -> bf16 in the same instruction, then * affinity (bf16), + acc.
# Weights go to SBUF as nl.float8_e4m3 tiles (nkilib's resolve_fp8_e4m3_dtype(NON_OCP)): the matmul
# is FP8 stationary x bf16 moving, fp32 PSUM, as in nkilib. Extra inputs (HBM, fp32):
#   ``gate_up_scale_t`` [E_L, 128, 2, I / 128]: [e, i0, g, i1] = scale of column g * I + i1 * 128 + i0
#       (the [I0, 2, NI] SBUF layout of the gate/up PSUM tile, transposed once at load: one
#       contiguous 2 * NI * 4-byte row per partition instead of 4-byte element DMAs);
#   ``down_scale`` [E_L, H]: this program reads [h_off, h_off + H/2) as [128, H1s] (h = h0 * H1s + h1).
# Scale buffers are loop-carried like the weights but loaded for the NEXT routed expert only after
# the current expert's PSUM drain has read them (one buffer each; the DMAs are 128 x 40 B).


def _f8_tile_dtype(dtype):
    """SBUF dtype of an FP8 HBM weight (nkilib: float8e4 -> nl.float8_e4m3, trn2 e4m3, max 240)."""
    return nl.float8_e4m3 if str(dtype) == "float8e4" else dtype


@nki.jit
def moe_block_tkg_fp8(
    x,
    router_wt,
    gate_up,
    down,
    gate_up_scale_t,
    down_scale,
    shared_gu,
    shared_down,
    rank_id,
    top_k: int = 10,
    with_shared: bool = True,
    return_affinities: bool = False,
    gu_chunks: int = 5,
):
    """``moe_block_tkg`` with FP8 routed experts + per-output-channel fp32 scales (see the block above)."""
    _, n_prgs, prg = get_verified_program_sharding_info("moe_block_tkg_fp8", (0, 1))
    T, H = x.shape
    E = router_wt.shape[1]
    E_L, _, _, I = gate_up.shape
    P = _P
    H1 = H // P
    kernel_assert(H % P == 0 and H1 % n_prgs == 0, f"H={H} must split into 128-row tiles per program")
    H1s = H1 // n_prgs  # 128-row tiles of H per program
    Hs = H1s * P
    h_off = prg * Hs
    NI = I // P
    kernel_assert(I % P == 0, f"I={I} must be a multiple of 128")
    kernel_assert(1 <= T <= 32, f"T={T}: 1..32 rows (PSUM tiles hold T columns per H tile)")
    kernel_assert(E <= 512 and E % E_L == 0, f"E={E} must be <= 512 (one fp32 PSUM bank) and a multiple of E_L={E_L}")
    kernel_assert(1 <= top_k <= 16, f"top_k={top_k}: 1..16 (two max8)")
    kernel_assert(H1s % gu_chunks == 0, f"gu_chunks={gu_chunks} must divide H1/n_prgs={H1s}")
    kernel_assert(tuple(gate_up_scale_t.shape) == (E_L, P, 2, NI),
                  f"gate_up_scale_t {tuple(gate_up_scale_t.shape)} must be [E_L, 128, 2, I/128]=({E_L}, {P}, 2, {NI})")
    kernel_assert(tuple(down_scale.shape) == (E_L, H), f"down_scale {tuple(down_scale.shape)} must be [E_L, H]=({E_L}, {H})")
    R = E // E_L  # expert blocks (EP ranks)
    other = 1 - prg
    dt = x.dtype

    # ---- 0. Loads that do not depend on anything (issued first) --------------------------------
    x_sb = nl.ndarray((P, T, H1s), dtype=dt, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_sb, src=x.ap(pattern=[[H1s, P], [H, T], [1, H1s]], offset=h_off))
    wr_sb = nl.ndarray((P, H1s, E), dtype=router_wt.dtype, buffer=nl.sbuf)
    half = H1s // 2 if H1s >= 2 else H1s
    for c0, c1 in ((0, half), (half, H1s)) if half < H1s else ((0, H1s),):
        nisa.dma_copy(
            dst=wr_sb[:, c0:c1, :],
            src=router_wt.ap(pattern=[[H1s * E, P], [E, c1 - c0], [1, E]], offset=(h_off + c0) * E),
        )
    rank_i = nl.ndarray((T, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=rank_i, src=rank_id.ap(pattern=[[0, T], [1, 1]], offset=0))
    if with_shared:
        I_s = shared_down.shape[0]
        C = shared_gu.shape[1]
        kernel_assert(C == 2 * I_s + 1 and I_s <= P, f"shared_gu [H, {C}] must be [H, 2 * I_s + 1], I_s={I_s} <= 128")
        sgu_sb = nl.ndarray((P, H1s, C), dtype=shared_gu.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=sgu_sb, src=shared_gu.ap(pattern=[[H1s * C, P], [C, H1s], [1, C]], offset=h_off * C))
        sd_sb = nl.ndarray((I_s, Hs), dtype=shared_down.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=sd_sb, src=shared_down[0:I_s, h_off : h_off + Hs])

    iota_r = nl.ndarray((T, R), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=iota_r, pattern=[[1, R]], offset=0, channel_multiplier=0)
    ones_t = nl.ndarray((T, 1), dtype=dt, buffer=nl.sbuf)
    nisa.memset(ones_t, 1.0)
    routed_sb = nl.ndarray((P, E_L), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(routed_sb, 0.0)
    acc = nl.ndarray((P, H1s, T), dtype=dt, buffer=nl.sbuf)
    nisa.memset(acc, 0)

    # ---- 1. Router logits (H half per program, partial sums exchanged) -------------------------
    lg_ps = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.psum)
    for h1 in range(H1s):
        nisa.nc_matmul(dst=lg_ps, stationary=x_sb[:, :, h1], moving=wr_sb[:, h1, :], accumulate=(h1 > 0))
    lg = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=lg, src=lg_ps)
    if n_prgs > 1:
        lg_rv = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.sbuf)
        nisa.sendrecv(src=lg, dst=lg_rv, send_to_rank=other, recv_from_rank=other, pipe_id=0)
        nisa.tensor_tensor(dst=lg, data1=lg, data2=lg_rv, op=nl.add)

    # ---- 2. Top-k and the local affinities -----------------------------------------------------
    top = nl.ndarray((T, 16), dtype=nl.float32, buffer=nl.sbuf)
    nisa.max8(dst=top[:, 0:8], src=lg)
    if top_k > 8:
        lg_m = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.sbuf)
        nisa.nc_match_replace8(dst=lg_m, data=lg, vals=top[:, 0:8], imm=_NEG)
        nisa.max8(dst=top[:, 8:16], src=lg_m)
    thr = top[:, top_k - 1 : top_k]
    neg_max = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=neg_max, data=top[:, 0:1], op0=nl.multiply, operand0=-1.0)
    ex_top = nl.ndarray((T, top_k), dtype=nl.float32, buffer=nl.sbuf)
    den = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation_reduce(dst=ex_top, op=nl.exp, data=top[:, 0:top_k], reduce_op=nl.add, reduce_res=den, bias=neg_max)
    rank_f = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=rank_f, src=rank_i)
    onehot = nl.ndarray((T, R), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=onehot, data=iota_r, op0=nl.equal, operand0=rank_f)
    picked = nl.ndarray((T, E_L, R), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=picked,
        data1=lg.reshape_dim(dim=1, shape=(R, E_L)).permute(dims=(0, 2, 1)),
        data2=onehot.expand_dim(dim=1).broadcast(dim=1, size=E_L),
        op=nl.multiply,
    )
    lg_loc = nl.ndarray((T, E_L), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=lg_loc, op=nl.add, data=picked, axis=(2,))
    sel = nl.ndarray((T, E_L), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=sel, data=lg_loc, op0=nl.greater_equal, operand0=thr)
    ex_loc = nl.ndarray((T, E_L), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=ex_loc, op=nl.exp, data=lg_loc, bias=neg_max)
    rden = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=rden, data=den)
    aff = nl.ndarray((T, E_L), dtype=dt, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=aff, data=ex_loc, op0=nl.multiply, operand0=rden, op1=nl.multiply, operand1=sel)

    aff_hbm = nl.ndarray((n_prgs * T, E_L), dtype=dt, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=aff_hbm[nl.ds(prg * T, T), 0:E_L], src=aff)

    # ---- 3. Routed local expert ids (ascending), their count ------------------------------------
    nz = nl.ndarray((T, E_L), dtype=dt, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=nz, data=aff, op0=nl.not_equal, operand0=0.0)
    cnt_ps = nl.ndarray((1, E_L), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=cnt_ps, stationary=ones_t, moving=nz, accumulate=False)
    nisa.tensor_copy(dst=routed_sb[0:1, 0:E_L], src=cnt_ps)
    ids_sb = nl.ndarray((P, E_L + 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.nonzero_with_count(src=routed_sb, dst=ids_sb, padding_val=_PAD_ID)
    n_first_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=n_first_sb, data=ids_sb[0:1, E_L : E_L + 1], op0=nl.minimum, operand0=1)
    n_rest_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=n_rest_sb, data=ids_sb[0:1, E_L : E_L + 1], op0=nl.add, operand0=-1,
                       op1=nl.maximum, operand1=0)
    n_first_reg = nisa.register_alloc()
    nisa.register_load(src=n_first_sb, dst=n_first_reg)
    n_rest_reg = nisa.register_alloc()
    nisa.register_load(src=n_rest_sb, dst=n_rest_reg)
    cur_ids = nl.ndarray((1, E_L + 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(src=ids_sb[0:1, :], dst=cur_ids)
    shift_tmp = nl.ndarray((1, E_L), dtype=nl.int32, buffer=nl.sbuf)

    # ---- 4. Shared expert (BF16, as in moe_block_tkg) ------------------------------------------
    if with_shared:
        sh_ps = nl.ndarray((I_s, 3, T), dtype=nl.float32, buffer=nl.psum)
        for h1 in range(H1s):
            nisa.nc_matmul(dst=sh_ps[:, 0, :], stationary=sgu_sb[:, h1, 0:I_s], moving=x_sb[:, :, h1])
            nisa.nc_matmul(dst=sh_ps[:, 1, :], stationary=sgu_sb[:, h1, I_s : 2 * I_s], moving=x_sb[:, :, h1])
            nisa.nc_matmul(dst=sh_ps[0:1, 2, :], stationary=sgu_sb[:, h1, 2 * I_s : C], moving=x_sb[:, :, h1])
        sh = nl.ndarray((I_s, 3, T), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(sh[:, 2, :], 0.0)
        nisa.tensor_copy(dst=sh[:, 0:2, :], src=sh_ps[:, 0:2, :])
        nisa.tensor_copy(dst=sh[0:1, 2, :], src=sh_ps[0:1, 2, :])
        if n_prgs > 1:
            sh_rv = nl.ndarray((I_s, 3, T), dtype=nl.float32, buffer=nl.sbuf)
            nisa.sendrecv(src=sh, dst=sh_rv, send_to_rank=other, recv_from_rank=other, pipe_id=0)
            nisa.tensor_tensor(dst=sh, data1=sh, data2=sh_rv, op=nl.add)
        sg = nl.ndarray((1, T), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=sg, op=nl.exp, data=sh[0:1, 2, :], scale=-1.0)
        nisa.tensor_scalar(dst=sg, data=sg, op0=nl.add, operand0=1.0)
        nisa.reciprocal(dst=sg, data=sg)
        sg_bc = nl.ndarray((P, T), dtype=nl.float32, buffer=nl.sbuf)
        _bcast_row(sg_bc, sg)
        nisa.activation(dst=sh[:, 0, :], op=nl.silu, data=sh[:, 0, :], scale=1.0)
        act_s = nl.ndarray((I_s, T), dtype=dt, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=act_s, data1=sh[:, 0, :], data2=sh[:, 1, :], op=nl.multiply)
        sd_ps = nl.ndarray((P, H1s, T), dtype=nl.float32, buffer=nl.psum)
        sd_v = sd_sb.reshape_dim(dim=1, shape=(P, H1s))
        for h1 in range(H1s):
            nisa.nc_matmul(dst=sd_ps[:, h1, :], stationary=sd_v.select(dim=2, index=h1), moving=act_s, accumulate=False)
        sh_out = nl.ndarray((P, H1s, T), dtype=dt, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=sh_out, data1=sd_ps, data2=sg_bc.expand_dim(dim=1).broadcast(dim=1, size=H1s), op=nl.multiply)

    # ---- 5. Routed FP8 experts: L0 load first, L1 compute + prefetch next, L2 compute last -----
    wgu = nl.ndarray((P, H1s, 2, I), dtype=_f8_tile_dtype(gate_up.dtype), buffer=nl.sbuf)  # loop-carried
    wd = nl.ndarray((P, NI, Hs), dtype=_f8_tile_dtype(down.dtype), buffer=nl.sbuf)
    gus_sb = nl.ndarray((P, 2, NI), dtype=nl.float32, buffer=nl.sbuf)  # loop-carried scales
    ds_sb = nl.ndarray((P, H1s), dtype=nl.float32, buffer=nl.sbuf)
    gu_sb = nl.ndarray((P, 2, NI, T), dtype=nl.float32, buffer=nl.sbuf)
    gu_rv = nl.ndarray((P, 2, NI, T), dtype=nl.float32, buffer=nl.sbuf)
    act = nl.ndarray((P, NI, T), dtype=dt, buffer=nl.sbuf)
    d_bf = nl.ndarray((P, H1s, T), dtype=dt, buffer=nl.sbuf)
    aff_row = nl.ndarray((1, T), dtype=dt, buffer=nl.sbuf)
    aff_bc = nl.ndarray((P, T), dtype=dt, buffer=nl.sbuf)
    cs = H1s // gu_chunks

    def eid(pos):
        return cur_ids.ap(pattern=[[E_L + 1, 1], [1, 1]], offset=pos)

    def load_gate_up(e_ap):
        src = gate_up.select(dim=0, index=e_ap).slice(dim=0, start=h_off, end=h_off + Hs).reshape_dim(dim=0, shape=(P, H1s))
        for c in range(gu_chunks):
            nisa.dma_copy(dst=wgu[:, c * cs : (c + 1) * cs], src=src.slice(dim=1, start=c * cs, end=(c + 1) * cs),
                          dge_mode=nisa.dge_mode.hwdge)

    def load_down(e_ap):
        src = down.select(dim=0, index=e_ap).slice(dim=1, start=h_off, end=h_off + Hs)
        for i1 in range(NI):
            nisa.dma_copy(dst=wd[:, i1, :], src=src.slice(dim=0, start=i1 * P, end=(i1 + 1) * P),
                          dge_mode=nisa.dge_mode.hwdge)

    def load_gu_scale(e_ap):
        nisa.dma_copy(dst=gus_sb, src=gate_up_scale_t.select(dim=0, index=e_ap), dge_mode=nisa.dge_mode.hwdge)

    def load_d_scale(e_ap):
        src = down_scale.select(dim=0, index=e_ap).slice(dim=0, start=h_off, end=h_off + Hs).reshape_dim(dim=0, shape=(P, H1s))
        nisa.dma_copy(dst=ds_sb, src=src, dge_mode=nisa.dge_mode.hwdge)

    def expert(prefetch: bool):
        e_ap = eid(0)
        nisa.dma_copy(dst=aff_row,
                      src=aff_hbm.ap(pattern=[[1, 1], [E_L, T]], offset=prg * T * E_L, scalar_offset=e_ap, indirect_dim=1))
        _bcast_row(aff_bc, aff_row)
        gu_ps = nl.ndarray((P, 2, NI, T), dtype=nl.float32, buffer=nl.psum)
        for h1 in range(H1s):
            for g in range(2):
                for i1 in range(NI):
                    nisa.nc_matmul(
                        dst=gu_ps[:, g, i1, :],
                        stationary=wgu.select(dim=1, index=h1).select(dim=1, index=g).slice(dim=1, start=i1 * P, end=(i1 + 1) * P),
                        moving=x_sb[:, :, h1],
                    )
        if prefetch:
            load_gate_up(eid(1))  # WAR on wgu: after the matmuls above
        # dequant at the PSUM drain (per program, before the sendrecv sum: nkilib's ROW order)
        nisa.tensor_tensor(dst=gu_sb, data1=gu_ps, data2=gus_sb.expand_dim(dim=3).broadcast(dim=3, size=T), op=nl.multiply)
        if prefetch:
            load_gu_scale(eid(1))  # WAR on gus_sb: after the drain above
        if n_prgs > 1:
            nisa.sendrecv(src=gu_sb, dst=gu_rv, send_to_rank=other, recv_from_rank=other, pipe_id=0)
            nisa.tensor_tensor(dst=gu_sb, data1=gu_sb, data2=gu_rv, op=nl.add)
        nisa.activation(dst=gu_sb[:, 0], op=nl.silu, data=gu_sb[:, 0], scale=1.0)
        nisa.tensor_tensor(dst=act, data1=gu_sb[:, 0], data2=gu_sb[:, 1], op=nl.multiply)
        d_ps = nl.ndarray((P, H1s, T), dtype=nl.float32, buffer=nl.psum)
        for i1 in range(NI):
            wv = wd.select(dim=1, index=i1).reshape_dim(dim=1, shape=(P, H1s))
            for h1 in range(H1s):
                nisa.nc_matmul(dst=d_ps[:, h1, :], stationary=wv.select(dim=2, index=h1), moving=act[:, i1, :])
        if prefetch:
            load_down(eid(1))
        # PSUM * row scale -> bf16 (one instruction, as nkilib's ROW down drain), * affinity, + acc
        nisa.tensor_tensor(dst=d_bf, data1=d_ps, data2=ds_sb.expand_dim(dim=2).broadcast(dim=2, size=T), op=nl.multiply)
        if prefetch:
            load_d_scale(eid(1))  # WAR on ds_sb: after the drain above
        nisa.tensor_tensor(dst=d_bf, data1=d_bf, data2=aff_bc.expand_dim(dim=1).broadcast(dim=1, size=H1s), op=nl.multiply)
        nisa.tensor_tensor(dst=acc, data1=d_bf, data2=acc, op=nl.add)

    def shift_ids():
        nisa.tensor_copy(src=cur_ids[0:1, 1 : E_L + 1], dst=shift_tmp[0:1, 0:E_L])
        nisa.tensor_copy(src=shift_tmp[0:1, 0:E_L], dst=cur_ids[0:1, 0:E_L])

    def _l0(_):
        load_gate_up(eid(0))
        load_gu_scale(eid(0))
        load_down(eid(0))
        load_d_scale(eid(0))

    def _l1(_):
        expert(prefetch=True)
        shift_ids()

    def _l2(_):
        expert(prefetch=False)

    nl.fori_loop(0, n_first_reg, _l0)
    nl.fori_loop(0, n_rest_reg, _l1)
    nl.fori_loop(0, n_first_reg, _l2)

    # ---- 6. out = routed + shared ---------------------------------------------------------------
    out_sb = nl.ndarray((P, T, H1s), dtype=dt, buffer=nl.sbuf)
    out_v = out_sb.permute(dims=(0, 2, 1))
    if with_shared:
        nisa.tensor_tensor(dst=out_v, data1=acc, data2=sh_out, op=nl.add)
    else:
        nisa.tensor_copy(dst=out_v, src=acc)
    out = nl.ndarray((T, H), dtype=dt, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out.ap(pattern=[[H1s, P], [H, T], [1, H1s]], offset=h_off), src=out_sb)
    if return_affinities:
        return out, aff_hbm
    return out
