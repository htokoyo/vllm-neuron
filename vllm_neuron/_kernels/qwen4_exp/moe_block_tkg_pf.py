# SPDX-License-Identifier: Apache-2.0
"""``moe_block_tkg_fp8`` with the routed-expert weights pipelined inside one loop body (``moe_block_tkg_pf``).

Same inputs, outputs and arithmetic as ``moe_block_tkg.moe_block_tkg_fp8`` (steps 0-4 and 6 are a copy; the routed sum
is accumulated over the routed local experts in ascending id with one bf16 add each, the scales applied where nkilib's
moe_tkg FP8 ROW applies them), so for the same inputs the output is bit-identical to it (NKI simulator and neuronx-cc
BIRSim of the compiled kernel, TP32 / TP16 / TP8 rank shapes, 0 .. E_L routed experts, 1 .. 32 rows). Only the schedule
of step 5 differs.

Why (a TP16 decode device profile): in ``moe_block_tkg_fp8`` every routed local expert costs about 40 us. Its
weights live in loop-carried SBUF buffers (written by DMA in one ``fori_loop`` body, read in the next region); in the
model NEFF the compiler saves each of them to HBM at the end of the writing body and reloads it in the next one
(``_mem_47_41`` / ``_mem_48_42`` / ``_mem_49_43`` / ``_mem_50_44 ..._SpillSave``: 1,739 save names in the debug info
of the B=1 decode NEFF; 0 in a stand-alone compile of the kernel). Per expert and program that is 2.46 MB of FP8 weight
loads + 2.46 MB saved + 2.46 MB reloaded = the 7.4 MB of DMA packets measured per expert, all on the same queue as the
weight loads. Each ``fori_loop`` region is also a barrier that drains every engine, so the next expert's weights
prefetched in iteration k never overlap iteration k + 1 anyway.

Step 5 here:
- no SBUF tensor is written in one loop region and read in another, except ``acc`` / ``cur_ids`` (written by compute
  engines, as before): the expert weights and scales are allocated inside the loop body;
- one body handles ``experts_per_iter`` (U) routed experts: it issues the weight DMAs of all U experts first and then
  computes them in ascending order, so the loads of expert u + 1 run under the matmuls of expert u (U buffers of
  gate/up [128, H1/2, 2, I] + down [128, I/128, H/2] FP8: 19.2 KB per partition each at H = 2560, I = 640);
- loops: ``n // U`` iterations of the U-expert body, then for r = U-1 .. 1 a 0/1-trip body of r experts
  (n = number of routed local experts, a runtime value; U = 1 is one expert per body);
- fewer, larger weight DMAs: gate/up in ``gu_chunks`` HWDGE DMAs (default 2), down in ``d_chunks`` (default 1);
  with ``dma_spread`` the down DMAs are issued from the Scalar (Activation) engine and the gate/up DMAs from Sync;
- ``scales_once``: the gate/up and down scales of all E_L local experts are loaded once per layer (two static DMAs,
  before the router), and each expert's [128, 2, I/128] / [128, H1/2] scale tile is a dynamic-offset tensor_copy.

Extra (trailing, defaulted) arguments: ``experts_per_iter``, ``d_chunks``, ``dma_spread``, ``scales_once``.
"""

from __future__ import annotations

import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.kernel_assert import kernel_assert
from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info

from vllm_neuron._kernels.qwen4_exp.moe_block_tkg import _NEG, _P, _PAD_ID, _bcast_row, _f8_tile_dtype

_ = _PAD_ID  # same nonzero_with_count padding as moe_block_tkg


@nki.jit
def moe_block_tkg_fp8_pf(
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
    gu_chunks: int = 2,
    experts_per_iter: int = 2,
    d_chunks: int = 1,
    dma_spread: bool = True,
    scales_once: bool = True,
    row_tile: int = 32,
):
    """``moe_block_tkg_fp8`` with the routed experts pipelined in one loop body (see the module docstring)."""
    _, n_prgs, prg = get_verified_program_sharding_info("moe_block_tkg_fp8_pf", (0, 1))
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
    kernel_assert(1 <= T <= 128, f"T={T}: 1..128 rows (router / top-k hold the rows on partitions)")
    kernel_assert(1 <= row_tile <= 32, f"row_tile={row_tile}: 1..32 (PSUM tiles hold row_tile columns per H tile)")
    # Row tiles of the shared / routed expert matmuls: each routed expert's weights are loaded once per call and
    # used for every tile (T <= row_tile: one tile, the instructions of the T <= 32 kernel).
    tiles = []
    for r0 in range(0, T, row_tile):
        tiles.append((r0, min(T, r0 + row_tile) - r0))
    kernel_assert(E <= 512 and E % E_L == 0, f"E={E} must be <= 512 (one fp32 PSUM bank) and a multiple of E_L={E_L}")
    kernel_assert(1 <= top_k <= 16, f"top_k={top_k}: 1..16 (two max8)")
    kernel_assert(H1s % gu_chunks == 0, f"gu_chunks={gu_chunks} must divide H1/n_prgs={H1s}")
    kernel_assert(NI % d_chunks == 0, f"d_chunks={d_chunks} must divide I/128={NI}")
    kernel_assert(1 <= experts_per_iter <= 4 and experts_per_iter <= E_L,
                  f"experts_per_iter={experts_per_iter}: 1..min(4, E_L={E_L})")
    kernel_assert(tuple(gate_up_scale_t.shape) == (E_L, P, 2, NI),
                  f"gate_up_scale_t {tuple(gate_up_scale_t.shape)} must be [E_L, 128, 2, I/128]=({E_L}, {P}, 2, {NI})")
    kernel_assert(tuple(down_scale.shape) == (E_L, H), f"down_scale {tuple(down_scale.shape)} must be [E_L, H]=({E_L}, {H})")
    R = E // E_L  # expert blocks (EP ranks)
    other = 1 - prg
    dt = x.dtype
    U = experts_per_iter

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
    if scales_once:
        # all local experts' scales, once per layer (read-only in the loops): [i0, e, g * NI + i1] and [h0, e, h1]
        gus_all = nl.ndarray((P, E_L, 2 * NI), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=gus_all, src=gate_up_scale_t.ap(pattern=[[2 * NI, P], [P * 2 * NI, E_L], [1, 2 * NI]], offset=0))
        ds_all = nl.ndarray((P, E_L, H1s), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ds_all, src=down_scale.ap(pattern=[[H1s, P], [H, E_L], [1, H1s]], offset=h_off))

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

    # ---- 3. Routed local expert ids (ascending), their count, the loop trip counts ---------------
    nz = nl.ndarray((T, E_L), dtype=dt, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=nz, data=aff, op0=nl.not_equal, operand0=0.0)
    cnt_ps = nl.ndarray((1, E_L), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=cnt_ps, stationary=ones_t, moving=nz, accumulate=False)
    nisa.tensor_copy(dst=routed_sb[0:1, 0:E_L], src=cnt_ps)
    ids_sb = nl.ndarray((P, E_L + 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.nonzero_with_count(src=routed_sb, dst=ids_sb, padding_val=_PAD_ID)
    # n (routed local experts) -> q = n // U (U-expert bodies) and rem = n - U * q, in fp32 (exact: n <= E_L)
    n_f = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=n_f, src=ids_sb[0:1, E_L : E_L + 1])
    J = E_L // U  # q <= J
    mult = nl.ndarray((1, J), dtype=nl.float32, buffer=nl.sbuf)  # U, 2U, .., J * U
    nisa.iota(dst=mult, pattern=[[U, J]], offset=U, channel_multiplier=0)
    le = nl.ndarray((1, J), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=le, data=mult, op0=nl.less_equal, operand0=n_f)
    q_f = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=q_f, op=nl.add, data=le, axis=(1,))
    q_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=q_sb, src=q_f)
    q_reg = nisa.register_alloc()
    nisa.register_load(src=q_sb, dst=q_reg)
    rem_regs = {}
    if U > 1:
        uq = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=uq, data=q_f, op0=nl.multiply, operand0=-float(U))
        rem_f = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=rem_f, data1=n_f, data2=uq, op=nl.add)
        for r in range(U - 1, 0, -1):
            tr_f = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=tr_f, data=rem_f, op0=nl.equal, operand0=float(r))
            tr_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=tr_sb, src=tr_f)
            reg = nisa.register_alloc()
            nisa.register_load(src=tr_sb, dst=reg)
            rem_regs[r] = reg
    cur_ids = nl.ndarray((1, E_L + 1), dtype=nl.int32, buffer=nl.sbuf)  # [0, u] = u-th expert of the current body
    nisa.tensor_copy(src=ids_sb[0:1, :], dst=cur_ids)
    shift_tmp = nl.ndarray((1, E_L), dtype=nl.int32, buffer=nl.sbuf)

    # ---- 4. Shared expert (BF16, as in moe_block_tkg), per row tile -----------------------------
    if with_shared:
        sh_out = nl.ndarray((P, H1s, T), dtype=dt, buffer=nl.sbuf)
        sd_v = sd_sb.reshape_dim(dim=1, shape=(P, H1s))
        for ti in range(len(tiles)):
            r0 = tiles[ti][0]
            rn = tiles[ti][1]
            sh_ps = nl.ndarray((I_s, 3, rn), dtype=nl.float32, buffer=nl.psum)
            for h1 in range(H1s):
                nisa.nc_matmul(dst=sh_ps[:, 0, :], stationary=sgu_sb[:, h1, 0:I_s], moving=x_sb[:, r0 : r0 + rn, h1])
                nisa.nc_matmul(dst=sh_ps[:, 1, :], stationary=sgu_sb[:, h1, I_s : 2 * I_s], moving=x_sb[:, r0 : r0 + rn, h1])
                nisa.nc_matmul(dst=sh_ps[0:1, 2, :], stationary=sgu_sb[:, h1, 2 * I_s : C], moving=x_sb[:, r0 : r0 + rn, h1])
            sh = nl.ndarray((I_s, 3, rn), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(sh[:, 2, :], 0.0)
            nisa.tensor_copy(dst=sh[:, 0:2, :], src=sh_ps[:, 0:2, :])
            nisa.tensor_copy(dst=sh[0:1, 2, :], src=sh_ps[0:1, 2, :])
            if n_prgs > 1:
                sh_rv = nl.ndarray((I_s, 3, rn), dtype=nl.float32, buffer=nl.sbuf)
                nisa.sendrecv(src=sh, dst=sh_rv, send_to_rank=other, recv_from_rank=other, pipe_id=0)
                nisa.tensor_tensor(dst=sh, data1=sh, data2=sh_rv, op=nl.add)
            sg = nl.ndarray((1, rn), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=sg, op=nl.exp, data=sh[0:1, 2, :], scale=-1.0)
            nisa.tensor_scalar(dst=sg, data=sg, op0=nl.add, operand0=1.0)
            nisa.reciprocal(dst=sg, data=sg)
            sg_bc = nl.ndarray((P, rn), dtype=nl.float32, buffer=nl.sbuf)
            _bcast_row(sg_bc, sg)
            nisa.activation(dst=sh[:, 0, :], op=nl.silu, data=sh[:, 0, :], scale=1.0)
            act_s = nl.ndarray((I_s, rn), dtype=dt, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=act_s, data1=sh[:, 0, :], data2=sh[:, 1, :], op=nl.multiply)
            sd_ps = nl.ndarray((P, H1s, rn), dtype=nl.float32, buffer=nl.psum)
            for h1 in range(H1s):
                nisa.nc_matmul(dst=sd_ps[:, h1, :], stationary=sd_v.select(dim=2, index=h1), moving=act_s, accumulate=False)
            nisa.tensor_tensor(dst=sh_out[:, :, r0 : r0 + rn], data1=sd_ps,
                               data2=sg_bc.expand_dim(dim=1).broadcast(dim=1, size=H1s), op=nl.multiply)

    # ---- 5. Routed FP8 experts: U per loop body, all U experts' weights loaded first ----------
    wdt_gu, wdt_d = _f8_tile_dtype(gate_up.dtype), _f8_tile_dtype(down.dtype)
    cs = H1s // gu_chunks
    dcs = NI // d_chunks
    hw = nisa.dge_mode.hwdge

    def eid(pos):
        return cur_ids.ap(pattern=[[E_L + 1, 1], [1, 1]], offset=pos)

    def load_expert(pos):
        """Body-local weight / scale tiles of the expert at cur_ids[0, pos], their DMAs issued now."""
        e_ap = eid(pos)
        wgu = nl.ndarray((P, H1s, 2, I), dtype=wdt_gu, buffer=nl.sbuf)
        wd = nl.ndarray((P, NI, Hs), dtype=wdt_d, buffer=nl.sbuf)
        gus = nl.ndarray((P, 2, NI), dtype=nl.float32, buffer=nl.sbuf)
        ds = nl.ndarray((P, H1s), dtype=nl.float32, buffer=nl.sbuf)
        src = gate_up.select(dim=0, index=e_ap).slice(dim=0, start=h_off, end=h_off + Hs).reshape_dim(dim=0, shape=(P, H1s))
        for c in range(gu_chunks):
            nisa.dma_copy(dst=wgu[:, c * cs : (c + 1) * cs], src=src.slice(dim=1, start=c * cs, end=(c + 1) * cs),
                          dge_mode=hw)
        dsrc = down.select(dim=0, index=e_ap).slice(dim=1, start=h_off, end=h_off + Hs)  # [I, Hs]
        for c in range(d_chunks):
            part = dsrc.slice(dim=0, start=c * dcs * P, end=(c + 1) * dcs * P).reshape_dim(dim=0, shape=(dcs, P))
            if dma_spread:
                nisa.dma_copy(dst=wd[:, c * dcs : (c + 1) * dcs, :], src=part.permute(dims=(1, 0, 2)), dge_mode=hw,
                              engine=nisa.engine.scalar)
            else:
                nisa.dma_copy(dst=wd[:, c * dcs : (c + 1) * dcs, :], src=part.permute(dims=(1, 0, 2)), dge_mode=hw)
        if scales_once:
            # compute-engine dynamic offsets need a uint32 index (ids are >= 0: same bits as the int32 id)
            e_u = cur_ids.ap(pattern=[[E_L + 1, 1], [1, 1]], offset=pos, dtype=nl.uint32)
            nisa.tensor_copy(dst=gus, src=gus_all.ap(pattern=[[E_L * 2 * NI, P], [NI, 2], [1, NI]], offset=0,
                                                     scalar_offset=e_u, indirect_dim=1))
            nisa.tensor_copy(dst=ds, src=ds_all.ap(pattern=[[E_L * H1s, P], [1, H1s]], offset=0,
                                                   scalar_offset=e_u, indirect_dim=1))
        else:
            nisa.dma_copy(dst=gus, src=gate_up_scale_t.select(dim=0, index=e_ap), dge_mode=hw)
            dss = down_scale.select(dim=0, index=e_ap).slice(dim=0, start=h_off, end=h_off + Hs).reshape_dim(dim=0, shape=(P, H1s))
            nisa.dma_copy(dst=ds, src=dss, dge_mode=hw)
        return wgu, wd, gus, ds

    def compute_expert(pos, wts):
        """moe_block_tkg_fp8's expert() on body-local tiles (same instructions, same order of the arithmetic)."""
        wgu, wd, gus, ds = wts
        e_ap = eid(pos)
        aff_row = nl.ndarray((1, T), dtype=dt, buffer=nl.sbuf)
        aff_bc = nl.ndarray((P, T), dtype=dt, buffer=nl.sbuf)
        nisa.dma_copy(dst=aff_row,
                      src=aff_hbm.ap(pattern=[[1, 1], [E_L, T]], offset=prg * T * E_L, scalar_offset=e_ap, indirect_dim=1))
        _bcast_row(aff_bc, aff_row)
        for ti in range(len(tiles)):
            r0 = tiles[ti][0]
            rn = tiles[ti][1]
            gu_ps = nl.ndarray((P, 2, NI, rn), dtype=nl.float32, buffer=nl.psum)
            for h1 in range(H1s):
                for g in range(2):
                    for i1 in range(NI):
                        nisa.nc_matmul(
                            dst=gu_ps[:, g, i1, :],
                            stationary=wgu.select(dim=1, index=h1).select(dim=1, index=g).slice(dim=1, start=i1 * P, end=(i1 + 1) * P),
                            moving=x_sb[:, r0 : r0 + rn, h1],
                        )
            gu_sb = nl.ndarray((P, 2, NI, rn), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=gu_sb, data1=gu_ps, data2=gus.expand_dim(dim=3).broadcast(dim=3, size=rn), op=nl.multiply)
            if n_prgs > 1:
                gu_rv = nl.ndarray((P, 2, NI, rn), dtype=nl.float32, buffer=nl.sbuf)
                nisa.sendrecv(src=gu_sb, dst=gu_rv, send_to_rank=other, recv_from_rank=other, pipe_id=0)
                nisa.tensor_tensor(dst=gu_sb, data1=gu_sb, data2=gu_rv, op=nl.add)
            nisa.activation(dst=gu_sb[:, 0], op=nl.silu, data=gu_sb[:, 0], scale=1.0)
            act = nl.ndarray((P, NI, rn), dtype=dt, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=act, data1=gu_sb[:, 0], data2=gu_sb[:, 1], op=nl.multiply)
            d_ps = nl.ndarray((P, H1s, rn), dtype=nl.float32, buffer=nl.psum)
            for i1 in range(NI):
                wv = wd.select(dim=1, index=i1).reshape_dim(dim=1, shape=(P, H1s))
                for h1 in range(H1s):
                    nisa.nc_matmul(dst=d_ps[:, h1, :], stationary=wv.select(dim=2, index=h1), moving=act[:, i1, :])
            d_bf = nl.ndarray((P, H1s, rn), dtype=dt, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=d_bf, data1=d_ps, data2=ds.expand_dim(dim=2).broadcast(dim=2, size=rn), op=nl.multiply)
            nisa.tensor_tensor(dst=d_bf, data1=d_bf,
                               data2=aff_bc[:, r0 : r0 + rn].expand_dim(dim=1).broadcast(dim=1, size=H1s), op=nl.multiply)
            nisa.tensor_tensor(dst=acc[:, :, r0 : r0 + rn], data1=d_bf, data2=acc[:, :, r0 : r0 + rn], op=nl.add)

    def experts(k):
        wts = [load_expert(u) for u in range(k)]  # all k experts' DMAs first
        for u in range(k):
            compute_expert(u, wts[u])  # ascending expert id: acc order as moe_block_tkg_fp8

    def _main(_):
        experts(U)
        keep = E_L + 1 - U  # cur_ids[0, U:] -> cur_ids[0, :keep]
        nisa.tensor_copy(src=cur_ids[0:1, U : E_L + 1], dst=shift_tmp[0:1, 0:keep])
        nisa.tensor_copy(src=shift_tmp[0:1, 0:keep], dst=cur_ids[0:1, 0:keep])

    nl.fori_loop(0, q_reg, _main)
    for r in range(U - 1, 0, -1):
        def _rem(_, _r=r):
            experts(_r)

        nl.fori_loop(0, rem_regs[r], _rem)

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
