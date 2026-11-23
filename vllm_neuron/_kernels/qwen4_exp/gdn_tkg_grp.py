# SPDX-License-Identifier: Apache-2.0
"""``gdn_tkg_grp``: one-token gated delta-rule decode with the per-head work vectorized over all heads of a core.

Same contract and math as :func:`gdn_tkg.gdn_tkg` (q / k RAW and GQA-expanded, beta = sigmoid(b),
g = -exp(A_log) * softplus(a + dt_bias), l2norm(q, k), RMSNormGated output with a selectable gate), different
schedule. ``gdn_tkg`` loops over the heads of its core (Python-unrolled), and every head runs ~20 small instructions on
one partition ([1, V] vectors) plus two activation-table reloads; it also keeps the fp32 state of every head and a
bf16 copy in SBUF, which spills at 64 rows at TP=8 (192 heads per core; 101.6 MB of HBM traffic per call vs 12.8 MB at
32 rows). Here, after the grouped delta rule of a GDN decode block written for Qwen3.5 on Trainium:

* every per-head scalar lives in a column of a [128, J] tile (K = V = 128 on partitions, J = the core's heads), so
  l2norm, the gates, delta and the RMSNorm are a handful of instructions over all heads;
* per head ONE matmul gives both k^T S and q^T S ([2, V] moving = [k | q] columns, S_bf stationary): with
  S1 = S * eg, kv = eg * (k^T S), and out = q^T S_new = eg * (q^T S) + (q . k) * delta, so S is read once before the
  update;
* the state updates S_new = eg * S + k (x) delta run four heads per matmul (k rows stationary, delta rows times a
  block-diagonal mask moving, [128, 4 * 128] into PSUM) plus two vector ops;
* heads are processed in groups of ``G`` (default 16): the group's fp32 state and its bf16 copy are the only state in
  SBUF (G * 128 * 4 + G * 128 * 2 = 12 KB per partition at G = 16), streamed in and out per group, so the SBUF use
  does not grow with the row count.

Output layout and dtypes as ``gdn_tkg``: out [BH, V] bf16, state_out [BH, K, V] fp32. Numerics: kv and out are formed
from k^T S (bf16 S, fp32 accumulate) times eg instead of k^T (S * eg); the RMSNorm sum of squares runs on the
TensorEngine over bf16 squares. Not bit-identical to ``gdn_tkg``; checked against an fp64
reference (cosine >= 0.9999 on out and state) in the NKI simulator.
"""

import math

import nki
import nki.isa as nisa
import nki.language as nl

from nkilib.core.utils.kernel_assert import kernel_assert

F32, BF = nl.float32, nl.bfloat16


def _sb(shape, dt=F32):
    return nl.ndarray(shape, dtype=dt, buffer=nl.sbuf)


def _identity(dst, n):
    io = nl.ndarray((n, n), dtype=F32, buffer=nl.sbuf)
    nisa.iota(dst=io, pattern=[[-1, n]], offset=0, channel_multiplier=1)
    nisa.tensor_scalar(dst=dst, data=io, op0=nl.equal, operand0=0.0)


@nki.jit
def gdn_tkg_grp(q, k, v, b, a, A_log, dt_bias, z, norm_weight, state_in, gate_activation: str = "silu",
                group: int = 16):
    """See the module docstring. ``group``: heads per streamed state group (trace-time constant)."""
    kernel_assert(gate_activation in ("silu", "sigmoid"), f"unsupported gate_activation {gate_activation!r}")
    gate_op = nl.sigmoid if gate_activation == "sigmoid" else nl.silu
    BH, K = q.shape
    _, V = v.shape
    HV = A_log.shape[0]
    B = BH // HV
    kernel_assert(K == 128 and V == 128, f"K and V must be 128, got {K=}, {V=}")
    kernel_assert(state_in.shape == (BH, K, V), f"state_in must be [BH, K, V], got {state_in.shape}")
    n_prgs = nl.num_programs(0)
    prg = nl.program_id(0)
    J0 = BH // n_prgs
    h0 = prg * J0
    J = BH - h0 if prg == n_prgs - 1 else J0
    kernel_assert(J <= 512, f"{J} heads per core: at most 512 (one PSUM bank of fp32 columns)")
    G = min(group, J)

    out = nl.ndarray((BH, V), dtype=BF, buffer=nl.shared_hbm)
    state_out = nl.ndarray((BH, K, V), dtype=F32, buffer=nl.shared_hbm)

    ident = _sb((128, 128), BF)
    _identity(ident, 128)
    ones = _sb((128, 128), BF)
    nisa.memset(ones, 1.0)

    # ---- per-head vectors, heads on columns: [128 (k or v), J] ----
    qT = _sb((128, J), BF)
    kT = _sb((128, J), BF)
    vT = _sb((128, J))
    zT = _sb((128, J))
    nisa.dma_copy(dst=qT, src=q.ap(pattern=[[1, K], [K, J]], offset=h0 * K))
    nisa.dma_copy(dst=kT, src=k.ap(pattern=[[1, K], [K, J]], offset=h0 * K))
    vtb = _sb((128, J), BF)
    nisa.dma_copy(dst=vtb, src=v.ap(pattern=[[1, V], [V, J]], offset=h0 * V))
    nisa.tensor_copy(dst=vT, src=vtb)
    ztb = _sb((128, J), BF)
    nisa.dma_copy(dst=ztb, src=z.ap(pattern=[[1, V], [V, J]], offset=h0 * V))
    nisa.activation(dst=zT, data=ztb, op=gate_op)
    # gates, broadcast to all partitions: column j = head h0 + j (v-head (h0 + j) % HV)
    braw = _sb((128, J))
    araw = _sb((128, J))
    nisa.dma_copy(dst=braw, src=b.ap(pattern=[[0, 128], [1, J]], offset=h0))
    nisa.dma_copy(dst=araw, src=a.ap(pattern=[[0, 128], [1, J]], offset=h0))
    alf = _sb((128, B * HV))
    dtf = _sb((128, B * HV))
    nisa.dma_copy(dst=alf, src=A_log.ap(pattern=[[0, 128], [0, B], [1, HV]], offset=0))
    nisa.dma_copy(dst=dtf, src=dt_bias.ap(pattern=[[0, 128], [0, B], [1, HV]], offset=0))
    nwc = _sb((128, 1))
    nisa.dma_copy(dst=nwc, src=norm_weight.ap(pattern=[[1, 128], [1, 1]], offset=0))

    beta = _sb((128, J))
    nisa.activation(dst=beta, data=braw, op=nl.sigmoid)
    t1 = _sb((128, J))
    nisa.tensor_tensor(dst=t1, data1=araw, data2=dtf[:, h0:h0 + J], op=nl.add)
    nisa.activation(dst=t1, data=t1, op=nl.softplus)
    ea = _sb((128, J))
    nisa.activation(dst=ea, data=alf[:, h0:h0 + J], op=nl.exp)
    gg = _sb((128, J))
    nisa.scalar_tensor_tensor(dst=gg, data=t1, op0=nl.multiply, operand0=-1.0, op1=nl.multiply, operand1=ea)
    eg = _sb((128, J))
    nisa.activation(dst=eg, data=gg, op=nl.exp)

    # ---- l2norm q, k over the partitions (column sums by a ones matmul) ----
    sq = _sb((128, 2 * J), BF)
    nisa.activation(dst=sq[:, 0:J], data=qT, op=nl.square)
    nisa.activation(dst=sq[:, J:2 * J], data=kT, op=nl.square)
    ss = _sb((128, 2 * J))
    for c0 in range(0, 2 * J, 512):
        cn = min(512, 2 * J - c0)
        ps = nl.ndarray((128, cn), dtype=F32, buffer=nl.psum)
        nisa.nc_matmul(ps, ones, sq[:, c0:c0 + cn])
        nisa.tensor_copy(dst=ss[:, c0:c0 + cn], src=ps)
    rs = _sb((128, 2 * J))
    nisa.activation(dst=rs, data=ss, op=nl.rsqrt, bias=1e-6)
    qn = _sb((128, J))
    kn = _sb((128, J))
    nisa.scalar_tensor_tensor(dst=qn, data=qT, op0=nl.multiply, operand0=1.0 / math.sqrt(K), op1=nl.multiply,
                              operand1=rs[:, 0:J])
    nisa.tensor_tensor(dst=kn, data1=kT, data2=rs[:, J:2 * J], op=nl.multiply)
    KQ = _sb((128, J, 2), BF)  # per head: [k | q] moving columns
    nisa.tensor_copy(dst=KQ[:, :, 0], src=kn)
    nisa.tensor_copy(dst=KQ[:, :, 1], src=qn)
    kbf = _sb((128, J), BF)
    nisa.tensor_copy(dst=kbf, src=kn)
    pqk = _sb((128, J), BF)
    nisa.tensor_tensor(dst=pqk, data1=qn, data2=kn, op=nl.multiply)
    qkd = _sb((128, J))
    for c0 in range(0, J, 512):
        cn = min(512, J - c0)
        ps = nl.ndarray((128, cn), dtype=F32, buffer=nl.psum)
        nisa.nc_matmul(ps, ones, pqk[:, c0:c0 + cn])
        nisa.tensor_copy(dst=qkd[:, c0:c0 + cn], src=ps)
    # block-diagonal mask for the grouped rank-1 update: mk[i, j * 128 + v] = (i == j)
    mk = _sb((G, G * 128), BF)
    nisa.tensor_copy(dst=mk, src=ident.ap(pattern=[[128, G], [1, G], [0, 128]], offset=0))

    kq = _sb((128, J, 2))  # per head: [k^T S | q^T S] over v (partitions)
    o = _sb((128, J))
    delta = _sb((128, J))
    for g0 in range(0, J, G):
        gn = min(G, J - g0)
        S = _sb((128, G * 128))
        nisa.dma_copy(dst=S[:, 0:gn * 128], src=state_in.ap(pattern=[[V, K], [K * V, gn], [1, V]], offset=(h0 + g0) * K * V))
        Sb = _sb((128, G * 128), BF)
        nisa.tensor_copy(dst=Sb[:, 0:gn * 128], src=S[:, 0:gn * 128], engine=nisa.scalar_engine)
        # [V, 2] = S^T [k q]: stationary S_bf [K, V] -> out partitions = V
        psA = nl.ndarray((128, G, 2), dtype=F32, buffer=nl.psum)
        for j in range(gn):
            nisa.nc_matmul(psA[:, j, 0:2], Sb[:, j * 128:(j + 1) * 128], KQ[:, g0 + j, 0:2])
        nisa.tensor_copy(dst=kq[:, g0:g0 + gn, :], src=psA[:, 0:gn, :])
        # delta = (v - eg * kS) * beta ; o = eg * qS + (q . k) * delta     (columns g0 .. g0 + gn)
        cs = slice(g0, g0 + gn)
        kv = _sb((128, G))
        nisa.tensor_tensor(dst=kv[:, 0:gn], data1=kq[:, cs, 0], data2=eg[:, cs], op=nl.multiply)
        nisa.tensor_tensor(dst=delta[:, cs], data1=vT[:, cs], data2=kv[:, 0:gn], op=nl.subtract)
        nisa.tensor_tensor(dst=delta[:, cs], data1=delta[:, cs], data2=beta[:, cs], op=nl.multiply)
        nisa.tensor_tensor(dst=o[:, cs], data1=kq[:, cs, 1], data2=eg[:, cs], op=nl.multiply)
        t2 = _sb((128, G))
        nisa.tensor_tensor(dst=t2[:, 0:gn], data1=qkd[:, cs], data2=delta[:, cs], op=nl.multiply)
        nisa.tensor_tensor(dst=o[:, cs], data1=o[:, cs], data2=t2[:, 0:gn], op=nl.add)
        # rank-1 updates of the group: k and delta as rows (head j on partition j), then 4 heads per matmul with the
        # block-diagonal mask: po[kk, j * 128 + vv] = sum_i K[i, kk] * (D[i, vv] * (i == j)) = k_j[kk] * delta_j[vv]
        dbf = _sb((128, G), BF)
        nisa.tensor_copy(dst=dbf[:, 0:gn], src=delta[:, cs])
        pd = nl.ndarray((G, 128), dtype=F32, buffer=nl.psum)
        nisa.nc_matmul(pd[0:gn, :], dbf[:, 0:gn], ident)
        drow = _sb((G, 128), BF)
        nisa.tensor_copy(dst=drow[0:gn, :], src=pd[0:gn, :])
        pk = nl.ndarray((G, 128), dtype=F32, buffer=nl.psum)
        nisa.nc_matmul(pk[0:gn, :], kbf[:, g0:g0 + gn], ident)
        krow = _sb((G, 128), BF)
        nisa.tensor_copy(dst=krow[0:gn, :], src=pk[0:gn, :])
        dblk = _sb((G, G * 128), BF)
        nisa.tensor_tensor(dst=dblk[0:gn, 0:gn * 128], data1=drow.ap(pattern=[[128, gn], [0, gn], [1, 128]], offset=0),
                           data2=mk[0:gn, 0:gn * 128], op=nl.multiply)
        for c0 in range(0, gn, 4):
            cn = min(4, gn - c0)
            po = nl.ndarray((128, 512), dtype=F32, buffer=nl.psum)
            nisa.nc_matmul(po[:, 0:cn * 128], krow[0:gn, :], dblk[0:gn, c0 * 128:(c0 + cn) * 128])
            Sc = S[:, c0 * 128:(c0 + cn) * 128]
            nisa.tensor_tensor(dst=Sc, data1=Sc, data2=eg.ap(pattern=[[J, 128], [1, cn], [0, 128]], offset=g0 + c0),
                               op=nl.multiply)
            nisa.tensor_tensor(dst=Sc, data1=Sc, data2=po[:, 0:cn * 128], op=nl.add)
        nisa.dma_copy(dst=state_out.ap(pattern=[[V, K], [K * V, gn], [1, V]], offset=(h0 + g0) * K * V), src=S[:, 0:gn * 128])

    # ---- RMSNormGated over v (partitions) ----
    o2 = _sb((128, J), BF)
    nisa.activation(dst=o2, data=o, op=nl.square)
    ms = _sb((128, J))
    for c0 in range(0, J, 512):
        cn = min(512, J - c0)
        ps = nl.ndarray((128, cn), dtype=F32, buffer=nl.psum)
        nisa.nc_matmul(ps, ones, o2[:, c0:c0 + cn])
        nisa.tensor_copy(dst=ms[:, c0:c0 + cn], src=ps)
    rr = _sb((128, J))
    nisa.activation(dst=rr, data=ms, op=nl.rsqrt, scale=1.0 / V, bias=1e-6)
    y = _sb((128, J))
    nisa.scalar_tensor_tensor(dst=y, data=o, op0=nl.multiply, operand0=nwc, op1=nl.multiply, operand1=rr)
    yb = _sb((128, J), BF)
    nisa.tensor_tensor(dst=yb, data1=y, data2=zT, op=nl.multiply)
    nisa.dma_copy(dst=out.ap(pattern=[[1, V], [V, J]], offset=h0 * V), src=yb)
    return out, state_out
