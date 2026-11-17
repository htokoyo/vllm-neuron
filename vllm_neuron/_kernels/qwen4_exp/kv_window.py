# SPDX-License-Identifier: Apache-2.0
"""Block-table KV window read for the dense decode (``layers._paged_decode``).

``cache.index_select(0, table.flatten())`` on the ``[blocks, 1, 32, 256]`` cache lowered to one software-DGE DMA per
(row, token) with one 512 B descriptor per block: B * 32 GpSimd DMAs per gather (256 at B=8, 0.28 ms each, a TP16 decode
profile). This kernel moves whole blocks: per program (LNC) and per <= 128 blocks, one DMA loads the block ids, one
indirect DMA (``vector_offset`` on the partition dim) gathers the 16 KB block rows HBM -> SBUF, one DMA stores them to
the HBM window. ``out[j] = cache[table[j]]`` bit for bit (NKI simulator, grid 1 and 2, 16 to 512 rows).
"""
from __future__ import annotations

import nki
import nki.isa as nisa
import nki.language as nl

P_MAX = 128


@nki.jit
def kv_window_tkg(cache, table):
    """cache [NB, R] (one block of R = block_size * head_dim elements per row), table [n, 1] int32 (ids >= 0, n
    divisible by the grid) -> out [n, R] (HBM), out[j] = cache[table[j]]."""
    NB, R = cache.shape
    n = table.shape[0]
    n_prgs = nl.num_programs(0)
    prg = nl.program_id(0)
    rows = n // n_prgs
    r_start = prg * rows
    out = nl.ndarray((n, R), dtype=cache.dtype, buffer=nl.shared_hbm)
    for t in range((rows + P_MAX - 1) // P_MAX):
        m = min(P_MAX, rows - t * P_MAX)
        r0 = r_start + t * P_MAX
        idx = nl.ndarray((m, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=idx, src=table.ap(pattern=[[1, m], [1, 1]], offset=r0))
        buf = nl.ndarray((m, R), dtype=cache.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=buf, src=cache.ap(pattern=[[R, m], [1, R]], offset=0, vector_offset=idx, indirect_dim=0))
        nisa.dma_copy(dst=out.ap(pattern=[[R, m], [1, R]], offset=r0 * R), src=buf)
    return out
