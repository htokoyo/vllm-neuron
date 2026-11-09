# SPDX-License-Identifier: Apache-2.0
"""Device-side (torch, int32-only) n-gram ids for the Qwen4-Exp PLE.

Bit-exact re-implementation of :func:`.ngram.compute_ngram_ids_windows` that avoids
int64 and, more strictly, keeps every *arithmetic* value (add / sub / mul operands and
results, compares, selects) below 2**24 in magnitude, so the result is exact whether the
engine that runs an op does native int32 math or casts to fp32 (the Vector / Scalar
engines compute arithmetic in fp32). Values wider than 24 bits only ever pass through
bitwise ops (and / or / xor / shifts of non-negative values), which act on bit patterns.

Math (all constants come from :class:`.ngram.NGramLayout`):

* digits are base ``R = 2**11``;
* product ``P = t * m`` (t < 2**18, m odd < 2**45, so P < 2**63): ``t`` is split into two
  11-bit digits and ``m`` into 5, column sums are < 2**22 + 2**18, then carried into six
  11-bit limbs;
* ``X = P_0 ^ P_1 [^ P_2]`` limb-wise (bit fields are aligned, so this is exact);
* ``X mod p`` by Horner over the limbs, top first: ``v = r * R + L`` with ``r < p``; the
  quotient ``q = floor(v / p) < R`` is estimated in fp32 (``v * (1/p)``; off by at most
  one in either direction, whatever the float->int rounding), ``v - q * p`` is formed
  exactly from 11-bit digits of ``p`` (``q * digit < 2**22``), normalised with carries, and
  one conditional ``+p`` / ``-p`` brings it into ``[0, p)``;
* ``id = r + offset`` with the high part assembled by ``(hi << 11) | lo``.

``arith`` hooks let the CPU checker run the same code with arithmetic emulated in fp32
(both float->int rounding modes) and with a bound tracker; on the device it is ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .ngram import NGramLayout

RADIX_BITS = 11
RADIX = 1 << RADIX_BITS
MASK = RADIX - 1
_PROD_LIMBS = 6  # 63-bit products / xors in 11-bit limbs
_M_DIGITS = 5  # multipliers < 2**55
_BIAS = 1 << 23  # makes a value >= -2**23 non-negative before a shift (multiple of RADIX)


class Arith:
    """Arithmetic used by the kernel. The default is plain torch (what the device runs)."""

    def add(self, a, b):
        return a + b

    def sub(self, a, b):
        return a - b

    def mul(self, a, b):
        return a * b

    def f2i(self, x):  # fp32 -> int32 of a non-negative value
        return x.to(torch.int32)

    def where(self, c, a, b):
        return torch.where(c, a, b)

    def cmp(self, name, a, b):
        return getattr(torch, name)(a, b)


_PLAIN = Arith()


def _digits(value: int, n: int) -> list[int]:
    out = []
    for _ in range(n):
        out.append(value & MASK)
        value >>= RADIX_BITS
    assert value == 0
    return out


@dataclass
class DeviceNGramConsts:
    """Constants of one PLE layout as small int32 / fp32 tensors (graph literals)."""

    ngram_size: int
    heads_per_ngram: int
    eos_token_id: int
    mult_lo: torch.Tensor  # [ngram_size, 6] int32: digit k of m at column k
    mult_hi: torch.Tensor  # [ngram_size, 6] int32: digit k-1 of m at column k
    p0: torch.Tensor  # [H] int32 digit 0 of p
    p1: torch.Tensor  # [H] digit 1
    p2: torch.Tensor  # [H] p >> 22
    p_hi: torch.Tensor  # [H] p >> 11
    inv_p: torch.Tensor  # [H] fp32 1/p
    off_lo: torch.Tensor  # [H] offset & MASK
    off_hi: torch.Tensor  # [H] offset >> 11
    skip_top_step: bool  # top limb < every p: start Horner from it

    @property
    def num_heads(self) -> int:
        return self.p0.shape[0]

    @property
    def context_len(self) -> int:
        return self.ngram_size - 1

    @classmethod
    def from_layout(cls, layout: NGramLayout, device=None) -> "DeviceNGramConsts":
        ns = layout.ngram_size
        for m in layout.multipliers:
            assert 0 < m < (1 << (RADIX_BITS * _M_DIGITS)), m
        sizes = layout.head_vocab_sizes
        # q * p2 and the high parts must stay small; also id < 2**31.
        assert max(sizes) < (1 << 25) and layout.padded_vocab_size < (1 << 31)
        lo = [[0] * _PROD_LIMBS for _ in range(ns)]
        hi = [[0] * _PROD_LIMBS for _ in range(ns)]
        for k, m in enumerate(layout.multipliers):
            d = _digits(m, _M_DIGITS)
            for c in range(_M_DIGITS):
                lo[k][c] = d[c]
                hi[k][c + 1] = d[c]
        i32 = dict(dtype=torch.int32, device=device)
        return cls(
            ngram_size=ns,
            heads_per_ngram=layout.heads_per_ngram,
            eos_token_id=layout.eos_token_id,
            mult_lo=torch.tensor(lo, **i32),
            mult_hi=torch.tensor(hi, **i32),
            p0=torch.tensor([p & MASK for p in sizes], **i32),
            p1=torch.tensor([(p >> RADIX_BITS) & MASK for p in sizes], **i32),
            p2=torch.tensor([p >> (2 * RADIX_BITS) for p in sizes], **i32),
            p_hi=torch.tensor([p >> RADIX_BITS for p in sizes], **i32),
            inv_p=torch.tensor([1.0 / p for p in sizes], dtype=torch.float32, device=device),
            off_lo=torch.tensor([o & MASK for o in layout.head_offsets], **i32),
            off_hi=torch.tensor([o >> RADIX_BITS for o in layout.head_offsets], **i32),
            # 63-bit values: the top limb holds 8 bits.
            skip_top_step=min(sizes) > (1 << (63 - RADIX_BITS * (_PROD_LIMBS - 1))),
        )


def _floor_split(x, A: Arith):
    """``x`` (int32, |x| < 2**23) -> (x mod R, floor(x / R)) with non-negative shifts only."""
    b = A.add(x, _BIAS)
    return b & MASK, A.sub(b >> RADIX_BITS, _BIAS >> RADIX_BITS)


def shifted_tokens(c: DeviceNGramConsts, windows: torch.Tensor, A: Arith = _PLAIN,
                   eos_block: bool = True) -> torch.Tensor:
    """``windows`` [N, ngram_size] int32 (last = token) -> [N, ngram_size] tokens per
    n-gram position (0 = the token, s = s back), EOS where an EOS lies in between."""
    ctx = c.ngram_size - 1
    eos = c.eos_token_id
    cols = [windows[:, ctx]]
    blocked = None
    for s in range(1, c.ngram_size):
        prev = windows[:, ctx - s]
        if eos_block:
            hit = A.cmp("eq", prev, eos)
            blocked = hit if blocked is None else (blocked | hit)
            prev = A.where(blocked, torch.full_like(prev, eos), prev)
        cols.append(prev)
    return torch.stack(cols, dim=1)


def product_limbs(c: DeviceNGramConsts, toks: torch.Tensor, A: Arith = _PLAIN) -> list:
    """``toks`` [N, ngram_size] int32 (< 2**22) -> six [N, ngram_size] limbs of t * m."""
    t0 = (toks & MASK).unsqueeze(-1)
    t1 = (toks >> RADIX_BITS).unsqueeze(-1)
    cols = A.add(A.mul(t0, c.mult_lo), A.mul(t1, c.mult_hi))  # [N, ns, 6], < 2**22 + 2**18
    limbs = list(cols.unbind(-1))
    for k in range(_PROD_LIMBS - 1):
        carry = limbs[k] >> RADIX_BITS
        limbs[k] = limbs[k] & MASK
        limbs[k + 1] = A.add(limbs[k + 1], carry)
    return limbs


def xor_limbs(c: DeviceNGramConsts, limbs: list) -> torch.Tensor:
    """-> [N, H, 6] limbs of the mixed value of each head's n-gram."""
    P = torch.stack(limbs, dim=-1)  # [N, ns, 6]
    mixed, acc = [], P[:, 0]
    for pos in range(1, c.ngram_size):
        acc = acc ^ P[:, pos]
        mixed.append(acc)
    X = torch.stack(mixed, dim=1)  # [N, ns-1, 6]
    n, g, l = X.shape
    return X.unsqueeze(2).expand(n, g, c.heads_per_ngram, l).reshape(n, g * c.heads_per_ngram, l)


def mod_limbs(c: DeviceNGramConsts, X: torch.Tensor, A: Arith = _PLAIN, correct: bool = True):
    """``X`` [N, H, 6] 11-bit limbs of a value < 2**63 -> (r_hi, r_lo) with
    ``X mod p_h = r_hi * R + r_lo``, ``r_lo`` in [0, R)."""
    L = X.unbind(-1)
    if c.skip_top_step:
        r_hi = torch.zeros_like(L[-1])
        r_lo = L[-1]
        order = range(_PROD_LIMBS - 2, -1, -1)
    else:
        r_hi = torch.zeros_like(L[-1])
        r_lo = torch.zeros_like(L[-1])
        order = range(_PROD_LIMBS - 1, -1, -1)
    two22 = float(1 << (2 * RADIX_BITS))
    for k in order:
        d = L[k]
        # v = r_hi * 2**22 + (r_lo * R + d); r_lo * R + d < 2**22 via bit ops.
        low = (r_lo << RADIX_BITS) | d
        vf = A.add(A.mul(r_hi.to(torch.float32), two22), low.to(torch.float32))
        q = A.f2i(A.mul(vf, c.inv_p))  # floor(v / p) +- 1, in [0, R]
        e0 = A.sub(d, A.mul(q, c.p0))
        e1 = A.sub(r_lo, A.mul(q, c.p1))
        e2 = A.sub(r_hi, A.mul(q, c.p2))
        e0, carry = _floor_split(e0, A)
        e1, carry = _floor_split(A.add(e1, carry), A)
        e2 = A.add(e2, carry)
        D_hi = A.add(A.mul(e2, RADIX), e1)  # D = D_hi * R + e0 in [-p, 2p)
        if not correct:
            r_hi, r_lo = D_hi, e0
            continue
        # D + p
        s0 = A.add(e0, c.p0)
        sc = A.cmp("ge", s0, RADIX)
        s0 = A.where(sc, A.sub(s0, RADIX), s0)
        s_hi = A.add(A.add(D_hi, c.p_hi), sc.to(torch.int32))
        # D - p
        m0 = A.sub(e0, c.p0)
        mb = A.cmp("lt", m0, 0)
        m0 = A.where(mb, A.add(m0, RADIX), m0)
        m_hi = A.sub(A.sub(D_hi, c.p_hi), mb.to(torch.int32))
        neg = A.cmp("lt", D_hi, 0)
        ge = A.cmp("ge", m_hi, 0)
        r_hi = A.where(neg, s_hi, A.where(ge, m_hi, D_hi))
        r_lo = A.where(neg, s0, A.where(ge, m0, e0))
    return r_hi, r_lo


def ngram_ids_from_windows(c: DeviceNGramConsts, windows: torch.Tensor, A: Arith = _PLAIN,
                           *, eos_block: bool = True, correct: bool = True) -> torch.Tensor:
    """Device twin of :func:`.ngram.compute_ngram_ids_windows`.

    ``windows`` [N, ngram_size] int32 -> [N, num_heads] int32 rows of the PLE table.
    """
    toks = shifted_tokens(c, windows, A, eos_block)
    X = xor_limbs(c, product_limbs(c, toks, A))
    r_hi, r_lo = mod_limbs(c, X, A, correct)
    lo = A.add(r_lo, c.off_lo)
    lc = A.cmp("ge", lo, RADIX)
    lo = A.where(lc, A.sub(lo, RADIX), lo)
    hi = A.add(A.add(r_hi, c.off_hi), lc.to(torch.int32))
    return (hi << RADIX_BITS) | lo


def windows_from_context(c: DeviceNGramConsts, context: torch.Tensor, tokens: torch.Tensor):
    """Per-row n-gram windows of consecutive tokens.

    ``context`` [B, ngram_size - 1] int32: the tokens before ``tokens[:, 0]`` (oldest first;
    EOS where the request has none). ``tokens`` [B, T] int32. Returns [B * T, ngram_size].
    """
    seq = torch.cat([context, tokens], dim=1)  # [B, ctx + T]
    T = tokens.shape[1]
    w = torch.stack([seq[:, j:j + T] for j in range(c.ngram_size)], dim=-1)  # [B, T, ns]
    return w.reshape(-1, c.ngram_size)
