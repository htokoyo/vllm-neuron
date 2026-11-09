# SPDX-License-Identifier: Apache-2.0
"""Host-side n-gram ids for the Qwen4-Exp per-layer embedding (PLE).

The n-gram id of a token is a pure function of the token and the ``ngram_size - 1``
tokens before it, so it is computed on the host from the request's token history
and handed to the device as int32. This keeps 64-bit multiply/xor/modulo off the
device and removes the n-gram context from the per-request device state.

Attribution: the n-gram hash and layout (and parts of their code) follow the Hugging Face
Transformers 5.17 reference, ``transformers/models/qwen4_exp/modeling_qwen4_exp.py`` (Copyright 2026
The Qwen Team and The HuggingFace Inc. team, Apache License 2.0).

The hash matches the checkpoint: ``layer_multipliers``, ``ngram_heads_vocab_sizes``
and ``ngram_heads_offsets`` stored in the safetensors are recomputed exactly by
:func:`build_ngram_layout`.

History rule (same as the reference): the n-gram window does not cross an EOS. A
token ``s`` positions back is used only if no EOS lies in the ``s`` positions
before the current token; otherwise, or before the start of the sequence, EOS is
used in its place.
"""

import math
from dataclasses import dataclass

import numpy as np

_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    return all(value % d for d in range(3, math.isqrt(value) + 1, 2))


@dataclass(frozen=True)
class NGramLayout:
    """Hash constants of one PLE module."""

    ngram_size: int
    heads_per_ngram: int
    eos_token_id: int
    multipliers: tuple[int, ...]  # one per position in the n-gram window
    head_vocab_sizes: tuple[int, ...]  # one prime per head
    head_offsets: tuple[int, ...]  # row offset of each head in the shared table
    padded_vocab_size: int  # rows of the embedding table

    @property
    def num_heads(self) -> int:
        return len(self.head_vocab_sizes)

    @property
    def context_len(self) -> int:
        return self.ngram_size - 1


def build_ngram_layout(
    vocab_size: int,
    ngram_size: int,
    heads_per_ngram: int,
    ngram_vocab_size_base: int,
    make_ngram_vocab_size_divisible_by: int,
    eos_token_id: int,
    ple_layer_index: int = 0,
    seed: int = 1234,
) -> NGramLayout:
    """Recompute the hash constants of the ``ple_layer_index``-th PLE module."""
    half_bound = max(1, (((1 << 63) - 1) // max(vocab_size, 1)) // 2)
    base_seed = seed + _PRIME_1 * ple_layer_index
    multipliers = tuple(
        2 * (_splitmix64((base_seed + _SPLITMIX_GAMMA * (i + 1)) & _MASK64) % half_bound) + 1
        for i in range(ngram_size)
    )
    num_heads = (ngram_size - 1) * heads_per_ngram
    sizes, offsets, total, prime = [], [], 0, ngram_vocab_size_base - 1
    # Primes are drawn globally across PLE modules, so skip the earlier modules' heads.
    for _ in range(ple_layer_index * num_heads):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    for _ in range(num_heads):
        prime += 1
        while not _is_prime(prime):
            prime += 1
        sizes.append(prime)
        offsets.append(total)
        total += prime
    div = make_ngram_vocab_size_divisible_by
    return NGramLayout(
        ngram_size=ngram_size,
        heads_per_ngram=heads_per_ngram,
        eos_token_id=eos_token_id,
        multipliers=multipliers,
        head_vocab_sizes=tuple(sizes),
        head_offsets=tuple(offsets),
        padded_vocab_size=-(-total // div) * div,
    )


def compute_ngram_ids_windows(layout: NGramLayout, windows: np.ndarray) -> np.ndarray:
    """n-gram table rows from per-token windows.

    Args:
        layout: Hash constants from :func:`build_ngram_layout`.
        windows: ``[N, context_len + 1]`` int64; the last column is the token, the
            columns before it the tokens preceding it (EOS where the request has none).

    Returns:
        ``[N, num_heads]`` int32 row indices into the PLE embedding table. Bigram
        heads come first, then trigram heads, each ``heads_per_ngram`` wide.
    """
    ctx = layout.context_len
    eos = layout.eos_token_id
    windows = np.asarray(windows, dtype=np.int64)
    # shifted[s] = the token s positions back, or EOS if an EOS lies in between.
    is_eos = windows == eos
    shifted = [windows[:, ctx]]
    blocked = np.zeros(windows.shape[0], dtype=bool)
    for s in range(1, layout.ngram_size):
        blocked |= is_eos[:, ctx - s]
        shifted.append(np.where(blocked, eos, windows[:, ctx - s]))

    mult = layout.multipliers
    sizes = np.asarray(layout.head_vocab_sizes, dtype=np.int64)
    offsets = np.asarray(layout.head_offsets, dtype=np.int64)
    blocks = []
    for n in range(2, layout.ngram_size + 1):
        mixed = shifted[0] * mult[0]
        for pos in range(1, n):
            mixed = np.bitwise_xor(mixed, shifted[pos] * mult[pos])
        lo = (n - 2) * layout.heads_per_ngram
        hi = lo + layout.heads_per_ngram
        blocks.append(np.remainder(mixed[:, None], sizes[None, lo:hi]) + offsets[None, lo:hi])
    ids = np.concatenate(blocks, axis=-1)
    assert layout.padded_vocab_size < 2**31
    return ids.astype(np.int32)


def compute_ngram_ids(
    layout: NGramLayout, tokens: np.ndarray, history: np.ndarray | None = None
) -> np.ndarray:
    """Return the n-gram table rows for consecutive ``tokens`` of one request.

    Args:
        layout: Hash constants from :func:`build_ngram_layout`.
        tokens: ``[T]`` token ids of the positions to embed.
        history: The up-to ``context_len`` tokens that precede ``tokens[0]`` in the
            same request (oldest first). ``None`` or shorter means the request starts
            within the window, which is padded with EOS as the reference does.

    Returns:
        ``[T, num_heads]`` int32, see :func:`compute_ngram_ids_windows`.
    """
    ctx = layout.context_len
    tokens = np.asarray(tokens, dtype=np.int64).reshape(-1)
    hist = np.asarray(history if history is not None else [], dtype=np.int64)[-ctx:]
    seq = np.concatenate([np.full(ctx - hist.size, layout.eos_token_id, np.int64), hist, tokens])
    windows = np.lib.stride_tricks.sliding_window_view(seq, ctx + 1)
    return compute_ngram_ids_windows(layout, windows)
