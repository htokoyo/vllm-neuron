# SPDX-License-Identifier: Apache-2.0
"""Host-resident PLE n-gram table (below TP 32; ``model_bf16._qwen4_exp_tp_init``).

Device placement (TP >= 32): the PLE n-gram table is a model parameter (row-sharded over TP, FP8 bytes for the FP8
checkpoint), the n-gram ids are hashed in the graph from the step's input ids and the per-slot n-gram context pool
(``model_bf16``, "Device-side n-gram ids"), and the looked-up rows are all-reduced. Nothing in this module runs.

Host placement (TP < 32): the table is never on the device. Every worker process memory-maps the checkpoint's
``ngram_embedding.shard_{0..S-1}.weight`` read-only (one page cache shared by all TP ranks and DP replicas), and
every step the graph gets, per PLE layer, the gathered rows ``[T, heads * D]`` as one more request-state input
(``ngram_ids.{layer}``, the key that held the ids before): the raw e4m3fn bytes (``uint8``) for the FP8 checkpoint,
decoded on the device by the same exact arithmetic decode as the device placement
(``parallel.e4m3fn_bytes_to`` x ``ngram_scale``), or the rows in the model dtype for a BF16 checkpoint. Every rank
gets the full rows (no row sharding, no all-reduce), so the PLE embedding output equals the device placement's bit
for bit (except a ``-0.0`` the device all-reduce turns into ``+0.0`` at TP > 1).

Which tokens (the async question). The n-gram window of a token is the token and the ``ngram_size - 1`` tokens
before it, so the host needs each step's real input tokens and the tokens before them:

- the n-gram context pool ``_ngram_ctx_pool`` stays in the graphs and is committed there as before (prefill,
  decode, MTP verify with the accepted count, prefix-cache snapshot copies): it remains the source of truth;
- prefill-shaped steps (``max_query_len > decode_token_threshold``): the windows come from the worker's token table
  (``token_ids_cpu``: prompt tokens, or the scheduler's tokens of a resumed request; built at input preparation,
  :meth:`request_state_host_inputs`), which equals the pool row for a chunk that starts at ``p > 0`` (the pool holds
  the tokens before ``p``; a prefix-cache restore copies the row stored under the hash of the same tokens);
- plain decode steps (one token per row): the windows come from the worker's token table as well, read at the
  *join* right before the model call (the runner's ``_execute_model_forward``, ``RequestStateSlots.join_host_inputs``),
  after the runner has materialized the previous step's output on this rank (see :func:`_join_decode`);
- MTP-verify-shaped steps: the join reads back the step's actual ``input_ids`` (a device future under async
  scheduling: the draft NEFF's ``[bonus, d1, .., dk]``, which exist only on the device) and the pool, and builds the
  windows ``pool[state_indices] ++ input_ids`` exactly as the graph's ``_device_ngram_prepare`` does (padding tokens
  beyond a row's ``num_valid``: all-EOS windows; ``clamp(min=0)``).

The hash is :func:`.ngram.compute_ngram_ids_windows` (numpy int64; bit-identical to ``ngram_device``), the gather is
a record-view copy per shard (single-threaded below 4,096 rows, a thread pool above).

Everything here is bound per instance (a class swap of the model and of its ``Qwen4ExpNGramEmbedding`` modules at
construction, :func:`enable`) and the runner / request-state hooks are installed only then, so device placement runs
the model classes and the runner unchanged.

Attribution: the memory-mapped table and its gather follow the host-memory n-gram table design of an independent
qwen4_exp port (Apache License 2.0), adapted to FP8 bytes and to this model's loader.
"""

from __future__ import annotations

import collections
import ctypes
import functools
import json
import logging
import math
import mmap
import os
import re
import struct
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
from torch import nn

from .ngram import NGramLayout, compute_ngram_ids_windows

logger = logging.getLogger(__name__)

DEFAULT_THREADS = 8  # gather threads for large (prefill) gathers
# Request-state extras key of the per-step host object the join reads (never sent to the device).
JOIN_KEY = "__join__"

_MADV_RANDOM = 1
_MADV_WILLNEED = 3
# Below this many gathered rows (tokens x heads) the gather stays on the calling thread.
_MIN_PARALLEL_GATHER_ROWS = 4096
_ST_DTYPES = {"F8_E4M3": torch.uint8, "BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
_PLE_TABLE = re.compile(r"^(.*\.ngram_embedding)\.(shard_(\d+)\.weight|weight_scale)$")


# ── The memory-mapped table ──────────────────────────────────────────────────────────────────────────────
_libc = None


def _madvise(addr: int, length: int, advice: int) -> int:
    """libc madvise through ctypes (releases the GIL). Returns errno (0 = ok)."""
    global _libc
    if _libc is None:
        _libc = ctypes.CDLL(None, use_errno=True)
        _libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    page = mmap.PAGESIZE
    start = addr // page * page
    return 0 if _libc.madvise(start, length + addr - start, advice) == 0 else ctypes.get_errno()


def _safetensors_header(path: str) -> tuple[dict, int]:
    """(tensor metadata, absolute byte offset of the data section)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header, 8 + n


class _MappedFile:
    """A checkpoint file mapped ``MAP_PRIVATE`` (``PROT_WRITE`` only so torch gets a writable buffer without a copy;
    nothing writes into it, so no page is ever copied on write)."""

    def __init__(self, path: str):
        self.path = path
        fd = os.open(path, os.O_RDONLY)
        try:
            self.size = os.fstat(fd).st_size
            self.mm = mmap.mmap(fd, self.size, flags=mmap.MAP_PRIVATE, prot=mmap.PROT_READ | mmap.PROT_WRITE)
        finally:
            os.close(fd)
        self._anchor = ctypes.c_char.from_buffer(self.mm)  # pins the mapping; gives its address
        self.addr = ctypes.addressof(self._anchor)
        _madvise(self.addr, self.size, _MADV_RANDOM)  # gathers touch random rows: no readahead on faults

    def tensor(self, offset: int, shape: Sequence[int], dtype: torch.dtype) -> torch.Tensor:
        return torch.frombuffer(self.mm, dtype=dtype, count=math.prod(shape), offset=offset).view(*shape)


class HostPLETable:
    """The PLE n-gram table ``[V, D]`` in host memory as row shards (their row concatenation is the table).

    ``shards`` are 2-D contiguous CPU tensors of one dtype (``uint8`` for e4m3fn bytes) and width: views into
    memory-mapped checkpoint files (:meth:`from_checkpoint`) or into a tensor (:meth:`from_tensor`).
    :meth:`gather` returns ``[N, H * D]`` (head ``h`` of row ``n`` at columns ``h * D : (h + 1) * D``, the column
    order of the device lookup's ``.flatten(-2)``).
    """

    def __init__(self, shards: Sequence[torch.Tensor], num_threads: int = DEFAULT_THREADS, _files=(), _regions=()):
        if not shards:
            raise ValueError("no table shards")
        self.shards = [s.view(torch.uint8) if s.dtype == torch.float8_e4m3fn else s for s in shards]
        self.dim = int(self.shards[0].shape[1])
        self.dtype = self.shards[0].dtype
        for s in self.shards:
            if s.dim() != 2 or s.shape[1] != self.dim or s.dtype != self.dtype or not s.is_contiguous():
                raise ValueError("table shards must be contiguous 2-D tensors of one width and dtype")
        rows = [int(s.shape[0]) for s in self.shards]
        self.shard_starts = np.cumsum([0] + rows[:-1]).astype(np.int64)
        self.num_rows = int(sum(rows))
        # Uniform shards (the checkpoint layout): route by a division instead of searchsorted.
        self._uniform = rows[0] if all(r == rows[0] for r in rows[:-1]) and rows[-1] <= rows[0] else None
        # One numpy record per row: fancy indexing then copies whole rows and releases the GIL.
        self._row = np.dtype((np.void, self.dim * self.shards[0].element_size()))
        self._np = [self._records(s) for s in self.shards]
        self._files = list(_files)
        self._regions = list(_regions)  # (mapped file, byte offset, byte length) per shard, for prefetch
        self.num_threads = max(1, int(num_threads))
        self._pool = None
        self._pool_lock = threading.Lock()

    def _records(self, t: torch.Tensor) -> np.ndarray:
        return t.view(torch.uint8).numpy().view(self._row).reshape(-1)

    @classmethod
    def from_tensor(cls, weight: torch.Tensor, num_shards: int = 1, num_threads: int = 1) -> "HostPLETable":
        """An in-memory table (tests); ``num_shards`` row splits exercise the shard routing."""
        w = weight.detach().contiguous()
        rows = -(-w.shape[0] // num_shards)
        return cls(list(torch.split(w, rows, dim=0)), num_threads=num_threads)

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: str, entries: Sequence[tuple[str, str]],
                        num_threads: int = DEFAULT_THREADS) -> "HostPLETable":
        """Map the shards ``entries`` = ``[(tensor name, file)]`` (in row order). No row is read here."""
        files: dict[str, _MappedFile] = {}
        headers: dict[str, tuple[dict, int]] = {}
        shards, regions = [], []
        for name, fname in entries:
            path = os.path.join(checkpoint_dir, fname)
            if fname not in headers:
                headers[fname] = _safetensors_header(path)
            header, data_start = headers[fname]
            meta = header.get(name)
            if meta is None:
                raise KeyError(f"{name} not in {path}")
            if meta["dtype"] not in _ST_DTYPES:
                raise ValueError(f"{name}: unsupported table dtype {meta['dtype']}")
            if fname not in files:
                files[fname] = _MappedFile(path)
            mf = files[fname]
            begin, end = meta["data_offsets"]
            t = mf.tensor(data_start + begin, meta["shape"], _ST_DTYPES[meta["dtype"]])
            if t.numel() * t.element_size() != end - begin:
                raise ValueError(f"{name}: {end - begin} bytes for shape {meta['shape']} {meta['dtype']}")
            shards.append(t)
            regions.append((mf, data_start + begin, end - begin))
        table = cls(shards, num_threads=num_threads, _files=files.values(), _regions=regions)
        table.checkpoint_dtype = {headers[f][0][n]["dtype"] for n, f in entries}
        return table

    @property
    def nbytes(self) -> int:
        return self.num_rows * self.dim * self.shards[0].element_size()

    def _executor(self):
        with self._pool_lock:  # created on first use (not before a fork)
            if self._pool is None:
                self._pool = ThreadPoolExecutor(self.num_threads, thread_name_prefix="ple-gather")
            return self._pool

    def gather(self, ids) -> torch.Tensor:
        """``ids`` ``[N, H]`` int rows of the table -> ``[N, H * D]`` in the table dtype."""
        flat = np.asarray(ids.numpy() if isinstance(ids, torch.Tensor) else ids)
        n, h = flat.shape
        flat = flat.reshape(-1).astype(np.int64, copy=False)
        out = torch.empty(n, h * self.dim, dtype=self.dtype)
        if flat.size == 0:
            return out
        lo, hi = int(flat.min()), int(flat.max())
        if lo < 0 or hi >= self.num_rows:
            raise IndexError(f"n-gram rows [{lo}, {hi}] outside the table [0, {self.num_rows})")
        dst = self._records(out.view(n * h, self.dim))
        s = len(self.shards)
        shard = flat // self._uniform if self._uniform is not None else (
            np.searchsorted(self.shard_starts, flat, side="right") - 1)
        shard = shard.astype(np.int32)
        order = np.argsort(shard, kind="stable")
        bounds = np.searchsorted(shard[order], np.arange(s + 1)).tolist()
        local = flat[order]
        work = [k for k in range(s) if bounds[k + 1] > bounds[k]]

        def run(ks):
            for k in ks:
                a, b = bounds[k], bounds[k + 1]
                dst[order[a:b]] = self._np[k][local[a:b] - self.shard_starts[k]]

        if self.num_threads == 1 or len(work) < 2 or flat.size < _MIN_PARALLEL_GATHER_ROWS:
            run(work)
        else:
            m = min(self.num_threads, len(work))
            for f in [self._executor().submit(run, work[i::m]) for i in range(m)]:
                f.result()
        return out

    def prefetch(self, num_threads: int = 16, block: bool = False, chunk_bytes: int = 1 << 28):
        """Warm the page cache for the whole table (``MADV_WILLNEED`` per chunk, in a background thread unless
        ``block``). No ``MADV_POPULATE_READ``: at TP8 x DP8 every process would build page tables for 51 GB.
        No-op for an in-memory table."""
        jobs = [(mf.addr + off + a, min(chunk_bytes, length - a))
                for mf, off, length in self._regions for a in range(0, length, chunk_bytes)]
        if not jobs:
            return None

        def run():
            with ThreadPoolExecutor(num_threads, thread_name_prefix="ple-prefetch") as pool:
                list(pool.map(lambda j: _madvise(*j, _MADV_WILLNEED), jobs))

        if block:
            run()
            return None
        t = threading.Thread(target=run, name="ple-prefetch", daemon=True)
        t.start()
        return t


# ── Windows (numpy twins of model_bf16._device_ngram_prepare) ──────────────────────────────────────────────
def prefill_windows(token_ids: np.ndarray, rows: np.ndarray, positions: np.ndarray, context_len: int,
                    eos: int) -> np.ndarray:
    """``[N, context_len + 1]`` int64: token ``k`` = ``token_ids[rows[k], positions[k]]`` and the ``context_len``
    tokens before it in its request (EOS before the start); a padding token (``rows[k] < 0``): all EOS."""
    rows = np.asarray(rows, dtype=np.int64)
    positions = np.asarray(positions, dtype=np.int64)
    out = np.full((rows.shape[0], context_len + 1), eos, dtype=np.int64)
    real = rows >= 0
    for c in range(context_len + 1):
        p = positions - (context_len - c)
        ok = real & (p >= 0)
        out[ok, c] = token_ids[rows[ok], p[ok]]
    return out


def decode_windows(ids: np.ndarray, ctx: np.ndarray, num_valid: np.ndarray, context_len: int,
                   eos: int) -> np.ndarray:
    """``ids`` ``[b, T]``, ``ctx`` ``[b, w]`` (the pool rows, oldest first) -> ``[b * T, context_len + 1]``:
    the windows of ``ctx[:, w - context_len:] ++ ids``; tokens at or beyond a row's ``num_valid``: all EOS."""
    b, t = ids.shape
    seq = np.concatenate([ctx[:, ctx.shape[1] - context_len:].astype(np.int64), ids.astype(np.int64)], axis=1)
    win = np.lib.stride_tricks.sliding_window_view(seq, context_len + 1, axis=1)[:, :t].reshape(b * t, -1).copy()
    real = (np.arange(t).reshape(1, t) < np.asarray(num_valid[:b]).reshape(-1, 1)).reshape(-1)
    win[~real] = eos
    return win


# ── Per-model host state ─────────────────────────────────────────────────────────────────────────────────
@dataclass
class _Layer:
    layout: NGramLayout
    dim: int  # columns per head
    dtype: torch.dtype  # row dtype on the graph input (uint8 for the FP8 table; else the table's checkpoint dtype)
    table: HostPLETable | None = None


@dataclass
class HostPLEState:
    layers: dict[int, _Layer]
    eos: int
    threads: int = DEFAULT_THREADS
    prefetch: bool = True  # MADV_WILLNEED the table in the background after load
    stats: collections.Counter = field(default_factory=collections.Counter)
    last: dict | None = None  # the last join's windows / ids per layer (tests)


# Kept off the module (torch.compile would guard on / trace through an attribute).
_STATE: "weakref.WeakKeyDictionary[nn.Module, HostPLEState]" = weakref.WeakKeyDictionary()


def state(model) -> HostPLEState:
    return _STATE[getattr(model, "_orig_mod", model)]


def is_host(model) -> bool:
    return getattr(model, "_orig_mod", model) in _STATE


@dataclass
class HostTokens:
    """The step's windows from the token table (built at input preparation) and the token layout, for the join."""

    windows: dict[int, np.ndarray]  # context_len -> [num_tokens, context_len + 1]
    token_ids: np.ndarray | None = None  # the worker's token table (the live array: read at the join)
    rows: np.ndarray | None = None  # table row per token (-1: padding)
    positions: np.ndarray | None = None  # position per token


class _TablePlaceholder(nn.Module):
    """``ngram_embedding`` of a host-placed table: no rows on the device, ``weight`` ``[0, D]``."""

    def __init__(self, dim: int, dtype: torch.dtype):
        super().__init__()
        self.register_buffer("weight", torch.empty(0, dim, dtype=dtype, device="cpu"), persistent=False)


def _ple_modules(model):
    """``{layer: Qwen4ExpNGramEmbedding}`` of the model's PLE layers."""
    return {i: model.model.layers[i].ple.ple_embedding for i in model.model.ngram_layouts}


def enable(model, *, threads: int = DEFAULT_THREADS, prefetch: bool = True,
           install_runner: bool = True) -> HostPLEState:
    """Switch a constructed (not yet loaded) ``Qwen4ExpForCausalLM`` to the host-placed table."""
    from . import model_bf16 as MB

    if is_host(model):
        return state(model)
    if type(model) is not MB.Qwen4ExpForCausalLM:
        raise TypeError(f"host PLE placement needs a model_bf16.Qwen4ExpForCausalLM, got {type(model).__name__}")
    layers = {}
    for i, emb in _ple_modules(model).items():
        w = emb.ngram_embedding.weight
        layers[i] = _Layer(model.model.ngram_layouts[i], int(w.shape[1]), w.dtype)
        emb.ngram_embedding = _TablePlaceholder(int(w.shape[1]), w.dtype)  # frees the (meta or CPU) table
        emb.__class__ = HostPLENGramEmbedding
    model.__class__ = HostPLEQwen4ExpForCausalLM
    st = _STATE[model] = HostPLEState(layers=layers, eos=int(model.config.eos_token_id), threads=threads,
                                      prefetch=prefetch)
    if install_runner:
        install_join()
    logger.info("[qwen4_exp] PLE n-gram table placement: host (layers %s, gather threads %d, prefetch %s)",
                sorted(layers), st.threads, st.prefetch)
    return st


def install_join() -> None:
    """The runner's join of the per-step host inputs (installed once per process, host placement only)."""
    from vllm_neuron.vllm.worker import neuron_model_runner as R

    R.install_request_state_join()


def attach_tables(model, tables: dict[int, HostPLETable], scales: dict[int, torch.Tensor | None]) -> None:
    """Give a host-placed model its tables and (FP8) scales, with the checks the device loader makes."""
    st = state(model)
    mods = _ple_modules(model)
    if set(tables) != set(st.layers):
        raise ValueError(f"host PLE tables for layers {sorted(tables)}, model PLE layers {sorted(st.layers)}")
    for i, lay in st.layers.items():
        t, s, emb = tables[i], scales.get(i), mods[i]
        heads = lay.layout.num_heads
        if t.num_rows != lay.layout.padded_vocab_size or t.dim != lay.dim:
            raise ValueError(f"layer {i}: host table [{t.num_rows}, {t.dim}] vs n-gram layout "
                             f"[{lay.layout.padded_vocab_size}, {lay.dim}] ({heads} heads)")
        fp8 = "ngram_scale" in emb._buffers
        if fp8:
            if t.dtype != torch.uint8 or getattr(t, "checkpoint_dtype", {"F8_E4M3"}) != {"F8_E4M3"}:
                raise ValueError(f"layer {i}: the FP8 PLE table must be e4m3fn, got {t.dtype}")
            if s is None:
                raise ValueError(f"layer {i}: FP8 table without ngram_embedding.weight_scale")
            # As the device loader (_ple_rows_fp8): the scale in the model dtype, set explicitly (it used to be a side
            # effect of loading the shards; without it the buffer stays NaN).
            emb.ngram_scale = s.reshape(()).to(model.config.torch_dtype)
            if not bool(torch.isfinite(emb.ngram_scale)):
                raise ValueError(f"layer {i}: PLE scale {emb.ngram_scale} is not finite")
        else:
            if s is not None:
                raise ValueError(f"layer {i}: weight_scale given for a {t.dtype} table")
            if not t.dtype.is_floating_point:
                raise ValueError(f"layer {i}: host table {t.dtype} for a {lay.dtype} model")
            # Rows are sent in the checkpoint dtype and converted in the graph (HostPLENGramEmbedding), where the
            # device loader converts the whole table at load (_checked: tensor.to(dtype)).
        lay.dtype = t.dtype
        lay.table = t


def load_tables(model, checkpoint_path: str, layers=None) -> None:
    """Map the PLE shards of ``checkpoint_path`` for the model's PLE layers (in scope of ``layers``)."""
    from . import model_bf16 as MB

    st = state(model)
    wm = MB.Qwen4ExpForCausalLM._weight_map(checkpoint_path)
    shards: dict[int, list] = collections.defaultdict(list)
    scale_entries: dict[int, tuple[str, str]] = {}
    for ckpt_name, file in wm.items():
        name = MB.Qwen4ExpForCausalLM._map_name(ckpt_name)
        m = None if name is None else _PLE_TABLE.match(name)
        if m is None:
            continue
        li = int(name.split(".")[2])
        if li not in st.layers or (layers is not None and li not in {int(x) for x in layers}):
            continue
        if m.group(3) is not None:
            shards[li].append((int(m.group(3)), ckpt_name, file))
        else:
            scale_entries[li] = (ckpt_name, file)
    tables, scales = {}, {}
    for li in st.layers:
        if not shards.get(li):
            raise ValueError(f"layer {li}: no ngram_embedding.shard_*.weight in {checkpoint_path}")
        idx = sorted(k for k, _, _ in shards[li])
        if idx != list(range(len(idx))):
            raise ValueError(f"layer {li}: PLE shards {idx[:8]}... are not 0..{len(idx) - 1}")
        tables[li] = HostPLETable.from_checkpoint(
            checkpoint_path, [(n, f) for _, n, f in sorted(shards[li])], num_threads=st.threads)
        scales[li] = MB._read_ple_scale(checkpoint_path, scale_entries.get(li))
    attach_tables(model, tables, scales)
    for li, t in tables.items():
        logger.info("[qwen4_exp] host PLE table layer %d: %d rows x %d (%s, %.2f GB) mapped from %d shards", li,
                    t.num_rows, t.dim, t.dtype, t.nbytes / 1e9, len(t.shards))
        if st.prefetch:
            t.prefetch()


# ── Host-side inputs ─────────────────────────────────────────────────────────────────────────────────────
def _context_lens(st: HostPLEState) -> set[int]:
    return {lay.layout.context_len for lay in st.layers.values()}


def _rows_from_windows(st: HostPLEState, windows: dict[int, np.ndarray]) -> dict[str, torch.Tensor]:
    out, last = {}, {}
    for i, lay in st.layers.items():
        win = windows[lay.layout.context_len]
        ids = compute_ngram_ids_windows(lay.layout, win)
        if lay.table is None:
            raise RuntimeError(f"host PLE layer {i}: no table (load_weights / attach_tables not called)")
        rows = lay.table.gather(ids)
        if rows.dtype != lay.dtype:
            raise RuntimeError(f"host PLE layer {i}: rows {rows.dtype} vs graph input {lay.dtype}")
        out[f"ngram_ids.{i}"] = rows
        last[i] = dict(windows=win, ids=ids)
    st.last = last
    return out


def _read_ids(input_ids) -> np.ndarray:
    """The step's input ids on the host (waits for the producing graph when they are a device future)."""
    return input_ids.cpu().numpy().astype(np.int64).reshape(-1)


def _read_pool(model) -> np.ndarray:
    """The n-gram context pool on the host (``[slots, w]`` int32), as the graph about to run will read it."""
    return model._ngram_ctx_pool.cpu().numpy().copy()


def _step_shape(model, attn_metadata, num_tokens: int):
    meta0 = attn_metadata[f"layers.{model._attention_layers()[0][0]}.self_attn"]
    q = meta0["max_query_len"]
    is_prefill = q > meta0["decode_token_threshold"]
    b = 1 if is_prefill else num_tokens // q
    return is_prefill, b, num_tokens // b


def _join_prefill(model, st, input_ids, host, b, t):
    win_by_len = host["join"].windows
    num_valid = host["num_valid"]
    windows = {}
    for cl, win in win_by_len.items():
        if win.shape[0] < t:
            raise RuntimeError(f"host PLE: {win.shape[0]} prefill windows for {t} tokens")
        w = win[:t].copy()
        w[np.arange(t) >= int(num_valid[0])] = st.eos  # as the graph: tokens beyond num_valid are padding
        windows[cl] = w
    st.stats["prefill_steps"] += 1
    return windows


def _join_decode_pool(model, st, input_ids, host, b, t):
    """MTP verify: the windows of ``pool[state_indices] ++ input_ids`` (the drafts exist only on the device)."""
    ids = np.clip(_read_ids(input_ids), 0, None)
    if ids.shape[0] != b * t:
        raise RuntimeError(f"host PLE: {ids.shape[0]} input ids for a [{b}, {t}] step")
    ctx = _read_pool(model)[np.asarray(host["state_indices"][:b], dtype=np.int64)]
    st.stats["decode_steps"] += 1
    st.stats["verify_steps"] += int(t > 1)
    return {cl: decode_windows(ids.reshape(b, t), ctx, host["num_valid"], cl, st.eos) for cl in _context_lens(st)}


def _join_decode(model, st, input_ids, host, b, t):
    """A plain decode step (one token per row) from the worker's token table; MTP verify steps from the pool.

    Reading the step's input ids and the in-place n-gram pool on the host (as :func:`_join_decode_pool` does) on the
    worker's main thread raced, under async scheduling at TP > 1, with the upstream ``WorkerAsyncOutputCopy`` thread
    reading the same sampled-token future through ``get_output`` outside its per-object lock (TP16: the first decode's
    ``sample_tokens`` never returned). So the runner first materializes the previous step's output on EVERY rank
    through ``get_output`` (under its lock, so each future is read back once; upstream writes the sampled tokens into
    the token table on the output rank only), then this join builds ``token_ids[row, p - context_len .. p]`` for each
    real token at position ``p`` and refuses a token the table does not hold yet (``p >= num_tokens_no_spec``),
    instead of hashing a stale one on some ranks only. The device placement never reads either tensor on the host."""
    tok = host.get("join")
    if t > 1 or not isinstance(tok, HostTokens) or tok.token_ids is None:
        st.stats["decode_pool_joins"] += 1
        return _join_decode_pool(model, st, input_ids, host, b, t)
    n = b * t
    if int(input_ids.shape[0]) != n or tok.rows.shape[0] < n:
        raise RuntimeError(f"host PLE: {int(input_ids.shape[0])} input ids / {tok.rows.shape[0]} layout tokens "
                           f"for a [{b}, {t}] step")
    rows, pos = tok.rows[:n], tok.positions[:n]
    real = (rows >= 0) & (np.asarray(host["num_valid"][:b]).reshape(-1) > 0)
    batch = host.get("input_batch")
    if batch is not None and real.any():
        have = np.asarray(batch.num_tokens_no_spec)[rows[real]]
        if (pos[real] >= have).any():
            k = int(np.nonzero(pos[real] >= have)[0][0])
            raise RuntimeError(
                f"host PLE: decode token at position {int(pos[real][k])} of table row {int(rows[real][k])} is not in "
                f"this rank's token table (num_tokens_no_spec {int(have[k])}): the previous step's output was not "
                "materialized on this rank before the join")
    rows = np.where(real, rows, -1)
    st.stats["decode_steps"] += 1
    st.stats["decode_table_joins"] += 1
    return {cl: prefill_windows(tok.token_ids, rows, pos, cl, st.eos) for cl in _context_lens(st)}


def join_inputs(model, input_ids, attn_metadata, host: dict) -> dict[str, torch.Tensor]:
    """The step's ``ngram_ids.{layer}`` rows (CPU tensors), right before the model call."""
    st = state(model)
    t_all = int(input_ids.shape[0])
    is_prefill, b, t = _step_shape(model, attn_metadata, t_all)
    fn = _join_prefill if is_prefill else _join_decode
    windows = fn(model, st, input_ids, host, b, t)
    rows = _rows_from_windows(st, windows)
    st.stats["rows"] += t_all * sum(lay.layout.num_heads for lay in st.layers.values())
    return rows


# ── Model side (class swap at construction; device placement never sees these classes) ─────────────────────
from . import layers as _L  # noqa: E402
from . import model_bf16 as _MB  # noqa: E402


class HostPLENGramEmbedding(_L.Qwen4ExpNGramEmbedding):
    """PLE n-gram embedding of a host-placed table: the graph input is the gathered rows."""

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        if rows.dtype == torch.uint8:
            if "ngram_scale" not in self._buffers:
                raise TypeError("uint8 PLE rows (FP8 bytes) for a model without ngram_scale")
            from . import parallel as P

            # The device placement's decode (parallel.sharded_lookup_dequant), applied to the same bytes.
            return P.e4m3fn_bytes_to(rows, self.ngram_scale.dtype) * self.ngram_scale
        w = self.ngram_embedding.weight  # [0, D] placeholder in the device table's dtype
        return rows if rows.dtype == w.dtype else rows.to(w.dtype)


def _host_ngram_prepare(self, input_ids, attn_metadata, spec_decode_metadata):
    """``model_bf16._device_ngram_prepare`` without the hash: the pool update (``seq``) only; the rows are inputs."""
    rs = attn_metadata["request_state"]
    meta0 = attn_metadata[f"layers.{self._attention_layers()[0][0]}.self_attn"]
    q = meta0["max_query_len"]
    is_prefill = q > meta0["decode_token_threshold"]
    verify = spec_decode_metadata is not None and not is_prefill
    t = input_ids.shape[0]
    b = 1 if is_prefill else t // q
    T = t // b
    eos = self.config.eos_token_id
    slots = rs["state_indices"][:b].long(); read_slots = _MB._state_read_slots(rs, slots, b)
    has_init = rs["has_initial_state"][:b].bool()
    num_valid = rs["num_valid"][:b].to(torch.int32)
    pool = self._ngram_ctx_pool
    ctx = pool.index_select(0, read_slots) if b > 0 else pool[:0]
    if is_prefill:
        ctx = torch.where(has_init.view(-1, 1), ctx, torch.full_like(ctx, eos))
    toks = input_ids.to(torch.int32).view(b, T).clamp(min=0)
    missing = [i for i in self.model.ngram_layouts if f"ngram_ids.{i}" not in rs]
    if missing:
        raise RuntimeError(f"host PLE rows for layers {missing} missing: the runner's join did not run")
    return dict(b=b, T=T, slots=slots, seq=torch.cat([ctx, toks], dim=1), num_valid=num_valid,
                verify=verify, prefill=is_prefill)


@functools.wraps(_MB._host_forward)
def _forward_host_ngram(self, input_ids, positions, inputs_embeds=None, is_token_ids=None,
                        attn_metadata=None, sampling_positions=None, sampling_params=None,
                        spec_decode_metadata=None, logit_mask=None, rank=None,
                        prev_sampled_token_ids=None, prev_num_draft_tokens=None,
                        req_indices_per_token=None, **kwargs):
    # model_bf16._forward_device_ngram with the host rows: same async correction, same pool commit.
    positions, attn_metadata = _MB._async_spec_correct(
        self.config.vocab_size, positions, attn_metadata, prev_sampled_token_ids,
        prev_num_draft_tokens, req_indices_per_token)
    upd = _host_ngram_prepare(self, input_ids, attn_metadata, spec_decode_metadata)
    out = _MB._host_forward(self, input_ids, positions, inputs_embeds, is_token_ids, attn_metadata,
                            sampling_positions, sampling_params, spec_decode_metadata, logit_mask,
                            rank, **kwargs)
    tokens = out[0] if isinstance(out, tuple) else out
    _MB._device_ngram_commit(self, upd, tokens)
    _MB._verify_snapshot_ngram_write(self, upd, tokens, attn_metadata["request_state"], positions)
    return out


class HostPLEQwen4ExpForCausalLM(_MB.Qwen4ExpForCausalLM):
    """``Qwen4ExpForCausalLM`` with the host-placed PLE table (set by :func:`enable`, never constructed directly)."""

    forward = _forward_host_ngram

    @staticmethod
    def _weight_map(checkpoint_path: str) -> dict[str, str]:
        """The checkpoint without the PLE table (shards and scale): the device loader never reads them."""
        return {k: v for k, v in _MB.Qwen4ExpForCausalLM._weight_map(checkpoint_path).items()
                if _PLE_TABLE.match(k) is None}

    def load_weights(self, checkpoint_path: str, device=None, cache_dir=None, layers=None) -> set[str]:
        loaded = super().load_weights(checkpoint_path, device, cache_dir, layers)
        load_tables(self, checkpoint_path, layers)
        return loaded

    def request_state_host_inputs(self, token_ids, rows, positions) -> dict:
        st = state(self)
        rows = np.asarray(rows, dtype=np.int64).copy()
        positions = np.asarray(positions, dtype=np.int64).copy()
        win = {cl: prefill_windows(token_ids, rows, positions, cl, st.eos) for cl in _context_lens(st)}
        return {JOIN_KEY: HostTokens(win, token_ids=token_ids, rows=rows, positions=positions)}

    def request_state_warmup_inputs(self, num_tokens: int, *args, **kwargs) -> dict[str, torch.Tensor]:
        st = state(self)
        return {f"ngram_ids.{i}": torch.zeros(num_tokens, lay.layout.num_heads * lay.dim, dtype=lay.dtype)
                for i, lay in st.layers.items()}

    def request_state_join_inputs(self, input_ids, attn_metadata, host: dict) -> dict[str, torch.Tensor]:
        return join_inputs(self, input_ids, attn_metadata, host)
