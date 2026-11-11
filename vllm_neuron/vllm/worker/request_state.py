# SPDX-License-Identifier: Apache-2.0
"""Per-request state slots for models that keep their own recurrent state.

Hybrid models (linear attention, short convolutions) carry state that is not
paged KV. Instead of declaring it to the KV cache manager, such a model owns
fixed pools of ``num_slots`` rows and the runner assigns one slot per live
request. The model opts in by implementing:

- ``allocate_request_state(num_slots, device)``
- ``request_state_host_inputs(token_ids, rows, positions) -> dict[str, Tensor]``
  per-token host inputs (e.g. hashed n-gram ids), ``rows`` = -1 for padding
- ``request_state_warmup_inputs(num_tokens) -> dict[str, Tensor]`` with the same
  keys, shapes and dtypes

The runner then adds ``attn_metadata["request_state"]`` (appended after the
per-layer entries, which keeps ``next(iter(attn_metadata.values()))`` a layer)::

    state_indices      [rows] int32  slot per batch row; padding rows use the dummy slot
    has_initial_state  [rows] int32  1 if the request already has computed tokens
    num_valid          [rows] int32  real (unpadded) tokens of the row in this step
    <model extras>     per-token host inputs

Warmup emits the same keys and shapes so compiled graphs do not change.
Scheduling must be synchronous when the host inputs depend on the last sampled
token.

Prefix caching (align mode)
---------------------------
A cached KV prefix is only usable if the recurrent state at its end is also
available. With ``--enable-prefix-caching`` a model that declares
``supports_state_prefix_caching`` gets ``S`` extra **snapshot rows** in every
state pool::

    rows [0, max_num_seqs)                running slots (one per live request)
    rows [max_num_seqs, max_num_seqs + S) snapshot rows (scheduler-owned)
    row  max_num_seqs + S                 dummy slot (padding rows read it)
    (+ the model's own sink row after it)

The scheduler (:class:`StateSnapshotDirectory`, wired into the Neuron scheduler
by ``vllm_neuron/vllm/core/state_prefix_cache.py``) keeps a map from the block
hash at a ``block_size`` boundary to the snapshot row holding the state after
exactly that many tokens. A prefix-cache hit is truncated to the last boundary
that has a snapshot; prefill chunks are split at the boundaries worth keeping
(vLLM's align-mode split); a chunk that ends on one stores its state in a fresh
snapshot row. The per-step choice travels as
``SchedulerOutput.request_state_plan`` = ``{req_id: (read, write)}`` with
snapshot indices in ``[0, S)`` or ``RUNNING``. A published snapshot row is never
written again until it is evicted (LRU over rows that no request currently
reads).

The model graphs never see the plan: every row reads and writes its state at
``state_indices`` (its running slot), exactly as with prefix caching off. The
runner moves states between the running slots and the snapshot rows with row
copies outside the model graphs (:meth:`RequestStateSlots.restore_snapshots`
before the step's forward, :meth:`RequestStateSlots.save_snapshots` after it):

- read = snapshot row ``r``: before the step, every pool's row ``r`` is copied to
  the request's running slot, unless the slot already holds row ``r`` (the
  request stored ``r`` itself in its previous step and nothing wrote the slot
  since), so only a prefix-cache hit pays the copy;
- write = snapshot row ``r``: the graph writes the running slot as usual, and
  after the step the slot is copied to row ``r``.

(Earlier revisions passed the graphs a separate ``state_read_indices`` input and
read each row there while writing at ``state_indices``. On device (TP=32, no
MTP), gathering from and scattering into the same in-place pools through two
index inputs made identical steps run-to-run nondeterministic, even with both
indices equal; a copy ``pool[dst] = pool[src]`` has a true data dependency.)

The copies are issued from the worker thread in program order, on the same
(default) device stream as the model graphs, so a save runs after the step that
produced the state and before any later step that restores the row, with or
without async scheduling (the worker executes steps in submission order; the
scheduler only names rows ahead). One copy is one small graph over every pool
(:func:`copy_state_rows`, compiled once at the end of warm-up), or one eager
``copy_`` per pool with ``neuron_config.prefix_cache_state_copy = "eager"``.

Snapshot boundaries are multiples of ``G = lcm(block_size, A)``, where ``A`` is
the model's ``state_snapshot_alignment`` (the token grid of its chunked
recurrent prefill kernel; 128 for Qwen4-Exp's gated DeltaNet), see
:func:`state_snapshot_granularity`. Hits are truncated to the largest multiple
of ``G`` with a snapshot, and prefill chunks are cut on multiples of ``G``
(:meth:`StateSnapshotDirectory.split`), so a prefill that resumes from a hit
starts on the same kernel chunk grid as the cold run of the same tokens.

Which boundaries are kept (the scheduler passes ``long_prompt_tokens`` =
``LONG_PROMPT_CHUNKS`` x ``max_num_batched_tokens``):

- every prefill chunk end on the grid (the natural cuts of the token budget;
  they cost no extra step), for a prompt shorter than ``long_prompt_tokens``;
- for a longer prompt only the prompt-end boundary (the last one a lookup of the
  same tokens can return), where the cold prefill is stopped: one extra prefill
  step, small next to such a prompt's prefill;
- the shared-prefix junction (a KV hit that reached further than any state).

So a short prompt is chunked exactly as with prefix caching off.

Boundaries reached during decode and MTP verify (for multi-turn chat, whose next
prompt repeats the generated text) are kept only when
``neuron_config.prefix_cache_decode_snapshots`` is true (opt-in; default false):
with MTP every verify step writes one more state per row (the candidate at the
boundary inside the window, to a reserved row or the sink), which measured +12 %
ITL at B=16 on device. With the key false (or absent) states are kept only at
prefill boundaries (above) and the verify graphs have no ``state_snapshot_rows``
input. Without MTP the decode step that ends on a boundary stores its state like
a prefill chunk does (a row copy after the step). With MTP the graph writes the
candidate state through ``state_snapshot_rows``, a second write index into the
same pools inside the verify graph: that is the pattern the row copies removed
for prefill, and it may be nondeterministic on device (not measured since the
change); it stays opt-in. When enabled, the scheduler cannot
name such a snapshot when it plans the step: the tokens (and so the block hash)
and, with async scheduling and MTP, the number of accepted drafts are only known
from the step's output. So a row is reserved when a boundary *can* fall inside
the step, written by the step, and published (or released) when the step's output
arrives (:meth:`StateSnapshotDirectory.publish`):

- without speculative decoding a decode step is one token at a known position:
  the step that ends on a boundary stores its state in the reserved row (a row
  copy after the step, as for a prefill chunk);
- with MTP the graph gets ``state_snapshot_rows`` ``[rows]`` int32 (a reserved row
  or the dummy slot) and writes the candidate state at the boundary inside the
  ``1 + k`` window, ``offset = (-p0 - 1) mod block_size`` from the row's first
  position ``p0``, to that row when ``offset < count`` (the accepted tokens), else
  to the sink. Only one boundary fits in a window (``1 + k <= block_size``).
  Per request the latest decode boundary (and the one before it with MTP, whose
  KV hit drops a block) is kept; older ones are released.

``S`` is fixed per serve (it is part of every graph's pool shape):
``additional_config = {"neuron_config": {"prefix_cache_state_snapshots": S}}``,
default 16. With prefix caching off there are no snapshot rows, i.e. the pools
and graphs are those of a model without this feature; with it on only the pool
shapes differ (S more rows), plus the row-copy graph.
"""

import collections
import math

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

REQUEST_STATE_KEY = "request_state"
# SchedulerOutput attribute carrying the per-step snapshot plan.
PLAN_ATTR = "request_state_plan"
# neuron_config key of S, the number of snapshot rows, and its default.
SNAPSHOTS_KEY = "prefix_cache_state_snapshots"
DEFAULT_SNAPSHOTS = 16
# neuron_config key: also keep boundaries reached during decode / MTP verify (opt-in, default off).
DECODE_SNAPSHOTS_KEY = "prefix_cache_decode_snapshots"
# neuron_config key: how snapshot rows are copied ("graph": one compiled graph per copy, default; "eager": one eager
# copy_ per pool, the fallback if the graph does not compile or run on a stack).
STATE_COPY_KEY = "prefix_cache_state_copy"
STATE_COPY_MODES = ("graph", "eager")
# Plan value of "the request's running slot".
RUNNING = -1
# A prompt of at least this many max_num_batched_tokens chunks also stops at its prompt-end boundary (one extra
# prefill step); shorter prompts keep only the natural chunk ends, i.e. are chunked as with prefix caching off.
LONG_PROMPT_CHUNKS = 4
# Model class attribute: token grid of the model's chunked recurrent prefill (snapshot boundaries align to it).
ALIGNMENT_ATTR = "state_snapshot_alignment"


def state_snapshot_rows(vllm_config) -> int:
    """S for this serve: 0 unless prefix caching is on for a model with per-request state.

    The scheduler and every worker call this on the same config, so both sides agree.
    """
    if not getattr(vllm_config.cache_config, "enable_prefix_caching", False):
        return 0
    from vllm_neuron.vllm.platform import _STATE_PREFIX_CACHING_ARCHS

    hf_config = getattr(vllm_config.model_config, "hf_config", None)
    if not set(getattr(hf_config, "architectures", None) or ()) & _STATE_PREFIX_CACHING_ARCHS:
        return 0
    neuron_config = (getattr(vllm_config, "additional_config", None) or {}).get("neuron_config") or {}
    s = neuron_config.get(SNAPSHOTS_KEY, DEFAULT_SNAPSHOTS)
    if isinstance(s, bool) or not isinstance(s, int) or s < 1:
        raise ValueError(f"neuron_config.{SNAPSHOTS_KEY} must be a positive integer, got {s!r} "
                         "(turn prefix caching off with --no-enable-prefix-caching instead)")
    return s


def state_snapshot_granularity(vllm_config) -> int:
    """G: snapshot boundaries (and prefill chunk cuts) are multiples of ``lcm(block_size, alignment)``.

    ``alignment`` is the model class's ``state_snapshot_alignment`` (1 if it declares none): the token grid of
    its chunked recurrent prefill kernel, which counts its chunks from the start of each prefill step. A warm
    prefill must resume on that grid, or it partitions the tokens differently from the cold run (another fp32
    accumulation order).
    """
    from vllm_neuron.model.registry import get_models

    block_size = vllm_config.cache_config.block_size
    models = dict(get_models())
    hf_config = getattr(vllm_config.model_config, "hf_config", None)
    archs = getattr(hf_config, "architectures", None) or ()
    return math.lcm(block_size, *(int(getattr(models[a], ALIGNMENT_ATTR, 1)) for a in archs if a in models))


# ModelConfig attribute set by the EngineArgs probe below when the user gave neither
# --enable-prefix-caching nor --no-enable-prefix-caching (vLLM then turns prefix caching on).
PREFIX_CACHING_DEFAULTED_ATTR = "_neuron_prefix_caching_defaulted"


def probe_prefix_caching_default() -> None:
    """Make ``EngineArgs`` record on the ModelConfig whether ``enable_prefix_caching`` was unset.

    vLLM resolves an unset flag to the model's default before any config reaches the platform,
    which then cannot tell ``--enable-prefix-caching`` from no flag. Idempotent; installed from
    ``NeuronPlatform.pre_register_and_update``, which ``create_engine_config`` calls first.
    """
    from vllm.engine.arg_utils import EngineArgs

    orig = EngineArgs._set_default_chunked_prefill_and_prefix_caching_args
    if getattr(orig, "_neuron_prefix_caching_probe", False):
        return

    def probed(self, model_config):
        defaulted = self.enable_prefix_caching is None
        orig(self, model_config)
        if defaulted:
            setattr(model_config, PREFIX_CACHING_DEFAULTED_ATTR, True)

    probed._neuron_prefix_caching_probe = True
    EngineArgs._set_default_chunked_prefill_and_prefix_caching_args = probed


def segmented_prefill_off_reason(vllm_config) -> str | None:
    """Why the Neuron runner will not run segmented prefill for this config (None if it will).

    Segmented prefill is the runner's precondition for prefix caching (it raises at start-up
    otherwise). Mirrors ``NeuronModelRunner.__init__``: explicit ``kv_segment_size_buckets``
    turn it on (or fail there); the default is on only when ``max_num_batched_tokens`` is below
    ``max_model_len``, is a supported segment size, and the KV cache is not non-packed FP8.
    """
    from vllm_neuron.utils.bucket_utils import (
        SUPPORTED_KV_SEGMENT_SIZES,
        get_default_kv_segment_size_buckets,
        get_max_num_batched_tokens,
    )

    neuron_config = (getattr(vllm_config, "additional_config", None) or {}).get("neuron_config") or {}
    if "kv_segment_size_buckets" in neuron_config:
        return None
    cache_dtype = vllm_config.cache_config.cache_dtype
    if isinstance(cache_dtype, str) and cache_dtype.startswith("fp8") and not neuron_config.get(
            "fp8_packed_kv", False):
        return f"the KV cache is non-packed {cache_dtype}"
    mml = vllm_config.model_config.max_model_len
    mbt = get_max_num_batched_tokens(vllm_config.scheduler_config.max_num_batched_tokens, mml)
    if get_default_kv_segment_size_buckets(mbt, mml) is not None:
        return None
    if mbt >= mml:
        return f"prefill is single-shot: max_num_batched_tokens {mbt} >= max_model_len {mml}"
    return (f"max_num_batched_tokens {mbt} is not a supported segment size "
            f"{sorted(SUPPORTED_KV_SEGMENT_SIZES)}")


def single_shot_segment_buckets(vllm_config) -> list[int] | None:
    """``kv_segment_size_buckets`` that let prefix caching run with a single-shot prefill size, else None.

    With ``max_num_batched_tokens >= max_model_len`` the runner does not default segmented prefill on, and it refuses
    prefix caching without it. Models with snapshot rows (Qwen4-Exp) read the cached KV through the block table in
    every prefill graph; for them a segment size is only the label of the graphs a step dispatches. One explicit size
    (the smallest supported one >= ``max_model_len``, capped at the largest) keeps one prefill graph per
    ``num_batched_tokens`` bucket, as without prefix caching, and a cold prompt is still one prefill step. Returns None
    when the user set ``kv_segment_size_buckets``, when prefill is chunked anyway, or with a non-packed FP8 KV cache.
    """
    from vllm_neuron.utils.bucket_utils import SUPPORTED_KV_SEGMENT_SIZES, get_max_num_batched_tokens

    neuron_config = (getattr(vllm_config, "additional_config", None) or {}).get("neuron_config") or {}
    if "kv_segment_size_buckets" in neuron_config:
        return None
    cache_dtype = vllm_config.cache_config.cache_dtype
    if isinstance(cache_dtype, str) and cache_dtype.startswith("fp8") and not neuron_config.get(
            "fp8_packed_kv", False):
        return None
    mml = vllm_config.model_config.max_model_len
    if get_max_num_batched_tokens(vllm_config.scheduler_config.max_num_batched_tokens, mml) < mml:
        return None
    sizes = sorted(SUPPORTED_KV_SEGMENT_SIZES)
    return [next((x for x in sizes if x >= mml), sizes[-1])]


def prefix_caching_auto_off_reason(vllm_config) -> str | None:
    """Why prefix caching that vLLM turned on by default should be turned off again, else None.

    "Auto when usable": with neither flag given, prefix caching stays on only when the runner
    runs segmented prefill. An explicit ``--enable-prefix-caching`` is never overridden (an
    unusable config then fails at start-up with the runner's message).
    """
    if not getattr(vllm_config.model_config, PREFIX_CACHING_DEFAULTED_ATTR, False):
        return None
    return segmented_prefill_off_reason(vllm_config)


def decode_snapshots_enabled(vllm_config) -> bool:
    """Whether boundaries reached during decode / MTP verify are kept (prefix caching on; opt-in, default off)."""
    if not state_snapshot_rows(vllm_config):
        return False
    neuron_config = (getattr(vllm_config, "additional_config", None) or {}).get("neuron_config") or {}
    v = neuron_config.get(DECODE_SNAPSHOTS_KEY, False)
    if not isinstance(v, bool):
        raise ValueError(f"neuron_config.{DECODE_SNAPSHOTS_KEY} must be true or false, got {v!r}")
    return v


def state_copy_mode(vllm_config) -> str:
    """How snapshot rows are copied (``neuron_config.prefix_cache_state_copy``, default "graph")."""
    neuron_config = (getattr(vllm_config, "additional_config", None) or {}).get("neuron_config") or {}
    mode = neuron_config.get(STATE_COPY_KEY, "graph")
    if mode not in STATE_COPY_MODES:
        raise ValueError(f"neuron_config.{STATE_COPY_KEY} must be one of {STATE_COPY_MODES}, got {mode!r}")
    return mode


def state_snapshot_options(vllm_config) -> dict:
    """``RequestStateSlots`` keyword arguments for this serve (the runner passes them)."""
    rows = state_snapshot_rows(vllm_config)
    spec = getattr(vllm_config, "speculative_config", None)
    num_spec = int(getattr(spec, "num_speculative_tokens", 0) or 0) if spec is not None else 0
    return dict(num_snapshots=rows, block_size=vllm_config.cache_config.block_size,
                verify_snapshots=bool(rows and num_spec and decode_snapshots_enabled(vllm_config)),
                copy_mode=state_copy_mode(vllm_config) if rows else "graph")


def copy_state_rows(pools, src: torch.Tensor, dst: torch.Tensor) -> None:
    """``pool[dst] = pool[src]`` for every pool (``src`` / ``dst``: ``[1]`` int32 row indices on the pools' device).

    The row-copy graph of prefix caching (compiled once per serve). The value written is the row read, so the
    scatter depends on the gather; the read uses the flattened pool and the write the 1-D index form, as the
    model's ``_pool_rows`` / ``_pool_write`` do (an index_select on the [n, C, K] pool lowered to many small
    indirect DMAs; index_copy_ to element-wise scatters).
    """
    src, dst = src.long(), dst.long()
    for pool in pools:
        row = pool.view(pool.shape[0], -1).index_select(0, src)
        pool[dst] = row.view(1, *pool.shape[1:])


def copy_state_rows_eager(pools, src: int, dst: int) -> None:
    """The same copy as one eager ``copy_`` per pool (``neuron_config.prefix_cache_state_copy = "eager"``)."""
    for pool in pools:
        pool[dst].copy_(pool[src])


def model_uses_request_state(model) -> bool:
    inner = getattr(model, "_orig_mod", model)
    return hasattr(inner, "allocate_request_state")


class RequestStateSlots:
    def __init__(self, model, max_num_reqs: int, device, num_snapshots: int = 0, block_size: int = 0,
                 verify_snapshots: bool = False, copy_mode: str = "graph", compile_copy=None):
        self.model = getattr(model, "_orig_mod", model)
        if num_snapshots and not getattr(self.model, "supports_state_prefix_caching", False):
            raise ValueError(f"{type(self.model).__name__} does not support prefix caching with "
                             "per-request state; run with --no-enable-prefix-caching")
        if copy_mode not in STATE_COPY_MODES:
            raise ValueError(f"copy_mode must be one of {STATE_COPY_MODES}, got {copy_mode!r}")
        # One slot per concurrently scheduled request, then the snapshot rows (prefix
        # caching), plus a dummy slot for padding rows.
        self.num_snapshots = num_snapshots
        self.snapshot_base = max_num_reqs
        self.num_slots = max_num_reqs + num_snapshots + 1
        self.dummy_slot = max_num_reqs + num_snapshots
        self.device = device
        self._slot_of: dict[str, int] = {}
        self._free = list(range(max_num_reqs - 1, -1, -1))
        # Running slots that hold their request's state (written by a step: its forward, after any restore).
        self._written: set[int] = set()
        self._plan: dict[str, tuple[int, int]] = {}
        # Prefix caching: running slot -> snapshot index whose state it holds (the slot was copied to that row
        # after its last step, or restored from it), so the next step that reads the row needs no copy.
        self._holds: dict[int, int] = {}
        # The step being built (runtime_metadata) -> executed (restore_snapshots / save_snapshots):
        # [(req_id, slot, read, write)] of its real rows, and the copies [(src pool row, dst pool row)].
        self._step: list[tuple[str, int, int, int]] = []
        self._restores: list[tuple[int, int]] = []
        self._restored: set[tuple[int, int]] = set()
        self._saves: list[tuple[int, int]] = []
        self.copy_stats = collections.Counter()
        # MTP: the graph writes the state at a boundary inside the verify window to a reserved row.
        self.verify_snapshots = bool(verify_snapshots and num_snapshots)
        self._snap: dict[str, int] = {}
        if self.verify_snapshots:
            if not block_size:
                raise ValueError("verify snapshots need the block size")
            self.model.state_snapshot_block_size = int(block_size)
        self.model.allocate_request_state(self.num_slots, device)
        self.copy_mode = copy_mode
        self._copy_fn = None
        self._row_index: list[torch.Tensor] = []
        if num_snapshots:
            if not hasattr(self.model, "request_state_pools"):
                raise ValueError(f"{type(self.model).__name__} declares supports_state_prefix_caching but has no "
                                 "request_state_pools() (the pools the snapshot row copies move)")
            if copy_mode == "graph":
                # [1] int32 index of every pool row (the state_indices dtype), made once: a copy then needs no
                # host-to-device write.
                self._row_index = [torch.tensor([i], dtype=torch.int32).to(device) for i in range(self.num_slots)]
                self._copy_fn = compile_copy(copy_state_rows) if compile_copy is not None else copy_state_rows
        self.layout = None  # (rows [T], positions [T], num_valid per batch row), set per step
        logger.info("[request_state] %d slots + %d snapshot rows (+1 dummy) allocated%s",
                    max_num_reqs, num_snapshots,
                    f"; snapshot row copies: {copy_mode}" if num_snapshots else "")

    # ── Slot lifecycle ───────────────────────────────────────────────────
    def release(self, req_ids) -> None:
        for req_id in req_ids or ():
            slot = self._slot_of.pop(req_id, None)
            if slot is not None:
                self._written.discard(slot)
                self._holds.pop(slot, None)
                self._free.append(slot)

    def set_plan(self, plan) -> None:
        """The step's snapshot plan (``SchedulerOutput.request_state_plan``; absent = all running)."""
        plan = dict(plan or {})
        if plan and not self.num_snapshots:
            raise RuntimeError("request-state plan without snapshot rows (prefix caching is off)")
        snap = {}
        for req_id, entry in plan.items():
            if len(entry) == 3:  # (read, write, verify-snapshot row)
                if not self.verify_snapshots:
                    raise RuntimeError(f"request-state plan for {req_id}: a verify snapshot without "
                                       "state_snapshot_rows (MTP verify snapshots are off)")
                snap[req_id] = entry[2]
            if not all(RUNNING <= v < self.num_snapshots for v in entry) or len(entry) not in (2, 3):
                raise ValueError(f"request-state plan for {req_id}: {entry} outside "
                                 f"[{RUNNING}, {self.num_snapshots})")
        self._plan = {req_id: tuple(entry[:2]) for req_id, entry in plan.items()}
        self._snap = snap
        self._step, self._restores, self._saves = [], [], []

    def _row(self, snapshot: int, slot: int) -> int:
        return slot if snapshot == RUNNING else self.snapshot_base + snapshot

    def _slot(self, req_id: str) -> int:
        slot = self._slot_of.get(req_id)
        if slot is None:
            if not self._free:
                raise RuntimeError(
                    f"no free request-state slot for {req_id} ({self.num_slots - 1} in use)"
                )
            slot = self._slot_of[req_id] = self._free.pop()
        return slot

    # ── Per-step metadata ────────────────────────────────────────────────
    def record_layout(self, rows: np.ndarray, positions: np.ndarray, num_valid: np.ndarray) -> None:
        """Token layout of the step: table row (-1 = padding) and position per token,
        and real tokens per batch row."""
        self.layout = (rows, positions, num_valid)

    def runtime_metadata(self, input_batch, num_rows: int, num_tokens: int) -> dict:
        """The step's ``request_state`` inputs; also fixes the step's snapshot row copies (from the plan).

        Every row reads and writes its running slot. The copies are executed by :meth:`restore_snapshots`
        (before the forward) and :meth:`save_snapshots` (after it). Idempotent within a step (the runner may
        prepare a step's inputs twice).
        """
        rows, positions, num_valid = self.layout
        num_reqs = input_batch.num_reqs
        state_indices = np.full(num_rows, self.dummy_slot, dtype=np.int32)
        has_init = np.zeros(num_rows, dtype=np.int32)
        valid = np.zeros(num_rows, dtype=np.int32)
        snap_rows = np.full(num_rows, self.dummy_slot, dtype=np.int32)
        step, restores, saves = [], [], []
        for r in range(min(num_reqs, num_rows)):
            req_id = input_batch.req_ids[r]
            slot = self._slot(req_id)
            rd, wr = self._plan.get(req_id, (RUNNING, RUNNING))
            state_indices[r] = slot
            if req_id in self._snap:
                snap_rows[r] = self.snapshot_base + self._snap[req_id]
            has_init[r] = int(input_batch.num_computed_tokens_cpu[r] > 0)
            valid[r] = num_valid[r]
            if rd != RUNNING and self._holds.get(slot) != rd:
                restores.append((self._row(rd, slot), slot))
            if has_init[r] and rd == RUNNING and slot not in self._written:
                # A request that resumes from cached tokens must restore their snapshot: its
                # running slot was just assigned and holds another request's state.
                raise RuntimeError(f"request {req_id} has {int(input_batch.num_computed_tokens_cpu[r])} "
                                   f"computed tokens but its slot {slot} was never written "
                                   "(prefix-cache hit without a state snapshot)")
            # The forward writes every real row's running slot (restored first if the plan reads a snapshot).
            self._written.add(slot)
            if wr != RUNNING:
                saves.append((slot, self._row(wr, slot)))
            step.append((req_id, slot, rd, wr))
        if {src for src, _ in restores} & {dst for _, dst in saves}:
            raise RuntimeError(f"a snapshot row is restored and written in the same step: {restores} / {saves}")
        self._step, self._restores, self._saves = step, restores, saves
        # Tokens added by bucket padding after the layout was recorded are padding.
        tok_rows = np.full(num_tokens, -1, dtype=np.int64)
        tok_pos = np.zeros(num_tokens, dtype=np.int64)
        n = min(num_tokens, rows.shape[0])
        tok_rows[:n], tok_pos[:n] = rows[:n], positions[:n]
        extras = self.model.request_state_host_inputs(input_batch.token_ids_cpu, tok_rows, tok_pos)
        return self._to_device(state_indices, has_init, valid, extras, snap_rows)

    def warmup_metadata(self, num_rows: int, num_tokens: int) -> dict:
        state_indices = np.full(num_rows, self.dummy_slot, dtype=np.int32)
        zeros = np.zeros(num_rows, dtype=np.int32)
        extras = self.model.request_state_warmup_inputs(num_tokens)
        self._step, self._restores, self._saves = [], [], []
        return self._to_device(state_indices, zeros, zeros.copy(), extras, state_indices.copy())

    def _to_device(self, state_indices, has_init, num_valid, extras, snap_rows) -> dict:
        md = {
            "state_indices": torch.from_numpy(state_indices),
            "has_initial_state": torch.from_numpy(has_init),
            "num_valid": torch.from_numpy(num_valid),
            **extras,
        }
        if self.verify_snapshots:
            # MTP verify snapshots: the reserved row per batch row, the dummy slot for none.
            md["state_snapshot_rows"] = torch.from_numpy(snap_rows)
        return {k: v.to(self.device) for k, v in md.items()}

    # ── Prefix caching: snapshot row copies around the step ──────────────
    def copy_rows(self, src: int, dst: int) -> None:
        """Copy pool row ``src`` to pool row ``dst`` in every state pool of the model."""
        pools = self.model.request_state_pools()
        if self.copy_mode == "eager":
            copy_state_rows_eager(pools, src, dst)
        else:
            self._copy_fn(pools, self._row_index[src], self._row_index[dst])

    def restore_snapshots(self) -> int:
        """Before the step's forward: copy each read snapshot row to its request's running slot."""
        for src, dst in self._restores:
            self.copy_rows(src, dst)
            self.copy_stats["restores"] += 1
        n, self._restored, self._restores = len(self._restores), set(self._restores), []
        return n

    def save_snapshots(self) -> int:
        """After the step's forward: copy each running slot the plan stores to its snapshot row."""
        for _, slot, rd, wr in self._step:
            # The forward wrote every real row's slot: it holds the snapshot it is copied to now, or none.
            if wr != RUNNING:
                self._holds[slot] = wr
            else:
                self._holds.pop(slot, None)
            # A read the slot already held (the request stored that row in its previous step): no copy.
            self.copy_stats["reads_held"] += int(rd != RUNNING and (self._row(rd, slot), slot) not in self._restored)
        for src, dst in self._saves:
            self.copy_rows(src, dst)
            self.copy_stats["saves"] += 1
        n = len(self._saves)
        self._step, self._saves, self._restored = [], [], set()
        return n

    def warmup_copy(self) -> None:
        """Compile (and run once) the row copy: the dummy slot onto itself, a no-op on the values.

        The destination index is a second tensor: passing one tensor as both inputs would compile an
        aliased-input variant, not the graph that serving runs (source and destination always differ there).
        """
        if not self.num_snapshots:
            return
        if self.copy_mode == "eager":
            copy_state_rows_eager(self.model.request_state_pools(), self.dummy_slot, self.dummy_slot)
        else:
            dst = torch.tensor([self.dummy_slot], dtype=torch.int32).to(self.device)
            self._copy_fn(self.model.request_state_pools(), self._row_index[self.dummy_slot], dst)


# ── Prefix caching: the scheduler-side snapshot directory ─────────────────────
class StateSnapshotDirectory:
    """Which snapshot row holds the recurrent state at which prefix boundary (scheduler process).

    Keys are vLLM block hashes: ``request.block_hashes[b // hash_block_size - 1]`` names the
    exact first ``b`` tokens (and the request's extra keys), so a row found under it holds
    the state after exactly those tokens. Rows are evicted LRU; a row is pinned while it
    holds some request's current state (that request still has to read it) and while the
    step being planned reads it. Rows are assigned and published when a step is scheduled:
    every rank executes steps in submission order, so step N writes a row before any later
    step reads it, with or without async scheduling.

    Lifecycle per request: :meth:`lookup` at admission (prefix-cache hit) -> :meth:`plan`
    every scheduled step (+ :meth:`take_pending` / :meth:`publish` around the step's output when
    it may reach a boundary in decode) -> :meth:`release` on finish / abort / preemption.

    Decode snapshots (``decode_snapshots``): ``side_write`` (MTP) reserves a row the graph writes
    besides the running slot; otherwise the decode step that ends on the boundary stores its state in
    the row (the worker copies the running slot there after the step). ``lookahead`` is how far the scheduler's ``num_computed_tokens``
    can be ahead of the step's real start (async scheduling with MTP: the previous step's drafts
    count as accepted until its output arrives).
    """

    def __init__(self, num_rows: int, block_size: int, hash_block_size: int | None = None,
                 use_eagle: bool = False, decode_snapshots: bool = False, side_write: bool = False,
                 lookahead: int = 0, max_model_len: int | None = None, granularity: int | None = None,
                 long_prompt_tokens: int | None = None):
        hash_block_size = hash_block_size or block_size
        granularity = granularity or block_size
        if num_rows < 1 or block_size % hash_block_size or granularity % block_size:
            raise ValueError(f"snapshot rows {num_rows}, block {block_size}, hash block {hash_block_size}, "
                             f"granularity {granularity}")
        self.num_rows = num_rows
        self.block_size = block_size
        # G: every snapshot boundary (and every prefill chunk cut) is a multiple of it.
        self.granularity = granularity
        # Prompts with at least this many tokens stop at the prompt end; shorter ones keep the natural chunk ends
        # instead. None: every prompt stops at the prompt end and natural chunk ends are not kept.
        self.long_prompt_tokens = long_prompt_tokens
        self.hash_block_size = hash_block_size
        self.use_eagle = use_eagle
        self._entries: collections.OrderedDict = collections.OrderedDict()  # hash -> row, oldest first
        self._hash_of_row: dict[int, object] = {}
        self._free = list(range(num_rows - 1, -1, -1))
        self._current: dict[str, int] = {}  # req_id -> snapshot row holding its state (absent: running slot)
        self._pins: collections.Counter = collections.Counter()
        self._pending: dict[str, int] = {}  # req_id -> row of a hit whose request is not admitted yet
        self.stats = collections.Counter()
        self.decode_snapshots = decode_snapshots
        self.side_write = side_write
        self.lookahead = lookahead
        self.max_model_len = max_model_len
        # Decode snapshots: rows written by a step whose output has not arrived -> (req_id, epoch);
        # the rows reserved by the last plan(); per request the hashes of its published decode
        # snapshots (oldest first). The epoch counts reset() calls (an older write is dropped).
        self._inflight: dict[int, tuple[str, int]] = {}
        self._planned: list[tuple[str, int]] = []
        self._decode_hashes: dict[str, list] = {}
        self._epoch = 0

    # ── Boundaries ───────────────────────────────────────────────────────
    def _hash(self, request, boundary: int):
        return request.block_hashes[boundary // self.hash_block_size - 1]

    def _long(self, request) -> bool:
        return self.long_prompt_tokens is None or request.num_prompt_tokens >= self.long_prompt_tokens

    def keep_boundaries(self, request) -> set[int]:
        """Boundaries whose state is kept when a prefill chunk ends there (also the split stops).

        The two prompt-end ones only for a long prompt (:attr:`long_prompt_tokens`):

        - vLLM's ``last_cache_position``: the last full block of the known tokens, one block
          less with Eagle / MTP (the KV hit drops its last block for the drafter);
        - the largest boundary a lookup of the same tokens can return (the hit is capped at
          ``num_tokens - 1``): differs from the above only when ``num_tokens % block_size == 0``,
          where vLLM's position would be unreachable for an identical prompt;
        - the shared-prefix junction (a KV hit that reached further than any snapshot);

        each floored to a multiple of the granularity ``G`` (a hit can only return such a boundary).
        """
        bs, g, n = self.block_size, self.granularity, request.num_tokens
        back = bs if self.use_eagle else 0
        last = max(n - n % bs - back, 0) // g * g
        reach = max((n - 1) // bs * bs - back, 0) // g * g
        junction = getattr(request, "shared_prefix_boundary", 0) // g * g
        if not self._long(request):
            last = reach = 0
        return {b for b in (last, reach, junction) if 0 < b <= n}

    def keeps(self, request, end: int) -> bool:
        """Whether a prefill chunk ending at ``end`` keeps its state: a keep boundary, or (short prompt) any chunk
        end on the grid."""
        return end in self.keep_boundaries(request) or (
            not self._long(request) and end > 0 and end % self.granularity == 0)

    def extra_stops(self, request, start: int, num_new_tokens: int) -> int:
        """``num_new_tokens`` clipped so the chunk ends at the first keep boundary inside it."""
        end = start + num_new_tokens
        inside = [b for b in self.keep_boundaries(request) if start < b < end]
        return (min(inside) - start) if inside else num_new_tokens

    def split(self, request, start: int, num_new_tokens: int, max_prefill_tokens: int) -> int:
        """vLLM's align-mode split (``Scheduler._mamba_block_aligned_split``) on the grid ``G``.

        A prefill chunk that does not finish the prefill ends on a multiple of ``G`` (if the budget allows one;
        else, as vLLM does for a block wider than the chunk limit, it advances short and the next chunk stops at
        the next multiple of ``G``), and stops at every keep boundary inside it. Starting from 0 or from a hit
        (a multiple of ``G``), every chunk then starts on a multiple of ``G``. With ``G = block_size`` this is
        vLLM's split plus :meth:`extra_stops` (no partial-hash tail stop, no prefill checkpoint blocks).
        """
        prefill_end = max(request.num_prompt_tokens, request.num_tokens - 1)
        if start >= prefill_end:
            return num_new_tokens
        g = self.granularity
        end = start + num_new_tokens
        if end < prefill_end:
            aligned = end // g * g
            if aligned > start or g <= max_prefill_tokens:
                end = aligned
        stops = set(self.keep_boundaries(request))
        if start % g:
            stops.add((start // g + 1) * g)
        end = min((b for b in stops if start < b < end), default=end)
        return max(end - start, 0)

    # ── Lookup (admission) ───────────────────────────────────────────────
    def lookup(self, request, max_tokens: int) -> int:
        """The largest boundary ``<= max_tokens`` with a stored state (0 if none).

        ``max_tokens`` is the KV prefix-cache hit. The row is remembered for the request and
        becomes its read row if the request is admitted in this step (:meth:`plan`).
        """
        self.stats["lookups"] += 1
        self._pending.pop(request.request_id, None)
        b = max_tokens // self.granularity * self.granularity
        while b > 0:
            row = self._entries.get(self._hash(request, b))
            if row is not None:
                self._entries.move_to_end(self._hash(request, b))
                self._pending[request.request_id] = row
                return b
            b -= self.granularity
        return 0

    # ── Per-step plan ────────────────────────────────────────────────────
    def plan(self, scheduled) -> dict[str, tuple[int, int]]:
        """``{req_id: (read, write)}`` for the step; ``scheduled`` = ``[(request, start, num_tokens)]``.

        ``start`` is the request's computed tokens before the step. Requests reading and
        writing their running slot are left out.
        """
        # Admitted hits: their snapshot row becomes the request's current state.
        for request, start, _ in scheduled:
            row = self._pending.pop(request.request_id, None)
            if row is not None and start > 0:
                self._set_current(request.request_id, row)
                self.stats["hits"] += 1
                self.stats["hit_tokens"] += start
        self._pending.clear()  # lookups of requests that were not admitted are redone later
        reads = {request.request_id: self._current.get(request.request_id, RUNNING)
                 for request, _, _ in scheduled}
        step_pins = {row for row in reads.values() if row != RUNNING}
        plan = {}
        self._planned = []
        for request, start, num_tokens in scheduled:
            rid, end = request.request_id, start + num_tokens
            write = snap = RUNNING
            prefill_end = max(request.num_prompt_tokens, request.num_tokens - 1)
            if start < prefill_end:
                if self.keeps(request, end):
                    h = self._hash(request, end)
                    if h in self._entries:
                        self._entries.move_to_end(h)  # already stored (same tokens)
                    else:
                        write = self._allocate(step_pins)
                        if write != RUNNING:
                            self._entries[h] = write
                            self._hash_of_row[write] = h
                            step_pins.add(write)
                            self.stats["snapshots"] += 1
            elif self.decode_snapshots and self._decode_window(start, num_tokens):
                row = self._allocate(step_pins)
                if row != RUNNING:
                    step_pins.add(row)
                    self._inflight[row] = (rid, self._epoch)
                    self._planned.append((rid, row))
                    self.stats["decode_reserved"] += 1
                    if self.side_write:
                        snap = row
                    else:
                        write = row
            self._set_current(rid, write)
            if snap != RUNNING:
                plan[rid] = (reads[rid], write, snap)
            elif (reads[rid], write) != (RUNNING, RUNNING):
                plan[rid] = (reads[rid], write)
        return plan

    def _decode_window(self, start: int, num_tokens: int) -> bool:
        """Whether a decode / verify step scheduled at ``start`` can end on or cross a boundary (a multiple of G)."""
        g = self.granularity
        if not self.side_write:
            return num_tokens == 1 and (start + 1) % g == 0
        if self.lookahead and self.max_model_len is not None and (
                start + num_tokens > self.max_model_len - 3 - 3 * self.lookahead):
            return False  # near max_model_len async MTP stops drafting; positions are not exact there
        lo = start - self.lookahead  # the step's real start is in [lo, start]
        return (lo // g + 1) * g <= start + num_tokens

    # ── Decode snapshots: publication after the step's output ─────────────
    def take_pending(self) -> list[tuple[str, int]]:
        """``[(req_id, row)]`` reserved by the last :meth:`plan` (the scheduler keeps them with the step)."""
        out, self._planned = self._planned, []
        return out

    def publish(self, items) -> None:
        """After a step's output: ``items`` = ``[(request, num_tokens_before, row)]`` of its reserved rows.

        ``num_tokens_before`` is ``request.num_tokens`` before the output was applied, so the step
        started at ``p0 = num_tokens_before - 1`` (the last known token is its first input) and
        processed the tokens up to ``request.num_tokens - 1``. The row holds the state after the
        first boundary ``b > p0`` exactly when ``b <= request.num_tokens - 1`` (redirect: the step
        ended there; MTP: the graph's ``offset < count``). It is published under the hash of the
        first ``b`` tokens, else returned (or freed when its last reader lets go of it). The graph writes the
        first *block* boundary of the window; it is kept only if that is a multiple of G (with async MTP the
        window is a bound, so a reserved step can hold another block boundary instead).
        """
        bs, keep = self.block_size, 2 if self.use_eagle else 1
        for request, n_before, row in items:
            rid, epoch = self._inflight.pop(row)
            h = None
            if request is not None and n_before is not None and epoch == self._epoch:
                b = (n_before - 1) // bs * bs + bs
                if b % self.granularity == 0 and b <= request.num_tokens - 1 and len(
                        request.block_hashes) >= b // self.hash_block_size:
                    h = self._hash(request, b)
                    if h in self._entries:
                        h = None  # the same tokens are stored already
            if h is None:
                self.stats["decode_dropped"] += 1
                if not self._pins[row]:
                    self._free.append(row)
            else:
                self._entries[h] = row
                self._hash_of_row[row] = h
                self.stats["decode_snapshots"] += 1
                hashes = self._decode_hashes.setdefault(rid, [])
                hashes.append(h)
                while len(hashes) > keep:  # an older decode boundary of the same request
                    old = hashes.pop(0)
                    r = self._entries.get(old)
                    if r is not None and not self._pins[r]:
                        del self._entries[old]
                        del self._hash_of_row[r]
                        self._free.append(r)
                        self.stats["decode_rolled"] += 1
            if request is not None and request.is_finished():
                self._decode_hashes.pop(rid, None)

    def _allocate(self, step_pins) -> int:
        if self._free:
            return self._free.pop()
        for h, row in self._entries.items():  # least recently used first
            if not self._pins[row] and row not in step_pins:
                del self._entries[h]
                del self._hash_of_row[row]
                self.stats["evictions"] += 1
                return row
        self.stats["no_row"] += 1
        return RUNNING

    def _set_current(self, req_id: str, row: int) -> None:
        old = self._current.pop(req_id, RUNNING)
        if row != RUNNING:
            self._current[req_id] = row
            self._pins[row] += 1
        if old != RUNNING:
            self._pins[old] -= 1
            if not self._pins[old]:
                del self._pins[old]
                # Dropped by reset() while pinned, or a decode snapshot not published; one still
                # in flight is returned by publish().
                if old not in self._hash_of_row and old not in self._inflight:
                    self._free.append(old)

    # ── Request end / reset ──────────────────────────────────────────────
    def release(self, req_id: str) -> None:
        """Finish, abort or preemption: the request no longer needs its current row."""
        self._pending.pop(req_id, None)
        self._set_current(req_id, RUNNING)
        if not any(r == req_id for r, _ in self._inflight.values()):
            self._decode_hashes.pop(req_id, None)  # else publish() drops it after the last output

    def reset(self) -> None:
        """Forget every stored state (``reset_prefix_cache``); rows still pinned free on release."""
        for row in list(self._hash_of_row):
            if not self._pins[row]:
                self._free.append(row)
        self._entries.clear()
        self._hash_of_row.clear()
        self._pending.clear()
        self._decode_hashes.clear()
        self._epoch += 1  # rows still in flight are dropped when their output arrives
        self.stats["resets"] += 1

    # ── Introspection (tests, logs) ──────────────────────────────────────
    def num_stored(self) -> int:
        return len(self._entries)

    def current_row(self, req_id: str) -> int:
        return self._current.get(req_id, RUNNING)

    def check(self) -> None:
        """Internal consistency (tests): every row is free, stored or pinned, exactly once."""
        stored, free = set(self._hash_of_row), set(self._free)
        pinned = {r for r, n in self._pins.items() if n > 0}
        inflight = set(self._inflight)
        assert len(free) == len(self._free), "row freed twice"
        assert not (stored & free) and not (pinned & free), "a free row is in use"
        assert not (inflight & free) and not (inflight & stored), "an in-flight row is free or stored"
        assert stored | free | pinned | inflight == set(range(self.num_rows)), "row leaked"
        assert {v: k for k, v in self._entries.items()} == self._hash_of_row
        assert sorted(self._current.values()) == sorted(self._pins.elements())
