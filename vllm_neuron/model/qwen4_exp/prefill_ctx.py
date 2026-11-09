# SPDX-License-Identifier: Apache-2.0
"""Prefill context buckets for Qwen4-Exp (host side only).

The v1 runner hands every prefill-shaped step the block table at the full ``max_model_len`` width (only decode is
trimmed, by ``decode_context_length_buckets``). With QSA attention the cost of a prefill chunk follows that width,
not the request's context: at ``max_model_len`` 81,920 the first chunk of a short prompt scores and masks 81,920
positions. This module trims the block table of a **real prefill step** to the smallest context bucket that holds
every position the step (and the MTP draft that follows it in the same step) can touch, with the same
``index_select`` trim decode uses, and makes warm-up compile each prefill bucket ``T`` at every context bucket
``>= T`` as well as at ``max_model_len``.

Rule (no environment variables):

* enabled only when ``max_model_len > 32,768``, with data / decode-context / prefill-context parallel size 1;
* buckets: ``2,048``, ``8,192`` and ``32,768`` (those below ``max_model_len``); a step that needs more keeps the
  full table;
* a step is trimmed only if it is prefill-shaped: its query length is above the decode threshold of the runner
  (``1 + num_speculative_tokens``) **and** above ``1 +`` its own draft count. Decode, MTP verify and the MTP draft
  steps keep their graphs and tables unchanged (decode keeps ``decode_context_length_buckets``);
* the bucket is at least the step's padded query length, so every graph used at runtime is one warm-up compiled;
* segmented-prefill steps (``kv_segment_size > 0``; with the default ``kv_segment_size_buckets`` every chunk whose
  length is a multiple of 128 is one, the first included) are trimmed the same way: this model's attention reads only
  the block table and positions, ``kv_segment_size`` is just carried in the metadata (no segment layout depends on
  the table width), and warm-up builds every ``(T, kv_segment_size)`` prefill target at each bucket ``>= T``.

At ``max_model_len <= 32,768`` nothing is installed. The model's traced code is unchanged: the trim only changes
the block-table width the existing graphs are traced with. Inside a trimmed step every position is below the bucket,
so the attention reads the same cache rows as with the full table (the QSA forms mask by position; table entries
past the bucket are never read for valid queries).
"""

from __future__ import annotations

import functools
import logging

import torch

logger = logging.getLogger(__name__)

MIN_MAX_MODEL_LEN = 32768  # enabled only above this
CTX_BUCKETS = (2048, 8192, 32768)

STATS: dict = {"trimmed": 0, "full": 0, "decode_shaped": 0, "by_ctx": {}, "by_shape": {}, "warmup": []}


def buckets_for(max_model_len: int | None) -> list[int]:
    """Context buckets used at ``max_model_len`` (empty: the feature is off)."""
    if not max_model_len or max_model_len <= MIN_MAX_MODEL_LEN:
        return []
    return [b for b in CTX_BUCKETS if b < max_model_len]


def headroom(num_speculative_tokens: int) -> int:
    """Positions past the step the same step may write: the MTP draft after a prefill chunk (decode keeps 2k + 1)."""
    return 2 * num_speculative_tokens + 1


def table_width(ctx: int, block_size: int) -> int:
    """Block-table width for a context bucket: the width ``_build_warmup_attention_metadata`` builds for it."""
    n = (ctx + block_size - 1) // block_size
    al = 128 // block_size if block_size <= 128 else 1
    return (n + al - 1) // al * al


def pick(buckets: list[int], needed: int, query_len: int) -> int | None:
    """Smallest bucket holding ``needed`` positions and the padded query (warm-up compiles ``T`` at ``ctx >= T``)."""
    for b in buckets:
        if b >= needed and b >= query_len:
            return b
    return None


def is_prefill_step(max_query_len: int, max_num_draft_tokens: int, decode_threshold: int) -> bool:
    """A step is trimmed only if neither the runner's threshold nor its own draft count makes it decode-shaped."""
    return max_query_len > max(1 + max_num_draft_tokens, decode_threshold)


def _num_spec(runner) -> int:
    sc = getattr(runner, "speculative_config", None)
    return int(sc.num_speculative_tokens) if sc is not None else 0


def _trim(runner, md: dict, ctx: int) -> dict:
    out, done = {}, {}
    for name, meta in md.items():
        if not (isinstance(meta, dict) and "block_table_tensor" in meta and "block_size" in meta):
            out[name] = meta
            continue
        key = id(meta)
        if key not in done:
            w = table_width(ctx, meta["block_size"])
            table = meta["block_table_tensor"]
            if w < table.shape[1]:
                meta = dict(meta)
                idx = runner._cached_constant_tensor(
                    ("q4x_prefill_ctx_idx", w), lambda w=w: torch.arange(w, device=runner.device))
                meta["block_table_tensor"] = torch.index_select(table, 1, idx)  # fresh row-major (as decode)
                meta["max_blocks_per_seq"] = w
            done[key] = meta
        out[name] = done[key]
    return out


def _runner_patches(cls) -> None:
    orig_bam = cls._build_attention_metadata
    orig_wam = cls._build_warmup_attention_metadata
    orig_warmup = cls.warmup_prefill
    orig_extract = cls.extract_prefill_graphs

    @functools.wraps(orig_bam)
    def _build_attention_metadata(self, padded_num_reqs, total_num_scheduled_tokens, max_query_len,
                                  max_num_draft_tokens, cached_seq_len=0, max_decode_ctx_len=0):
        md = orig_bam(self, padded_num_reqs, total_num_scheduled_tokens, max_query_len, max_num_draft_tokens,
                      cached_seq_len, max_decode_ctx_len)
        buckets = getattr(self, "_q4x_prefill_ctx_buckets", None)
        if not md or not buckets:
            return md
        if not is_prefill_step(int(max_query_len), int(max_num_draft_tokens), self._decode_token_threshold()):
            STATS["decode_shaped"] += 1
            return md
        n = self.input_batch.num_reqs
        # Every position of this step is < min(longest request, largest cached + query); the MTP draft of this step
        # writes up to 2k + 1 positions further.
        ends = int(self.input_batch.num_tokens_no_spec[:n].max()) if n else 0
        needed = min(ends, int(cached_seq_len) + int(max_query_len)) + headroom(_num_spec(self))
        ctx = pick(buckets, needed, int(max_query_len))
        if ctx is None:
            STATS["full"] += 1
            return md
        STATS["trimmed"] += 1
        STATS["by_ctx"][ctx] = STATS["by_ctx"].get(ctx, 0) + 1
        first = next(iter(md.values()))
        seg = int(first.get("kv_segment_size", 0) or 0) if isinstance(first, dict) else 0
        key = (int(max_query_len), seg, ctx)
        STATS["by_shape"][key] = STATS["by_shape"].get(key, 0) + 1
        return _trim(self, md, ctx)

    @functools.wraps(orig_wam)
    def _build_warmup_attention_metadata(self, num_tokens, num_reqs, cached_seq_len=0, decode_token_threshold=None,
                                         ctx_bucket=None, device=None, null_kv=False, kv_segment_size=0):
        ctx = getattr(self, "_q4x_prefill_ctx", None)
        thr = decode_token_threshold if decode_token_threshold is not None else 1 + _num_spec(self)
        if ctx is not None and ctx_bucket is None and num_tokens // num_reqs > thr:
            ctx_bucket = ctx
        return orig_wam(self, num_tokens, num_reqs, cached_seq_len, decode_token_threshold, ctx_bucket, device,
                        null_kv, kv_segment_size)

    def _each_ctx(fn):
        @functools.wraps(fn)
        def run(self, bucket_size, kv_segment_size, *args, **kwargs):
            fn(self, bucket_size, kv_segment_size, *args, **kwargs)  # max_model_len first, as without buckets
            for ctx in getattr(self, "_q4x_prefill_ctx_buckets", None) or []:
                if ctx < bucket_size:
                    continue
                STATS["warmup"].append((fn.__name__, bucket_size, kv_segment_size, ctx))
                self._q4x_prefill_ctx = ctx
                try:
                    fn(self, bucket_size, kv_segment_size, *args, **kwargs)
                finally:
                    self._q4x_prefill_ctx = None

        return run

    cls._build_attention_metadata = _build_attention_metadata
    cls._build_warmup_attention_metadata = _build_warmup_attention_metadata
    cls.warmup_prefill = _each_ctx(orig_warmup)
    cls.extract_prefill_graphs = _each_ctx(orig_extract)


def _with_ctx(runner, ctx, fn, *args, **kwargs):
    runner._q4x_prefill_ctx = ctx
    try:
        return fn(*args, **kwargs)
    finally:
        runner._q4x_prefill_ctx = None


def _subset(buckets, ctx):
    return [(t, kv) for t, kv in buckets if t <= ctx]


def _worker_patches(cls) -> None:
    """The parallel-trace pool and the sequential drafter capture build prefill inputs without warmup_prefill /
    extract_prefill_graphs: add the context-bucket variants to the job lists they build."""
    orig_jobs = cls._build_prefill_trace_jobs
    orig_draft_jobs = cls._build_drafter_trace_jobs
    orig_draft_extract = cls._extract_prefill_drafter_graphs

    @functools.wraps(orig_jobs)
    def _build_prefill_trace_jobs(self, buckets, *args, **kwargs):
        jobs = orig_jobs(self, buckets, *args, **kwargs)
        r = self.model_runner
        for ctx in reversed(getattr(r, "_q4x_prefill_ctx_buckets", None) or []):
            sub = _subset(buckets, ctx)
            if sub:
                STATS["warmup"].extend(("trace_jobs", t, kv, ctx) for t, kv in sub)
                jobs += _with_ctx(r, ctx, orig_jobs, self, sub, *args, **kwargs)
        return jobs

    @functools.wraps(orig_draft_jobs)
    def _build_drafter_trace_jobs(self, prefill_buckets, decode_targets):
        jobs = orig_draft_jobs(self, prefill_buckets, decode_targets)
        r = self.model_runner
        for ctx in reversed(getattr(r, "_q4x_prefill_ctx_buckets", None) or []):
            sub = _subset(prefill_buckets, ctx)
            if sub:
                jobs += _with_ctx(r, ctx, orig_draft_jobs, self, sub, [])
        return jobs

    @functools.wraps(orig_draft_extract)
    def _extract_prefill_drafter_graphs(self, buckets):
        orig_draft_extract(self, buckets)
        r = self.model_runner
        for ctx in getattr(r, "_q4x_prefill_ctx_buckets", None) or []:
            sub = _subset(buckets, ctx)
            if sub:
                _with_ctx(r, ctx, orig_draft_extract, self, sub)

    cls._build_prefill_trace_jobs = _build_prefill_trace_jobs
    cls._build_drafter_trace_jobs = _build_drafter_trace_jobs
    cls._extract_prefill_drafter_graphs = _extract_prefill_drafter_graphs


_INSTALLED: list[int] = []


def maybe_install(vllm_config=None) -> list[int]:
    """Install the runner / worker patches if the serving config qualifies; returns the buckets ([] = off).

    Called while the runner builds the model (``factory``); the runner reads ``_q4x_prefill_ctx_buckets`` from its
    class, so the patches act on every runner of this process (one model per process).
    """
    if vllm_config is None:
        try:
            from vllm.config import get_current_vllm_config

            vllm_config = get_current_vllm_config()
        except Exception:  # no vLLM config in scope (model built outside the runner): off
            return []
    mml = getattr(getattr(vllm_config, "model_config", None), "max_model_len", None)
    buckets = buckets_for(mml)
    if not buckets:
        return []
    pc = getattr(vllm_config, "parallel_config", None)
    sizes = {k: int(getattr(pc, k, 1) or 1) for k in (
        "data_parallel_size", "decode_context_parallel_size", "prefill_context_parallel_size")}
    if any(v > 1 for v in sizes.values()):
        logger.info("[qwen4_exp] prefill context buckets off: %s (the bucket is not synchronised across ranks)", sizes)
        return []
    if _INSTALLED:
        return list(_INSTALLED)
    from vllm_neuron.vllm.worker import neuron_model_runner as R
    from vllm_neuron.vllm.worker import neuron_worker as W

    _runner_patches(R.NeuronModelRunner)
    _worker_patches(W.NeuronWorker)
    R.NeuronModelRunner._q4x_prefill_ctx_buckets = list(buckets)
    R.NeuronModelRunner._q4x_prefill_ctx = None
    _INSTALLED.extend(buckets)
    logger.info("[qwen4_exp] prefill context buckets %s (max_model_len %d)", buckets, mml)
    return list(buckets)
