# SPDX-License-Identifier: Apache-2.0
"""Prefix caching for models that own per-request recurrent state (align mode).

The Neuron scheduler, plus a :class:`~vllm_neuron.vllm.worker.request_state.StateSnapshotDirectory`
(see ``request_state.py`` for the pool layout and the plan). The platform selects these
classes instead of ``NeuronScheduler`` / ``NeuronAsyncScheduler`` when prefix caching is on
for such a model (``state_snapshot_rows(vllm_config) > 0``); every other configuration keeps
the plain schedulers.

What it changes, all through vLLM's own hooks:

- ``_get_local_prefix_cache_hit``: the KV hit is truncated to the last multiple of ``G`` that
  has a state snapshot; if the KV hit reaches further, that point becomes the request's
  ``shared_prefix_boundary`` (the junction), so the chunk stops there and its state is kept.
- ``need_mamba_block_aligned_split = True`` and ``_mamba_block_aligned_split``: vLLM's
  align-mode split, on the snapshot grid ``G = lcm(block_size, model alignment)`` instead of
  the block (:meth:`StateSnapshotDirectory.split`): prefill chunks end on multiples of ``G``
  and stop at the kept boundaries (the junction; for a prompt of at least
  ``LONG_PROMPT_CHUNKS`` x ``max_num_batched_tokens`` tokens also the prompt end), so a
  prefill resumed from a hit starts on the model's prefill chunk grid. A shorter prompt keeps
  the state at its natural chunk ends and is chunked as with prefix caching off.
- ``schedule``: attaches the step's ``{req_id: (read, write)}`` row plan to the output.
- ``update_from_output``: publishes the rows that the step reserved for a boundary reached
  during decode / MTP verify, now that its tokens (and accepted drafts) are known.
- ``_free_request`` / ``_preempt_request`` / ``reset_prefix_cache``: release / forget rows.
"""

import logging

from vllm_neuron.vllm.core.scheduler import NeuronAsyncScheduler, NeuronScheduler
from vllm_neuron.vllm.worker.request_state import (
    LONG_PROMPT_CHUNKS,
    PLAN_ATTR,
    StateSnapshotDirectory,
    decode_snapshots_enabled,
    state_snapshot_granularity,
    state_snapshot_rows,
)

logger = logging.getLogger(__name__)


def scheduled_chunks(scheduler, scheduler_output):
    """``[(request, start, num_tokens)]`` of a scheduled step; ``start`` as the worker sees it."""
    start = {r.req_id: r.num_computed_tokens for r in scheduler_output.scheduled_new_reqs}
    cached = scheduler_output.scheduled_cached_reqs
    start.update(zip(cached.req_ids, cached.num_computed_tokens))
    return [(scheduler.requests[req_id], start[req_id], n)
            for req_id, n in scheduler_output.num_scheduled_tokens.items()]


class StatePrefixCacheMixin:
    """Mixed in before a Neuron scheduler class (see the module docstring)."""

    def __init__(self, vllm_config, kv_cache_config, *args, **kwargs) -> None:
        super().__init__(vllm_config, kv_cache_config, *args, **kwargs)
        rows = state_snapshot_rows(vllm_config)
        if not rows or not self.cache_config.enable_prefix_caching:
            raise ValueError("StatePrefixCacheMixin needs prefix caching and snapshot rows")
        if self.connector is not None:
            raise NotImplementedError("prefix caching with per-request state does not support a KV "
                                      "connector (an external KV hit has no state)")
        spec = self.num_spec_tokens > 0
        granularity = state_snapshot_granularity(vllm_config)
        decode_snapshots = decode_snapshots_enabled(vllm_config)  # opt-in (default off)
        if decode_snapshots and spec and self.num_spec_tokens + 1 > self.cache_config.block_size:
            raise ValueError("decode snapshots need 1 + num_speculative_tokens <= block_size")
        self.state_snapshots = StateSnapshotDirectory(
            rows, self.cache_config.block_size, self.hash_block_size, use_eagle=self.use_eagle,
            decode_snapshots=decode_snapshots, side_write=spec,
            lookahead=self.num_spec_tokens if spec and self.scheduler_config.async_scheduling else 0,
            max_model_len=self.max_model_len, granularity=granularity,
            long_prompt_tokens=LONG_PROMPT_CHUNKS * self.max_num_scheduled_tokens)
        self._reserved_by_step: dict[int, list] = {}  # id(SchedulerOutput) -> its reserved rows
        self.need_mamba_block_aligned_split = True
        if granularity > self._max_prefill_tokens():
            logger.warning("[state prefix cache] snapshot granularity %d > max prefill chunk %d: chunks "
                           "after the first may start off the model's prefill grid", granularity,
                           self._max_prefill_tokens())
        logger.info("[state prefix cache] %d snapshot rows, block %d, granularity %d, eagle back-off %s, "
                    "decode snapshots %s", rows, self.cache_config.block_size, granularity, self.use_eagle,
                    self.state_snapshots.decode_snapshots)

    def _max_prefill_tokens(self) -> int:
        n = self.max_num_scheduled_tokens
        threshold = self.scheduler_config.long_prefill_token_threshold
        return min(n, threshold) if threshold > 0 else n

    def _get_local_prefix_cache_hit(self, request):
        blocks, num_kv, boundary, diverged = super()._get_local_prefix_cache_hit(request)
        num_state = self.state_snapshots.lookup(request, num_kv)
        if num_state < num_kv:
            blocks = self.kv_cache_manager.truncate_computed_blocks(blocks, num_state)
            boundary = max(boundary, num_kv)  # the junction: keep the state where the KV hit ends
        return blocks, num_state, boundary, diverged

    def _mamba_block_aligned_split(self, request, num_new_tokens, num_new_local_computed_tokens=0,
                                   num_external_computed_tokens=0):
        # Replaces vLLM's split (whose cuts and prompt-end stop are block-granular).
        start = request.num_computed_tokens + num_new_local_computed_tokens + num_external_computed_tokens
        return self.state_snapshots.split(request, start, num_new_tokens, self._max_prefill_tokens())

    def schedule(self, *args, **kwargs):
        out = super().schedule(*args, **kwargs)
        setattr(out, PLAN_ATTR, self.state_snapshots.plan(scheduled_chunks(self, out)))
        reserved = self.state_snapshots.take_pending()
        if reserved:
            self._reserved_by_step[id(out)] = reserved
        return out

    def update_from_output(self, scheduler_output, model_runner_output):
        reserved = self._reserved_by_step.pop(id(scheduler_output), None)
        if not reserved:
            return super().update_from_output(scheduler_output, model_runner_output)
        before = []
        for req_id, row in reserved:
            request = self.requests.get(req_id)
            before.append((request, None if request is None else request.num_tokens, row))
        out = super().update_from_output(scheduler_output, model_runner_output)
        self.state_snapshots.publish(before)
        return out

    def _free_request(self, request, delay_free_blocks: bool = False):
        self.state_snapshots.release(request.request_id)
        return super()._free_request(request, delay_free_blocks)

    def _preempt_request(self, request, *args, **kwargs):
        self.state_snapshots.release(request.request_id)
        return super()._preempt_request(request, *args, **kwargs)

    def reset_prefix_cache(self, *args, **kwargs) -> bool:
        ok = super().reset_prefix_cache(*args, **kwargs)
        if ok:
            self.state_snapshots.reset()
        return ok


class StatePrefixCacheNeuronScheduler(StatePrefixCacheMixin, NeuronScheduler):
    pass


class StatePrefixCacheNeuronAsyncScheduler(StatePrefixCacheMixin, NeuronAsyncScheduler):
    pass
