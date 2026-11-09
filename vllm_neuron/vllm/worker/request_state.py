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
Limits of this scheme: prefix caching must be off (a cached prefix has no state),
and scheduling must be synchronous when the host inputs depend on the last
sampled token.
"""

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

REQUEST_STATE_KEY = "request_state"


def model_uses_request_state(model) -> bool:
    inner = getattr(model, "_orig_mod", model)
    return hasattr(inner, "allocate_request_state")


class RequestStateSlots:
    def __init__(self, model, max_num_reqs: int, device):
        self.model = getattr(model, "_orig_mod", model)
        # One slot per concurrently scheduled request, plus a dummy slot for padding rows.
        self.num_slots = max_num_reqs + 1
        self.dummy_slot = max_num_reqs
        self.device = device
        self._slot_of: dict[str, int] = {}
        self._free = list(range(max_num_reqs - 1, -1, -1))
        self.model.allocate_request_state(self.num_slots, device)
        self.layout = None  # (rows [T], positions [T], num_valid per batch row), set per step
        logger.info("[request_state] %d slots (+1 dummy) allocated", max_num_reqs)

    # ── Slot lifecycle ───────────────────────────────────────────────────
    def release(self, req_ids) -> None:
        for req_id in req_ids or ():
            slot = self._slot_of.pop(req_id, None)
            if slot is not None:
                self._free.append(slot)

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
        rows, positions, num_valid = self.layout
        num_reqs = input_batch.num_reqs
        state_indices = np.full(num_rows, self.dummy_slot, dtype=np.int32)
        has_init = np.zeros(num_rows, dtype=np.int32)
        valid = np.zeros(num_rows, dtype=np.int32)
        for r in range(min(num_reqs, num_rows)):
            state_indices[r] = self._slot(input_batch.req_ids[r])
            has_init[r] = int(input_batch.num_computed_tokens_cpu[r] > 0)
            valid[r] = num_valid[r]
        # Tokens added by bucket padding after the layout was recorded are padding.
        tok_rows = np.full(num_tokens, -1, dtype=np.int64)
        tok_pos = np.zeros(num_tokens, dtype=np.int64)
        n = min(num_tokens, rows.shape[0])
        tok_rows[:n], tok_pos[:n] = rows[:n], positions[:n]
        extras = self.model.request_state_host_inputs(input_batch.token_ids_cpu, tok_rows, tok_pos)
        return self._to_device(state_indices, has_init, valid, extras)

    def warmup_metadata(self, num_rows: int, num_tokens: int) -> dict:
        state_indices = np.full(num_rows, self.dummy_slot, dtype=np.int32)
        zeros = np.zeros(num_rows, dtype=np.int32)
        extras = self.model.request_state_warmup_inputs(num_tokens)
        return self._to_device(state_indices, zeros, zeros.copy(), extras)

    def _to_device(self, state_indices, has_init, num_valid, extras) -> dict:
        md = {
            "state_indices": torch.from_numpy(state_indices),
            "has_initial_state": torch.from_numpy(has_init),
            "num_valid": torch.from_numpy(num_valid),
            **extras,
        }
        return {k: v.to(self.device) for k, v in md.items()}
