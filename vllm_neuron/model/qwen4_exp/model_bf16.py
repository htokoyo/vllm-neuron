# SPDX-License-Identifier: Apache-2.0
"""Qwen4-Exp text model (BF16), single rank.

Stage 1 of the port: the model math with explicit per-request state, validated
against the HF reference on CPU. Runner wiring (``attn_metadata``, paged KV and
the GDN/PLE state cache) and tensor/expert parallelism are added on top of this.
"""

import json
import logging
import os
import re
from collections.abc import Iterable

import numpy as np
import torch
from torch import nn

from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.model.neuron_config import NeuronConfig

from .config import FULL_ATTENTION, Qwen4ExpConfig
from .layers import Qwen4ExpDecoderLayer, Qwen4ExpGatedResidual
from .parallel import all_gather_last, sharded_lookup, shard_pieces, validate
from .ngram import NGramLayout, build_ngram_layout, compute_ngram_ids

logger = logging.getLogger(__name__)

# Checkpoint prefixes (composite Qwen4ExpForConditionalGeneration layout).
_TEXT_PREFIX = "model.language_model."
_SKIPPED_PREFIXES = ("model.visual.", "mtp.")
# The PLE table is stored as `split_ngram_parts` row shards.
_PLE_SHARD = re.compile(r"^(.*\.ngram_embedding)\.shard_(\d+)\.weight$")
# Experts saved one 2-D tensor per expert (transformers' save_pretrained) instead of
# the fused 3-D layout of the released checkpoint.
# The FP8 checkpoint stores this layout too, with 128x128 block scales ``weight_scale_inv``.
_PER_EXPERT = re.compile(r"^(.*\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|weight_scale_inv)$")
# FP8 checkpoint: the PLE table's single per-tensor scale (rows dequantized at load).
_PLE_SCALE = re.compile(r"^(.*\.ngram_embedding)\.weight_scale$")
# Hash constants stored as buffers; recomputed by build_ngram_layout and checked.
_PLE_BUFFERS = ("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets")


class Qwen4ExpTextModel(nn.Module):
    def __init__(self, config: Qwen4ExpConfig):
        super().__init__()
        self.config = config
        self.ngram_layouts: dict[int, NGramLayout] = {
            layer_idx: build_ngram_layout(
                config.vocab_size,
                config.ngram_size,
                config.heads_per_ngram,
                config.ngram_vocab_size_base,
                config.make_ngram_vocab_size_divisible_by,
                config.eos_token_id,
                ple_layer_index=i,
                seed=config.seed,
            )
            for i, layer_idx in enumerate(config.ple_layer_indices)
        }
        ngram_rows = {i: lay.padded_vocab_size for i, lay in self.ngram_layouts.items()}
        validate(config, config.tp, list(ngram_rows.values()), config.attn_tp)
        self.embed_tokens = nn.Embedding(config.vocab_size // config.tp_size, config.hidden_size)
        self.layers = nn.ModuleList(
            Qwen4ExpDecoderLayer(config, i, ngram_rows.get(i, 0))
            for i in range(config.num_hidden_layers)
        )
        self.hyper_connection_mixer = Qwen4ExpGatedResidual(config, use_combine=False)

    def init_state(self, batch: int, device=None) -> list[dict]:
        dtype = self.embed_tokens.weight.dtype
        return [layer.init_state(batch, dtype, device) for layer in self.layers]

    def forward(self, input_ids, positions, ngram_ids: dict[int, torch.Tensor], state: list[dict],
                num_valid: torch.Tensor | None = None, is_decode: bool = False):
        streams = sharded_lookup(self.embed_tokens.weight, input_ids, self.config.tp)
        streams = streams.repeat(1, 1, self.config.hc_count)
        new_state = []
        for i, layer in enumerate(self.layers):
            streams, s = layer(streams, positions, ngram_ids.get(i), state[i], num_valid, is_decode)
            new_state.append(s)
        return self.hyper_connection_mixer(streams), new_state


class Qwen4ExpForCausalLM(nn.Module):
    """Qwen4-Exp text model + LM head."""

    def __init__(self, config: Qwen4ExpConfig):
        super().__init__()
        self.config = config
        self.model = Qwen4ExpTextModel(config)
        if config.tp_size > 1:
            logger.info("[qwen4_exp] TP=%d (experts %d/rank, vocab rows sharded), attention/GDN group=%d",
                        config.tp_size, config.num_experts // config.tp_size, config.attn_tp_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size // config.tp_size, bias=False)
        nc = config.neuron_config
        ods = nc.on_device_sampling_config if nc else None
        # Full logits are also returned when logprobs (or a logits dump) are requested.
        self._gather_logits = nc is not None and (nc.max_logprobs != 0 or nc.debug_logits_dir is not None)
        self._sampler = None
        if ods is not None:
            from vllm.distributed.parallel_state import get_tp_group

            from vllm_neuron.nn.sampler import Sampler

            self._sampler = Sampler(ods, process_group=get_tp_group().device_group)

    @classmethod
    def from_configs(cls, hf_config, neuron_config: NeuronConfig | None):
        return cls(Qwen4ExpConfig.from_configs(hf_config, neuron_config))

    # ── Host side ────────────────────────────────────────────────────────
    def compute_ngram_ids(self, tokens: np.ndarray, history: np.ndarray | None = None):
        """n-gram ids of one request's ``tokens`` for every PLE layer (``{layer: [T, heads]}``)."""
        return {
            i: torch.from_numpy(compute_ngram_ids(layout, tokens, history))
            for i, layout in self.model.ngram_layouts.items()
        }

    # ── Reference forward with explicit state ────────────────────────────
    def init_state(self, batch: int, device=None) -> list[dict]:
        return self.model.init_state(batch, device)

    def forward_with_state(self, input_ids, positions, ngram_ids, state):
        """``input_ids`` / ``positions`` ``[B, T]``, ``ngram_ids`` ``{layer: [B, T, heads]}``.

        Returns logits ``[B, T, vocab]`` and the updated state.
        """
        hidden, state = self.model(input_ids, positions, ngram_ids, state)
        return self.lm_head(hidden), state

    # ── Runner interface: paged KV ───────────────────────────────────────
    def _attention_layers(self):
        return [
            (i, layer.self_attn) for i, layer in enumerate(self.model.layers)
            if layer.layer_type == FULL_ATTENTION
        ]

    def get_kv_spec(self) -> KVSpec:
        layers = []
        for i, attn in self._attention_layers():
            layers.append(LayerSpec(name=f"layers.{i}.self_attn", num_kv_heads=attn.num_kv_heads,
                                    head_size=attn.head_dim, dtype=self.config.torch_dtype))
            if not attn.dense:
                # QSA indexer: one raw key per token, replicated on every rank.
                layers.append(LayerSpec(name=f"layers.{i}.self_attn.indexer", num_kv_heads=1,
                                        head_size=self.config.indexer_head_dim,
                                        dtype=self.config.torch_dtype, key_only=True))
        return KVSpec(layers=layers)

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor]]):
        for i, attn in self._attention_layers():
            name = f"layers.{i}.self_attn"
            if name not in kv_caches:
                raise KeyError(f"KV cache for layer {name} not initialized")
            attn.k_cache, attn.v_cache = kv_caches[name][0], kv_caches[name][1]
            if not attn.dense:
                c = kv_caches[name + ".indexer"][0]
                attn.idx_cache = c.squeeze(0) if c.dim() == 5 else c  # [blocks, 1, block, D]

    # ── Runner interface: per-request state slots ────────────────────────
    # The runner owns the slot of each request (vllm_neuron/vllm/worker/request_state.py)
    # and passes ``attn_metadata["request_state"]`` with ``state_indices`` /
    # ``has_initial_state`` per row and ``num_valid`` per row.

    def allocate_request_state(self, num_slots: int, device) -> None:
        """Allocate the GDN / PLE state pools, ``num_slots + 1`` rows each (the last row: the write sink, _pool_write)."""
        self._state_pool = [
            {k: v  # pools start at 0
             for k, v in layer.init_state(num_slots + 1, self.config.torch_dtype, device).items() if k != "kv"}
            for layer in self.model.layers
        ]

    def request_state_host_inputs(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        """No per-token host inputs: the n-gram ids are computed in the graph (device n-gram ids below)."""
        return {}

    def request_state_warmup_inputs(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        return {}

    # ── Runner forward ───────────────────────────────────────────────────
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        is_token_ids: torch.Tensor | None = None,
        attn_metadata: dict | None = None,
        sampling_positions: torch.Tensor | None = None,
        sampling_params: torch.Tensor | None = None,
        spec_decode_metadata=None,
        logit_mask: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
        **kwargs,
    ):
        """Prefill is one request (``[T]`` tokens); decode is one token per row."""
        if spec_decode_metadata is not None or getattr(self, "mtp_enabled", False):  # _mtp_forward: end of module
            return _mtp_forward(self, input_ids, positions, attn_metadata, sampling_positions, sampling_params, spec_decode_metadata, logit_mask, rank)
        rs = attn_metadata["request_state"]
        first_attn = f"layers.{self._attention_layers()[0][0]}.self_attn"
        meta0 = attn_metadata[first_attn]
        is_prefill = meta0["max_query_len"] > meta0["decode_token_threshold"]
        t = input_ids.shape[0]
        b = 1 if is_prefill else t
        shape = (b, t // b)
        ids = input_ids.long().view(shape)
        pos = positions.long().view(shape)
        slots = rs["state_indices"][:b].long(); read_slots = _state_read_slots(rs, slots, b)  # prefix caching
        has_init = rs["has_initial_state"][:b].bool()
        num_valid = rs["num_valid"][:b]

        # Gather each row's state; a fresh request starts from zeros. torch.where
        # selects (no arithmetic), so garbage in an unwritten slot cannot leak in.
        state = []
        for i, layer in enumerate(self.model.layers):
            s = {}
            for key, pool in self._state_pool[i].items():
                rows = _pool_rows(pool, read_slots, b)  # _state_rows
                keep = has_init.view(-1, *([1] * (rows.dim() - 1)))
                s[key] = torch.where(keep, rows, torch.zeros_like(rows)) if is_prefill else rows  # decode: see _DECODE_ROWS
            if layer.layer_type == FULL_ATTENTION:
                s["paged"] = attn_metadata[f"layers.{i}.self_attn"]
                s["paged_idx"] = attn_metadata.get(f"layers.{i}.self_attn.indexer")
            state.append(s)
        ngram = {i: rs[f"ngram_ids.{i}"].view(*shape, -1) for i in self.model.ngram_layouts}

        hidden, state = self.model(ids, pos, ngram, state, num_valid, is_decode=not is_prefill)

        for i, s in enumerate(state):
            for key, pool in self._state_pool[i].items():
                _pool_write(pool, slots, b, s[key])  # _state_rows

        hidden = hidden.reshape(t, -1)
        # One-token decode is already row-aligned with sampling_positions.
        if is_prefill or hidden.shape[0] != sampling_positions.shape[0]:
            hidden = torch.index_select(hidden, 0, sampling_positions)
        logits = self.lm_head(hidden)  # [rows, vocab / TP]
        if self._sampler is None:
            return all_gather_last(logits, self.config.tp)
        sampled = self._sampler(logits, sampling_params, logit_mask=logit_mask, tp_rank=rank)
        return sampled, (all_gather_last(logits, self.config.tp) if self._gather_logits else None)

    # ── Weights ──────────────────────────────────────────────────────────
    def load_weights(
        self, checkpoint_path: str, device=None, cache_dir=None, layers: Iterable[int] | None = None
    ) -> set[str]:
        """Load a HF safetensors checkpoint (composite or text-only layout).

        Args:
            layers: Restrict loading to these decoder layers (for partial models in
                tests). ``None`` loads everything.

        Returns:
            The set of parameter names that were loaded.
        """
        from safetensors import safe_open

        weight_map = self._weight_map(checkpoint_path)
        params = dict(self.named_parameters())
        wanted = None if layers is None else {int(i) for i in layers}
        def in_scope(name: str) -> bool:
            # Layers past config.num_hidden_layers (a truncated config, e.g. via
            # hf_overrides for a bring-up run) are skipped like an explicit subset.
            m = re.match(r"model\.layers\.(\d+)\.", name)
            if m is None:
                return True
            i = int(m.group(1))
            return i < self.config.num_hidden_layers and (wanted is None or i in wanted)

        by_file: dict[str, list[tuple[str, str]]] = {}
        ple_shards: dict[str, list[tuple[int, str, str]]] = {}
        buffers: list[tuple[str, str, str]] = []
        per_expert: dict[str, list[tuple[int, str, str, str, str]]] = {}
        ple_scales: dict[str, tuple[str, str]] = {}
        for ckpt_name, file in weight_map.items():
            name = self._map_name(ckpt_name)
            if name is None:
                continue
            if not in_scope(name):
                continue
            if name.rsplit(".", 1)[-1] in _PLE_BUFFERS:
                buffers.append((name, ckpt_name, file))
                continue
            expert = _PER_EXPERT.match(name)
            if expert:
                per_expert.setdefault(expert.group(1), []).append(
                    (int(expert.group(2)), expert.group(3), expert.group(4), ckpt_name, file)
                )
                continue
            ple_scale = _PLE_SCALE.match(name)
            if ple_scale:
                ple_scales[ple_scale.group(1) + ".weight"] = (ckpt_name, file)
                continue
            shard = _PLE_SHARD.match(name)
            if shard:
                ple_shards.setdefault(shard.group(1) + ".weight", []).append(
                    (int(shard.group(2)), ckpt_name, file)
                )
                continue
            if name not in params:
                raise KeyError(f"checkpoint tensor {ckpt_name} has no parameter {name}")
            by_file.setdefault(file, []).append((name, ckpt_name))

        dtype = self.config.torch_dtype
        tp = self.config.tp
        state_dict: dict[str, torch.Tensor] = {}
        # Only this rank's slices are read (safetensors get_slice), so a TP group does
        # not load the checkpoint once per rank.
        for file, items in sorted(by_file.items()):
            with safe_open(os.path.join(checkpoint_path, file), framework="pt") as f:
                for name, ckpt_name in items:
                    sl = f.get_slice(ckpt_name)
                    pieces = shard_pieces(name, tuple(sl.get_shape()), self.config, tp,
                                          self.config.attn_tp)
                    t = torch.cat([sl[p] for p in pieces]) if len(pieces) > 1 else sl[pieces[0]]
                    state_dict[name] = self._checked(params[name], t, name, dtype)
        for name, shards in ple_shards.items():
            # This rank's rows [lo, hi) of the table stored as consecutive row shards.
            rows = params[name].shape[0]
            lo, hi, start, parts = tp.rank * rows, (tp.rank + 1) * rows, 0, []
            ple_scale = _read_ple_scale(checkpoint_path, ple_scales.pop(name, None))
            for _, ckpt_name, file in sorted(shards):
                with safe_open(os.path.join(checkpoint_path, file), framework="pt") as f:
                    sl = f.get_slice(ckpt_name)
                    n = sl.get_shape()[0]
                    a_, b_ = max(lo, start), min(hi, start + n)
                    if a_ < b_:
                        parts.append(_ple_rows(sl[a_ - start : b_ - start], ple_scale, dtype, name))
                start += n
            state_dict[name] = self._checked(params[name], torch.cat(parts), name, dtype)
            del parts
        if ple_scales:
            raise KeyError(f"PLE scales without a table: {sorted(ple_scales)}")
        for prefix, items in per_expert.items():
            state_dict.update(_load_per_expert(checkpoint_path, prefix, items, params, self.config, dtype))
        self._check_ngram_buffers(checkpoint_path, buffers)

        missing = {n for n in params if in_scope(n)} - set(state_dict)
        if missing:
            raise ValueError(f"{len(missing)} parameters not in checkpoint, e.g. {sorted(missing)[:5]}")
        # assign=True: the runner builds the model on the meta device.
        self.load_state_dict(state_dict, strict=False, assign=True)
        return set(state_dict)

    @staticmethod
    def _weight_map(checkpoint_path: str) -> dict[str, str]:
        """Tensor name -> file, from the index or from a single-file checkpoint."""
        index_path = os.path.join(checkpoint_path, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                return json.load(f)["weight_map"]
        from safetensors import safe_open

        with safe_open(os.path.join(checkpoint_path, "model.safetensors"), framework="pt") as f:
            return {k: "model.safetensors" for k in f.keys()}

    @staticmethod
    def _map_name(ckpt_name: str) -> str | None:
        if ckpt_name.startswith(_SKIPPED_PREFIXES):
            return None
        if ckpt_name.startswith(_TEXT_PREFIX):
            return "model." + ckpt_name[len(_TEXT_PREFIX) :]
        return ckpt_name  # text-only layout (model.* / lm_head.*)

    @staticmethod
    def _checked(param: nn.Parameter, tensor: torch.Tensor, name: str, dtype: torch.dtype) -> torch.Tensor:
        if tensor.dtype in _FP8_DTYPES or param.dtype in _FP8_DTYPES:
            # FP8 tensors carry scales; a plain cast would drop them (FP8 experts and the PLE
            # table go through _load_per_expert / _ple_rows).
            raise ValueError(f"{name}: no FP8 conversion for checkpoint {tensor.dtype} -> parameter "
                             f"{param.dtype} (FP8 expert parameters need the FP8 checkpoint's per-expert "
                             f"tensors with weight_scale_inv)")
        if (re.search(r"\.experts\.(gate_up_proj|down_proj)$", name)
                and tuple(tensor.shape) != tuple(param.shape)
                and tuple(tensor.transpose(-1, -2).shape) == tuple(param.shape)):
            tensor = tensor.transpose(-1, -2).contiguous()  # HF -> kernel layout
        if tuple(tensor.shape) != tuple(param.shape):
            raise ValueError(f"{name}: checkpoint {tuple(tensor.shape)} vs parameter {tuple(param.shape)}")
        return tensor.to(dtype)

    def _check_ngram_buffers(self, checkpoint_path, buffers) -> None:
        """The stored hash constants must equal the recomputed ones."""
        from safetensors import safe_open

        for name, ckpt_name, file in buffers:
            layer_idx = int(name.split(".")[2])
            layout = self.model.ngram_layouts[layer_idx]
            expected = {
                "layer_multipliers": layout.multipliers,
                "ngram_heads_vocab_sizes": layout.head_vocab_sizes,
                "ngram_heads_offsets": layout.head_offsets,
            }[name.rsplit(".", 1)[-1]]
            with safe_open(os.path.join(checkpoint_path, file), framework="pt") as f:
                stored = tuple(f.get_tensor(ckpt_name).tolist())
            if stored != tuple(expected):
                raise ValueError(f"{ckpt_name}: stored {stored} != recomputed {expected}")


# _state_rows: decode gathers / scatters the per-request state one row (slot) at a time, the way
# B=1 already lowers (a dynamic slice, ~7 us). A batched index_select / index_copy_ over the short-
# row pools (GDN conv [slots, 1280, 4], PLE conv [slots, 10240, 9]) lowered in the full model to one
# 4-byte indirect DMA per channel (1.66 ms per layer at B=2; ~83 of the +100 ms per TP64 decode
# step, profile of 2026-11-03), which a standalone microbenchmark did not reproduce. Viewing the
# pools 2-D instead made B=1 3.4x slower and changed the outputs on device (reverted). Duplicate
# slots (padding rows share the dummy slot) are written in order, so the last one wins.


# ── Speculative decoding (MTP) ───────────────────────────────────────────────
# Defined at the end of the module so that ``forward`` keeps every line (and so the
# existing graphs' NEFF keys); ``forward`` hands over when there is a spec-decode
# batch or when the MTP drafter is enabled (``mtp_enabled``), which also needs the
# target's pre-final-mixer streams of every token.


def _mtp_forward(self, input_ids, positions, attn_metadata, sampling_positions, sampling_params,
                 spec_decode_metadata, logit_mask, rank):
    """``forward`` for MTP speculative decoding.

    Prefill and one-token decode are those of ``forward``. A verify step (``spec_decode_metadata``
    and ``max_query_len = 1 + k`` tokens per row, each row's tokens ``[x0, d1..dk]`` at
    consecutive positions) runs the layers through :func:`verify_decoder_layer`, samples every
    token, applies the rejection sampler in the graph and commits, per row, the recurrent state
    after the last accepted token (:func:`select_verified_state`); the pools keep one slot per
    request. Returns ``(tokens, streams, logits)``: ``tokens`` is the sampled tokens (prefill /
    decode) or the rejection output ``[rows, 1 + k]`` (-1 = rejected), ``streams`` the
    pre-final-mixer multi stream ``[T, hc * H]`` of every input token (the MTP drafter's input)
    and ``logits`` the gathered logits (or ``None``).
    """
    from vllm_neuron.nn.rejection_sampler import rejection_sampler

    from .layers import select_verified_state, verify_decoder_layer

    rs = attn_metadata["request_state"]
    meta0 = attn_metadata[f"layers.{self._attention_layers()[0][0]}.self_attn"]
    q = meta0["max_query_len"]
    is_prefill = q > meta0["decode_token_threshold"]
    verify = spec_decode_metadata is not None and not is_prefill
    t = input_ids.shape[0]
    b = 1 if is_prefill else t // q
    shape = (b, t // b)
    ids = input_ids.long().view(shape)
    pos = positions.long().view(shape)
    slots = rs["state_indices"][:b].long(); read_slots = _state_read_slots(rs, slots, b)  # prefix caching
    has_init = rs["has_initial_state"][:b].bool()
    num_valid = rs["num_valid"][:b]

    state = []
    for i, layer in enumerate(self.model.layers):
        st = {}
        for key, pool in self._state_pool[i].items():
            rows = _pool_rows(pool, read_slots, b)
            keep = has_init.view(-1, *([1] * (rows.dim() - 1)))
            st[key] = torch.where(keep, rows, torch.zeros_like(rows)) if is_prefill else rows  # decode: see _DECODE_ROWS
        if layer.layer_type == FULL_ATTENTION:
            st["paged"] = attn_metadata[f"layers.{i}.self_attn"]
            st["paged_idx"] = attn_metadata.get(f"layers.{i}.self_attn.indexer")
        state.append(st)
    ngram = {i: rs[f"ngram_ids.{i}"].view(*shape, -1) for i in self.model.ngram_layouts}

    streams = sharded_lookup(self.model.embed_tokens.weight, ids, self.config.tp)
    streams = streams.repeat(1, 1, self.config.hc_count)
    new_state, cands = [], []
    for i, layer in enumerate(self.model.layers):
        if verify:
            streams, st, c = verify_decoder_layer(layer, streams, pos, ngram.get(i), state[i])
            cands.append(c)
        else:
            streams, st = layer(streams, pos, ngram.get(i), state[i], num_valid, is_decode=not is_prefill)
        new_state.append(st)
    hidden = self.model.hyper_connection_mixer(streams).reshape(t, -1)
    streams = streams.reshape(t, -1)
    if is_prefill or hidden.shape[0] != sampling_positions.shape[0]:
        hidden = torch.index_select(hidden, 0, sampling_positions)
    logits = self.lm_head(hidden)
    if self._sampler is None:
        full = all_gather_last(logits, self.config.tp)
        tokens, gathered = full.argmax(-1), full
    else:
        tokens = self._sampler(logits, sampling_params, logit_mask=logit_mask, tp_rank=rank)
        gathered = all_gather_last(logits, self.config.tp) if self._gather_logits else None
    if verify:
        tokens = rejection_sampler(spec_decode_metadata, tokens)  # [rows, 1 + k]
        count = (tokens[:b] != -1).sum(dim=1)
        commit = [select_verified_state(c, count) for c in cands]; _verify_snapshot_write(self, rs, pos, b, cands=cands, count=count)  # prefix caching
    else:
        commit = new_state; _verify_snapshot_write(self, rs, pos, b, states=None if is_prefill else new_state)  # prefix caching
    for i, st in enumerate(commit):
        for key, pool in self._state_pool[i].items():
            _pool_write(pool, slots, b, st[key])
    if verify:
        # The runner's spec-decode contract (as appended by @async_speculative_decoding):
        # the last accepted token per row, the next plain decode step's input.
        from vllm_neuron.vllm.spec_decode.decorator import _extract_last_accepted_token

        return tokens, streams, gathered, _extract_last_accepted_token(tokens, self.config.vocab_size)
    return tokens, streams, gathered


# The runner's auxiliary-hidden-state protocol (vLLM SupportsEagle3), used for the MTP
# drafter: the auxiliary output is the pre-final-mixer streams, so no layer list.
def _set_aux_hidden_state_layers(self, layers) -> None:
    self.mtp_enabled = True


def _get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
    return ()


Qwen4ExpForCausalLM.supports_eagle3 = True
Qwen4ExpForCausalLM.has_own_lm_head = False
Qwen4ExpForCausalLM.has_own_embed_tokens = False
Qwen4ExpForCausalLM.get_eagle3_default_aux_hidden_state_layers = _get_eagle3_aux_hidden_state_layers
Qwen4ExpForCausalLM.set_aux_hidden_state_layers = _set_aux_hidden_state_layers
Qwen4ExpForCausalLM.get_eagle3_aux_hidden_state_layers = _get_eagle3_aux_hidden_state_layers


# ── State pool rows ───────────────────────────────────────────────────────────
def _pool_rows(pool, slots, b):
    """Rows ``slots[:b]`` of ``pool`` one slot at a time (see ``_state_rows``)."""
    # Each slot read as one row of the flattened pool: on the [n, C, K] pool a 1-slot index_select lowered to
    # ≈ 138 indirect DMAs per GDN layer on GpSimd at B=1, delaying the layer's first instruction (2.4 ms per
    # verify step, 2026-11-05 B=1 profile).
    flat = pool.view(pool.shape[0], -1)
    rows = torch.cat([flat.index_select(0, slots[j : j + 1]) for j in range(b)]) if b > 1 else flat.index_select(0, slots[:1])
    return rows.view(b, *pool.shape[1:])


def _pool_write(pool, slots, b, value):
    """Write ``value[:b]`` to ``pool`` at ``slots[:b]``, one slot at a time; the dummy slot is never written.

    The dummy slot (the runner's ``dummy_slot`` = the pool's second-to-last row) is read by every padding
    row. On device a NaN in one row of a decode step does not stay in that row: some decode kernels mix rows
    (one NaN row makes every row of the all-expert ``moe_tkg`` output NaN), so a NaN in the dummy slot
    reaches the real rows, and through their state pools every later step of those requests. Warm-up graphs
    run every row on the dummy slot against synthetic inputs, so they may produce anything. The pools have
    one row more than the runner's slots (allocate_request_state): the last row is a write-only sink. A row
    whose slot is the dummy slot writes to the sink instead, so the dummy slot keeps the zeros it was
    allocated with, and nothing reads the sink. Only the write index changes (one int compare + add on the
    ``[b]`` slots); every write's value is the step's value, as before the dummy-slot fix.
    """
    # pool[slot] = v (one 1-D index) lowers to a windowed row scatter; index_copy_ was decomposed into an
    # element-wise scatter, 4-byte indirect DMAs inside neuronx-cc's reused layer modules (2026-11-05).
    # A select feeding the last row's write (6200f25: where(pad, 0, last)) made the B=2 / B=16 decode steps
    # 1.5x / 7x slower on device; the value operand is left untouched here.
    sink = pool.shape[0] - 1
    w = slots[:b] + (slots[:b] == sink - 1).to(slots.dtype)  # dummy slot -> sink
    for j in range(b):
        pool[w[j : j + 1]] = value[j : j + 1].to(pool.dtype)


# _DECODE_ROWS: a decode row either continues a request (its slot holds the state that the
# request's prefill wrote: has_initial_state is 1) or is padding (the dummy slot, whose
# output is discarded). So decode reads the slot as is, without the ``where(has_init)``
# select (3.0 ms per TP32 step at B=8, 2026-11-03 profile). That is safe only because the
# dummy slot holds zeros (``_pool_write`` never writes it): a NaN there does not stay in the padding rows on
# device. The K/V a padding row writes go to the null block, which attention masks with ``where``.


# ── Device-side n-gram ids ─────────────────────────────────────────────────────────────────
# The PLE n-gram ids are computed in the graph (int32 / fp32-safe hash of ngram_device.py) from
# the step's input_ids and a per-slot int32 pool holding each request's last ``ngram_size - 1``
# processed tokens, instead of on the host from the token table. The pool is read like the other
# state pools (EOS for a fresh request) and written after the step with the tokens whose state
# was committed: ``num_valid`` of a prefill chunk / decode row, the accepted count (1 + accepted
# drafts) of a verify row. This removes the host inputs that depend on the last sampled token, the
# precondition for async scheduling. Padding tokens (beyond a row's ``num_valid``) get the all-EOS
# window, as ngram.compute_ngram_ids does on the host.


def _device_ngram_consts(self, device):
    from .ngram_device import DeviceNGramConsts

    cache = getattr(self, "_ngram_dev_consts", None)
    if cache is None or cache[0] != device:
        cache = (device, {i: DeviceNGramConsts.from_layout(lay, device)
                          for i, lay in self.model.ngram_layouts.items()})
        self._ngram_dev_consts = cache
    return cache[1]


def _device_ngram_prepare(self, input_ids, attn_metadata, spec_decode_metadata):
    """Put ``ngram_ids.{i}`` ([T, heads] int32) into the request state; return the pool update."""
    from .ngram_device import ngram_ids_from_windows, windows_from_context

    rs = attn_metadata["request_state"]
    meta0 = attn_metadata[f"layers.{self._attention_layers()[0][0]}.self_attn"]
    q = meta0["max_query_len"]
    is_prefill = q > meta0["decode_token_threshold"]
    verify = spec_decode_metadata is not None and not is_prefill
    t = input_ids.shape[0]
    b = 1 if is_prefill else t // q
    T = t // b
    eos = self.config.eos_token_id
    slots = rs["state_indices"][:b].long(); read_slots = _state_read_slots(rs, slots, b)  # prefix caching
    has_init = rs["has_initial_state"][:b].bool()
    num_valid = rs["num_valid"][:b].to(torch.int32)
    pool = self._ngram_ctx_pool
    ctx = pool.index_select(0, read_slots) if b > 0 else pool[:0]
    if is_prefill:
        ctx = torch.where(has_init.view(-1, 1), ctx, torch.full_like(ctx, eos))
    toks = input_ids.to(torch.int32).view(b, T).clamp(min=0)
    real = torch.arange(T, device=toks.device, dtype=torch.int32).view(1, T) < num_valid.view(-1, 1)
    for i, c in _device_ngram_consts(self, toks.device).items():
        win = windows_from_context(c, ctx[:, ctx.shape[1] - c.context_len:], toks)
        win = torch.where(real.reshape(-1, 1), win, torch.full_like(win, eos))
        rs[f"ngram_ids.{i}"] = ngram_ids_from_windows(c, win)
    return dict(b=b, T=T, slots=slots, seq=torch.cat([ctx, toks], dim=1), num_valid=num_valid,
                verify=verify, prefill=is_prefill)


def _device_ngram_commit(self, upd, tokens):
    """Write the last ``ngram_size - 1`` committed tokens of every row into the pool."""
    b, seq = upd["b"], upd["seq"]
    if upd["verify"]:
        count = (tokens[:b] != -1).sum(dim=1).to(torch.int32)  # x0 + accepted drafts
    else:
        count = upd["num_valid"]
    w = self._ngram_ctx_pool.shape[1]
    idx = count.clamp(min=0).view(-1, 1) + torch.arange(w, device=seq.device, dtype=torch.int32).view(1, w)
    new = torch.gather(seq, 1, idx.long())  # seq = ctx ++ row: the w tokens before index ctx + count
    slots = upd["slots"]
    for j in range(b):
        self._ngram_ctx_pool[slots[j : j + 1]] = new[j : j + 1]


def _async_spec_correct(vocab_size, positions, attn_metadata, prev_sampled, prev_num_draft,
                        req_indices_per_token):
    """The @async_speculative_decoding correction (``correct_spec_decode_positions_and_slot_mapping``)
    over the paged layers only (the request-state entry is not a layer); a no-op unless the
    runner passed the prev_* kwargs. Positions are clamped at 0: a padding row keeps the previous
    step's padding output (``_remap_prev_sampled_by_req_id``), whose rejected count can exceed its
    position; a negative position then indexed the block table out of range (the drafter's
    ``compute_slot_mapping`` raised on CPU) or, through ``view(-1)``, the previous row's table.
    A real row is never clamped (its optimistic position is >= its rejected count)."""
    if prev_sampled is None or prev_num_draft is None or req_indices_per_token is None:
        return positions, attn_metadata
    from vllm_neuron.functional.spec_decode_correction import _compute_slot_mapping_per_token

    valid = (prev_sampled != -1) & (prev_sampled < vocab_size)
    rejected = (1 + prev_num_draft.to(torch.int32)) - valid.sum(dim=1).to(torch.int32)
    positions = (positions - rejected[req_indices_per_token].to(positions.dtype)).clamp(min=0)
    fixed = {}
    for name, meta in attn_metadata.items():
        if isinstance(meta, dict) and "block_table_tensor" in meta and "block_size" in meta:
            table = meta.get("full_block_table_tensor", meta["block_table_tensor"])
            meta = dict(meta)
            meta["slot_mapping"] = _compute_slot_mapping_per_token(
                positions, table, req_indices_per_token, meta["block_size"])
        fixed[name] = meta
    return positions, fixed


import functools

_host_allocate_request_state = Qwen4ExpForCausalLM.allocate_request_state
_host_forward = Qwen4ExpForCausalLM.forward

def _allocate_request_state_device_ngram(self, num_slots: int, device) -> None:
    _host_allocate_request_state(self, num_slots, device)
    width = max(lay.context_len for lay in self.model.ngram_layouts.values())
    self._ngram_ctx_pool = torch.full((num_slots, width), self.config.eos_token_id,
                                      dtype=torch.int32, device=device)

@functools.wraps(_host_forward)
def _forward_device_ngram(self, input_ids, positions, inputs_embeds=None, is_token_ids=None,
                          attn_metadata=None, sampling_positions=None, sampling_params=None,
                          spec_decode_metadata=None, logit_mask=None, rank=None,
                          prev_sampled_token_ids=None, prev_num_draft_tokens=None,
                          req_indices_per_token=None, **kwargs):
    # Async spec decode (the @async_speculative_decoding prologue): the scheduler's
    # num_computed_tokens assumed every draft of the previous step was accepted; shift
    # positions back by the rejected count and recompute the layers' slot mappings.
    positions, attn_metadata = _async_spec_correct(
        self.config.vocab_size, positions, attn_metadata, prev_sampled_token_ids,
        prev_num_draft_tokens, req_indices_per_token)
    upd = _device_ngram_prepare(self, input_ids, attn_metadata, spec_decode_metadata)
    out = _host_forward(self, input_ids, positions, inputs_embeds, is_token_ids, attn_metadata,
                        sampling_positions, sampling_params, spec_decode_metadata, logit_mask,
                        rank, **kwargs)
    tokens = out[0] if isinstance(out, tuple) else out
    _device_ngram_commit(self, upd, tokens); _verify_snapshot_ngram_write(self, upd, tokens, attn_metadata["request_state"], positions)  # prefix caching
    return out

Qwen4ExpForCausalLM.allocate_request_state = _allocate_request_state_device_ngram
# The runner injects the prev_* kwargs (async + spec decode only) for marked models.
Qwen4ExpForCausalLM._async_spec_decoded = True

from . import mtp as _mtp

_host_mtp_forward = _mtp.Qwen4ExpMTP.forward

@functools.wraps(_host_mtp_forward)
def _mtp_forward_async(self, input_ids, positions, initial_target_hidden_states,
                       attn_metadata=None, sampling_positions=None, rank=None,
                       raw_sampled_token_ids=None, prev_sampled_token_ids=None,
                       prev_num_draft_tokens=None, req_indices_per_token=None):
    positions, attn_metadata = _async_spec_correct(
        self.config.vocab_size, positions, attn_metadata, prev_sampled_token_ids,
        prev_num_draft_tokens, req_indices_per_token)
    return _host_mtp_forward(self, input_ids, positions, initial_target_hidden_states,
                             attn_metadata, sampling_positions, rank, raw_sampled_token_ids)

_mtp.Qwen4ExpMTP.forward = _mtp_forward_async
Qwen4ExpForCausalLM.forward = _forward_device_ngram


# ── FP8 checkpoint (Qwen/Qwen3.8-Flash-Next-FP8) ─────────────────────────────
# Routed experts: per-expert gate/up/down e4m3fn with 128x128 block scales ``weight_scale_inv``
# (bf16, W = q * s per block). FP8 experts (config.expert_fp8) are requantized per output channel
# (trn2 e4m3, |q| <= 240: the checkpoint's codes 256..448 would be inf there) -- one more rounding,
# ~2.6 % weight error on the real layers -- else dequantized exactly to the model dtype. The PLE
# n-gram table (e4m3fn, one bf16 ``weight_scale``) is dequantized to the model dtype at load, as
# transformers' FP8Embedding does per lookup (rows.to(bf16) * scale): the table is in BF16 on device
# as with the BF16 checkpoint. Everything else in the checkpoint is BF16.
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2, torch.float8_e5m2fnuz)


def _read_ple_scale(checkpoint_path: str, entry: tuple[str, str] | None) -> torch.Tensor | None:
    if entry is None:
        return None
    from safetensors import safe_open

    ckpt_name, file = entry
    with safe_open(os.path.join(checkpoint_path, file), framework="pt") as f:
        s = f.get_tensor(ckpt_name)
    if s.numel() != 1:
        raise ValueError(f"{ckpt_name}: expected one per-tensor scale, got shape {tuple(s.shape)}")
    return s.reshape(())


def _ple_rows(rows: torch.Tensor, scale: torch.Tensor | None, dtype: torch.dtype, name: str) -> torch.Tensor:
    """Rows of the PLE table in ``dtype``: FP8 rows times the per-tensor scale (in ``dtype``; the FP8
    value is exact in bf16, so bf16 rounds the product once, like FP8Embedding)."""
    if rows.dtype in _FP8_DTYPES:
        if scale is None:
            raise ValueError(f"{name}: FP8 table without ngram_embedding.weight_scale")
        return rows.to(dtype).mul_(scale.to(dtype))
    if scale is not None:
        raise ValueError(f"{name}: weight_scale given for a {rows.dtype} table")
    return rows


def _fp8_block_dequant(q: torch.Tensor, s: torch.Tensor, block: tuple[int, int], name: str) -> torch.Tensor:
    """Exact fp32 dequant of a block-quantized ``q`` ``[out, in]`` with ``s`` ``[ceil(out/bm), ceil(in/bn)]``."""
    bm, bn = block
    o, i = q.shape
    if tuple(s.shape) != (-(-o // bm), -(-i // bn)):
        raise ValueError(f"{name}: scale {tuple(s.shape)} does not match weight {tuple(q.shape)} in {block} blocks")
    sf = s.float().repeat_interleave(bm, 0)[:o].repeat_interleave(bn, 1)[:, :i]
    return q.float() * sf


def _load_per_expert(checkpoint_path, prefix, items, params, config, dtype) -> dict[str, torch.Tensor]:
    """State dict entries of ``<prefix>.{gate_up_proj,down_proj}`` (+ FP8 scales) from
    per-expert checkpoint tensors ``items`` = [(expert, proj, weight|weight_scale_inv, name, file)].

    Reads this rank's ``E_L`` experts only."""
    from safetensors import safe_open

    from .layers import fp8_row_quantize

    tp = config.tp
    block = tuple(config.fp8_block_size or (128, 128))
    gu_name, dn_name = prefix + ".gate_up_proj", prefix + ".down_proj"
    n = params[dn_name].shape[0]  # local experts
    base = tp.rank * n
    fp8_param = params[gu_name].dtype in _FP8_DTYPES
    files: dict[str, list[tuple[int, str, str, str]]] = {}
    for e, proj, kind, ckpt_name, file in items:
        if base <= e < base + n:
            files.setdefault(file, []).append((e, proj, kind, ckpt_name))
    raw: dict[tuple[int, str, str], torch.Tensor] = {}
    for file, entries in sorted(files.items()):
        with safe_open(os.path.join(checkpoint_path, file), framework="pt") as f:
            for e, proj, kind, ckpt_name in entries:
                raw[(e, proj, kind)] = f.get_tensor(ckpt_name)

    def weight(e: int, proj: str) -> torch.Tensor:
        """HF layout ``[out, in]``: fp32 exact dequant for FP8, else as stored."""
        key = f"{prefix}.{e}.{proj}"
        q = raw.get((e, proj, "weight"))
        if q is None:
            raise ValueError(f"{key}.weight not in checkpoint")
        s = raw.get((e, proj, "weight_scale_inv"))
        if q.dtype in _FP8_DTYPES:
            if s is None:
                raise ValueError(f"{key}: FP8 weight without weight_scale_inv")
            return _fp8_block_dequant(q, s, block, key)
        if s is not None:
            raise ValueError(f"{key}: weight_scale_inv given for a {q.dtype} weight")
        return q

    out: dict[str, torch.Tensor] = {}
    if fp8_param:
        e_l, h, two_i = params[gu_name].shape
        i = two_i // 2
        gu = torch.empty(e_l, h, two_i, dtype=torch.float8_e4m3fn)
        dn = torch.empty(e_l, i, h, dtype=torch.float8_e4m3fn)
        gus = torch.empty(e_l, 1, two_i, dtype=torch.float32)
        dns = torch.empty(e_l, 1, h, dtype=torch.float32)
        for j in range(e_l):
            g_q, g_s = fp8_row_quantize(weight(base + j, "gate_proj").float())  # [I, H], [I]
            u_q, u_s = fp8_row_quantize(weight(base + j, "up_proj").float())
            d_q, d_s = fp8_row_quantize(weight(base + j, "down_proj").float())  # [H, I], [H]
            if g_q.shape != (i, h) or d_q.shape != (h, i):
                raise ValueError(f"{prefix}.{base + j}: shapes {tuple(g_q.shape)} / {tuple(d_q.shape)} "
                                 f"vs parameter I={i}, H={h}")
            gu[j, :, :i], gu[j, :, i:] = g_q.T, u_q.T  # kernel layout [H, 2I], gate columns first
            dn[j] = d_q.T  # [I, H]
            gus[j, 0, :i], gus[j, 0, i:] = g_s, u_s
            dns[j, 0] = d_s
        out[gu_name], out[dn_name] = gu, dn
        out[prefix + ".gate_up_scale"], out[prefix + ".down_scale"] = gus, dns
    else:
        _M = Qwen4ExpForCausalLM
        gate_up = torch.stack([torch.cat([weight(e, "gate_proj"), weight(e, "up_proj")]).to(dtype)
                               for e in range(base, base + n)])
        down = torch.stack([weight(e, "down_proj").to(dtype) for e in range(base, base + n)])
        out[gu_name] = _M._checked(params[gu_name], gate_up, gu_name, dtype)
        out[dn_name] = _M._checked(params[dn_name], down, dn_name, dtype)
    return out


# ── FP8 PLE table (FP8 checkpoint; layers._ple_fp8_init) ─────────────────────────────────────────────
# For a model whose PLE parameter is the uint8 byte table (layers._ple_fp8_init), the PLE loop of load_weights
# keeps this rank's rows as their e4m3fn bytes (no dequant) and the per-tensor scale goes into the module's
# ``ngram_scale`` buffer (in the model dtype, as _ple_rows uses it). Keyed on the parameter dtype, so BF16-table
# models load exactly as before.
_PLE_FP8_TARGETS: set[str] = set()  # byte-table PLE parameter names of the model being loaded
_PLE_FP8_SCALES: dict[str, torch.Tensor] = {}


def _ple_rows_fp8(rows: torch.Tensor, scale: torch.Tensor | None, dtype: torch.dtype, name: str) -> torch.Tensor:
    if name not in _PLE_FP8_TARGETS:
        return _ple_rows_orig(rows, scale, dtype, name)
    if rows.dtype != torch.float8_e4m3fn:
        raise ValueError(f"{name}: the FP8 PLE table must be e4m3fn, got {rows.dtype}")
    if scale is None:
        raise ValueError(f"{name}: FP8 table without ngram_embedding.weight_scale")
    b = rows.view(torch.uint8)
    nan = int(((b & 0x7F) == 0x7F).sum())
    if nan:
        raise ValueError(f"{name}: {nan} NaN codes in the FP8 table")
    s = scale.to(dtype)
    prev = _PLE_FP8_SCALES.setdefault(name, s)
    if not torch.equal(prev, s):
        raise ValueError(f"{name}: two different scales {prev} / {s}")
    return b


def _checked_ple_fp8(param: nn.Parameter, tensor: torch.Tensor, name: str, dtype: torch.dtype) -> torch.Tensor:
    if name in _PLE_FP8_TARGETS and param.dtype == tensor.dtype == torch.uint8:
        if tuple(tensor.shape) != tuple(param.shape):
            raise ValueError(f"{name}: checkpoint {tuple(tensor.shape)} vs parameter {tuple(param.shape)}")
        return tensor
    return _checked_orig(param, tensor, name, dtype)


def _load_weights_ple_fp8(self, checkpoint_path: str, *args, **kwargs) -> set[str]:
    targets = {n for n, p in self.named_parameters()
               if n.endswith(".ple_embedding.ngram_embedding.weight") and p.dtype == torch.uint8}
    _PLE_FP8_TARGETS.clear()
    _PLE_FP8_TARGETS.update(targets)
    _PLE_FP8_SCALES.clear()
    try:
        loaded = _load_weights_orig(self, checkpoint_path, *args, **kwargs)
        modules = dict(self.named_modules())
        for name in sorted(targets & set(loaded)):
            if name not in _PLE_FP8_SCALES:
                raise ValueError(f"{name}: FP8 PLE table loaded without its scale")
            # "<...>.ple_embedding.ngram_embedding.weight" -> the Qwen4ExpNGramEmbedding module
            modules[name.rsplit(".", 2)[0]].ngram_scale = _PLE_FP8_SCALES[name]
    finally:
        _PLE_FP8_TARGETS.clear()
        _PLE_FP8_SCALES.clear()
    return loaded


_ple_rows_orig = _ple_rows
_checked_orig = Qwen4ExpForCausalLM._checked
_load_weights_orig = Qwen4ExpForCausalLM.load_weights


_ple_rows = _ple_rows_fp8
Qwen4ExpForCausalLM._checked = staticmethod(_checked_ple_fp8)
Qwen4ExpForCausalLM.load_weights = _load_weights_ple_fp8

from . import layers as _ple_layers  # noqa: E402


# [in, out] weight copies (layers.py, wio_wrap): lm_head reads an [in, out] copy for decode / verify /
# MTP-draft rows (+39.7 MB per rank at TP32).
def _wio_causal_lm_init(self, config):
    _wio_causal_lm_orig_init(self, config)
    _ple_layers.wio_wrap(self.lm_head)


_wio_causal_lm_orig_init = Qwen4ExpForCausalLM.__init__
Qwen4ExpForCausalLM.__init__ = _wio_causal_lm_init


# Device n-gram constants built before the first trace (11-06 device run): `_device_ngram_consts` cached its
# tensors on the model the first time it ran, i.e. inside the first traced graph (the 128-token prefill bucket in the
# warm-up). That graph's guard was `not hasattr(self, "_ngram_dev_consts")`, so the next 128-token call failed it and
# raised under the runner's fail_on_recompile stance. Build them when the request-state pool is allocated (before any
# trace) and return the cached ones regardless of the device object the trace sees.
_device_ngram_consts_build = _device_ngram_consts
_allocate_request_state_ngram_no_consts = Qwen4ExpForCausalLM.allocate_request_state

def _device_ngram_consts(self, device):  # noqa: F811 (rebinds the global used by _device_ngram_prepare)
    cache = getattr(self, "_ngram_dev_consts", None)
    if cache is not None:
        return cache[1]
    return _device_ngram_consts_build(self, device)

def _allocate_request_state_ngram_consts(self, num_slots: int, device) -> None:
    _allocate_request_state_ngram_no_consts(self, num_slots, device)
    _device_ngram_consts_build(self, self._ngram_ctx_pool.device)

Qwen4ExpForCausalLM.allocate_request_state = _allocate_request_state_ngram_consts


# ── Prefix caching (align mode; vllm_neuron/vllm/worker/request_state.py, "Prefix caching") ─────────────
# With prefix caching on, the runner adds S snapshot rows to the pools (allocate_request_state's num_slots
# includes them; the dummy slot stays the second-to-last row, so _pool_write is unchanged). The graphs now
# read and write every pool at ``state_indices`` only: the runner restores / saves a snapshot row with a
# row copy between graphs (request_state_pools at the end of the module; request_state.py), so it never passes
# ``state_read_indices`` (a separate read index made device steps nondeterministic) and the reads below use
# ``state_indices`` (the graphs of prefix caching off, with S more pool rows).
# The edits above keep every line number of the module (NEFF keys include source lines).
def _state_read_slots(rs, slots, b):
    read = rs.get("state_read_indices")
    return slots if read is None else read[:b].long()


Qwen4ExpForCausalLM.supports_state_prefix_caching = True



# Prefix caching, boundaries reached in decode / MTP verify (request_state.py, "Prefix caching"): with MTP the runner
# passes ``state_snapshot_rows`` [rows] (a row the scheduler reserved, or the dummy slot). A decode / verify row whose
# window of tokens at p0 .. p0 + n - 1 reaches the block boundary b = p0 + offset + 1, offset = (-p0 - 1) mod block,
# with offset < count (the tokens whose state is kept: 1 + accepted drafts, 1 in a plain decode step) writes the
# state after token ``offset`` to that row; every other row writes to the sink (the dummy slot maps there). Positions
# are the corrected ones (async: the previous step's rejections), so the offset is right where the host could only
# bound it; the host publishes the row once the step's output tells it which tokens were kept.
def _verify_snapshot_write(self, rs, pos, b, cands=None, count=None, states=None):
    snap = rs.get("state_snapshot_rows")
    if snap is None or (cands is None and states is None):
        return
    from .layers import select_verified_state

    off = torch.remainder(-pos[:b, 0] - 1, self.state_snapshot_block_size)
    crossed = off < (count[:b].to(off.dtype) if count is not None else 1)
    for i in range(len(self._state_pool)):
        st = states[i] if cands is None else select_verified_state(cands[i], off + 1)
        for key, pool in self._state_pool[i].items():
            rows = torch.where(crossed, snap[:b].long(), torch.full_like(snap[:b].long(), pool.shape[0] - 2))
            _pool_write(pool, rows, b, st[key])


def _verify_snapshot_ngram_write(self, upd, tokens, rs, positions):
    snap = rs.get("state_snapshot_rows")
    if snap is None or upd["prefill"]:
        return
    b, T, seq = upd["b"], upd["T"], upd["seq"]
    count = (tokens[:b] != -1).sum(dim=1).to(torch.int32) if upd["verify"] else upd["num_valid"]
    off = torch.remainder(-positions.view(b, T)[:, 0].to(torch.int32) - 1, self.state_snapshot_block_size)
    w = self._ngram_ctx_pool.shape[1]
    idx = (off + 1).clamp(max=T).view(-1, 1) + torch.arange(w, device=seq.device, dtype=torch.int32).view(1, w)
    new = torch.gather(seq, 1, idx.long())  # the w tokens before index ctx + offset + 1
    dummy = torch.full_like(snap[:b].long(), self._ngram_ctx_pool.shape[0] - 1)
    rows = torch.where(off < count, snap[:b].long(), dummy)
    for j in range(b):
        self._ngram_ctx_pool[rows[j : j + 1]] = new[j : j + 1]


# Prefix caching, snapshot rows (request_state.py, "Prefix caching"): every per-slot pool the runner's row copies move
# (dim 0 = pool row): the GDN / PLE state pools of every layer and the n-gram window pool.
def _request_state_pools(self):
    return [pool for layer in self._state_pool for pool in layer.values()] + [self._ngram_ctx_pool]


Qwen4ExpForCausalLM.request_state_pools = _request_state_pools
