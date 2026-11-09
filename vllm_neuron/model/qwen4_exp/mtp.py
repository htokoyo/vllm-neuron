"""Qwen4-Exp MTP (multi-token prediction) draft module.

The checkpoint's ``mtp.*`` block (``mtp_num_hidden_layers = 1``): the target's pre-final-mixer
multi stream ``[..., hc * H]`` at position ``t`` and the embedding of the token at ``t + 1``
are fused as ``fc_hidden(pre_fc_norm_hidden(streams))`` on every stream plus
``fc_embedding(pre_fc_norm_embedding(embed))``, then one full-attention hyper-connection
decoder layer (MoE MLP, no PLE) with its own KV, then ``hyper_connection_mixer`` and the
target's LM head. The layer's output streams feed the next draft step. Follows upstream vLLM
``vllm/models/qwen4_exp/nvidia/mtp.py`` (``pre_fc_norm_hidden`` normalises the whole
``hc * H`` vector once).

The embedding and LM head are the target's (``mtp_use_dedicated_embeddings = False``); call
:meth:`Qwen4ExpMTP.alias_target_weights` after loading both. The decoder layer is this
package's :class:`Qwen4ExpDecoderLayer` built as layer ``num_hidden_layers`` with type
full attention, so its tensor-parallel split (attention group, experts, shared expert) is the
target's.
"""

from __future__ import annotations

import copy
import os

import torch
from torch import nn

from .config import FULL_ATTENTION, Qwen4ExpConfig
from .layers import Qwen4ExpDecoderLayer, Qwen4ExpGatedResidual, Qwen4ExpRMSNorm
from .parallel import shard_pieces, sharded_lookup

MTP_PREFIX = "mtp."


class Qwen4ExpMTP(nn.Module):
    def __init__(self, config: Qwen4ExpConfig):
        super().__init__()
        if getattr(config, "mtp_num_hidden_layers", 1) != 1:
            raise ValueError("Qwen4-Exp MTP: only mtp_num_hidden_layers = 1 is supported")
        self.config = config
        h, hc = config.hidden_size, config.hc_count
        self.hidden, self.hc = h, hc
        layer_config = copy.copy(config)
        layer_config.layer_types = list(config.layer_types) + [FULL_ATTENTION]
        self.layer_idx = config.num_hidden_layers
        self.fc_embedding = nn.Linear(h, h, bias=False)
        self.fc_hidden = nn.Linear(h, h, bias=False)
        self.pre_fc_norm_embedding = Qwen4ExpRMSNorm(h, config.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen4ExpRMSNorm(hc * h, config.rms_norm_eps)
        self.layers = nn.ModuleList([Qwen4ExpDecoderLayer(layer_config, self.layer_idx, 0)])
        assert self.layers[0].ple is None
        self.hyper_connection_mixer = Qwen4ExpGatedResidual(config, use_combine=False)
        self._target: tuple = ()  # (target,) — a tuple, so its modules are not registered here

    def alias_target_weights(self, target) -> None:
        """Share the target's (vocab-sharded) embedding table and LM head."""
        self._target = (target,)

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return sharded_lookup(self._target[0].model.embed_tokens.weight, ids, self.config.tp)

    def fuse(self, streams: torch.Tensor, next_ids: torch.Tensor) -> torch.Tensor:
        """Layer input ``[..., hc * H]`` from target streams and the next tokens' ids."""
        e = self.fc_embedding(self.pre_fc_norm_embedding(self.embed(next_ids)))
        hs = self.pre_fc_norm_hidden(streams).unflatten(-1, (self.hc, self.hidden))
        return (self.fc_hidden(hs) + e.unsqueeze(-2)).flatten(-2)

    def forward_with_state(self, streams, next_ids, positions, kv):
        """Reference path: ``streams`` ``[B, T, hc*H]``, ``next_ids`` / ``positions`` ``[B, T]``,
        ``kv`` the layer's KV state (``None`` to start). Returns
        ``(logits [B, T, vocab / TP], out_streams [B, T, hc*H], kv)``."""
        out, st = self.layers[0](self.fuse(streams, next_ids), positions, None, {"kv": kv})
        return self._target[0].lm_head(self.hyper_connection_mixer(out)), out, st["kv"]

    # ── Draft-model interface of the spec-decode proposer ────────────────
    # (vllm_neuron/vllm/spec_decode: the Eagle fused-propose contract)
    num_speculative_tokens: int = 1
    kv_fill_only: bool = False

    @classmethod
    def from_configs(cls, config, start_layer_idx: int | None = None, neuron_config=None):
        """``config``: vLLM's draft config (the target checkpoint's, ``model_type`` qwen4_exp_mtp)."""
        c = Qwen4ExpConfig.from_configs(config, neuron_config)
        if start_layer_idx is not None and start_layer_idx != c.num_hidden_layers:
            raise ValueError(f"MTP layer index {start_layer_idx} != target layers {c.num_hidden_layers}")
        return cls(c)

    @property
    def target_hidden_size(self) -> int:
        """Width of the target's auxiliary hidden state: the pre-final-mixer streams."""
        return self.hc * self.hidden

    @property
    def attn_layer_name(self) -> str:
        return f"layers.{self.layer_idx}.self_attn"

    def get_kv_spec(self):
        from vllm_neuron.model.kv_cache import KVSpec, LayerSpec

        attn = self.layers[0].self_attn
        if not attn.dense:
            # The runner registers draft layers as plain full-attention caches; the QSA
            # indexer's key-only cache is not supported there yet.
            raise NotImplementedError(
                "Qwen4-Exp MTP needs dense attention: max_model_len must be <= "
                f"{self.config.dense_attention_exact_len} for now")
        return KVSpec(layers=[LayerSpec(name=self.attn_layer_name, num_kv_heads=attn.num_kv_heads,
                                        head_size=attn.head_dim, dtype=self.config.torch_dtype)])

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor]]):
        attn = self.layers[0].self_attn
        attn.k_cache, attn.v_cache = kv_caches[self.attn_layer_name][0], kv_caches[self.attn_layer_name][1]

    def _greedy(self, streams):
        from .parallel import all_gather_last

        logits = self._target[0].lm_head(self.hyper_connection_mixer(streams))
        return all_gather_last(logits, self.config.tp).argmax(-1).to(torch.int32)

    def forward(self, input_ids, positions, initial_target_hidden_states, attn_metadata=None,
                sampling_positions=None, rank=None, raw_sampled_token_ids=None,
                prev_sampled_token_ids=None, prev_num_draft_tokens=None, req_indices_per_token=None):
        """Fused propose (same contract as ``Eagle3LlamaForCausalLM.forward``, sync scheduling).

        ``input_ids`` ``[T]``: the target's tokens shifted by one (the last accepted position's
        next token is patched in from ``raw_sampled_token_ids``); ``initial_target_hidden_states``
        ``[T, hc*H]``: the target's pre-final-mixer streams of the same tokens. The first pass
        writes the MTP layer's K/V at every position; then ``num_speculative_tokens - 1`` one-token
        steps on the layer's own output streams. Returns ``(tokens [bs, 1+k] or [bs, k],
        drafts [bs, k], None)``; the first column of ``tokens`` is the bonus token on verify steps.
        """
        from vllm_neuron.model.llama3.eagle3_model import compute_slot_mapping, extract_accepted_tokens

        if raw_sampled_token_ids is not None:
            input_ids, sampling_positions, bonus = extract_accepted_tokens(
                input_ids, sampling_positions, raw_sampled_token_ids, self.config.vocab_size,
                self.num_speculative_tokens)
        else:
            bonus = None
        meta = attn_metadata[self.attn_layer_name]
        q = meta["max_query_len"]
        is_prefill = q > meta["decode_token_threshold"]
        t = input_ids.shape[0]
        b = 1 if is_prefill else t // q
        ids = input_ids.long().view(b, t // b)
        pos = positions.long().view(b, t // b)
        streams = initial_target_hidden_states.view(b, t // b, -1)
        layer = self.layers[0]
        out, _ = layer(self.fuse(streams, ids), pos, None, {"paged": meta, **_qsa_draft_idx(self, attn_metadata)}, None, not is_prefill)
        sp = sampling_positions.long()
        h = torch.index_select(out.reshape(t, -1), 0, sp)  # [bs, hc*H]
        cur = torch.index_select(positions.long(), 0, sp)
        if self.kv_fill_only:
            z = torch.zeros((sp.shape[0], self.num_speculative_tokens), dtype=torch.int32, device=input_ids.device)
            return z, z, None
        tok = self._greedy(h)
        drafts = [tok]
        table = meta["block_table_tensor"]
        block_size = attn_metadata[self.attn_layer_name].get("block_size", layer.self_attn.k_cache.shape[2])
        n = sp.shape[0]
        for _ in range(self.num_speculative_tokens - 1):
            cur = cur + 1
            step = dict(meta)
            step.update(slot_mapping=compute_slot_mapping(cur, table, block_size), max_query_len=1,
                        decode_token_threshold=1)
            o, _ = layer(self.fuse(h.view(n, 1, -1), tok.view(n, 1)), cur.view(n, 1), None, {"paged": step, **_qsa_draft_idx(self, attn_metadata, cur)}, None, True)
            h = o.reshape(n, -1)
            tok = self._greedy(h)
            drafts.append(tok)
        drafts_only = torch.stack(drafts, dim=1)
        tokens = torch.stack([bonus.to(torch.int32)] + drafts, dim=1) if bonus is not None else drafts_only
        return tokens, drafts_only, None

    # ── Weights ──────────────────────────────────────────────────────────
    def load_weights(self, checkpoint_path: str, device=None, download_dir=None) -> set[str]:
        """Load ``mtp.*`` (this rank's slices only) from a HF safetensors checkpoint."""
        from safetensors import safe_open

        from .model_bf16 import _PER_EXPERT, Qwen4ExpForCausalLM, _load_per_expert

        weight_map = Qwen4ExpForCausalLM._weight_map(checkpoint_path)
        params = dict(self.named_parameters())
        dtype = self.config.torch_dtype
        tp = self.config.tp
        by_file: dict[str, list[tuple[str, str]]] = {}
        per_expert: dict[str, list[tuple[int, str, str, str, str]]] = {}
        for ckpt_name, file in weight_map.items():
            if not ckpt_name.startswith(MTP_PREFIX):
                continue
            name = ckpt_name[len(MTP_PREFIX):]
            expert = _PER_EXPERT.match(name)  # per-expert layout (the FP8 checkpoint)
            if expert:
                per_expert.setdefault(expert.group(1), []).append(
                    (int(expert.group(2)), expert.group(3), expert.group(4), ckpt_name, file))
                continue
            if name not in params:
                raise KeyError(f"checkpoint tensor {ckpt_name} has no MTP parameter {name}")
            by_file.setdefault(file, []).append((name, ckpt_name))
        state_dict: dict[str, torch.Tensor] = {}
        for file, items in sorted(by_file.items()):
            with safe_open(os.path.join(checkpoint_path, file), framework="pt") as f:
                for name, ckpt_name in items:
                    sl = f.get_slice(ckpt_name)
                    pieces = shard_pieces(ckpt_name, tuple(sl.get_shape()), self.config, tp,
                                          self.config.attn_tp)
                    t = torch.cat([sl[p] for p in pieces]) if len(pieces) > 1 else sl[pieces[0]]
                    state_dict[name] = Qwen4ExpForCausalLM._checked(params[name], t, name, dtype)
        for prefix, items in per_expert.items():
            state_dict.update(_load_per_expert(checkpoint_path, prefix, items, params, self.config, dtype))
        missing = set(params) - set(state_dict)
        if missing:
            raise ValueError(f"{len(missing)} MTP parameters not in checkpoint, e.g. {sorted(missing)[:5]}")
        self.load_state_dict(state_dict, strict=False, assign=True)
        return set(state_dict)


# ── QSA in the draft layer (long context) ────────────────────────────────────
# Past ``dense_attention_exact_len`` (max_model_len > 2051) the MTP layer runs QSA like the
# target's full-attention layers (upstream ``nvidia/mtp.py`` builds it with a QSA indexer, and the
# checkpoint has ``mtp.layers.0.self_attn.indexer.*``): its raw indexer keys go to their own
# key-only cache ``layers.{N}.self_attn.indexer``, written at the same slots as its K/V, and each
# draft step selects its own blocks (upstream's ``index_share_for_mtp_iteration`` reuse of the step-0
# selection is not in this checkpoint's config). Drafts never change the output (the target
# verifies them); the draft's attention only changes the acceptance rate.


def _qsa_draft_idx(mtp: Qwen4ExpMTP, attn_metadata: dict, cur: torch.Tensor | None = None) -> dict:
    """Extra layer state for :meth:`Qwen4ExpMTP.forward`: ``{}`` when the draft layer is dense,
    else ``{"paged_idx": meta}`` with the indexer cache's metadata (``cur`` given: a one-token draft
    step at positions ``cur``, the slot mapping recomputed from the indexer cache's own block table)."""
    attn = mtp.layers[0].self_attn
    if attn.dense:
        return {}
    meta = attn_metadata[mtp.attn_layer_name + ".indexer"]
    if cur is None:
        return {"paged_idx": meta}
    from vllm_neuron.model.llama3.eagle3_model import compute_slot_mapping

    step = dict(meta)
    step.update(slot_mapping=compute_slot_mapping(cur, meta["block_table_tensor"],
                                                  meta.get("block_size", attn.idx_cache.shape[2])),
                max_query_len=1, decode_token_threshold=1)
    return {"paged_idx": step}


def _mtp_get_kv_spec(self):
    from vllm_neuron.model.kv_cache import KVSpec, LayerSpec

    attn = self.layers[0].self_attn
    layers = [LayerSpec(name=self.attn_layer_name, num_kv_heads=attn.num_kv_heads,
                        head_size=attn.head_dim, dtype=self.config.torch_dtype)]
    if not attn.dense:  # QSA indexer: one raw key per token, replicated on every rank (as the target's)
        layers.append(LayerSpec(name=self.attn_layer_name + ".indexer", num_kv_heads=1,
                                head_size=self.config.indexer_head_dim, dtype=self.config.torch_dtype,
                                key_only=True))
    return KVSpec(layers=layers)


def _mtp_bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor]]):
    attn = self.layers[0].self_attn
    attn.k_cache, attn.v_cache = kv_caches[self.attn_layer_name][0], kv_caches[self.attn_layer_name][1]
    if not attn.dense:
        c = kv_caches[self.attn_layer_name + ".indexer"][0]
        attn.idx_cache = c.squeeze(0) if c.dim() == 5 else c  # [blocks, 1, block, D]


def _mtp_kv_layer_names(self) -> list[str]:
    """Every cache the draft reads (the proposer passes these layers' metadata to ``forward``)."""
    return [layer.name for layer in self.get_kv_spec().layers]


Qwen4ExpMTP.get_kv_spec = _mtp_get_kv_spec
Qwen4ExpMTP.bind_kv_cache = _mtp_bind_kv_cache
Qwen4ExpMTP.kv_layer_names = property(_mtp_kv_layer_names)
