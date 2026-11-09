# SPDX-License-Identifier: Apache-2.0
"""MTP (multi-token prediction) proposer: ``speculative_config.method == "mtp"``.

The draft is the target checkpoint's own MTP block (vLLM rewrites the draft config's
architecture, e.g. ``Qwen4ExpMTP``) and follows the Eagle fused-propose contract, so this
reuses :class:`EagleProposer` and changes only what differs:

* one draft layer, named after the target's last layer (``layers.{N}.self_attn``);
* the target's auxiliary hidden state is whatever the draft model declares
  (``target_hidden_size`` attribute, e.g. ``hc_count * hidden_size`` for Qwen4-Exp's
  pre-final-mixer streams), not three concatenated layers;
* the embedding and LM head are the target's (:meth:`alias_target_weights`).
"""

from __future__ import annotations

import torch

from vllm_neuron.vllm.spec_decode.eagle import EagleProposer


class MTPProposer(EagleProposer):
    def load_model(self, target_hidden_size: int, target_hidden_pad_groups: int = 1,
                   compile_options: dict | None = None) -> None:
        text = self.vllm_config.model_config.hf_text_config
        n = text.num_hidden_layers
        draft = self.draft_model_config.hf_config
        if not hasattr(draft, "hidden_size"):  # composite config: EagleProposer reads it at the top level
            draft.hidden_size = draft.get_text_config().hidden_size
        self.model = self.compile_and_load_draft_model(
            start_layer_idx=n,
            target_hidden_size=target_hidden_size,
            target_hidden_pad_groups=target_hidden_pad_groups,
            compile_options=compile_options,
        )
        n_mtp = getattr(text, "mtp_num_hidden_layers", 1)
        self.attn_layer_names = [f"layers.{i}.self_attn" for i in range(n, n + n_mtp)]
        # Extra caches the draft reads (Qwen4-Exp past 2,051 tokens: the QSA indexer's key-only
        # cache): their metadata is passed to the draft too (``propose`` filters by this list).
        extra = [ln for ln in getattr(self.model, "kv_layer_names", ()) if ln not in self.attn_layer_names]
        self.attn_layer_names += extra

    def validate_same_kv_cache_group(self, kv_cache_config) -> None:
        """As EagleProposer's, over the draft's attention layers only: a key-only indexer cache has
        its own width and may sit in another group (its metadata is read under its own name)."""
        names = [ln for ln in self.attn_layer_names if not ln.endswith(".indexer")]
        groups = {ln: i for i, g in enumerate(kv_cache_config.kv_cache_groups) for ln in g.layer_names}
        assert len({groups[ln] for ln in names}) == 1, "All MTP attention layers should belong to the same kv cache group"

    def _build_synthetic_inputs(self, num_tokens: int, num_reqs: int, device=None) -> dict:
        kw = super()._build_synthetic_inputs(num_tokens, num_reqs, device)
        width = getattr(self.model, "target_hidden_size", None)
        if width is not None:
            h = kw["target_hidden_states"]
            kw["target_hidden_states"] = torch.ones(h.shape[0], width, dtype=h.dtype, device=h.device)
        return kw

    def alias_target_weights(self, target_model) -> None:
        self.model.alias_target_weights(target_model)
