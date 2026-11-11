# SPDX-License-Identifier: Apache-2.0
"""Factory for Qwen4-Exp (Qwen3.8-Flash-Next) model selection."""

import logging

import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig

from .layers import GDN_CTE_CHUNK

logger = logging.getLogger(__name__)


class Qwen4ExpForCausalLM(nn.Module):
    """Factory that validates config and selects the Qwen4-Exp implementation.

    Only the BF16 text model exists so far. The composite checkpoint
    (``Qwen4ExpForConditionalGeneration``) is served text-only: the vision tower
    and the MTP layer are not loaded.
    """

    # Prefix caching (vllm_neuron/vllm/worker/request_state.py): state snapshots sit on multiples of
    # lcm(block_size, this), the gated-DeltaNet prefill kernel's chunk, so a prefill resumed from a cache hit
    # runs on the same chunk grid as the cold run.
    state_snapshot_alignment = GDN_CTE_CHUNK

    def __init__(self, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None = None,
                 **kwargs) -> None:
        super().__init__()
        self._model = self.from_configs(hf_config, neuron_config, **kwargs)

    def forward(self, *args, **kwargs):
        """Delegate forward pass to the selected implementation."""
        return self._model(*args, **kwargs)

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        neuron_config: NeuronConfig | None = None,
        text_neuron_config: NeuronConfig | None = None,
        encoder_neuron_config: NeuronConfig | None = None,
    ) -> nn.Module:
        """Create model from configs. Returns the selected implementation directly.

        The runner passes ``text_neuron_config`` / ``encoder_neuron_config`` for a
        multimodal architecture; this port is text-only, so the encoder config is unused.
        """
        return cls._select_implementation(hf_config, text_neuron_config or neuron_config)

    @classmethod
    def _select_implementation(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        cls._validate_config(hf_config, neuron_config)
        from .model_bf16 import Qwen4ExpForCausalLM as Model

        logger.info("[qwen4_exp] model=model_bf16 (text-only)")
        model = Model.from_configs(hf_config, neuron_config)
        # Host-side prefill context buckets (max_model_len > 32,768 only; see prefill_ctx.py).
        from .prefill_ctx import maybe_install

        maybe_install()
        return model

    @classmethod
    def _validate_config(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None) -> None:
        quantization = neuron_config.quantization if neuron_config else None
        if quantization not in (None, "bf16"):
            raise ValueError(f"Qwen4-Exp supports bf16 only, got quantization={quantization!r}")


class Qwen4ExpForConditionalGeneration(Qwen4ExpForCausalLM):
    """Registry entry for the composite checkpoint; served text-only."""
