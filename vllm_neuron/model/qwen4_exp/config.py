# SPDX-License-Identifier: Apache-2.0
"""
Qwen4-Exp (Qwen3.8-Flash-Next) Configuration
============================================
<-- MODEL-SPECIFIC: All fields in this config are model-specific.

Architecture overview (text model):
- 48 decoder layers in a 3:1 pattern of Gated DeltaNet (linear attention) and
  gated full attention. The full-attention layers carry a QSA indexer that
  selects 4-token blocks; with a context of at most ``indexer_budget`` plus the
  open block tail the selection is the whole causal window, so dense causal
  attention is exact there.
- Every layer is a sparse MoE (softmax top-k router, sigmoid-gated shared expert).
- Hyper-connections ("gated residual"): the residual stream is ``hc_count``
  copies of ``hidden_size`` mixed by low-rank read gates and per-branch write gates.
- Per-layer embedding (PLE): hashed bigram/trigram embeddings injected before
  the 1-based layers listed in ``ple_layer_ids``.

The HF config can come from transformers (``Qwen4ExpTextConfig``, whose
``layer_types`` say ``qwen_sparse_attention``) or from vLLM's own config class
(``full_attention``). Both are accepted.
"""

import json
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig

LINEAR_ATTENTION = "linear_attention"
FULL_ATTENTION = "full_attention"
_FULL_ATTENTION_ALIASES = (FULL_ATTENTION, "qwen_sparse_attention", "attention")


def _text_config_dict(hf_config: Any) -> dict:
    """Return the text sub-config of a composite or text-only HF config as a dict."""
    if isinstance(hf_config, (str, bytes)):
        with open(hf_config) as f:
            hf_config = json.load(f)
    if isinstance(hf_config, PretrainedConfig):
        text = getattr(hf_config, "text_config", None) or hf_config
        return text.to_dict()
    return dict(hf_config.get("text_config", hf_config))


@dataclass
class Qwen4ExpConfig:
    """Configuration for the Qwen4-Exp text model."""

    # ── Model architecture (MODEL-SPECIFIC) ──────────────────────────────
    vocab_size: int = 248320
    hidden_size: int = 2560
    num_hidden_layers: int = 48
    layer_types: list[str] = field(default_factory=list)
    rms_norm_eps: float = 1e-6
    torch_dtype: torch.dtype = torch.bfloat16
    tie_word_embeddings: bool = False

    # ── Gated full attention ─────────────────────────────────────────────
    num_attention_heads: int = 24
    num_key_value_heads: int = 2
    head_dim: int = 256
    rope_theta: float = 10_000_000.0
    partial_rotary_factor: float = 0.25
    max_position_embeddings: int = 262144

    # ── QSA indexer (unused while attention runs dense) ──────────────────
    indexer_n_heads: int | None = None
    indexer_kv_heads: int | None = None
    indexer_head_dim: int | None = None
    indexer_budget: int | None = None
    indexer_compress_ratio: int | None = None

    # ── Gated DeltaNet ───────────────────────────────────────────────────
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 48
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    # Activation of the RMSNormGated output gate. Qwen3.5 uses silu; Qwen4-Exp uses sigmoid.
    output_gate_type: str = "sigmoid"

    # ── MoE ──────────────────────────────────────────────────────────────
    num_experts: int = 512
    num_experts_per_tok: int = 10
    moe_intermediate_size: int = 640
    shared_expert_intermediate_size: int = 640
    norm_topk_prob: bool = True

    # ── Hyper-connections ────────────────────────────────────────────────
    hc_count: int = 4
    hc_lowrank: int = 320

    # ── Per-layer n-gram embedding ───────────────────────────────────────
    ple_layer_ids: list[int] = field(default_factory=list)  # 1-based
    ple_embed_dim: int = 2560
    ple_conv_kernel_size: int = 4
    ngram_size: int = 3
    heads_per_ngram: int = 8
    ngram_vocab_size_base: int = 20_000_000
    make_ngram_vocab_size_divisible_by: int = 128
    split_ngram_parts: int = 128
    seed: int = 1234

    # ── Special tokens ───────────────────────────────────────────────────
    eos_token_id: int = 248044

    # ── Framework config (not model-specific) ────────────────────────────
    neuron_config: NeuronConfig | None = None
    # Tensor/expert-parallel group of this rank (see parallel.py); 1 rank by default.
    tp_size: int = 1
    tp_rank: int = 0
    # The serving max_model_len (from the vLLM config); None outside the runner.
    max_model_len: int | None = None
    # The group attention / Gated DeltaNet are split over (== TP group unless smaller).
    attn_tp_size: int = 1
    attn_tp_rank: int = 0
    # Routed experts stored as FP8 (trn2 e4m3, |q| <= 240) with per-output-channel fp32 scales
    # (requantized at load from the checkpoint's 128x128 block scales; see model_bf16 load_weights).
    # Set by from_configs for an FP8 checkpoint on the kernel path (neuron_config given).
    expert_fp8: bool = False
    # The checkpoint's FP8 weight block (quantization_config.weight_block_size), when it is FP8.
    fp8_block_size: tuple[int, int] | None = None

    def __post_init__(self):
        if not self.layer_types:
            raise ValueError("layer_types is required")
        self.layer_types = [
            FULL_ATTENTION if t in _FULL_ATTENTION_ALIASES else t for t in self.layer_types
        ]
        unknown = set(self.layer_types) - {LINEAR_ATTENTION, FULL_ATTENTION}
        if unknown:
            raise ValueError(f"Unsupported Qwen4-Exp layer types: {sorted(unknown)}")
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"{len(self.layer_types)} layer_types for {self.num_hidden_layers} layers"
            )
        if self.output_gate_type not in ("sigmoid", "silu"):
            raise ValueError(f"Unsupported output_gate_type: {self.output_gate_type}")
        if self.hc_count <= 1:
            raise ValueError(f"Qwen4-Exp requires hc_count > 1, got {self.hc_count}")
        if self.ple_embed_dim % self.ngram_heads:
            raise ValueError(
                f"ple_embed_dim {self.ple_embed_dim} is not divisible by {self.ngram_heads} n-gram heads"
            )

    # ── Derived quantities ───────────────────────────────────────────────
    @property
    def tp(self):
        from .parallel import TPInfo

        return TPInfo(self.tp_size, self.tp_rank)

    @property
    def attn_tp(self):
        from .parallel import TPInfo

        return TPInfo(self.attn_tp_size, self.attn_tp_rank, attn=True)

    @property
    def shared_expert_tp(self) -> bool:
        """Split the shared expert's intermediate over the whole TP group (summed by the
        routed all-reduce), so each rank reads 1/W of it per step."""
        return self.tp_size > 1 and self.shared_expert_intermediate_size % self.tp_size == 0

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def hc_hidden_size(self) -> int:
        return self.hc_count * self.hidden_size

    @property
    def ngram_heads(self) -> int:
        return (self.ngram_size - 1) * self.heads_per_ngram

    @property
    def ple_layer_indices(self) -> list[int]:
        """0-based indices of the decoder layers that own a PLE module."""
        return sorted({int(i) - 1 for i in self.ple_layer_ids})

    @property
    def dense_attention_exact_len(self) -> int:
        """Longest context for which dense causal attention equals QSA exactly."""
        if self.indexer_budget is None:
            return self.max_position_embeddings
        return self.indexer_budget + self.indexer_compress_ratio - 1

    @classmethod
    def from_configs(cls, hf_config: Any, neuron_config: NeuronConfig | None):
        """Create config from a HuggingFace config (composite or text-only) + NeuronConfig."""
        d = _text_config_dict(hf_config)
        rope = d.get("rope_parameters") or d.get("rope_scaling") or {}
        if rope.get("rope_type", "default") != "default":
            raise NotImplementedError(f"rope_type {rope.get('rope_type')!r} is not supported yet")

        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs = {k: v for k, v in d.items() if k in field_names and v is not None}
        kwargs["rope_theta"] = rope.get("rope_theta", d.get("rope_theta", cls.rope_theta))
        kwargs["partial_rotary_factor"] = rope.get(
            "partial_rotary_factor", d.get("partial_rotary_factor", cls.partial_rotary_factor)
        )
        eos = d.get("eos_token_id", cls.eos_token_id)
        kwargs["eos_token_id"] = int(eos[0] if isinstance(eos, (list, tuple)) else eos)
        dtype = d.get("torch_dtype") or d.get("dtype")
        if isinstance(dtype, str):
            kwargs["torch_dtype"] = getattr(torch, dtype)
        elif isinstance(dtype, torch.dtype):
            kwargs["torch_dtype"] = dtype
        if kwargs.get("ple_embed_dim") is None:
            kwargs["ple_embed_dim"] = kwargs.get("hidden_size", cls.hidden_size)
        kwargs["neuron_config"] = neuron_config
        fp8_block = fp8_checkpoint_block_size(hf_config)
        if fp8_block is not None:
            kwargs["fp8_block_size"] = fp8_block
        if neuron_config is not None and fp8_block is not None:
            kwargs["expert_fp8"] = True
        if neuron_config is not None:
            from .parallel import TPInfo

            tp = TPInfo.current(enabled=True)
            kwargs["tp_size"], kwargs["tp_rank"] = tp.size, tp.rank
            at = TPInfo.current(enabled=True, attn=True)
            kwargs["attn_tp_size"], kwargs["attn_tp_rank"] = at.size, at.rank
            try:
                from vllm.config import get_current_vllm_config

                kwargs["max_model_len"] = get_current_vllm_config().model_config.max_model_len
            except Exception:  # no vLLM config in scope
                pass
        return cls(**kwargs)


def fp8_checkpoint_block_size(hf_config: Any) -> tuple[int, int] | None:
    """``weight_block_size`` of a fine-grained FP8 checkpoint (``quantization_config`` with
    ``quant_method == "fp8"``, top level or in the text config), else ``None``."""
    if isinstance(hf_config, (str, bytes)):
        with open(hf_config) as f:
            hf_config = json.load(f)
    cands = []
    for c in (hf_config, getattr(hf_config, "text_config", None)
              if not isinstance(hf_config, dict) else hf_config.get("text_config")):
        if c is None:
            continue
        q = c.get("quantization_config") if isinstance(c, dict) else getattr(c, "quantization_config", None)
        if q is not None and not isinstance(q, dict):
            q = q.to_dict() if hasattr(q, "to_dict") else vars(q)
        if q:
            cands.append(q)
    for q in cands:
        if q.get("quant_method") != "fp8":
            continue
        block = q.get("weight_block_size")
        if not block or len(block) != 2:
            raise NotImplementedError(f"FP8 checkpoint without 2-D weight_block_size: {q}")
        if q.get("activation_scheme", "dynamic") != "dynamic":
            raise NotImplementedError(f"FP8 activation_scheme {q.get('activation_scheme')!r} is not supported")
        return int(block[0]), int(block[1])
    return None
