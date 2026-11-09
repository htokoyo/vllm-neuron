# qwen4_exp (Qwen3.8-Flash-Next, hybrid linear / sparse attention MoE, text path)

vllm-neuron model implementation for `Qwen/Qwen3.8-Flash-Next` and its FP8 release (`model_type:
qwen4_exp`, architecture `Qwen4ExpForConditionalGeneration`, served on its text path; `Qwen4ExpMTP` is the
MTP draft). A 48-layer hybrid decoder — gated DeltaNet ×3 → gated full attention with a block indexer ×1,
repeated 12 times — with a 512-expert MoE in every layer, a 4-stream hyper-connection residual, an n-gram
per-layer embedding and one MTP layer.

> **The single source of truth for this port** — architecture, setup, serving, verification, and measured
> performance — is the bundle README:
> [`Qwen3.8-Flash-Next/README.md`](../../../Qwen3.8-Flash-Next/README.md). This file documents only the
> module structure of the package.

## Module Structure

| File | Purpose |
|------|---------|
| `__init__.py` | Package exports: `Qwen4ExpForCausalLM`, `Qwen4ExpForConditionalGeneration`, `Qwen4ExpMTP`. |
| `config.py` | HF → Neuron config translation (layer pattern, DeltaNet / attention / indexer / MoE / hyper-connection / n-gram dimensions); FP8 checkpoint detection (FP8 experts and n-gram table follow the checkpoint). |
| `factory.py` | Model builder wiring for the registry entries. |
| `prefill_ctx.py` | Prefill context buckets, host side only (runner / worker hooks installed from `factory.py`; no traced code). On when `max_model_len > 32,768` (data / context parallel size 1): a prefill step's block table (segmented prefill steps included) is trimmed to the smallest of 2,048 / 8,192 / 32,768 that holds the step, the MTP draft headroom and the padded query, and warm-up also builds each prefill bucket at those widths. Decode, MTP verify and draft steps keep their tables. Off (nothing installed) at `max_model_len <= 32,768`. |
| `layers.py` | The model math, single rank, in PyTorch (also the CPU path): gated DeltaNet, gated full attention with the block indexer and its paged dense / sparse forms, MoE, hyper-connections, n-gram embedding. Also installs the kernel paths (each with its PyTorch fallback for CPU) and selects the sparse attention forms from `max_model_len`; the model needs no environment knobs. |
| `model_bf16.py` | The decoder: per-request state pools (DeltaNet recurrent / convolution, n-gram convolution) behind `vllm_neuron/vllm/worker/request_state.py` — each pool has one extra, never-read sink row that the padding rows of a batched step write, so the dummy slot they read is never written — on-device n-gram ids, and the loader (FP8 128×128 block scales re-quantized to per-output-channel scales for the routed experts; FP8 n-gram table). |
| `mtp.py` | The MTP draft model (`Qwen4ExpMTP`): the checkpoint's one MTP layer on the hyper-connection streams. |
| `ngram.py` | Host-side n-gram ids (the reference). |
| `ngram_device.py` | On-device n-gram ids (what the runner uses), int32-only and bit-exact with `ngram.py`, which lets async scheduling stay on. |
| `parallel.py` | TP / EP layout over one TP group: attention and DeltaNet in attention groups of up to 8 ranks (`ep_degree = TP / group`), experts and vocabulary rows over all ranks. |

The model's NKI kernels are in [`vllm_neuron/_kernels/qwen4_exp/`](../../_kernels/qwen4_exp/), and the
framework pieces it relies on (the MTP proposer, the per-request state slots) are carried on this branch
in-tree; the bundle README lists every path the port touches.

## Running

Offline example:
[`examples/vllm_neuron/models/qwen4_exp/run.py`](../../../examples/vllm_neuron/models/qwen4_exp/run.py)
(applies the runtime settings via `os.environ.setdefault`). The serving recipes, the required settings and
the long-context configuration are in the [bundle README](../../../Qwen3.8-Flash-Next/README.md).
