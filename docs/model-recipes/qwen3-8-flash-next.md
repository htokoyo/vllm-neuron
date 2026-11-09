# Qwen3.8-Flash-Next (Hybrid MoE, FP8) Model Recipe

<!-- meta: description: Model recipe for deploying Qwen3.8-Flash-Next with vLLM on
Neuron: a hybrid gated-DeltaNet / sparse-attention Mixture-of-Experts model served from
its FP8 checkpoint as one TP=32 replica on trn2.48xlarge, text only, with MTP
speculative decoding and an opt-in 81,920-token configuration. Pointer to the
consolidated Qwen3.8-Flash-Next port README, which is the single source of truth for
architecture, the FP8 path, configuration, long context, verification, and performance
on Trn2. -->
<!-- meta: keywords: vLLM, Neuron, Qwen3.8, Qwen3.8-Flash-Next, Qwen3.8-Flash-Next-FP8,
qwen4_exp, MoE, expert parallelism, FP8, gated DeltaNet, linear attention, hybrid,
sparse attention, MTP, speculative decoding, async scheduling, long context, model
recipe, model card, LLM serving, Trn2, Trainium -->
<!-- meta: date_updated: 2026-11-09 -->
<!-- Content type: model-card -->

> **The full model recipe for [`Qwen/Qwen3.8-Flash-Next-FP8`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8) lives in the port's consolidated README, which is the
> single source of truth:**
>
> **➜ [`Qwen3.8-Flash-Next/README.md`](https://github.com/htokoyo/vllm-neuron/tree/add-qwen38-flash-next/Qwen3.8-Flash-Next/README.md)**

See also the [deployment tutorial](../tutorials/tutorial-qwen3-8-flash-next.md).
