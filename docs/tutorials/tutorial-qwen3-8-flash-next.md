# Tutorial: Deploy Qwen3.8-Flash-Next with vLLM Neuron (FP8, TP=32)

<!-- meta: description: End-to-end tutorial for deploying the Qwen3.8-Flash-Next hybrid
Mixture-of-Experts model with vLLM on Neuron from its FP8 checkpoint, as one TP=32
replica with expert parallelism on trn2.48xlarge, text only, with MTP speculative
decoding and async scheduling. Pointer to the consolidated Qwen3.8-Flash-Next port
README, which is the single source of truth for setup, serving, long context,
verification, and performance on Trn2. -->
<!-- meta: keywords: vLLM, Neuron, Qwen3.8, Qwen3.8-Flash-Next, Qwen3.8-Flash-Next-FP8,
qwen4_exp, MoE, expert parallelism, FP8, gated DeltaNet, hybrid, MTP, speculative
decoding, long context, tutorial, LLM serving, Trn2, Trainium -->
<!-- meta: date_updated: 2026-11-09 -->
<!-- Content type: procedural-tutorial -->

> **The full end-to-end walkthrough of [`Qwen/Qwen3.8-Flash-Next-FP8`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8) lives in the port's consolidated README, which is the
> single source of truth:**
>
> **➜ [`Qwen3.8-Flash-Next/README.md`](https://github.com/htokoyo/vllm-neuron/tree/add-qwen38-flash-next/Qwen3.8-Flash-Next/README.md)**

See also the [model recipe / card](../model-recipes/qwen3-8-flash-next.md).
