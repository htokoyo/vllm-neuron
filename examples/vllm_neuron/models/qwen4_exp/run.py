# SPDX-License-Identifier: Apache-2.0
"""Offline inference example for Qwen3.8-Flash-Next (qwen4_exp), FP8 checkpoint.

Hybrid decoder: 48 layers in a 3:1 pattern of gated DeltaNet (linear attention) and gated full
attention with a block indexer, a 512-expert MoE in every layer, a 4-stream hyper-connection
residual, an n-gram per-layer embedding and one MTP layer. Text path only.

    python examples/vllm_neuron/models/qwen4_exp/run.py \
        --model-checkpoint /path/to/models/Qwen3.8-Flash-Next-FP8

Builds the primary configuration of the bundle README (Qwen3.8-Flash-Next/README.md): one TP=32
replica with expert parallelism (ep_degree 4, attention in groups of 8 ranks), max_model_len 2,048,
MTP k=2, async scheduling. The model needs no environment knobs; the runtime / compile settings of the
README's Step 1 are applied via os.environ.setdefault() below, so an explicit export still wins. Set the
compile-cache variables (VLLM_CACHE_ROOT, NKI_COMPILE_CACHE_URL) yourself; the first run compiles every
graph.
"""
import argparse
import os

# Runtime / compile settings of the README's Step 1 (set before importing vllm).
_ENV = {
    # KV-cache share of free HBM (the measured value; a larger share leaves too little for a batch).
    "VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION": "0.09",
    # Host path: keep the step readers spinning for the length of a decode step.
    "VLLM_NEURON_SHM_READER_BUSY_LOOP_S": "0.02",
    # A cold TP=32 compile: bounded compile pool, compile lock and rank barrier longer than the build.
    "NEURON_LIBTORCH_PARALLEL_COMPILE_WORKERS": "20",
    "TORCH_NEURONX_NEFF_CACHE_LOCK_TIMEOUT": "14400",
    "VLLM_NEURON_BARRIER_TIMEOUT": "14400",
}
for _k, _v in _ENV.items():
    os.environ.setdefault(_k, _v)

from vllm import LLM, SamplingParams  # noqa: E402

PREFILL_BUCKETS = [128, 256, 512, 1024, 1536, 1792, 2048]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-checkpoint", type=str, required=True,
                        help="Path to the Qwen3.8-Flash-Next-FP8 checkpoint.")
    parser.add_argument("--max-num-seqs", type=int, default=16,
                        help="Max concurrent sequences (16 is the measured value with MTP).")
    parser.add_argument("--num-speculative-tokens", type=int, default=2,
                        help="MTP draft tokens (2 is the measured value; 0 disables MTP).")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--prompts", type=str, default=None,
                        help="'@@'-separated prompt list overriding the default ones.")
    args = parser.parse_args()

    kwargs = {}
    if args.num_speculative_tokens > 0:
        kwargs["speculative_config"] = {"method": "mtp",
                                        "num_speculative_tokens": args.num_speculative_tokens}

    llm = LLM(
        model=args.model_checkpoint,
        tensor_parallel_size=32,
        # Attention / DeltaNet run in four groups of 8 ranks; experts are split over all 32.
        enable_expert_parallel=True,
        max_model_len=2048,
        max_num_batched_tokens=PREFILL_BUCKETS[-1],
        max_num_seqs=args.max_num_seqs,
        block_size=32,
        # Per-request recurrent state: the platform turns prefix caching off anyway.
        enable_prefix_caching=False,
        async_scheduling=True,
        # Text path only: the checkpoint's vision tower is not built.
        limit_mm_per_prompt={"image": 0, "video": 0},
        additional_config={
            "neuron_config": {
                "ep_degree": 4,
                "decode_context_length_buckets": [1024],
                "num_batched_tokens_buckets": PREFILL_BUCKETS,
            }
        },
        **kwargs,
    )

    if args.prompts:
        prompts = args.prompts.split("@@")
    else:
        prompts = [
            "The capital of France is",
            "In a hybrid model, linear attention layers carry a recurrent state per request, "
            "while full attention layers read a paged key-value cache. This means that",
            "def fibonacci(n):",
        ]
    sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)

    outputs = llm.generate(prompts, sampling_params)
    for o in outputs:
        print(repr(o.prompt), "->", repr(o.outputs[0].text))


if __name__ == "__main__":
    main()
