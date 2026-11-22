# Contributed Model: Qwen3.8-Flash-Next (hybrid linear / sparse attention MoE, FP8)

vllm-neuron port of `Qwen/Qwen3.8-Flash-Next-FP8` (`model_type` `qwen4_exp`) for AWS Trainium2, on the
**Neuron 2.33.1 stack (vLLM 0.29 / plugin 0.29.0.1.2.0)**, serving the **FP8 checkpoint** on a `trn2.48xlarge`
in three layouts: **TP=32 with MTP** (the default, for latency) and **four TP=16 or eight TP=8 replicas**
(for throughput).

It adds the [`qwen4_exp`](../vllm_neuron/model/qwen4_exp/README.md) model package, its NKI kernels, an MTP
speculative-decoding proposer, and per-request state slots for models that own their recurrent state. The
port is **text only**: the checkpoint's vision tower is not built.

## Change History

Newest first.

| Date | Change |
|---|---|
| 2026-11-18 | Two throughput configurations next to the TP=32 default: four TP=16 replicas or eight TP=8 replicas on one `trn2.48xlarge`, without MTP and with the n-gram table in host memory ([Throughput configurations](#throughput-configurations-tp16-and-tp8)). Faster decode and prefill, always on and chosen by the tensor-parallel degree ([Built-in optimisations](#built-in-optimisations)). TP=32 at B=1 against the 2026-11-12 revision ([measured](#primary-configuration-tp32-max_model_len-2048-max_num_seqs-16)): with a 1,635-token prompt, TPOT 12.91 → 9.78 ms (−24.2 %) and TTFT 0.379 → 0.275 s (−27.4 %). Batches of 1,635-token prompts: up to B=2 before (B=4 ran out of HBM), now up to B=16 (`max_num_seqs`). Long context, TP=32 at B=1 ([measured](#long-context-configuration-max_model_len-81920-max_num_batched_tokens-1024-max_num_seqs-4)): with a 29,971-token prompt, TPOT 33.35 → 30.15 ms (−9.6 %). The serve commands now use a KV-cache share of 0.07 and `NEURON_RT_IO_RING_CACHE_SIZE=4`. |
| 2026-11-12 | Prefix caching (vLLM align mode) is supported and on by default in both configurations, the single-shot primary configuration included (`kv_segment_size_buckets` is set automatically there); it was turned off before. Each state pool gets snapshot rows (`neuron_config.prefix_cache_state_snapshots`, default 16) that keep the state on a 128-token grid, a first request shorter than four prefill chunks is chunked exactly as with prefix caching off, and decode-time states are opt-in (`neuron_config.prefix_cache_decode_snapshots`). Measured on device: a repeated prompt's TTFT drops from 0.39 s to 0.12 s (1,635 tokens) and from 11.97 s to 0.86 s (7.8K tokens, long context), with time per output token and first-request TTFT unchanged and GSM8K within 0.5 pt. See [Prefix caching](#prefix-caching-align-mode). |
| 2026-11-09 | Initial contribution, on vllm-neuron 0.29 / Neuron 2.33. Adds the `qwen4_exp` package — a 48-layer hybrid decoder (gated DeltaNet linear attention ×3 → sparse full attention ×1, repeated 12 times), a 512-expert MoE in every layer, a 4-stream hyper-connection residual, an n-gram per-layer embedding and one MTP layer — with FP8 routed experts resident on HBM, MTP speculative decoding, async scheduling with on-device n-gram ids, and a long-context configuration at `max_model_len` 81,920 (sparse attention and prefill context buckets, selected from `max_model_len`). No environment knobs are needed. See [Verification scope](#verification-scope). |

## Introduction

Qwen3.8-Flash-Next is a large hybrid-attention Mixture-of-Experts model. Three of every four layers are
gated DeltaNet (linear attention with a fixed-size recurrent state); the fourth is a gated full-attention
layer with a block indexer that, past 2,051 tokens of context, restricts each query to the 512
highest-scoring 4-token blocks. Every layer has a 512-expert MoE (top-10) plus a shared expert, and the
residual stream is four streams wide (hyper-connections).

This port serves the architecture `Qwen4ExpForConditionalGeneration` on its **text path**. Image and video
inputs are not supported; serve with `limit_mm_per_prompt` set to 0 for both.

**Compatible checkpoints:**

| Model | HuggingFace | Hardware | Quantization | Verified on this branch |
|-------|-------------|----------|--------------|-------------------------|
| Qwen3.8-Flash-Next-FP8 | [Qwen/Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8) | Trn2 | FP8 routed experts and n-gram table (128×128 block scales), BF16 elsewhere | ✅ the configurations measured in this document |

There are four serving configurations — TP=32, TP=16, TP=8 and long context — and each is compiled separately:

- **Primary (low latency), TP=32 with MTP k=2:** `max_model_len` 2,048. The default; one or two replicas per
  `trn2.48xlarge`. See [Online serving](#online-serving--fp8-primary-configuration-max_model_len-2048).
- **Throughput, TP=16 × 4 or TP=8 × 8:** `max_model_len` 2,048, no MTP, the n-gram table in host memory; each
  replica is its own `vllm serve` on its own cores. See [Throughput configurations](#throughput-configurations-tp16-and-tp8).
- **Long context, TP=32:** `max_model_len` 81,920; the model selects its sparse attention forms from
  `max_model_len` and, above 32,768, gives each prefill chunk only the context bucket that holds it (see
  [Long context](#long-context-max_model_len-81920)). Time to first token is prefill-bound and grows with the
  prompt.

Which one to use: the TP=32 configuration has the lowest time per output token and TTFT for a single request;
the TP=16 and TP=8 layouts give more total output per box when many requests run at once, at a higher time per
output token per request ([Serving benchmark](#serving-benchmark-vllm-bench-serve)).

## Verification scope

Everything below was measured on real hardware: a single **trn2.48xlarge** (64 logical NeuronCores =
16 Trainium2 devices, LNC=2), the **FP8 checkpoint**, on the **Neuron 2.33.1 stack (vLLM 0.29.0, vllm-neuron
0.29.0.1.2.0)**, with a **warm compile cache** and greedy decoding, in three layouts: **TP=32** with expert
parallelism `ep_degree` 4 and MTP k=2; **TP=16** with `ep_degree` 2, no MTP; **TP=8**, no MTP. **Every number in
this document is this port's own measurement.**

| Configuration | Status |
|---|---|
| TP=32, `max_model_len` 2,048, B=1, MTP k=2, async scheduling | ✅ 9.21 ms per output token, TTFT 0.111 s (30-token prompt); 9.78 ms, TTFT 0.275 s (1,635-token prompt) |
| TP=32, B = 2 / 4 / 8 / 16 (`max_num_seqs` 16), 30-token and 1,635-token prompts | ✅ see [Measured performance](#measured-performance) |
| TP=16, B = 1 / 8 / 16 / 32 (`max_num_seqs` 32), 30-token and 1,635-token prompts | ✅ see [Throughput configurations, per replica](#throughput-configurations-per-replica-tp16-and-tp8) |
| TP=8, B = 1 / 8 / 16 (`max_num_seqs` 16), 30-token and 1,635-token prompts | ✅ see [Throughput configurations, per replica](#throughput-configurations-per-replica-tp16-and-tp8) |
| Serving benchmark (`vllm bench serve`, random 1,024 in / 512 out), TP=8 × 8 / TP=16 × 4 / TP=32 × 2 | ✅ every request of every client successful; see [Serving benchmark](#serving-benchmark-vllm-bench-serve) |
| Accuracy, GSM8K full test set (1,319 questions), TP=32 / TP=16 / TP=8 | ✅ see [GSM8K-CoT](#gsm8k-cot-thinking-disabled) |
| Primary, batches with padding rows (3 / 5 / 7 concurrent requests) | ✅ previous revision: no degenerate output; 13 of the 15 outputs equal the same prompt served alone, and the other two (one prompt, in the 5- and the 7-request batch) give a different, well-formed answer |
| Long context, TP=32, `max_model_len` 81,920, B=1, MTP k=2 | ✅ 12.3–17.0 ms per output token up to 7,823 prompt tokens, 30.2 ms at 29,971; TTFT 0.93 s at 1,635 tokens, 50.8 s at 29,971 — see [Measured performance](#measured-performance) |
| Long context, retrieval of a fact placed in a 14K-token prompt | ✅ the fact is retrieved (the model reasons and then answers with it) |
| MTP k=3 | ✅ runs at B=1 on an earlier revision of this port; no k=3 figure is quoted here |
| `max_num_seqs` with MTP k=2 (TP=32) | ✅ verified up to 16 (B=1–16 above); 32 not tried; 64 runs out of HBM in the decode warm-up (previous revision). How many requests fit at once depends on the prompt length — see [Known limits](#known-limits) |

Which revision each figure is from:

- **TP=32, TP=16 and TP=8 decode and TTFT figures, the long-context figures and the retrieval check, GSM8K and
  the serving benchmarks:** this tree.
- **The prefix-caching table** ([Prefix caching](#prefix-caching-align-mode)), **the padding-row check** and
  **`max_num_seqs` 64:** the previous revision of this port (Neuron 2.33, KV-cache share 0.09 in the primary
  configuration), at TP=32. **MTP k=3:** an earlier revision.

**Not verified:**

- **The generic logit-comparison scripts** (`examples/vllm_neuron/accuracy/`) **were not run** for this
  model. Correctness evidence is GSM8K and the batched-output checks on device; see
  [Accuracy Evaluation](#accuracy-evaluation) for how to run the logit comparison yourself.
- `max_model_len` 4,096 / 8,192 (or any value other than 2,048 and 81,920); long context at TP=16 or TP=8.
- MTP at TP=16 or TP=8; MTP k=3 with more than one concurrent request (B > 1).
- Tensor-parallel degrees other than 32 / 16 / 8; vLLM's own data parallelism (`--data-parallel-size`; the
  replicas here are independent servers); multi-node.
- A sustained 30-minute load test on this revision.
- Vision (image / video) input — not implemented.
- The BF16 checkpoint at the configurations above.
- On-device top-k / top-p sampling (every run here was greedy).
- Serving with real text: the serving benchmark uses random tokens and `--ignore-eos`, and MTP accepts far fewer
  drafts on random tokens than on text.

## Model Architecture

| Property | Value |
|---|---|
| Architecture | `Qwen4ExpForConditionalGeneration` (`model_type` `qwen4_exp`); text path only |
| Layers | 48 = 12 × (3 gated DeltaNet → 1 gated full attention) |
| Hidden size | 2,560, carried as a 4-stream hyper-connection residual (4 × 2,560) |
| Full attention | 24 query / 2 KV heads, head dim 256, sigmoid output gate, QK-norm, partial RoPE (64 of 256 dims) |
| Sparse block indexer | 4 query heads + 1 key head, dim 128; keys pooled every 4 tokens; top 512 blocks (budget 2,048 tokens) plus the open block |
| Gated DeltaNet | 16 QK heads / 48 V heads, dim 128, conv kernel 4, sigmoid output gate |
| MoE | 512 routed experts, top-10, intermediate 640, plus a shared expert with a sigmoid gate — in every layer |
| n-gram embedding | Second layer only: hashed bigram + trigram tables (FP8 in the FP8 checkpoint) and a short dilated convolution |
| MTP | 1 layer, on the hyper-connection streams |
| Vocabulary | 248,320 |
| KV cache | The 12 full-attention layers plus their indexer keys; the DeltaNet and n-gram states are fixed-size per request |

### Dense up to 2,051 tokens, sparse beyond

A query at position *p* keeps the 512 top-scoring complete 4-token blocks plus the open block it sits in, so
with at most 2,048 + 3 tokens of context the selection is the whole causal window and the layer is exactly
dense causal attention. With `max_model_len` ≤ 2,051 (the primary and throughput configurations) the model
therefore runs dense attention. Above 2,051 the block selection is live, and the model selects the sparse forms
that fit HBM and compile at 81,920 by itself: the block-level mask, sparse attention over the selected blocks for
steps of up to 4 tokens, a tiled sparse prefill, a row-chunked block top-k and partial RoPE without a concatenate.

### FP8 path

The FP8 checkpoint quantizes the routed experts (all 48 layers and the MTP layer) and the n-gram embedding
table, with 128×128 block scales; `modules_to_not_convert` covers everything else. At load time the experts'
block scales are re-quantized to **per-output-channel scales**, the format the decode and prefill MoE kernels
take, and the experts stay FP8 on HBM. At TP=32 the n-gram table also stays FP8 on HBM (1.60 instead of 3.20 GB
per rank) and is decoded after the row gather, bit-identically to the dequantized table; at TP=16 and TP=8 the
table is in host memory and the device decodes the gathered FP8 bytes with the same arithmetic
([Host-resident n-gram table](#host-resident-n-gram-table)). Both follow the checkpoint: with a BF16 checkpoint
the experts and the table are BF16.

### Parallel layout: attention groups of 8

The attention and DeltaNet head counts allow at most 8 ranks, so attention and DeltaNet run in **attention
groups of 8 ranks**, while the routed experts and the vocabulary rows are split over all ranks of the replica:

| TP | Expert parallelism | Attention groups | Routed experts per rank |
|---|---|---|---|
| 32 | `--enable-expert-parallel`, `ep_degree` 4 | 4 groups of 8 that compute the same attention and DeltaNet | 16 |
| 16 | `--enable-expert-parallel`, `ep_degree` 2 | 2 groups of 8 | 32 |
| 8 | not set | the whole replica | 64 |

vLLM's rule that the attention head count divide the TP size is relaxed for this architecture only, because the
model checks its own split.

### Per-request state, prefix caching and async scheduling

The DeltaNet recurrent and convolution states and the n-gram convolution state are not paged KV: the model
owns fixed pools of slots and the runner assigns one slot per live request
(`vllm_neuron/vllm/worker/request_state.py`). Three consequences:

- **Prefix caching** needs the state at the end of the cached prefix. With prefix caching on (the default) the
  pools get extra snapshot rows that keep it at selected block boundaries
  ([Prefix caching](#prefix-caching-align-mode)). The performance figures of this revision were measured with it
  on (the long-context figures with it off); no measured request hits the cache (see [Measured performance](#measured-performance)).
- **Padding rows** of a batched step read the pools' dummy slot, which is never written: their state writes go
  to one extra, never-read sink row of each pool (built in; nothing to set).
- **The n-gram ids are computed inside the graph** from a per-request pool of the last tokens, so no host
  input depends on the last sampled token and async scheduling works. With the table in host memory the host
  joins each decode step's actual input ids right before the model call ([Host-resident n-gram table](#host-resident-n-gram-table)).

#### Prefix caching (align mode)

The DeltaNet and convolution states cannot be rebuilt from cached KV, so a cached prefix is usable only up to a
point whose state was kept. Prefix caching runs in vLLM's align mode (`--mamba-cache-mode align`) and is **on by
default**. `--no-enable-prefix-caching` turns it off, when outputs must not depend on earlier requests or to save
the snapshot rows' HBM. KV connectors are not supported with it.

- **Snapshot rows.** Each state pool gets `S` extra rows for kept states (about 14 MiB of HBM per row per rank at
  TP=32): `"neuron_config": {"prefix_cache_state_snapshots": S}`, default 16.
- **Which request hits.** States are kept on a 128-token grid (the DeltaNet prefill kernel's chunk). With
  single-shot prefill (`max_model_len` 2,048: the primary and throughput configurations) the second request on a prefix keeps its state and the third
  and later ones hit; with 1,024-token chunks (long context) the second request hits. A first request shorter
  than four prefill chunks is prefilled exactly as with prefix caching off. For single-shot prefill the platform
  sets `kv_segment_size_buckets` to one size (2,048 there), which a hit needs.
- **Decode-time states (opt-in).** `"prefix_cache_decode_snapshots": true` also keeps states reached during
  decode, so a multi-turn follow-up that repeats the answer hits beyond the previous prompt. With MTP it adds a
  state write per verify row on every step; checked on the CPU engine only.
- **Outputs.** Repeating a request gives the same greedy tokens. A hit changes where the prefill is split, which
  on device alone can change greedy tokens (bf16 rounding); the GSM8K accuracy is unchanged.
- **Measured on device** (previous revision: TP=32, FP8, MTP k=2, async scheduling, greedy, warm compile cache):

  | | Prefix caching off | Prefix caching on |
  |---|---|---|
  | Primary, TTFT of a repeated 1,635-token prompt | 0.39 s (every request cold) | 0.12 s (hit, 1,536 cached) |
  | Long context, TTFT of a repeated 7.8K-token prompt | 11.97 s (every request cold) | 0.86 s (hit, 7,680 cached) |
  | Primary, time per output token, B=1 / 8 / 16 | 12.21 / 30.47 / 51.03 ms | 12.22 / 30.24 / 51.13 ms |
  | Primary, GSM8K full test set, strict / flexible | 0.934 / 0.941 | 0.932 / 0.936 |

  A first request costs the same as with prefix caching off (TTFT 0.392–0.399 s at 1,635 tokens, 0.391–0.397 s
  off, same-day runs).
- Turning prefix caching on or off changes the graphs, and the first serve after a change recompiles every
  graph once. Not measured: `S` above 16, decode-time states on device, random sampling with prefix caching,
  more than 16 concurrent requests with MTP.

### Host-resident n-gram table

Below TP=32 the model keeps the n-gram embedding table off the device; nothing needs to be set. Every worker
process memory-maps the checkpoint's n-gram table files read-only (about 51 GB in the FP8 checkpoint; one page
cache shared by all ranks and all replicas on the box), and every step the host gathers the rows of the step's
n-gram ids and passes them to the graph as one more input. For the FP8 checkpoint they travel as FP8 bytes and the
device decodes them with the same arithmetic as at TP=32, so the embedding output is the same bit for bit. Prefill
steps take the n-gram windows from the worker's token table; decode and MTP-verify steps join the step's actual
input ids right before the model call, so async scheduling stays on.

- **Why:** row-sharded on the device at TP=8, the table would take 6.4 GB of each rank's HBM next to the weights.
  At TP=32 it stays on the device.
- Large gathers (prefill) run on 8 host threads, and after load the table is read into the page cache in the
  background.
- Keep the checkpoint on a local volume, with host memory to spare for the page cache of the table.

## Required knobs

Every measured configuration sets these; the right column says what happens without each.

| Setting | What happens without it |
|---|---|
| `--async-scheduling` | Synchronous scheduling. Async scheduling is the largest single B=1 lever: it hides the host work of a step behind the device step (about −4.7 ms per output token measured during development). |
| `--enable-expert-parallel` + `"ep_degree"`: 4 at TP=32, 2 at TP=16 (neither at TP=8) | Required above TP=8 (see [Parallel layout](#parallel-layout-attention-groups-of-8)). |
| `--limit-mm-per-prompt '{"image": 0, "video": 0}'` | The text path only; the vision tower is never built. |
| `--block-size 32` | The value every measurement used. |
| `NEURON_RT_IO_RING_CACHE_SIZE=4` | A runtime setting; the compiled graphs do not depend on it. The serve commands at `max_model_len` 2,048 (TP=32, TP=16, TP=8) set it, as every serving benchmark, GSM8K run and decode figure at that length did; serving them without it was not measured on this revision. The long-context command and its figures do not use it. |
| `NEURON_LIBTORCH_PARALLEL_COMPILE_WORKERS=20` for one cold build (8 for long context) | Without an explicit count the compile pool is sized from the vCPU count; a fully cold TP=32 build then runs enough compiler processes at once to exceed the host RAM of a `trn2.48xlarge`. Each long-context graph needs more host memory to compile: there a pool of 12 already reached about 1.55 TB within 20 minutes, while 8 completed. The builds of this revision ran three at once, each with 4 workers and `NEURON_LIBTORCH_PARALLEL_TRACE_WORKERS=2`. |
| `TORCH_NEURONX_NEFF_CACHE_LOCK_TIMEOUT=14400`, `VLLM_NEURON_BARRIER_TIMEOUT=14400` | The NEFF compile lock is broken after 20 minutes by default, after which every rank compiles the same graph; the rank barrier default (1 hour) is shorter than a cold compile of this model. |
| `VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.07` | The KV cache takes a larger share of free HBM and leaves too little for the batch's activations (see [Known limits](#known-limits)). Every measurement of this revision used 0.07; the previous revision's primary-configuration figures used 0.09. |

## Feature status

| Category | Feature | Status |
|---|---|---|
| **Inputs** | Text | ✅ |
| | Image / video | ❌ Not implemented (the vision tower is not built) |
| **Checkpoint** | FP8 (`Qwen/Qwen3.8-Flash-Next-FP8`) | ✅ Every measurement in this document |
| | BF16 (`Qwen/Qwen3.8-Flash-Next`) | ⚠️ Loader path present; not measured on this branch |
| **Quantization** | FP8 routed experts on HBM (per-channel scales, FP8 checkpoint) | ✅ |
| | FP8 n-gram table, on HBM or in host memory (FP8 checkpoint) | ✅ HBM at TP=32, host memory at TP=16 / TP=8 |
| | FP8 KV cache (`kv_cache_dtype` fp8) | ❌ Not implemented — the FP8 checkpoint quantizes weights only; the KV cache is BF16 |
| **Parallelism** | Tensor parallelism, TP=32 (`ep_degree` 4) | ✅ Default; MTP k=2 |
| | TP=16 (`ep_degree` 2) | ✅ Throughput configuration, no MTP, `max_num_seqs` 32 |
| | TP=8 | ✅ Throughput configuration, no MTP, `max_num_seqs` 16 |
| | Several replicas on one box (one `vllm serve` per replica, cores chosen with `NEURON_VISIBLE_DEVICES`) | ✅ TP=32 × 2, TP=16 × 4, TP=8 × 8 |
| | Other TP degrees, pipeline / context parallelism, vLLM data parallelism (`--data-parallel-size`) | ❌ Not verified |
| **Attention** | Gated DeltaNet (linear attention) | ✅ |
| | Gated full attention, dense up to 2,051 tokens | ✅ Primary and throughput configurations |
| | Sparse block attention beyond 2,051 tokens (selected from `max_model_len`) | ✅ Long-context configuration (TP=32) |
| **Speculative decoding** | MTP k=2 | ✅ TP=32: primary and long-context configurations |
| | MTP at TP=16 / TP=8 | ⚠️ Not measured |
| | MTP k=3 | ✅ Runs at B=1 (earlier revision of this port); B > 1 not verified |
| **Scheduling** | Async scheduling (n-gram ids computed on device, or joined on the host) | ✅ |
| | Prefix caching (align mode, states kept on a 128-token grid) | ✅ On by default in every configuration. On device: time per output token and first-request TTFT unchanged, GSM8K within 0.5 pt (previous revision); with single-shot prefill the third request of a prefix hits, with 1,024-token chunks the second ([Prefix caching](#prefix-caching-align-mode)) |
| **Context** | `max_model_len` 2,048 | ✅ Primary and throughput configurations |
| | `max_model_len` 81,920 | ✅ Long-context configuration, TP=32 (1,024-token prefill chunks) |
| | Prefill context buckets (automatic above `max_model_len` 32,768) | ✅ Long-context configuration |
| | Other `max_model_len` values | ⚠️ Not measured |
| **Batching** | `max_num_seqs` 16 with MTP k=2 (TP=32) | ✅ |
| | `max_num_seqs` 32 without MTP (TP=16) | ✅ |
| | Padding rows in a batched step (dummy slot never written) | ✅ 3 / 5 / 7 concurrent requests checked (previous revision) |
| | `max_num_seqs` 32 with MTP | ⚠️ Not tried |
| | `max_num_seqs` 64 with MTP | ❌ Out of HBM (previous revision) |
| **Sampling** | Greedy (temperature 0) | ✅ Every measurement here (latency, batches, GSM8K, long context) |
| | Random sampling (temperature > 0, top-k, top-p) | ⚠️ Not verified — never run on this branch |

Legend: **✅** verified on device · **⚠️** present but not verified here · **❌** not supported, not
implemented, or does not fit.

### Built-in optimisations

These run by themselves; there is nothing to set. Where a choice depends on the tensor-parallel degree, the
model makes it once, when it is built. The NKI kernels run on device (or under the NKI simulator); in CPU mode
each falls back to its PyTorch path.

| Path | What it does |
|---|---|
| Prefill MoE | Skips the compute of empty expert blocks (dynamic block loop in a copy of the nki-library MoE CTE kernel) |
| Decode MoE, up to 128 / 64 / 32 rows at TP=32 / 16 / 8 (FP8 checkpoint) | Router, top-k, routed experts (FP8 or BF16, from the checkpoint) and the shared expert in one kernel, one all-reduce per step; above 32 rows at TP=32 the kernel runs row tiles of 32 and loads each routed expert once per step, at TP=16 it runs 16-row calls; TP=8 steps above 32 rows run as 32-row chunks |
| Decode MoE, FP8 experts | The fused kernel pipelines the routed experts in one loop body: two experts per iteration with both weight loads issued first, and every expert's scales loaded once per layer |
| Decode attention projections, ≤ 32 rows | q / k / v projection, q / k norm and partial RoPE, and o_proj in NKI kernels |
| Recurrent-state read | Each request's DeltaNet recurrent state is read from the 4-D pool `[slots, heads, k, v]` one row per DMA, by an NKI kernel whose DMAs the Sync engine issues, so the read does not wait behind the state-pool writes; a byte copy |
| Decode KV window | One NKI kernel gathers the decode bucket's KV blocks (up to 128 whole blocks per indirect DMA) instead of an XLA gather per token, and the current token is placed in its own column |
| MTP verify KV window | Steps of 2–4 tokens per row (the MTP verify step and the draft's first pass) read their KV window with the same block-gather kernel; the attention inputs are the same bit for bit |
| MTP draft argmax | Each rank takes the argmax over its vocabulary shard and the ranks gather one (value, index) pair per row instead of the full logits; the same token |
| DeltaNet decode, one token, up to 16 rows | The kernel runs the gate activation once for all heads and the RMSNorm reciprocal square root once per 8 heads, so the activation engine does not reload its tables at every head; decode steps of more than 16 rows take the per-head kernel |
| Decode / verify projections, ≤ 32 rows | Read `[in, out]` copies of the DeltaNet, hyper-connection, router, shared-expert, n-gram and LM-head weights |
| Hyper-connection weights below TP=32 | One copy of the hyper-connection transposes instead of two (−1.18 GiB per rank), where the weights need the room |
| Hyper-connection read, prefill | One NKI kernel instead of the XLA ops; for prefill steps of at least 512 tokens each rank of the attention group computes 1/8 of the tokens and the results are all-gathered |
| DeltaNet prefill | The chunk mask pack is a device buffer, built once on the host |
| MTP verify | The DeltaNet recurrence of the 1 + k tokens in one kernel call |
| Decode / verify steps of more than 32 rows | Row-wise parts run in chunks of 32 rows (with MTP, verify steps above 32 rows accepted no drafts on device otherwise) |
| n-gram ids | Computed inside the graph (async scheduling) |
| n-gram table | On the device at TP=32, in host memory below ([Host-resident n-gram table](#host-resident-n-gram-table)) |
| Prefill at `max_model_len` > 32,768 | Prefill context buckets: each prefill chunk attends only to the smallest of 2,048 / 8,192 / 32,768 that holds its positions instead of the whole `max_model_len` window (host side only; see [Long context](#long-context-max_model_len-81920)) |

### Known limits

| Limit | Detail |
|---|---|
| **TP=32 / 16 / 8 only** | No other degree was verified. |
| **MTP and long context at TP=32 only** | TP=16 and TP=8 were measured without MTP and at `max_model_len` 2,048 only. |
| **Host memory below TP=32** | The n-gram table is memory-mapped from the checkpoint files (about 51 GB, shared by all ranks and replicas on the box). |
| **Text only** | The checkpoint's vision weights are present but not served. |
| **`max_num_seqs` 64 with MTP** | Out of HBM in the B=64 decode warm-up (previous revision). The tested maximum with MTP k=2 is `max_num_seqs` 16. |
| **Concurrent requests depend on the prompt length** | At TP=32 with the KV-cache share 0.07, 1,635-token prompts ran up to B=16; on the previous revision, with 0.09, B=4 × 1,635-token prompts failed to allocate device memory (`Allocation Failure`, a runtime HBM allocation, not KV-cache exhaustion). Longer prompts in a batch were not measured. |
| **Prefill chunk at `max_model_len` 81,920** | Measured with 1,024-token chunks; 4,096-token chunks do not fit TP=32 HBM. |
| **Decode cost at long context** | Decode reads the whole decode bucket that holds the context, so its cost grows with the bucket width. |
| **TTFT at long context** | Prefill-bound; it grows with the prompt length. |

## Setup

### Native integration (nothing to apply)

The model package, its NKI kernels and every framework touch-point are committed on this branch. Check the
branch out and install as usual; there is no patch to apply.

- **Model package:** [`vllm_neuron/model/qwen4_exp/`](../vllm_neuron/model/qwen4_exp/README.md)
- **NKI kernels:** `vllm_neuron/_kernels/qwen4_exp/`
- **Shared files touched:** see [Paths this port touches](#paths-this-port-touches-across-the-repository).

**Prerequisites:** a `trn2.48xlarge` with Neuron SDK 2.33.1 (see the
[setup guide](../docs/getting-started/setup-guide.md)); this branch installed into the vLLM 0.29 environment;
the FP8 checkpoint (about 186 GB on disk) on a local volume.

### Step 1: Environment Setup

The model itself needs no environment knobs. The variables below are runtime and compile settings; every
measured run set them (the cache directories are optional).

```bash
# Confirm the devices are visible
neuron-ls

# Optional: compile caches on a large local volume (the instance store is lost on stop / terminate)
export VLLM_CACHE_ROOT=/path/to/scratch/vllm_cache
export NKI_COMPILE_CACHE_URL=/path/to/scratch/nki_cache

# KV-cache share of free HBM: 0.07 in every measurement of this revision. A larger share leaves too little
# HBM for the activations of a batch (see Known limits).
export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.07

# Host path: keep each step reader spinning for up to 20 ms (about one decode step) before it blocks;
# the plugin's default window is 0.2 ms (set in every measurement)
export VLLM_NEURON_SHM_READER_BUSY_LOOP_S=0.02

# Cold compile: a bounded compiler pool (the default, sized from the vCPU count, exceeds the host RAM;
# 8 at max_model_len 81,920), a NEFF lock and a rank barrier longer than the build (defaults: 20 minutes
# / 1 hour)
export NEURON_LIBTORCH_PARALLEL_COMPILE_WORKERS=20
export TORCH_NEURONX_NEFF_CACHE_LOCK_TIMEOUT=14400
export VLLM_NEURON_BARRIER_TIMEOUT=14400
```

If `torch-neuronx` is installed in the same environment, also set `TORCH_DEVICE_BACKEND_AUTOLOAD=0` (see the
[setup guide](../docs/getting-started/setup-guide.md)).

The first start compiles every prefill and decode graph; later starts with the same cache directories load
them from the cache. Replicas of one configuration share the cache: the compiled graphs do not depend on
which cores a replica runs on.

### Step 2: Download the Model

```bash
hf download Qwen/Qwen3.8-Flash-Next-FP8 --local-dir /path/to/models/Qwen3.8-Flash-Next-FP8
```

## Serving

### Offline inference (`llm.generate()`)

[`examples/vllm_neuron/models/qwen4_exp/run.py`](../examples/vllm_neuron/models/qwen4_exp/run.py) applies the
Step 1 settings and `NEURON_RT_IO_RING_CACHE_SIZE=4` itself via `os.environ.setdefault` and builds `LLM` with the
TP=32 configuration of the next section (prefix caching on), so the only required argument is the checkpoint:

```bash
python examples/vllm_neuron/models/qwen4_exp/run.py --model-checkpoint /path/to/models/Qwen3.8-Flash-Next-FP8
```

An explicit `export` still wins, since `setdefault` only fills unset variables. Set the compile-cache
variables of Step 1 yourself.

### Online serving — FP8, primary configuration (`max_model_len` 2,048)

TP=32 with MTP k=2, with the Step 1 environment:

```bash
NEURON_RT_IO_RING_CACHE_SIZE=4 vllm serve /path/to/models/Qwen3.8-Flash-Next-FP8 \
  --served-model-name Qwen3.8-Flash-Next-FP8 \
  --tensor-parallel-size 32 \
  --enable-expert-parallel \
  --max-model-len 2048 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 16 \
  --block-size 32 \
  --enable-prefix-caching \
  --async-scheduling \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}' \
  --additional-config '{"neuron_config": {"ep_degree": 4, "decode_context_length_buckets": [1024], "num_batched_tokens_buckets": [128, 256, 512, 1024, 1536, 1792, 2048]}}'
```

- `num_batched_tokens_buckets` are the prefill chunk shapes; the last must equal `--max-num-batched-tokens`.
  A prompt is prefilled in chunks of at most 2,048 tokens.
- `decode_context_length_buckets: [1024]` adds a decode graph for contexts up to 1,024 tokens next to the one
  for `max_model_len`, so short contexts do not read the whole 2,048-token window.
- `--enable-prefix-caching` is the default for this model; it is spelled out because the measured commands
  passed it.
- A second replica fits on the other 32 cores: prefix the command with `NEURON_VISIBLE_DEVICES=0-31` and
  `NEURON_VISIBLE_DEVICES=32-63`, and give each its own `--port`.
- MTP k=3 (`"num_speculative_tokens": 3`) also runs; it was checked at B=1 only, on an earlier revision of this
  port.

### Throughput configurations (TP=16 and TP=8)

When the total output of the box matters more than the latency of one request, run four TP=16 or eight TP=8
replicas instead, each its own `vllm serve` on its own cores and port, and put a load balancer or a client-side
split in front. Neither uses MTP, and both keep the n-gram table in host memory.

| | TP=32 (default) | TP=16 × 4 | TP=8 × 8 |
|---|---|---|---|
| Cores per replica / replicas per box | 32 / 1 or 2 | 16 / 4 | 8 / 8 |
| MTP | k=2 | — | — |
| `--max-num-seqs` | 16 | 32 | 16 |
| Expert parallelism | `ep_degree` 4 | `ep_degree` 2 | — |
| n-gram table | device | host | host |
| Prefill chunk shapes | 128 … 2,048 (7) | 128 … 2,048 (7) | 512 / 1,024 / 2,048 |
| Decode context buckets | 1,024 and 2,048 | 1,024 and 2,048 | 2,048 |

One TP=8 replica on cores 0–7, with the Step 1 environment; replica *i* uses cores 8*i* … 8*i* + 7 and its own
port:

```bash
NEURON_RT_IO_RING_CACHE_SIZE=4 NEURON_VISIBLE_DEVICES=0-7 vllm serve /path/to/models/Qwen3.8-Flash-Next-FP8 \
  --served-model-name Qwen3.8-Flash-Next-FP8 \
  --port 8000 \
  --tensor-parallel-size 8 \
  --max-model-len 2048 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 16 \
  --block-size 32 \
  --enable-prefix-caching \
  --async-scheduling \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --additional-config '{"neuron_config": {"num_batched_tokens_buckets": [512, 1024, 2048]}}'
```

One TP=16 replica on cores 0–15; replica *i* uses cores 16*i* … 16*i* + 15 and its own port:

```bash
NEURON_RT_IO_RING_CACHE_SIZE=4 NEURON_VISIBLE_DEVICES=0-15 vllm serve /path/to/models/Qwen3.8-Flash-Next-FP8 \
  --served-model-name Qwen3.8-Flash-Next-FP8 \
  --port 8000 \
  --tensor-parallel-size 16 \
  --enable-expert-parallel \
  --max-model-len 2048 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 32 \
  --block-size 32 \
  --enable-prefix-caching \
  --async-scheduling \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --additional-config '{"neuron_config": {"ep_degree": 2, "decode_context_length_buckets": [1024], "num_batched_tokens_buckets": [128, 256, 512, 1024, 1536, 1792, 2048]}}'
```

- **Starting the replicas.** On a cold cache start the first replica alone and the others after it is ready:
  they load its compiled graphs instead of compiling the same graphs at once. On a warm cache the measurements
  started the TP=16 and TP=8 replicas 60 seconds apart and the two TP=32 replicas one after the other.
- `NEURON_VISIBLE_DEVICES` selects the logical NeuronCores of a replica (a range or a comma list);
  `NEURON_RT_VISIBLE_CORES` is not accepted with vLLM's multiprocessing.
- The TP=8 prefill chunk shapes are 512 / 1,024 / 2,048 tokens, the shapes every TP=8 measurement used.
- Host memory: the replicas share one page cache of the n-gram table (about 51 GB).

### Long context (`max_model_len` 81,920)

TP=32, measured on this revision with the arguments below (see [Measured performance](#measured-performance)). The
Step 1 environment with a smaller compile pool; the sparse attention forms are selected from `max_model_len` (see
[Dense up to 2,051 tokens, sparse beyond](#dense-up-to-2051-tokens-sparse-beyond)):

```bash
export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.07
export NEURON_LIBTORCH_PARALLEL_COMPILE_WORKERS=8
```

```bash
vllm serve /path/to/models/Qwen3.8-Flash-Next-FP8 \
  --served-model-name Qwen3.8-Flash-Next-FP8 \
  --tensor-parallel-size 32 \
  --enable-expert-parallel \
  --max-model-len 81920 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 4 \
  --block-size 32 \
  --async-scheduling \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}' \
  --additional-config '{"neuron_config": {"ep_degree": 4, "decode_context_length_buckets": [1024, 2048, 4096, 8192, 32768], "num_batched_tokens_buckets": [128, 512, 1024]}}'
```

- Prefill chunks are 1,024 tokens here; 4,096-token chunks do not fit TP=32 HBM at this length.
- Decode picks the smallest decode bucket that holds the context (contexts above 32,768 use the
  `max_model_len` graph), and its cost grows with the bucket width.
- **Prefill context buckets** (built in, nothing to set): when `max_model_len` is above 32,768 and data,
  decode-context and prefill-context parallelism are all 1, the buckets are those of 2,048 / 8,192 / 32,768
  below `max_model_len`. Each prefill chunk — segmented prefill chunks included — attends only to the smallest
  bucket that holds its positions (plus the MTP draft that follows it in the same step), instead of the whole
  `max_model_len` window; a chunk that needs more keeps the full width. Warm-up compiles each prefill shape at
  those widths as well. Decode and MTP verify / draft steps are unaffected (decode keeps
  `decode_context_length_buckets`). This is host-side code only: the traced model is unchanged, and at
  `max_model_len` ≤ 32,768 nothing is installed.
- Time to first token is prefill-bound and grows with the prompt.

## Accuracy Evaluation

### GSM8K-CoT (thinking disabled)

Measured through the OpenAI-compatible endpoint with
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) 0.4.12 `local-completions`,
against one replica of each configuration's serve command
([TP=32](#online-serving--fp8-primary-configuration-max_model_len-2048),
[TP=16 / TP=8](#throughput-configurations-tp16-and-tp8)), task `gsm8k_cot_nothink` (below), full test set
(1,319 questions), 4-shot chain-of-thought, greedy:

| Configuration | MTP | Concurrency | strict-match | flexible-extract |
|---|---|---|---|---|
| TP=32 | k=2 | 8 | 0.931 | 0.936 |
| TP=16 | — | 32 | 0.936 | 0.939 |
| TP=8 | — | 16 | 0.932 | 0.937 |

The checkpoint is a thinking model, and `local-completions` sends raw text without the chat template, so
`enable_thinking` cannot be passed. The task therefore closes the think block in the prompt, exactly as the
chat template's `enable_thinking=false` branch does; without it the model opens a reasoning trace that the
stock stop strings never end, and strict-match collapses. The task is lm_eval's stock `gsm8k_cot` with one
field changed. Put two files in one directory:

1. a copy of `lm_eval/tasks/gsm8k/gsm8k-cot.yaml` from your lm_eval installation, with its `task:` line
   renamed to `gsm8k_cot_stock_copy` (so it does not collide with the packaged task);
2. `gsm8k-cot-nothink.yaml`:

```yaml
include: gsm8k-cot.yaml
task: gsm8k_cot_nothink
doc_to_text: "Q: {{question}}\n\nA: <think>\n\n</think>\n\n"
```

Then, with the server up (`num_concurrent` as in the table):

```bash
lm_eval --model local-completions \
  --model_args base_url=http://localhost:8000/v1/completions,model=Qwen3.8-Flash-Next-FP8,num_concurrent=8,max_retries=2,tokenizer=/path/to/models/Qwen3.8-Flash-Next-FP8,tokenized_requests=False,timeout=3600,max_length=2048 \
  --include_path /path/to/task_dir --tasks gsm8k_cot_nothink --num_fewshot 4 \
  --batch_size 1 --output_path /path/to/results
```

Before trusting a score, run the stock `gsm8k_cot` against the same server as well: it must score far lower
(the reasoning trace is never closed). If both score the same, the marker is not taking effect.

### Logit validation — not run

The generic 3-way logit validation (`examples/vllm_neuron/accuracy/run_logit_validation_offline.py` /
`run_logit_validation_online.py`) was **not run** for this model. To run it:

- serve the **BF16** checkpoint (the script compares the device against an FP32 and a BF16 CPU run of the
  same weights, so the FP8 path would fold its own quantization error into the comparison);
- add the expert-parallel arguments to the script's `LLM(...)` call (`enable_expert_parallel=True` and
  `additional_config={"neuron_config": {"ep_degree": 4}}`), which it does not pass by default, and keep TP=32;
- budget host memory for an FP32 CPU copy of the full model (several hundred GB) and a long CPU baseline run.

## Measured performance

Conditions for the decode figures: one replica on a `trn2.48xlarge` with its serve configuration (TP=32: MTP k=2,
`ep_degree` 4; TP=16: `ep_degree` 2, no MTP; TP=8: no MTP), FP8 checkpoint, async scheduling, prefix caching on,
warm compile cache, greedy, offline `LLM.generate` with closed batches of B requests, the Step 1 environment.
`NEURON_RT_IO_RING_CACHE_SIZE=4` was set for every figure, as in the serve commands. No measured request hits the prefix cache: the 30-token prompts are shorter than the
128-token state grid, and the prefix cache is cleared before each timed generate of the 1,635-token prompts.

What is measured, per batch size B (128 output tokens, `ignore_eos`):

- **TTFT** = wall time of a `max_tokens=1` generate of the whole batch. At B > 1 the prompts are prefilled
  one after another, so this is the time until the **last** request has its first token.
- **Time per output token** = (t128 − t1) / 127, where t128 and t1 are the wall times of the 128-token and
  1-token generates. At B > 1 every request gains one token in each such interval, so this is the
  **per-request inter-token latency**; with MTP it is not the device step time — one step can emit up to
  k + 1 tokens per request.
- **Aggregate output** = B × 127 / (t128 − t1), the decode-phase output rate of the whole batch.

### Primary configuration, TP=32 (`max_model_len` 2,048, `max_num_seqs` 16)

**Short prompt** — 30 prompt tokens, B identical requests:

| B | TTFT (s) | Time per output token, per request (ms) | Aggregate output (tok/s) | Previous revision, time per output token (ms) |
|---|---:|---:|---:|---:|
| 1 | **0.111** | **9.21** | 108.6 | 12.23 |
| 2 | 0.219 | 17.24 | 116.0 | 18.87 |
| 4 | 0.398 | 20.09 | 199.1 | 23.89 |
| 8 | 0.752 | 20.01 | 399.8 | 30.43 |
| 16 | 1.53 | 48.08 | 332.8 | 51.39 |

**Long prompt** — 1,635 prompt tokens (prefilled in one 2,048 chunk), B distinct requests:

| B | TTFT (s) | Time per output token, per request (ms) | Aggregate output (tok/s) | Previous revision, time per output token (ms) |
|---|---:|---:|---:|---:|
| 1 | **0.275** | **9.78** | 102.2 | 12.91 |
| 2 | 0.543 | 17.11 | 116.9 | 21.66 |
| 4 | 1.07 | 19.66 | 203.4 | ❌ out of HBM |
| 8 | 2.15 | 24.03 | 332.9 | — |
| 16 | 4.30 | 47.68 | 335.6 | — |

The last column is the previous revision of this README (Neuron 2.33, KV-cache share 0.09, prefix caching off),
whose B=1 TTFT was 0.123 s with the short prompt and 0.379 s with the long one. There, B=4 × 1,635-token prompts
did not fit.

The time per output token rises most from B=1 to B=2 in the table above (measured when the fused decode-MoE kernel
covered only four rows per step at TP=32, so a B=2 verify step of six rows took the all-expert path). With the fused
kernel extended to 128 rows, `vllm bench serve` random 1024/512 on one TP=32 replica gives a time per output token
of 10.1 / 12.5 / 16.9 / 21.6 ms at concurrency 1 / 2 / 4 / 8 (it was 10.8 / 26.7 / 28.4 / 24.9 ms).

Batches with padding rows (3, 5 and 7 concurrent 30-token requests, so the batch graphs carry padding rows;
previous revision) give no degenerate output; 13 of the 15 outputs equal the same prompt served alone, and the
other two (one prompt, in the 5- and the 7-request batch) give a different, well-formed answer.

### Throughput configurations, per replica (TP=16 and TP=8)

One replica, the decode conditions above, no MTP (each step emits one token per request, so the time per output
token is the step time); prompts of 30 and of 1,635 tokens:

| B | TP=16, 30-token prompt (ms) | TP=16, 1,635-token prompt (ms) | TP=8, 30-token prompt (ms) | TP=8, 1,635-token prompt (ms) |
|---|---:|---:|---:|---:|
| 1 | 21.0 | 20.8 | 20.2 | 20.2 |
| 8 | 31.8 | 37.1 | 32.3 | 36.0 |
| 16 | 47.2 | 51.9 | 48.1 | 54.7 |
| 32 | 101.8 | 104.8 | — | — |

TTFT of one 1,635-token prompt: TP=16 0.30 s, TP=8 0.36 s.

### Serving benchmark (`vllm bench serve`)

Every replica of a layout serves at once on one `trn2.48xlarge`, with the serve commands of
[Serving](#serving) on a warm cache; one `vllm bench serve` client per replica runs in parallel with
`--dataset-name random --random-input-len 1024 --random-output-len 512 --ignore-eos`, `--max-concurrency` C and
4 × C prompts. Box output is the sum of the clients' output token throughput; TPOT and TTFT are the medians a
client reports (the replicas of a layout agree within 6 %). Before each benchmark every replica answered a greedy
check prompt with fluent text, and every client reported every request successful.

| Layout | MTP | Concurrency per replica | Box output (tok/s) | Median TPOT (ms) | Median TTFT (s) |
|---|---|---:|---:|---:|---:|
| TP=8 × 8 | — | 8 | 1,323.7 | 45.9 | 1.29 |
| TP=8 × 8 | — | 16 | 1,677.6 | 70.1 | 2.82 |
| TP=16 × 4 | — | 16 | 966.7 | 62.6 | 1.94 |
| TP=16 × 4 | — | 32 | 1,020.9 | 116.9 | 4.16 |
| TP=32 × 2 | k=2 | 8 | 559.1 | 24.5 | 0.30 |
| TP=32 × 2 | k=2 | 16 | 522.5 | 55.0 | 0.50 |

Random tokens with `--ignore-eos` measure throughput only: they do not show whether the output is correct (hence
the check prompt), and MTP accepts far fewer drafts on random tokens than on text, so the TP=32 rows understate
what MTP gives on real prompts.

### Long-context configuration (`max_model_len` 81,920, `max_num_batched_tokens` 1,024, `max_num_seqs` 4)

TP=32, B=1, the [long-context serve arguments](#long-context-max_model_len-81920) (decode buckets 1,024 / 2,048 /
4,096 / 8,192 / 32,768, KV-cache share 0.07), offline `LLM.generate` with prefix caching off and without
`NEURON_RT_IO_RING_CACHE_SIZE`, warm compile cache; the last column is the previous revision of this README:

| Prompt tokens | Decode bucket | TTFT (s) | Time per output token (ms) | Previous revision, TTFT (s) / time per output token (ms) |
|---:|---:|---:|---:|---:|
| 30 | 1,024 | 0.144 | 12.80 | 0.153 / 15.84 |
| 1,635 | 2,048 | 0.925 | 12.31 | 1.010 / 15.14 |
| 3,847 | 4,096 | 5.200 | 15.09 | 5.387 / 17.85 |
| 7,823 | 8,192 | 11.58 | 17.04 | 11.97 / 19.70 |
| 29,971 | 32,768 | 50.76 | 30.15 | 52.72 / 33.35 |

The decode bucket is the smallest that holds the prompt, the 128 output tokens and the MTP headroom. The
prefill context buckets are active in this configuration.

Retrieval at long context: a fact placed at the start of a 14,034-token prompt is retrieved
(the model reasons in `<think>` and then answers with it). MTP acceptance in this run: 608 of 856 draft tokens
(71 %).

Decode cost grows with the width of the decode bucket that holds the context. TTFT at this `max_model_len` is
prefill-bound and grows with the prompt; the prefill context buckets keep the first 2,048 positions of a prompt
on a 2,048-wide table, which is where most of the reduction for short and mid-length prompts comes from.

## Contents of this bundle

The model is integrated in-tree (committed directly on this branch), so the top-level `Qwen3.8-Flash-Next/`
bundle is just this source-of-truth README. There is no `src/` copy (that would duplicate the in-tree
package) and no bundle `test/`: the scripts used for the device measurements are not part of the bundle — the
figures above are transcribed from them.

```text
Qwen3.8-Flash-Next/
└── README.md            # This file — the source of truth for the Qwen3.8-Flash-Next port
```

### Paths this port touches across the repository

Everything below is stated against the branch this port is based on, `release-0.29.0.1.2.0`
(vllm-neuron 0.29 / Neuron 2.33).

**New — model package** (`vllm_neuron/model/qwen4_exp/`)

```text
vllm_neuron/model/qwen4_exp/
├── __init__.py          # Package exports: Qwen4ExpForCausalLM, Qwen4ExpForConditionalGeneration, Qwen4ExpMTP
├── config.py            # HF → Neuron config translation; FP8 checkpoint detection
├── factory.py           # Model builder wiring
├── layers.py            # The model math: DeltaNet, gated attention + block indexer (dense / sparse forms), MoE, hyper-connections, n-gram embedding, the kernel paths
├── model_bf16.py        # The decoder, the per-request state pools (with the padding rows' sink row), the loader (FP8 block → per-channel requantization), the options chosen by the TP degree
├── mtp.py               # The MTP draft model
├── ngram.py             # Host-side n-gram ids
├── ngram_device.py      # On-device n-gram ids, int32-only, bit-exact with the host
├── parallel.py          # TP / EP layout: attention groups, expert and vocabulary sharding
├── ple_host.py          # The n-gram table in host memory below TP=32: memory-mapped table, per-step row gather and the decode-step join
├── prefill_ctx.py       # Prefill context buckets for max_model_len > 32,768 (host-side runner / warm-up hooks, installed from factory.py)
└── README.md            # Module structure (points here)
```

**New — NKI kernels** (`vllm_neuron/_kernels/qwen4_exp/`)

```text
vllm_neuron/_kernels/qwen4_exp/
├── __init__.py
├── attn_tkg.py               # Decode / MTP-verify attention projections
├── gdn_tkg.py                # Single-token gated delta-rule decode (decode steps of more than 16 rows)
├── gdn_tkg_act.py            # The same with the activation-table work grouped (decode steps of up to 16 rows)
├── gdn_tkg_multi.py          # The same for the 1 + k tokens of an MTP verify step in one call
├── gdn_tkg_multi_torch.py    # PyTorch reference of gdn_tkg_multi (CPU path)
├── hc_fused.py               # Hyper-connection read of a prefill chunk in one kernel
├── hc_fused_sel.py           # The same on the rows of one rank (prefill read split over the attention group)
├── kv_window.py              # KV window of decode and MTP verify: whole KV blocks per indirect DMA
├── moe_block_tkg.py          # Decode MoE block in one kernel: router, top-k, routed (BF16 / FP8) and shared experts
├── moe_block_tkg_pf.py       # The FP8 decode MoE block with the routed experts pipelined in one loop body
├── moe_cte_dynamic.py        # Prefill MoE that skips its empty expert blocks
├── moe_cte_dyn/
│   ├── __init__.py
│   └── bwmm_shard_on_block.py  # Copy of the public nki-library MoE CTE kernel with a static + dynamic block loop and the FP8 row-scaled path
├── moe_tkg_fp8.py            # All-expert decode MoE on FP8 per-channel expert weights (nki-library moe_tkg)
└── state_rows.py             # Rows of a 4-D state pool by DMAs on the Sync queue
```

**New — framework**

```text
vllm_neuron/vllm/spec_decode/mtp.py          # MTP proposer (speculative_config method "mtp"), on the Eagle fused-propose contract
vllm_neuron/vllm/worker/request_state.py     # Per-request state slots for models that own their recurrent state; prefix-caching snapshot rows and directory; per-step host inputs joined before the model call
vllm_neuron/vllm/core/state_prefix_cache.py  # Neuron scheduler with the state snapshot directory (selected only with prefix caching on)
```

**Modified — shared framework files**

```text
vllm_neuron/functional/moe/moe_blockwise.py         # Choose a flatten length that divides the token count (1,792-token prefill chunks)
vllm_neuron/functional/topk.py                      # Pad the NKI top-k rows to a multiple of the LNC programs (3-row MTP verify at B=1)
vllm_neuron/model/registry.py                       # Register Qwen4ExpForCausalLM / Qwen4ExpForConditionalGeneration / Qwen4ExpMTP
vllm_neuron/vllm/patches/shm_reader_spin_patch.py   # VLLM_NEURON_SHM_READER_BUSY_LOOP_S override (default unchanged)
vllm_neuron/vllm/platform.py                        # Accept quantization "fp8"; attention-group TP check; prefix caching in align mode for state-owning models (scheduler selection; a single-size kv_segment_size_buckets when prefill is single-shot); no vision buckets for the text-only port
vllm_neuron/vllm/spec_decode/eagle.py               # Accept method "mtp"
vllm_neuron/vllm/worker/neuron_model_runner.py      # Request-state slot assignment (+ snapshot rows and the per-step plan), MTP wiring, key-only indexer KV spec for the draft, the host-input join before the model call (installed only with the host n-gram table)
vllm_neuron/vllm/worker/neuron_model_runner_v2.py   # Quantization "fp8" enables the trn2 FP8 compiler flag
vllm_neuron/vllm/worker/neuron_worker.py            # Compile the prefix-cache snapshot row copy after warm-up
```

**New — offline inference example**

```text
examples/vllm_neuron/models/qwen4_exp/run.py   # Applies the Step 1 settings and NEURON_RT_IO_RING_CACHE_SIZE=4 via os.environ.setdefault; --model-checkpoint is the only required argument
```

**Documentation**

```text
README.md                                        # Contributed Models link (modified)
docs/model-recipes/index.md                      # Grid card + toctree entry (modified)
docs/tutorials/index.md                          # Grid card + toctree entry (modified)
docs/model-recipes/qwen3-8-flash-next.md         # Model recipe pointer to this README (new)
docs/tutorials/tutorial-qwen3-8-flash-next.md    # Deployment tutorial pointer to this README (new)
```
