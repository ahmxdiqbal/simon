# Local Inference Strategy

SiMon runs a local LLM on Apple Silicon to summarize Telegram messages into geopolitical event digests. This document explains the configuration choices, what each parameter does at the hardware level, and how we arrived at the current values.

## Hardware context

The target machine is an M1 Pro MacBook with 16GB unified memory and ~200 GB/s memory bandwidth. Apple Silicon shares memory between CPU and GPU, so the model weights and the GPU's working memory compete for the same 16GB.

The model (Qwen3.5-9B at 4-bit quantization) occupies ~5GB. Each generated token requires reading the full 5GB of weights through the 200 GB/s memory bus. That sets a hard floor of ~25ms per token, or about 40 tokens/second. With overhead, we see ~33 tokens/second in practice.

## Model selection

**`MODEL = "mlx-community/Qwen3.5-9B-MLX-4bit"`**

We tested three Qwen 3.5 sizes: 0.8B, 4B, and 9B.

The 0.8B model runs fast but produces garbage on structured extraction tasks. Country fields are wrong, deduplication doesn't happen, and the output requires heavy post-processing that negates the speed advantage.

The 4B model handles individual messages well but can't merge related events across a large input. It treats each message as its own event.

The 9B model can compress and merge at a level approaching DeepSeek-V3 (the cloud API). It collapses thread arcs, deduplicates across channels, and produces usable citation-attributed summaries. The tradeoff is speed: ~33 tokens/second on M1 Pro versus near-instant API responses from DeepSeek.

The model is 4-bit quantized, meaning each of the 9 billion weight parameters is stored as a 4-bit integer instead of the native 16-bit float. This cuts the file from ~18GB to ~5GB with modest quality loss. Lower quantization (3-bit, 2-bit) would further reduce size and increase decode speed but degrades output quality, particularly on structured JSON tasks.

## Inference phases and their bottlenecks

LLM inference has two phases with different performance characteristics:

**Prefill** processes all input tokens (system prompt + user message + Telegram messages) through the model to build the KV cache. This is compute-bound because many tokens are processed simultaneously, reusing the same weight matrices across the entire batch. The GPU's compute units are fully occupied.

**Decode** generates output tokens one at a time. Each token requires reading the full ~5GB of model weights from memory. This is memory-bandwidth-bound because the GPU finishes the math before the next batch of weights arrives from memory.

For our typical workload (114 messages, ~20-30k input tokens, ~8k output tokens), the split is roughly 40% prefill and 60% decode.

## Configuration parameters

### PREFILL_STEP_SIZE = 4096

During prefill, input tokens are processed in batches. This parameter controls the batch size.

**How it works:** With 20,000 input tokens and a step size of 4096, the model processes them in ~5 GPU dispatches instead of ~10 at the default of 2048. Each dispatch has fixed overhead (Metal kernel launch, memory allocation, synchronization). Fewer dispatches means less overhead, and larger batches better saturate the GPU's parallel compute units.

**Why 4096:** The default (2048) is conservative. We tested 8192, which crashed with an out-of-memory error on 16GB — the attention score matrices within each batch grow quadratically with batch size, and 8192 exceeded available memory. 4096 is the largest value that runs reliably on 16GB.

**Impact:** ~10% reduction in total inference time, driven by faster prefill. Decode time is unaffected since decode is always batch-size-1.

**Caveat:** Changing the prefill batch size alters the floating-point arithmetic order, which produces slightly different token sequences. At temperature=0 (greedy decoding), this caused the model to enter a repetition loop much earlier. The temperature and repetition penalty settings below were introduced to address this.

### TEMPERATURE = 0.1

Controls randomness in token selection during decode.

**How it works:** After each layer processes a token, the final output is a probability distribution over ~150,000 possible next tokens. At temperature=0 (greedy), the model always picks the highest-probability token. At temperature=1.0, it samples proportionally to the probabilities. At 0.1, the distribution is "sharpened" — the highest-probability token is still overwhelmingly likely, but the model occasionally picks the second or third choice.

**Why 0.1:** At 0.0 the model was deterministic and prone to repetition loops, where the same sequence of tokens gets regenerated indefinitely (the model enters a fixed point in its probability landscape). Even a small amount of randomness lets the model escape these loops. We chose 0.1 because it's low enough that output remains coherent and factual, but high enough to prevent loops. Higher values (0.3+) would introduce noticeable randomness — fabricated details, inconsistent formatting.

**Impact:** Eliminated repetition loops entirely. Previous runs consistently hit loops between 4,000 and 10,000 characters of output, requiring a JSON repair pass that lost data. With temperature=0.1, the model generates complete, valid JSON every time across multiple test runs.

**Tradeoff:** Output is no longer deterministic. Running the same input twice produces slightly different summaries (17-20 events, different wording). For a news digest use case this is acceptable.

### REPETITION_PENALTY = 1.1

Reduces the probability of tokens the model has recently generated.

**How it works:** Before selecting each token, the model checks the last `REPETITION_PENALTY_CONTEXT` tokens it generated. Any token that appears in that window has its probability divided by 1.1 (if the probability was positive) or multiplied by 1.1 (if negative in log-space). This makes recently-used tokens slightly less likely to be chosen again.

**Why 1.1:** The penalty is multiplicative, so it compounds. A token appearing 3 times in the context window doesn't get penalized 3x — it's a single check of presence. 1.1 is a light touch: it nudges the model away from repetition without forcing it to avoid words it genuinely needs to repeat (country names, common phrases). Values above 1.3 cause visible degradation — the model starts using unusual synonyms to avoid repeating words.

### REPETITION_PENALTY_CONTEXT = 256

How many recent tokens the repetition penalty examines.

**Why 256:** Our repetition loops typically had a period of ~100-200 tokens (one complete event object). A context window of 256 tokens means the penalty covers at least one full loop cycle. Smaller windows (20, the default) would miss the pattern because the repeated segment is longer than the window. Larger windows are unnecessary and add marginal compute overhead.

### REPETITION_WINDOW = 600

A separate, coarser safety net that detects when the model has entered a hard repetition loop and terminates generation.

**How it works:** After each token, the system checks whether the last 600 characters of output appear anywhere earlier in the output. If they do, generation stops immediately.

**Why it exists alongside the repetition penalty:** The penalty and temperature make loops rare, but not impossible. This is a backstop. When it triggers, the output is truncated and a JSON repair pass attempts to salvage complete event objects from the partial output.

**Why 600:** Event objects in our output average ~400-500 characters. A window of 600 catches a full repeated event regardless of where the repeat boundary falls.

### CHUNK_SIZE = 150

Number of Telegram messages per model call.

**Why 150:** Our typical refresh window produces 50-150 messages. At 150, this fits in a single model call, avoiding the need for a cross-chunk deduplication pass (which costs a second model call and ~60-70 seconds).

We tested smaller chunks (75 messages) that produced two chunks requiring dedup. Total time was 270s vs ~190s for a single chunk. The dedup pass also lost information — some events from chunk boundaries were dropped during merging.

If message volume grows beyond 150, the code automatically splits into multiple chunks and runs a dedup pass. This is slower but maintains coverage.

### MAX_TOKENS = 8192

Maximum number of tokens the model can generate in one call.

**Why 8192:** Qwen3.5-9B supports up to 128k context, but generation time scales linearly with output length. At 33 tokens/second, 8192 tokens takes ~4 minutes. Our typical output (15-20 events with citations) fits in 3,000-5,000 tokens. The 8192 ceiling provides headroom for unusually busy news periods without setting an impractical upper bound.

## Pipeline architecture

The local pipeline mirrors the DeepSeek cloud pipeline:

1. Format all messages into a structured prompt with channel names and timestamps
2. Prepend the shared SYSTEM_PROMPT (compression rules, citation format, output schema)
3. Send to the model as a single call (or chunked if >150 messages)
4. Parse JSON output; if truncated, repair by extracting complete event objects
5. Deduplicate events with identical text (catches any residual repetition)
6. Convert named citations ([Reuters], [@channel]) to numbered citations with URLs

The persistent model server (`model_server.py`) is optional. When running, it keeps the model loaded in memory between calls, saving ~20 seconds of model loading per invocation. The summarizer checks for it and falls back to loading the model directly if it's not running.

## Benchmark results

Test fixture: 114 messages from a 6-hour window (April 11, 2026).

| Configuration | Time | Events | Parse | Notes |
|---|---|---|---|---|
| DeepSeek-V3 (cloud API) | 42s | 15 | Clean | $0.005 per call |
| Local, default settings | 209s | 20 | Repaired (loop at 10547 chars) | Greedy decode, prefill=2048 |
| Local, current settings | 176-190s | 17-20 | Clean | temp=0.1, rep penalty, prefill=4096 |
| Local, prefill=4096 only | 155s | 9 | Repaired (loop at 3691 chars) | Faster but quality collapsed |
| Local, 2 chunks + dedup | 270s | 17 | Mixed | CHUNK_SIZE=75, dedup pass costly |

The current configuration represents a 10-15% speed improvement over defaults with better output quality (no loops, no repair needed, more diverse event coverage).

## What we tested and ruled out

**KV cache quantization (TurboQuant, mlx_lm kv_bits):** Compresses the key-value cache during inference. No measurable speedup at our context lengths (~20-30k tokens). The KV cache is a small fraction of memory traffic compared to the 5GB of model weights. This optimization pays off at 128k+ token contexts.

**prefill_step_size=8192:** Out-of-memory crash on 16GB M1 Pro. Would work on 32GB+ machines.

**Two-model pipeline (0.8B triage + 4B synthesis):** The 0.8B model produced unusable structured output (wrong countries, missed events). The 4B model couldn't deduplicate across its small input chunks. Combined quality was far worse than a single 9B pass.

## Future optimization paths

**Mixed-bit weight quantization:** mlx_lm supports recipes like `mixed_2_6` (2-bit MLP layers, 6-bit attention). This would reduce weight size from ~5GB to ~3.7GB, directly speeding up decode by ~30%. Requires quality testing on our summarization task.

**Speculative decoding:** Use the 0.6B Qwen as a draft model to propose tokens, with the 9B verifying in batches. Theoretical 1.5-2x decode speedup. Recent mlx_lm versions fixed a bug with Qwen3.5's hybrid attention architecture that previously corrupted speculative output. Requires temperature=0, which conflicts with our current anti-loop strategy.

**Faster hardware:** M4 Max (546 GB/s) would cut decode time by ~2.7x. M3 Ultra (819 GB/s) by ~4x. The local pipeline becomes competitive with DeepSeek API latency on high-bandwidth Apple Silicon.

**Better models:** Future Qwen releases (or competing models) may compress more effectively at the same parameter count, reducing output token count and total decode time. The infrastructure built here — chunking, repetition handling, citation linking — transfers directly to any new model.
