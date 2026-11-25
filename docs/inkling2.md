# Inkling 2

Status: Inkling 2 Mini's text decoder runs on the CPU (`model/inkling2`), from
the checkpoint as Thinking Machines ship it, with its tokenizer and chat
template. Checked against transformers on a tiny random checkpoint in the
original layout; not yet run on the real weights.

## The model

[thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
(Apache 2.0) is `Inkling2LoopForConditionalGeneration`: a vision tower and a
26B-parameter text decoder, of which about 3.4B are used per token. Only the
text decoder is implemented. Its reference is transformers'
`models/inkling2_loop` (5.24), which this follows operation by operation.

| | |
|---|---|
| layers | 24, hidden 2304, no rotary embedding |
| attention | 20 sliding-window layers (window 1024) and 4 full ones (5, 11, 17, 23); 24 query heads over 8 key/value heads of 128 |
| positions | a learned relative bias: each head's 16 relative states, from the layer input, L2-clipped, weigh a bank of profiles over the last 1024 distances |
| keys, values | short causal convolutions (kernel 4) after the projections, with per-token taps in the full layers; keys L2-clipped, values RMS-normalized |
| queries | scaled per channel, L2-clipped; in full layers scaled up logarithmically past position 8192 |
| softmax | one extra logit per head and token, the sink, from the layer input |
| output | gated by a sigmoid of the layer input, then a short convolution |
| residuals | attention residuals: each layer's input is a softmax mix, with learned queries, of the RMS-normalized block states (the inputs of layers 0, 6, 12 and 18) and the running sum |
| experts | 8 of 128 per token, sigmoid router with a selection bias, weights renormalized and scaled by 8; SwiGLU with the gate's linear factor softcapped at 4 and up at 25 |
| loop | layers 6–17 run three times with shared weights; passes 2 and 3 restart from layer 6's input and mix messages from the previous pass into their query, key, value and expert inputs through rank-64 tanh gates |
| head | 200,058 logits, softcapped at 30 |

Each token runs through 48 layer visits; every visit has its own key/value
and convolution caches.

## Weights

The checkpoint is one 19.5 GB file under the original names (`model.llm.*`),
not transformers' (`Load` reads those names directly):

| | format | size |
|---|---|---|
| experts of layers 0–19 | NVFP4: two E2M1 codes per byte, an E4M3 scale per 16 values, a float32 factor per expert (amax / outer denominator) | 11.5 GB |
| experts of layers 20–23 | MXFP8: E4M3 values, a power-of-two scale per 32 | 4.2 GB |
| embeddings, unembedding | bf16 | 1.85 GB |
| attention, gates, router | bf16 | 1.3 GB |
| residual and message queries | float32 | |

`ops.Microscaled` wraps an expert's bytes where they lie in the memory-mapped
file: nothing is converted at load. The gate and up projections are stored as
one matrix with their rows interleaved (gate j, up j), which the experts read
in pairs. The bf16 weights are packed as for the other models, and the
unembedding's padding rows (to 201,024) are dropped.

Resident memory is the mapped file (19.5 GB, paged in as experts are used)
plus about 2.3 GB of packed bf16 weights, so it fits a 32 GB Mac. Each
position adds 393 KB of keys and values across the 48 visits; the 40 sliding
visits keep only their last 1,023 positions (trimmed once 2,048 build up).

## Products over microscaled weights

`ops.MicroLinearInto` multiplies rows by a microscaled matrix, decoding each
block's values once for up to 16 rows and summing each block before applying
its scale. One row (each decoding step) has its own unrolled loops. On an M2
Max, one expert's gate, up and down projections (2304 × 2304 and 2304 × 1152,
NVFP4) take 0.40 ms for one row, about 10 GB/s of weights: about 0.15 s per
generated token for the 384 expert products of its 48 visits.

From 32 rows per expert (long prompts), the expert is dequantized to float32
and runs through the dense kernels instead: dequantizing and packing cost
about 7 ms, as much as 28 rows decoded block by block. Weights are dequantized
as transformers dequantizes them, (e2m1 · scale) · factor rounded once to
float32, so the dense path computes with the reference's weights.

Plain Go is the limit here: the decode loop is compute-bound at 10 GB/s, well
below the memory bandwidth a NEON kernel could use.

## How it was checked

Running the real model in transformers needs its experts dequantized to
float32: about 110 GB. So `scripts/dump_golden_inkling2.py tiny` writes a tiny
random checkpoint (`testdata/golden/inkling2_tiny`, 600 KB) in the original
layout, with every feature reachable: 6 layers in blocks of 2, the core
(layers 2–3) looped three times, full-attention layers inside and outside the
core, NVFP4 and MXFP8 experts, a window (8) and relative extent (6) shorter
than the 21-token input, log scaling from position 4, and L2 clips and
softcaps that bind. transformers loads it through its own converter (the
same renames, query merges and dequantization as for the real checkpoint),
and the script dumps each layer visit's input, the logits and greedy
generation with the cache.

| check | result |
|---|---|
| each of the 10 layer visits' inputs and the final output | within 5.1e-6 of the range (float32 rounding, growing with depth) |
| logits | within 2.1e-5 (softcapped at 5, which squeezes the range more than the error) |
| greedy generation with the caches, 6 tokens | identical tokens, each step's logits within 2.1e-5 |
| experts through the dense path | the same logits bounds |
| decoding 4 windows one token at a time, trimming the sliding caches | each step's logits within 2.1e-5 of the whole sequence's |
| tokenizer (`tokenizer/bpe`) | identical ids and decoding on the edge cases and 400 random strings |
| chat template (`tokenizer/chat`) | identical text and ids on 7 conversations: efforts by name and number, reasoning, a system-only conversation, tools with a call and its response |

The tests on the real weights (`model/inkling2/mini_test.go`) skip until the
weights are in `testdata/models/inkling2-mini`: the configuration and expert
formats, a greedy answer to a question, and, if goldens dumped on a machine
with the memory are in `testdata/golden`, the same comparisons as above.

## Not done

- Running on the real weights, and goldens from them.
- The vision tower and the multi-token prediction draft head (`mtp.safetensors`).
- The GPU: everything runs on the CPU.
- A NEON kernel for the microscaled products, and dequantizing straight into
  packed panels for long prompts.
- A session that reuses a shared prompt prefix, as `llama.Session` does.
