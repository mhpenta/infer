# Llama-family decoder: plan

Goal: run decoder-only language models in the Llama family, checked against
PyTorch the same way as BERT and T5. First model: MiniCPM5-1B for development,
then MiniCPM5-2B, which has the same architecture at a larger size. The same
code then covers Qwen3, SmolLM3, Phi-4-mini and Llama with small additions
(see [After MiniCPM5](#after-minicpm5)).

## The model

| | MiniCPM5-1B | MiniCPM5-2B |
|---|---|---|
| Architecture | `LlamaForCausalLM` | `LlamaForCausalLM` |
| Parameters | 1.08B | 2.52B |
| Layers | 24 | 42 |
| Hidden size | 1536 | 2048 |
| Query heads / key-value heads | 16 / 2 | 16 / 2 |
| Head dimension | 128 (not hidden ÷ heads = 96) | 128 |
| Feed-forward | SwiGLU, 4608 | SwiGLU, 6144 |
| Vocabulary | 130,560 (BPE, 130,072 + 510 added) | same |
| Output layer | separate (`lm_head`), 200M params | separate, 267M params |
| RoPE | θ = 5,000,000, no scaling | same |
| RMSNorm ε | 1e-6 | 1e-6 |
| Weights | bf16, 2.2 GB | bf16, 5 GB |
| End tokens | `</s>` (1), `<|im_end|>` (130073) | same |
| License | Apache-2.0 | Apache-2.0 |

Pinned in `scripts/fetch_models.sh` (`minicpm5`): `openbmb/MiniCPM5-1B` at
revision `87179e5c1f455ef22e6223592d2d61351b525bfc`.

## Reference outputs (done)

`scripts/dump_golden_llama.py`, run in the environment pinned by
`scripts/requirements-llama.txt` (torch 2.14.1, transformers 5.21.0, Python
3.13, in a gitignored `.venv`), writes `testdata/golden/minicpm5_1b_*`
(10.7 MB):

| File | Contents |
|---|---|
| `_tokens.json` | 22 tokenizer edge cases: contractions, digit runs, whitespace runs, newlines, CJK, emoji, combining marks, code, special tokens inside text; ids with and without the default `<s>`; decoded text |
| `_chat.json` | 5 chat-template cases: system/user/assistant turns, generation prompt, `enable_thinking` unset/false/true |
| `_model.json` | 4 greedy generations: a raw prompt (6 → 8 tokens), three chat prompts (27 → 8, 65 → 34, 2,461 → 24) |
| `_model.safetensors` | `rope.inv_freq`; per case, the top 64 logits and their ids at every position (teacher forcing); full logits rows for cases 0–1; every layer's hidden states for cases 0–1 |

The model runs in float32 (bf16 weights widen to float32 exactly, which is
what Go will compute with) and eager attention. The script checks that the
teacher-forced argmax reproduces greedy generation, including the 2,461-token
case.

Findings from the dump that the Go side must match:

- `<s>` is prepended by `tokenizer.json`'s post-processor even though
  `tokenizer_config.json` says `add_bos_token: false`; the chat template also
  starts with `<s>`.
- Digits are split into groups of at most three before the main split, and a
  space before a number becomes its own token (`242`).
- Some weights are smaller than fp16's normal range (about 1 in 8,000 in the
  layers, nearly 1% of the embedding table), so `PackHalf` would not hold
  them exactly: bf16 needs its own panels (step 1).

## Steps

Each step ends with a test against the goldens; nothing moves on until it
passes.

### 1. bf16 weight panels (`ops`) — done

`ops.PackBF16`: weights packed as bf16 and widened to float32 in registers,
in the same panel layout and summation order as the float32 kernels, so
products are bit-identical to packing the widened float32 weights.
`ops.BF16Panels()` reports whether this build widens them in SIMD registers.

| Kernels | Widening of one 16-wide panel row |
|---|---|
| AVX2 6×16, AVX2 one-row, AVX-512 12×32 | `VPMOVZXWD` from memory, then `VPSLLD $16` |
| NEON 6×16 and one-row | `ZIP1`/`ZIP2` against a zero register (4 instructions per row) |
| portable Go | `bits << 16` |

fp16 and bf16 share one "narrow panel" path in `ops` (a format tag on
`Packed`), and the tests are a table over both: bit-identical to float32
panels on AVX2 (Docker) and NEON, close on the portable kernels; one swapped
instruction in a NEON kernel fails them. The AVX-512 bf16 kernel cannot run
under Rosetta, so it is covered by the startup self-check (`ops.KernelCheck`),
which now includes bf16 panels.

Measured on an M2 Max (NEON):

| Product | float32 panels | bf16 panels |
|---|---|---|
| prefill-shaped, 64×4608×1536, 1 thread | 49.8 GFLOPS | 42.7 GFLOPS |
| prefill-shaped, 256×1536×4608, 1 thread | 50.6 GFLOPS | 45.6 GFLOPS |
| one row × 32768×2048 weight, 1 thread | 8.96 ms | 5.82 ms |
| one row × 32768×2048 weight, 12 threads | 3.55 ms | 2.06 ms |

Decoding-shaped products are 1.5–1.7× faster; prefill-shaped ones pay 10–14%
for the widening, which the accelerator work would take over anyway.

Memory: the 1B model's weights stay at 2.2 GB instead of 4.3 GB. Loading still
widens one tensor at a time through `safetensors` (largest: the 800 MB
embedding table) before packing; a raw bf16 accessor can remove that later.

The embedding table is a lookup, not a product: keep it as raw bf16 and widen
the rows a prompt uses.

### 2. Byte-level BPE tokenizer (`tokenizer/bpe`) — done

`bpe.LoadDir`, `Encode` (with the post-processor's `<s>`), `EncodeRaw`
(add_special_tokens=False, for text a chat template already wrapped),
`Decode`, `TokenID`. The pipeline, matching the tokenizers library:

1. Added tokens: the 30 matched on the raw text first, then the 480 marked
   "normalized" in the stretches between them, each pass leftmost and, at one
   position, longest. This lives in `tokenizer/internal/hf` with the
   post-processor template and transformers' clean-up, now shared with
   `tokenizer/unigram`, which also gets the faster matching.
2. Pre-tokenization: `\p{N}{1,3}`, then the GPT-4-style split, both as
   hand-written scanners keyed by their exact patterns (RE2 has no lookahead,
   and its `\s` is ASCII-only where Oniguruma's is Unicode White_Space).
   Unknown patterns are rejected, not approximated.
3. BPE with a priority queue: lowest rank first, leftmost among equals, as
   the tokenizers library's `Word::merge_all` does; a 10,000-piece cache.
4. Decode: the byte alphabet back to bytes, UTF-8 with Rust's
   `from_utf8_lossy` replacement rules (one U+FFFD per maximal invalid
   subpart), which matters when a generated character is split across tokens.

Tests, all exact against `tokenizers` 0.23.3 / transformers 5.21.0:

- the 22 edge cases in `_tokens.json` and 400 seeded random strings in
  `_tokens_random.json` (mixed scripts, Unicode spaces and digits, combining
  marks, contractions, fragments of added tokens): ids with and without
  `<s>`, decoding with and without special tokens;
- the chat goldens' rendered text to ids, and back;
- a one-off check of 288 KB of mixed text (this repository's docs and
  scripts): all 112,098 ids identical, decoding round-trips.

Speed on that text, cold caches, one thread on an M2 Max: 24 ms here against
190 ms for `tokenizers` (Rust, called from Python).

### 3. Chat formatting (`tokenizer/chat`) — done

No Jinja in Go: each supported template is a Go translation, and
`chat.Load(dir)` recognizes the model's template by the SHA-256 of its exact
source (`chat_template.jinja`, else `tokenizer_config.json`), so a changed or
unknown template is an error rather than a near miss. `chat.MiniCPM5` covers
system, user and assistant turns, the generation prompt and
`enable_thinking` (unset, false, true), and the template's reasoning rules:
reasoning from `reasoning_content` or split out of `<think>…</think>` in the
content, kept only after the last user query (a user turn that is just a tool
response does not count), dropped before it. Tool definitions, tool calls and
`tool` messages are rejected.

Test: the 12 cases in `_chat.json` (multi-turn, reasoning before and after
the last query, several think tags, a second system message, a tool-response
turn, no user turn, empty and non-ASCII content), rendered text and its ids
exactly.

### 4. Kernels (`ops`) — done

- **Grouped-query attention:** `AttentionArgs.KVHeads` (0 means `Heads`, so
  BERT and T5 are unchanged). `AttentionStrided` packs each key/value head
  once for the query heads that share it; `NewKVCache` takes the key/value
  heads, so MiniCPM5's cache holds 2 heads instead of 16. On a decoding step
  (one query per head) a group's 8 query rows are consecutive, so they go
  through each product together and the cached keys and values are read once
  per group instead of once per head. This is bit-identical to one head at a
  time, since the kernels sum each row the same way whatever the row count.
- **RoPE:** `ops.NewRoPE(headDim, theta)`, rotate-half convention.
  `rope.Rotation(pos0, seq)` computes the cosines and sines once per forward
  pass; `Apply` rotates queries or keys in place, each value as torch's two
  rounded products and a rounded sum (explicit conversions stop Go's arm64
  compiler from fusing them). The inverse frequencies match transformers' for
  MiniCPM5 in 63 of 64 values; the 64th is 1 ulp away, from torch's float32
  pow.
- **SiLU and SwiGLU:** `ops.SiLU`, and `ops.SwiGLUInto(out, gateUp)` over the
  gate and up projections stacked in one Linear, so the feed-forward is two
  products.

Tests: grouped attention against the float64 definition and bit for bit
against attention on copied heads (per head, batched for one query, and
through a KVCache), including multi-query attention; RoPE against float64
rotations, one position at a time against a whole sequence (bitwise), the
relative-position property, and the inverse frequencies against the golden;
SwiGLU and SiLU against float64, including saturated and infinite inputs.
All pass natively, with `-tags purego`, with the race detector, and on amd64
AVX2.

### 5. The model (`model/llama`) — done

`llama.Load(dir)` reads `config.json` and the weights (`model.safetensors`
or the shards in `model.safetensors.index.json`), rejecting what it does not
implement (biases, rope_scaling, other activations). q/k/v and gate/up are
stacked into single products; bf16 weights go into bf16 panels (2.2 GB for
the 1B instead of 4.3), and the embedding table stays bf16, widened row by
row. `HiddenStates(ids)`, `LogitsInto`, `Generate(ids, maxNew)` and
`GenerateFunc` (a callback per token, which can stop early) share one
forward pass that runs from the current position over per-layer KV caches,
so a whole-sequence pass is the same code on a fresh state.

Tests against the goldens, all passing natively, with `-tags purego`, with
the race detector and on amd64 AVX2:

| Check | Worst difference |
|---|---|
| every layer's input and the final output, cases 0–1 | 5.8e-6 of the tensor's range |
| full logits where generation starts, cases 0–1 | within the same |
| top-64 logits at every position (the long case: every 64th prompt position and all generated ones) | 1e-5 of the row's range |
| greedy tokens with the cache, all 4 cases including 2,461 → 24 tokens | identical |
| each cached step's logits against the teacher-forced top-64 | 4.6e-6 |

Loading went from 9.7 s to 1.9 s (file in the page cache) by packing
narrow panels in parallel and widening bf16 through a direct view.

### 6. Speed

MiniCPM5-1B on an M2 Max, a 65-token prompt then 32 generated tokens,
against PyTorch 2.14.1 with the same thread count:

| | infer | PyTorch float32 | PyTorch bf16 |
|---|---|---|---|
| prompt, 1 thread | 2.32 s | 232 ms | 4.97 s |
| prompt, 12 threads | 374 ms | 241 ms | 873 ms |
| decode, 1 thread | 82 ms/token | 98 ms/token | 83 ms/token |
| decode, 12 threads | 29 ms/token (34 tokens/s) | 97 ms/token | 36 ms/token |

- **Generation is bandwidth-bound,** and already ahead of PyTorch: each
  token reads the 2.2 GB of weights (29 ms is about 75 GB/s, close to the
  one-row kernels' 65 GB/s in isolation). The 400 MB output layer is a
  fifth of that; int8 weights would halve it all again.
- **Prompts are compute-bound.** PyTorch's float32 prompt time does not
  change with threads because it runs on Apple's AMX matrix unit
  (Accelerate); that is the gap the Accelerate and Metal spikes in
  `benchcmp/` would close.

### 7. Decode profile

A decoding step was profiled op by op (MiniCPM5-1B, M2 Max). The five
matrix products take 96% of it, and attention, norms, RoPE and sampling the
rest. The products were the place to look.

- **Where the time went:** the products streamed weights at only 50–70 GB/s.
  - The kernels measured alone looked faster (92–121 GB/s), but that
    benchmark reused one matrix small enough for the 48 MB system cache.
  - Cycling through 1 GB of distinct weights, the bf16 kernel ran at 66 GB/s,
    against about 115 GB/s for plain reads on all cores.
- **The fix**, keeping every result bit-identical:
  - `gemv` reads each four-panel group start to finish, so the weights stream
    sequentially. It used to visit every panel of its chunk per 256-deep slice.
  - It cuts the work into four chunks per thread, claimed as threads free up,
    instead of one fixed chunk each, which the efficiency cores held back.
  - Together: 96 GB/s from DRAM.
- **Tried and removed:** a persistent worker pool that spins between jobs.
  Starting goroutines per product looked expensive in the CPU profile
  (`pthread_cond_wait`), but measured within one process the pool gained at
  most 3%, and longer spins were slower.

Measured old against new in the same process, best of 6:

| | before | after |
|---|---|---|
| MiniCPM5-1B bf16 | 27.5 ms/token | 22.2 ms/token (45 tokens/s), 1.24× |
| MiniCPM5-1B int8 | 24.5 ms/token | 20.6 ms/token (49 tokens/s), 1.19× |
| Qwen3-1.7B bf16 | 50.1 ms/token | 39.2 ms/token (25.5 tokens/s), 1.28× |

`BenchmarkDecode` in `model/llama` times decoding steps for each model.

## After MiniCPM5

- MiniCPM5-2B: loads unchanged; its 82 greedy tokens on three of the 1B
  prompts match PyTorch, with top-8 logits within 3e-6 of the range. Its chat
  template differs (`chat.MiniCPM5For2B`: every assistant turn keeps its
  reasoning, or gets an empty `<think>` block), with its own goldens from
  `dump_golden_llama.py --chat-only`. About 14.7 tokens/s on 12 threads. With
  thinking off it often declines simple questions, in PyTorch too. Still to do:
  layer-by-layer goldens and the speed table.
- Qwen3: RMSNorm on queries and keys per head before RoPE.
- SmolLM3: no RoPE on every 4th layer (`no_rope_layers`).
- Sampling: done, in the `sample` package and `Model.GenerateWith`.
  - **Settings:** temperature, top-k, top-p and repetition penalty, with
    transformers' meaning and order. `sample.LoadConfig` reads
    `generation_config.json` with transformers' defaults (MiniCPM5-1B:
    temperature 0.9, top-k 50, top-p 0.95; MiniCPM5-2B: temperature 1).
  - **Checked:** against transformers' logits processors on 21 cases
    (`scripts/dump_golden_sample.py`), including real MiniCPM5 logits over the
    full vocabulary. A draw takes 0.3 ms with the model defaults.
  - **Loops:** on MiniCPM5-2B with thinking on, sampling ends int8's greedy
    loop ("Or 18.6% growth. Or 18.6% increase." until the token limit). The
    growth-rate prompt still produces long, repetitive reasoning, in bf16 as
    well as int8. 5 seeds × 4 settings, capped at 500 tokens: many runs
    hadn't finished. Repetition penalties of 1.05 and 1.1 did not help; they
    also penalize the digits the answer repeats. An open-ended prompt finished
    cleanly in every run.
- int8 weights: done; see [int8-weights.md](int8-weights.md). Quality is
  close to bf16 (greedy tokens unchanged on every golden case). Decoding is
  only 1.13–1.14× faster on NEON, where widening int8 costs more
  instructions than bf16; amd64 is still to be measured. Next would be int8
  activations with `SDOT`.
- Prefix reuse: done, in `llama.Session` (with `ops.KVCache.Truncate`).
  Each call runs only what its prompt adds to the longest prefix it shares
  with the session's tokens, with logits bit-identical to a fresh run
  (checked against `GenerateFunc` on every golden case, bf16 and int8). On
  Qwen3-1.7B tool-calling conversations, where every round re-renders a
  2,000–7,500-token prompt of tools, four multi-round conversations went
  from 1,902 s to 389 s (one of them from 1,189 s to 59 s), with identical
  transcripts. Prefill itself is still 67–105 tokens/s, so a cold
  7,500-token prompt still takes about 2 minutes.
- A streaming `Generate` (a callback or iterator per token) for a goinfer
  endpoint.

## Open questions

- **CI:** `fetch_models.sh` with no arguments now downloads 3.2 GB, all
  cached in CI. The purego run on a 1B model is slow; the 2,461-token case
  should be skipped under `-short`.
- **Repo size:** the MiniCPM5 goldens add 10.7 MB. The license allows
  committing them; this is the largest golden set so far.
