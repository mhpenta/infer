# Qwen3 in infer

Qwen3-1.7B as the second decoder in `model/llama`, checked against PyTorch the
same way as MiniCPM5. It is a widely used small model, good at instruction
following and tool calling. Supporting it adds the machinery MiniCPM5 did not
need: query/key normalization, NFC text normalization, a different digit
split, tied embeddings, and (as a later step) tool calls.

| | MiniCPM5-1B | Qwen3-1.7B |
|---|---|---|
| architecture | `LlamaForCausalLM` | `Qwen3ForCausalLM` |
| hidden, layers | 1536, 24 | 2048, 28 |
| heads (query / key-value), head dim | 16 / 2, 128 | 16 / 8, 128 |
| feed-forward | 4608 | 6144 |
| vocabulary | 130,560 | 151,936 |
| RoPE theta | 5e6 | 1e6 |
| output layer | separate | tied to the embeddings |
| per-head RMSNorm on q and k, before RoPE | no | yes (`q_norm`, `k_norm`) |
| weights | bf16, 2.2 GB | bf16, 4.1 GB (3.4 GB packed: the tied output layer is shared) |
| end tokens (`generation_config.json`) | `</s>`, `<\|im_end\|>` | `<\|im_end\|>`, `<\|endoftext\|>` (config.json lists only the first) |
| sampling defaults | temperature 0.9, top-p 0.95 | temperature 0.6, top-k 20, top-p 0.95 |

Pinned: `Qwen/Qwen3-1.7B` at `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`
(Apache 2.0), `scripts/fetch_models.sh qwen3`.

## Differences that need code

**Tokenizer** (`tokenizer/bpe`). This is the same byte-level BPE as
MiniCPM5's, with three changes:
- **NFC normalizer** before splitting. Our loader rejects normalizers today.
  Use `golang.org/x/text/unicode/norm` (the Go team's implementation).
  The risk is a Unicode version gap between it and Rust's
  `unicode-normalization`; the random-string goldens include combining marks
  to catch that.
- **Digits:** the split pattern is MiniCPM5's with `\p{N}` (one digit per
  piece) in place of `\p{N}+`, and no separate digit pre-split. It is one
  more entry in `splitPatterns`, with the hand-written word scanner
  parameterized on digit runs.
- **Post-processor:** `ByteLevel`, which changes nothing for ids. No
  template, and no BOS token.

**Chat template** (`tokenizer/chat`). It is embedded in
`tokenizer_config.json` (sha256 `a55ee1b1…`), which `chat.Load` already
reads. It differs from MiniCPM5-1B's in three ways:
- no `<s>`;
- after the last user query, the final assistant turn always gets a
  `<think>` block, even an empty one;
- `enable_thinking=true` adds nothing to the generation prompt (the model
  opens `<think>` itself), while `false` adds an empty block.

Tools come later (step 5).

**Model** (`model/llama`):
- `model_type: qwen3`: an RMSNorm over each head's 128 values of q and k,
  with weights `self_attn.q_norm.weight` and `self_attn.k_norm.weight`,
  after the projections and before RoPE.
- Tied embeddings: the bf16 embedding rows and the packed output layer
  come from one tensor (already supported).
- End tokens from `generation_config.json`, which transformers' `generate`
  uses, rather than only `config.json`.

## Steps

1. **Goldens:** done. `dump_golden_llama.py` on Qwen3-1.7B wrote
   `testdata/golden/qwen3_1.7b_*`:
   - 22 tokenizer cases plus 400 random ones, and 12 chat cases;
   - 4 model cases with hidden states and top-64 logits. Greedy, PyTorch
     answers " Paris. The capital of Italy is Rome", "391", a faithful
     one-sentence summary, and the long prompt's summary at 2,579 tokens in.
   - 14.3 MB in all.

   The raw-text case now starts with the tokenizer's BOS token, if it has
   one. MiniCPM5's goldens regenerate byte-identical.
2. **Tokenizer:** done.
   - **NFC:** `golang.org/x/text/unicode/norm`, applied between added tokens.
   - **Split:** `splitWords(false)`, Qwen's \p{N}-per-piece variant of the
     word scanner.
   - **Post-processor:** ByteLevel, which adds nothing.
   - **Checked:** all 422 token cases and 12 chat texts match transformers.
     Turning NFC off fails 45 of the random cases.
   - **Stress test:** 654 KB of docs, Go source, multilingual text and an
     NFD-decomposed copy of it. All 224,870 ids are identical to Rust
     tokenizers', encoded in 43 ms.
   - **Digits:** the single-digit split cannot change ids with this
     vocabulary. Qwen3 has no merges between digits, so a run of digits ends
     up as single-digit tokens either way. `TestSplitWordsSingleNumbers`
     checks the scanner itself.
3. **Chat template:** done. `chat.Qwen3`, recognized by its hash in
   `tokenizer_config.json`.
   - **One renderer:** the three ChatML templates (MiniCPM5-1B, MiniCPM5-2B,
     Qwen3) now share `renderChatML`, with a style choosing the opening token,
     which assistant turns get a `<think>` block, and what `ThinkingOn` adds.
   - **Checked:** all 12 conversations match transformers' text and ids.
   - **Mutations:** each of Qwen3's three rules, mutated on its own, fails 2–3
     of them: the final turn's think block, the think block for reasoning,
     and the empty generation prompt.
4. **Model:** done. `model_type: qwen3` loads in `model/llama`.
   - **q/k norms:** each layer's `q_norm`/`k_norm` RMSNorm runs over each
     head's 128 values after the projection, before RoPE.
   - **End tokens:** read from `generation_config.json`
     (`<|im_end|>`, `<|endoftext|>`), as `generate` does.
   - **Checked against PyTorch:**

     | Check | Result |
     |---|---|
     | every layer's input and the final output, cases 0–1 | within 7.3e-6 of the tensor's range |
     | top-64 logits at every position (the long case: every 64th prompt position and all generated ones) | within 1.2e-5 after position 0 |
     | greedy tokens with the cache, all 4 cases including 2,579 → 24 | identical |
     | each cached step's logits | within 4.2e-6 |

   - **Position 0 tolerance:** the first token is an attention sink. Its
     residual values reach 12,000–15,000 in the middle layers, against about
     100 elsewhere, and its logits are 2.2e-5 of their range from PyTorch's.
     That position alone gets a tolerance of 3e-5; the rest keep 2e-5.
   - **Speed** (M2 Max, 12 threads, the 66-token summary prompt, on a busy
     machine with load average about 11):

     | | prompt | decode |
     |---|---|---|
     | infer bf16 | 625 ms | 49.8 ms/token (20.1 tokens/s) |
     | infer int8 | 715 ms | 42.0 ms/token (23.8 tokens/s) |
     | PyTorch 2.14.1 bf16 | 1,435 ms | 53.7 ms/token (18.6 tokens/s) |
     | PyTorch float32 | 441 ms | 237 ms/token (likely swapping: 7 GB of float32 weights) |

   - **Playground:** runs Qwen3 with its own sampling defaults. The server
     tells the page whether the prompt opened `<think>`; Qwen3 opens it
     itself. "18.6%" after about 2,200 characters of reasoning, at
     16 tokens/s on the busy machine.
5. **Tool calls:** done.
   - **API:** `chat.Options.Tools` (JSON tool definitions), `Message.ToolCalls`
     and the `"tool"` role render in `chat.Qwen3`, and
     `Template.ParseToolCalls` reads `<tool_call>{json}</tool_call>` blocks
     back from a reply.
   - **tojson:** the template writes JSON with transformers' tojson, which
     is Python's `json.dumps(ensure_ascii=False)`. `pythonJSON` reproduces
     it: key order kept, `", "` and `": "` separators, only quotes,
     backslashes and control characters escaped, integers exact, floats as
     Python's repr (`1e-05`, `100.0`, `1e+16`). It matches Python on 19,992
     random numbers across every magnitude.
   - **Checked:** 8 conversations from `dump_golden_llama.py --tools` match
     transformers' text and ids. They cover tool definitions with and without
     a system message; calls with and without the `function` wrapper; string
     and nested arguments; text and reasoning before calls; consecutive tool
     responses; and tool turns before the last user query. Mutating the
     newline before a call, or the grouping of tool responses, fails the case
     that covers it.
   - **Real use:** asked whether Paris or Oslo is warmer, Qwen3-1.7B made two
     correct parallel `get_weather` calls, with thinking on or off. Both
     answers used the tool results. With thinking off it then concluded
     "Oslo is warmer" from 3°C against 14°C; with thinking on it answered
     correctly.
   - **Not supported:** MiniCPM5's templates have their own XML tool format,
     which is not implemented; they reject tools.

## Qwen3-4B

The next size up, and the largest model tested so far. Pinned:
`Qwen/Qwen3-4B` at `1cfa9a7208912126459214e8b04321603b3df60c`,
`scripts/fetch_models.sh qwen3-4b` (8 GB, not in "all").

| | Qwen3-1.7B | Qwen3-4B |
|---|---|---|
| hidden, layers | 2048, 28 | 2560, 36 |
| heads (query / key-value) | 16 / 8 | 32 / 8 |
| feed-forward | 6144 | 9728 |
| weights | 4.1 GB (3.4 GB packed) | 8.0 GB (7.9 GB packed) |

Its `tokenizer.json`, `tokenizer_config.json` (with the chat template),
`vocab.json`, `merges.txt` and `generation_config.json` are byte-identical to
Qwen3-1.7B's, so its goldens are the model's alone:
`dump_golden_llama.py --model-only` writes `testdata/golden/qwen3_4b_model.*`
(15.6 MB), with hidden states for case 1 only. PyTorch in float32 peaked at
21 GB. It loads with no code changes.

**Checked against PyTorch** (`TestMatchesPython/qwen3-4b`, skipped with
`-short`), on the CPU and Metal:

| Check | CPU / Metal |
|---|---|
| every layer's input and the final output, case 1 | within 5.4e-6 of the tensor's range (Metal) |
| top-64 logits at every position, including position 0 | within 6.4e-6 |
| greedy tokens with the cache, all 4 cases including 2,579 → 24 | identical |
| sessions reusing a prefix | the same tokens; bit-identical to an empty cache on the CPU |

PyTorch's answers: " Paris. The capital of Germany is Berlin", "391", and
one-sentence summaries of both summary prompts. The 4:1 query-to-key-value
grouping (1.7B's is 2:1) needed nothing new.

**Speed** (M2 Max, 32 GB, `BenchmarkDecodeWeights` and `BenchmarkPrefill`,
the 66-token summary prompt; the machine was busy, with 13–17 GB of swap
in use):

| | Qwen3-1.7B | Qwen3-4B |
|---|---|---|
| decode, Metal bf16 | 13–18 ms/token | 29–37 ms/token (27–34 tokens/s) |
| decode, Metal int8 | 9.3 ms/token | 17.5 ms/token (57 tokens/s) |
| decode, CPU bf16 | 40–42 ms/token | 126–495 ms/token (see below) |
| decode, CPU int8 | 27.5 ms/token | 70.5 ms/token (14 tokens/s) |
| prefill, Metal, 66 / 2,579 tokens | 122 ms / 2.5 s | 260 ms / 6.8 s (381 tokens/s) |
| prefill, CPU, 66 / 2,579 tokens | 0.87 s / 24 s | 1.5 s / 72 s |

Metal streams the 4B's weights at about 215 GB/s (1.7B: about 195), and
int8 on the CPU scales with the weights' size (2.6× the 1.7B's time for
2.3× the bytes). bf16 on the CPU does not: three runs gave 178, 495 and
126 ms/token. During the slowest, `vm_stat` showed about 64 MB free and the
compressor moving 8–16 GB every 5 seconds. Its 7.9 GB of packed weights are
anonymous memory, which macOS compresses and swaps when the machine is
short of memory. Metal's buffers and int8's 4 GB stayed resident. On a
32 GB Mac that runs other large programs, run Qwen3-4B on the GPU, or with
int8 weights on the CPU.

## Open questions

- **Repo size:** Qwen3's goldens will add roughly 14 MB (hidden states for
  29 × 2048 values per position), on top of MiniCPM5's 10.7 MB. Storing
  hidden states for fewer layers, or as bf16, would cut that.
- **CI:** the checkpoint is 4.1 GB. Should it be in `fetch_models.sh all`,
  like MiniCPM5-1B, or opt-in, like MiniCPM5-2B?
