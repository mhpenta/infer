# Qwen3.5 and OpenJev typed decisions

Status: Qwen3.5's text decoder runs on the CPU (`model/qwen35`), with the
sequence-classification head, and `openjev` makes typed decisions with
OpenJev's NLI checkpoints. Checked against PyTorch on OpenJev's 0.8B and its
4B v5, whose linear layers share each key head between two value heads.

## Why

[Jev](https://typesafe.ai) is a hosted decision model: given a state and
typed questions (a choice, a score, yes or no), it returns a probability for
each option without generating text. Open reproductions exist;
[AlexWortega/openjev](https://huggingface.co/AlexWortega/openjev) (MIT) is
the closest in quality. Its 4B v5 scores 0.814 on JevBench's public items
against Jev 1.13's 0.866, by its model card. Decisions need one forward pass
per option and no decoding, so they suit a CPU far better than chat does.

OpenJev is a Qwen3.5 decoder used as a cross-encoder:
`Qwen3_5ForSequenceClassification`, a 3-way head (contradiction, entailment,
neutral) on the last token of `"Premise: {premise}\nHypothesis: {hypothesis}"`.

## Qwen3.5's text decoder

| | Qwen3 | Qwen3.5 |
|---|---|---|
| token mixing | attention in every layer | linear attention (Gated DeltaNet) in 3 of every 4 layers, attention in the 4th |
| attention | GQA, q/k RMSNorm, RoPE on the whole head | the same, plus a sigmoid output gate (q_proj's rows alternate a head's queries and its gate), and RoPE on the first 25% of each head |
| RMSNorm | scale w | zero-centered: scale 1+w |
| rotary | 1D | 3D "mrope" sections, which reduce to 1D for text: every section sees the same position |

A linear layer projects to queries, keys and values (`in_proj_qkv`), z, b and
a. The queries, keys and values pass through a depthwise causal convolution
(kernel 4) and SiLU; queries and keys are L2-normalized per head and the
queries scaled by 1/√d. Then the gated delta rule:

    S ← exp(g)·S
    S ← S + k ⊗ (β·(v − Sᵀk))
    o ← Sᵀq

with β = sigmoid(b) and g = −exp(A_log)·softplus(a + dt_bias). Each head's
output is RMSNorm'd, scaled by the norm's weight and SiLU(z), and projected
back. S is a key-by-value matrix per value head: a fixed-size state instead
of a cache that grows with the sequence. When there are more value heads
than key heads (the 4B: 32 over 16), each group of value heads reads one key
head (`repeat_interleave`).

`ops.GatedDeltaRule` runs the recurrence one position at a time, value heads
in parallel. transformers runs prefill with the chunked form (64 positions
at a time, a triangular solve per chunk), which is the same math in another
order: the outputs stay within 4.4e-6 of PyTorch's through 2,575 positions.

### The tokenizer

transformers' Qwen3.5 tokenizer builds its own pre-tokenizer, with a split
pattern that keeps combining marks in words (`[\p{L}\p{M}]+`), and ignores
the Qwen3 pattern its checkpoints' `tokenizer.json` carries; Hindi or Bengali
text tokenizes differently under the two. `tokenizer_config.json` records the
one it uses as `pretokenize_regex`, and `bpe.LoadDir` uses it in place of
`tokenizer.json`'s (`bpe.Config.SplitPattern`). It matches Python on 425
tokenizer cases.

## Checked against PyTorch

`scripts/dump_golden_qwen35.py` writes the goldens, with the model in
float32 and eager attention. OpenJev 0.8B (`qwen3.5-0.8b-nli-v2s-long`,
`scripts/fetch_models.sh openjev-0.8b`, 1.7 GB, not in CI):

| Check | Result |
|---|---|
| every layer's input and the final output, 22 and 94 tokens | within 4.4e-6 of the tensor's range |
| the head's logits, 22 to 2,575 tokens | within 2.5e-6 |
| options scored from their shared prefix (`ClassifyShared`) | bit-identical to scoring each alone |
| 3 decisions (`openjev.Decide`): hypothesis text, ids, probabilities | identical text and ids; probabilities within 1e-5 |
| int8 weights | probabilities within 0.0031 of bf16 |

OpenJev 4B v5 (`qwen3.5-4b-nli-v5`, `scripts/fetch_models.sh openjev-4b`,
9.1 GB, not in CI; `--model-only` goldens, 7.4 MB: hidden states for one
case). PyTorch in float32 peaked at 21 GB.

| Check | Result |
|---|---|
| every layer's input and the final output, 22 tokens | within 7.9e-6 of the tensor's range |
| the head's logits, 22 to 2,575 tokens | within 2.6e-6 |
| options scored from their shared prefix | bit-identical to scoring each alone; logits within 1e-5 |

## Decisions

`openjev.Decide` follows OpenJev's `code/openjev_decide.py`: each option
becomes the hypothesis `The answer to "{instructions}" is {option}:
{rubric}`, its P(entailment) is normalized over the question's options, and a
state over 24,000 characters is scored in overlapping windows, taking each
option's best. Every option of every question shares the text up to the
hypothesis, so the prefix runs once: a 3-option decision takes 0.66 s instead
of 1.35 s. The prefix is found in the token ids, since the tokenizer can merge
characters across the boundary.

Unlike the Python code, nothing is truncated: Python cuts inputs at 4,096
tokens, which for a full 24,000-character window removes the hypothesis.

## Speed

M2 Max CPU, bf16, with the machine busy (load average 4–5):

| | |
|---|---|
| 0.8B, 22 / 94 / 2,575 tokens | 123 / 157 / 144 tokens/s |
| 0.8B, 3-option decision, 47-token state | 0.66 s |
| 4B, 3 questions of 2–3 options, a 55-token state (`cmd/decide`) | 6.7 s |

The products take 65% of the time and the delta rule 18%: it runs in scalar
Go. int8 weights are no faster here, since prompts are bound by arithmetic.

## Quality: test before relying on it

The 0.8B answers policy questions confidently wrong; the 4B gets them right.
Given "refunds require a receipt and purchase within 30 days", asked whether
the refund is permitted (rubric: "A condition is missing" or "Every condition
holds"):

| Customer | 0.8B | 4B |
|---|---|---|
| receipt, bought 12 days ago | no (1.000) | **yes** (0.998) |
| receipt, bought 45 days ago | no (1.000) | no (1.000) |
| no receipt, bought 12 days ago | no (1.000) | no (1.000) |

PyTorch gives the 0.8B's same 0.9997 on the first, so this is the model, not
infer. Even the 4B scores 0.62 on JevBench's hard tier, by its model card, so
measure a checkpoint on your own decisions before relying on it. On a CPU the
4B takes seconds per decision; it wants a GPU for anything interactive.

## Not done

- **The GPU.** The other decoders run on Metal; Qwen3.5 needs a delta-rule
  kernel and the convolution there.
- **A faster delta rule on the CPU**: NEON/AVX2 inner loops, or the chunked
  form, whose work is matrix products.
- **Generation.** The language-model head is not loaded, and the chat
  template is not in `tokenizer/chat`.
- **JSON states.** Python's `json.dumps` writes `", "` and `": "` separators
  and keeps key order; pass a state as the text you want the model to read.
