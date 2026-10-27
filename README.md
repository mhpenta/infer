# infer

Transformer inference on the CPU in Go: load Hugging Face weights, tokenize,
run the model, pool or quantize the result. No cgo, no ONNX, no Python.

Outputs match PyTorch: every model is checked layer by layer against
reference outputs dumped from `transformers` (see [Testing](#testing)).

## What it runs today

| Model | Output | Check against PyTorch |
|-------|--------|-----------------------|
| [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (BERT, 22M) | 384-dim mean-pooled embedding | every layer within 5e-6, cosine 1.000000000 |
| [mxbai-embed-large-v1](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1) (BERT, 335M, fp16) | 512-bit binary embedding (CLS, normalize, Matryoshka 512, sign bits) | bit-identical on 20 texts |
| [t5-small](https://huggingface.co/google-t5/t5-small) (T5, 60M) and T5 fine-tunes | greedy generation with a KV cache | every encoder and decoder layer within 1.5e-6 of its range; generated tokens identical on 5 inputs up to 512 tokens |
| [MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B) (Llama architecture, 1.1B, bf16) | greedy generation with a KV cache, chat template | every layer within 5.8e-6 of its range; top-64 logits within 1e-5 at every position; generated tokens identical on 4 prompts up to 2,461 tokens |
| [MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B) (same architecture, 42 layers, bf16) | the same, with its own chat template | chat template matches transformers on 12 conversations; generated tokens identical on 3 prompts (not in CI: 5 GB, `scripts/fetch_models.sh minicpm5-2b`) |
| [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) (`Qwen3ForCausalLM`: per-head q/k RMSNorm, tied embeddings, bf16) | greedy or sampled generation, chat template, NFC tokenizer | every layer within 7.3e-6 of its range; top-64 logits within 1.2e-5 after the first position (3e-5 there, an attention sink); generated tokens identical on 4 prompts up to 2,579 tokens (not in CI: 4.1 GB, `scripts/fetch_models.sh qwen3`) |

## Packages

| Package | |
|---------|---|
| `tensor` | dense, row-major float32 tensors |
| `ops` | matrix products and the rest of the kernels: LayerNorm, GELU, softmax, attention |
| `safetensors` | memory-mapped `.safetensors` reader (F32 zero-copy, F16/BF16 upcast) |
| `tokenizer/wordpiece` | BERT's tokenizer, matching Hugging Face's fast `BertTokenizer` |
| `tokenizer/unigram` | SentencePiece Unigram from `tokenizer.json` (T5), including the precompiled NFKC character map |
| `tokenizer/bpe` | byte-level BPE from `tokenizer.json` (MiniCPM5, Qwen3 and other GPT-2-style tokenizers): NFC normalization, the GPT-4-style split patterns as hand-written scanners |
| `tokenizer/chat` | chat templates in Go, recognized by the hash of the model's Jinja template (MiniCPM5-1B and -2B, Qwen3), with Qwen3 tool calling: tool definitions, tool calls and responses, and parsing calls from a reply |
| `model/bert` | the BERT encoder |
| `model/t5` | the T5 encoder-decoder: relative position bias, cross-attention, greedy decoding with a KV cache |
| `model/llama` | Llama-family decoders (`LlamaForCausalLM`, `Qwen3ForCausalLM`): grouped-query attention, rotary embeddings, SwiGLU, bf16 or int8 weights (`llama.LoadWith(dir, llama.Options{Int8: true})`), greedy or sampled decoding with a KV cache |
| `sample` | choosing the next token: greedy, or temperature, top-k, top-p and repetition penalty with transformers' meaning and order, defaults read from `generation_config.json`; the distributions match transformers' logits processors |
| `pool` | mean and CLS pooling, L2 normalization |
| `quant` | sentence-transformers' binary quantization |
| `cmd/embed` | a small CLI: `go run ./cmd/embed <model_dir> "some text"` |

```go
tok, err := wordpiece.LoadDir(dir) // vocab.txt + tokenizer_config.json
model, err := bert.Load(dir)       // config.json + model.safetensors
defer model.Close()

hidden, err := model.Forward(tok.Encode("Revenue grew 12% year over year."))
embedding := pool.Mean(hidden)
```

```go
tok, err := unigram.LoadDir(dir) // tokenizer.json + tokenizer_config.json
model, err := t5.Load(dir)       // config.json + model.safetensors
defer model.Close()

out, err := model.Generate(tok.Encode("translate English to German: The house is wonderful."), 64)
fmt.Println(tok.Decode(out, true)) // Das Haus ist wunderbar.
```

```go
tok, err := bpe.LoadDir(dir)    // tokenizer.json + tokenizer_config.json
tmpl, err := chat.Load(dir)     // chat_template.jinja, recognized by its hash
model, err := llama.Load(dir)   // config.json + safetensors shards
defer model.Close()

prompt, err := tmpl.Render([]chat.Message{{Role: "user", Content: "What is 17 * 23? Answer with just the number."}},
	chat.Options{AddGenerationPrompt: true, Thinking: chat.ThinkingOff})
out, err := model.Generate(tok.EncodeRaw(prompt), 64)
fmt.Println(tok.Decode(out, true)) // 17 * 23 = 391

// Or sample, with the model's own settings (MiniCPM5-2B: temperature 1, top-k 50, top-p 0.95).
cfg, err := sample.LoadConfig(dir) // generation_config.json, with transformers' defaults
s, err := sample.New(cfg, seed)
err = model.GenerateWith(tok.EncodeRaw(prompt), 512, s, func(token int, _ []float32) bool {
	out = append(out, token)
	return true
})
```

## How the matrix products work

`ops` follows the design of BLIS and OpenBLAS. Weights are packed once, at load
time, into 16-column panels; inputs are packed per product into 6-row strips;
an outer-product microkernel keeps a 6×16 block of the output in registers,
and loops are blocked for the caches.

| CPU | Microkernel | Notes |
|-----|-------------|-------|
| x86-64 with AVX2 + FMA | 6×16, 12 YMM accumulators | |
| x86-64 with AVX-512 | 12×32, 24 ZMM accumulators | enabled only after a startup self-check against the portable kernel passes (`ops.KernelCheck`) |
| arm64 | 6×16 NEON, 24 accumulators | |
| anything else, or `-tags purego` | portable Go | |

Weights stored as fp16 (like mxbai's) or bf16 (like most language models) can
stay in that format in memory (`ops.PackHalf`, `ops.PackBF16`) and be widened
to float32 in registers: half the memory and bandwidth, the same float32
values, so outputs are bit-identical. fp16 is widened by the x86 kernels; bf16
by the x86 and NEON kernels. Any weight can instead be quantized to int8
(`ops.PackInt8`, one scale per output column), which halves bf16's memory
again; products are bit-identical to the float32 product on the integers,
then scaled. The x86 and NEON kernels widen int8 in registers, but on NEON
that costs enough instructions that generation is only 1.13–1.14× faster
than bf16 (see [docs/int8-weights.md](docs/int8-weights.md), which also has
the quality measurements). Softmax and GELU use AVX2 vector loops on
x86. `ops.PeakGFLOPS`, `ops.KernelGFLOPS` and `ops.MatMulGFLOPS` measure the
kernels on the machine they run on.

A product with a single row (each decoding step) skips the strip packing and
runs a matrix-vector kernel that streams four panels at a time; it sums in the
same order as the matrix kernels, so a row gives the same bits either way.
Each group of four panels is read start to finish, so the weights stream from
memory sequentially, and the work is cut into four chunks per thread, claimed
as threads free up, so Apple's slower efficiency cores do not hold back the
rest. Decoding is bound by memory bandwidth: on an M2 Max the bf16 kernel
streams about 96 GB/s from DRAM, against roughly 115 GB/s for plain reads.
`ops.KVCache` keeps attention keys and values in the packed layout the kernels
read, appended once per position, so a decoder step packs nothing but its own
new key and value.

The amd64 assembly is generated by `scripts/gen_amd64_kernels.py`.

### T5 generation speed

t5-small, greedy, one thread on an M2 Max, against PyTorch 2.1.1 (which uses
Apple's Accelerate BLAS, running large products on the AMX matrix unit):

| Input / output tokens | infer | PyTorch |
|-----------------------|-------|---------|
| 11 / 6 | 42 ms | 50 ms |
| 19 / 31 | 179 ms | 226 ms |
| 45 / 31 | 197 ms | 233 ms |
| 50 / 62 | 361 ms | 457 ms |
| 512 / 41 | 845 ms | 398 ms |

Decoding is faster than PyTorch; a long input's encoder pass (570 ms of the
last row) is where Accelerate's AMX products pull ahead.

## Testing

Weights are not checked in. Fetch the pinned revisions (with hash checks) into
`testdata/models`, then run the tests; tests that need a model skip without it.

```bash
scripts/fetch_models.sh
```

```bash
go test ./...
```

`-tags purego` runs everything on the portable kernels. On Apple silicon, an
amd64 Docker container exercises the AVX2 kernels (Rosetta exposes AVX2, not
AVX-512); `INFER_REQUIRE_AVX2=1` makes the tests fail rather than skip without
it:

```bash
docker run --rm --platform linux/amd64 -v "$PWD":/src -w /src -e INFER_REQUIRE_AVX2=1 golang:1.25 go test -short ./...
```

The reference outputs in `testdata/golden` come from `scripts/dump_golden_bert.py`,
`scripts/dump_golden_mxbai.py`, `scripts/dump_golden_t5.py`,
`scripts/dump_golden_llama.py` and `scripts/dump_golden_sample.py` (PyTorch +
transformers). `benchcmp/` is a
separate module comparing the matrix product with gonum and vek.

## License

MIT
