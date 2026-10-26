"""
Dump reference outputs for mxbai-embed-large-v1 used as 512-dimension binary
embeddings, so the Go implementation can be checked.

    python scripts/dump_golden_mxbai.py <model_dir> <out_prefix>
    python scripts/dump_golden_mxbai.py testdata/models/mxbai-embed-large-v1 testdata/golden/mxbai

Writes:
  <out_prefix>_tokens.json        text -> input ids for tokenizer edge cases
  <out_prefix>_model.json         model cases: text, ids and the binary output
  <out_prefix>_model.safetensors  per case i: case{i}.hidden.{j} for j in
                                  HIDDEN_LAYERS (0 is the embedding output),
                                  case{i}.cls and case{i}.mrl (normalized,
                                  first 512 dims)
  <out_prefix>_binary.json        text -> binary output for a wider set of
                                  texts, to measure sign flips

The binary output is sentence-transformers' recipe: SentenceTransformer.encode with
normalize_embeddings=True, the first 512 dimensions (Matryoshka), then
quantize_embeddings(precision="binary").
"""
import json
import os
import sys

import torch
from safetensors.torch import save_file
from sentence_transformers import SentenceTransformer
from sentence_transformers.quantization import quantize_embeddings
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dump_golden_bert import LONG_TEXT, MODEL_CASES, TOKENIZER_CASES  # noqa: E402

HIDDEN_LAYERS = [0, 1, 12, 24]
MRL_DIMS = 512

# Typical inputs: search queries, filing sentences, table rows.
BINARY_CASES = [
    "apple dividend",
    "revenue recognition policy changes",
    "Total revenue increased 8% to $94.9 billion, driven by Services.",
    "The Company repurchased $23.5 billion of its common stock during the quarter.",
    "Operating income | 2023 | 2022 | 2021\nProducts | 114,301 | 123,216 | 105,126",
    "Risk Factors: The Company's business can be impacted by political events, trade disputes and pandemics.",
    "We expect gross margin to be between 45.5% and 46.5%.",
    "Net cash provided by operating activities was $110.5 billion.",
    "EBITDA margins (ex-SBC) ~23–25% — guidance reiterated…",
    "naïve café résumé — 中文 and emoji 😀",
    "hello",
    "",
] + MODEL_CASES


def binary(model, texts):
    embeddings = model.encode(texts, normalize_embeddings=True)[..., :MRL_DIMS]
    return quantize_embeddings(embeddings, precision="binary").tolist()


def main():
    if len(sys.argv) != 3:
        sys.exit(f"usage: python {sys.argv[0]} <model_dir> <out_prefix>")
    model_dir, out = sys.argv[1], sys.argv[2]

    st = SentenceTransformer(model_dir, device="cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True, torch_dtype=torch.float32)
    model.eval()

    tokens = [
        {"text": t, "ids": tokenizer(t, truncation=True, max_length=st.max_seq_length)["input_ids"]}
        for t in TOKENIZER_CASES + [LONG_TEXT]
    ]
    with open(f"{out}_tokens.json", "w") as f:
        json.dump(tokens, f, ensure_ascii=False, indent=1)

    tensors = {}
    cases = []
    for i, text in enumerate(MODEL_CASES):
        enc = tokenizer(text, truncation=True, max_length=st.max_seq_length, return_tensors="pt")
        with torch.no_grad():
            output = model(**enc, output_hidden_states=True)
        for j in HIDDEN_LAYERS:
            tensors[f"case{i}.hidden.{j}"] = output.hidden_states[j][0].clone()
        cls = output.last_hidden_state[0, 0]
        tensors[f"case{i}.cls"] = cls.clone()
        tensors[f"case{i}.mrl"] = torch.nn.functional.normalize(cls, dim=0)[:MRL_DIMS].contiguous()
        cases.append({"text": text, "ids": enc["input_ids"][0].tolist(), "binary": binary(st, [text])[0]})

    save_file(tensors, f"{out}_model.safetensors")
    with open(f"{out}_model.json", "w") as f:
        json.dump(cases, f, ensure_ascii=False, indent=1)
    with open(f"{out}_binary.json", "w") as f:
        json.dump([{"text": t, "binary": b} for t, b in zip(BINARY_CASES, binary(st, BINARY_CASES))],
                  f, ensure_ascii=False, indent=1)
    print(f"wrote {len(tokens)} tokenizer cases, {len(cases)} model cases, {len(BINARY_CASES)} binary cases to {out}_*")


if __name__ == "__main__":
    main()
