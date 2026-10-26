"""
Dump reference outputs from PyTorch for a BERT embedding model so the Go
implementation can be checked token-for-token and layer-by-layer.

    python scripts/dump_golden_bert.py <model_dir> <out_prefix>
    python scripts/dump_golden_bert.py testdata/models/all-MiniLM-L6-v2 testdata/golden/minilm

Writes:
  <out_prefix>_tokens.json        text -> input ids for tokenizer edge cases
  <out_prefix>_model.json         the model cases (text + ids), in order
  <out_prefix>_model.safetensors  per case i: case{i}.hidden.{j} for every
                                  hidden state (j=0 is the embedding output)
                                  and case{i}.mean_pooled

Uses the plain transformers path: AutoTokenizer with
truncation=True, AutoModel, masked mean pooling, no normalization.
"""
import json
import sys

import torch
from safetensors.torch import save_file
from transformers import AutoModel, AutoTokenizer

TOKENIZER_CASES = [
    "",
    "hello",
    "Hello, World!",
    "The quick brown fox jumps over the lazy dog.",
    "Revenue grew 12.5% year-over-year to $1,234.56 million in Q3 FY2024.",
    "naïve café résumé Ångström façade",
    "ÀÉÎÕÜ çñ ß ﬁ İstanbul",
    "中文字符 and 日本語 mixed with English",
    "emoji 😀 and symbols ™ © ® ° ± § ¶ €",
    "tabs\tand\nnewlines\r\nand   multiple    spaces",
    "control\x00chars\x07here​zero﻿width�replacement",
    "don't won't it's y'all",
    "e-mail: test@example.com, url: https://example.com/a?b=c&d=e",
    "unaffable antidisestablishmentarianism supercalifragilisticexpialidocious",
    "a" * 150,
    "<html><body>[CLS] [SEP] [MASK] [PAD]</body></html>",
    "EBITDA margins (ex-SBC) ~23–25% — guidance reiterated…",
    "Ω ≈ ç √ ∫ ∑ ∂ ∆ ∏ π",
    "Привет мир, Γειά σου Κόσμε, مرحبا بالعالم, שלום עולם",
    "한국어 텍스트 ภาษาไทย हिन्दी",
]

LONG_TEXT = " ".join(
    [
        "Net sales increased primarily due to higher iPhone and Services revenue,",
        "partially offset by lower Mac and iPad sales; gross margin percentage",
        "expanded 120 basis points, driven by cost savings and a favorable mix.",
    ]
    * 30
)

MODEL_CASES = [
    "hello",
    "Revenue grew 12.5% year-over-year to $1,234.56 million in Q3 FY2024.",
    "naïve café résumé — 中文 and emoji 😀",
    LONG_TEXT,  # longer than 512 tokens: exercises truncation and every position
]


def mean_pooling(token_embeddings, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)


def main():
    if len(sys.argv) != 3:
        sys.exit(f"usage: python {sys.argv[0]} <model_dir> <out_prefix>")
    model_dir, out = sys.argv[1], sys.argv[2]

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True)
    model.eval()

    tokens = [
        {"text": t, "ids": tokenizer(t, truncation=True)["input_ids"]}
        for t in TOKENIZER_CASES + [LONG_TEXT]
    ]
    with open(f"{out}_tokens.json", "w") as f:
        json.dump(tokens, f, ensure_ascii=False, indent=1)

    tensors = {}
    cases = []
    for i, text in enumerate(MODEL_CASES):
        enc = tokenizer(text, truncation=True, return_tensors="pt")
        with torch.no_grad():
            output = model(**enc, output_hidden_states=True)
        for j, hidden in enumerate(output.hidden_states):
            tensors[f"case{i}.hidden.{j}"] = hidden[0].contiguous()
        tensors[f"case{i}.mean_pooled"] = mean_pooling(output[0], enc["attention_mask"])[0].contiguous()
        cases.append({"text": text, "ids": enc["input_ids"][0].tolist()})

    save_file(tensors, f"{out}_model.safetensors")
    with open(f"{out}_model.json", "w") as f:
        json.dump(cases, f, ensure_ascii=False, indent=1)
    print(f"wrote {len(tokens)} tokenizer cases and {len(cases)} model cases to {out}_*")


if __name__ == "__main__":
    main()
