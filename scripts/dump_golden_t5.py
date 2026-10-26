"""
Dump reference outputs for a T5 checkpoint, so the Go implementation can be
checked token for token and layer by layer.

    python scripts/dump_golden_t5.py <model_dir> <out_prefix>
    python scripts/dump_golden_t5.py testdata/models/t5-small testdata/golden/t5small

Writes:
  <out_prefix>_tokens.json        tokenizer edge cases: text, ids, and the ids
                                  decoded back (skipping special tokens)
  <out_prefix>_tokens_nonorm.json the same with the tokenizer's normalizer
                                  removed, as some re-saved T5 tokenizers have
  <out_prefix>_buckets.json       relative-position buckets for positions
                                  -1024..1024, bidirectional (encoder) and not
                                  (decoder), from the model's own function
  <out_prefix>_model.json         per case: text, input ids, generated ids,
                                  decoded output
  <out_prefix>_model.safetensors  per case i:
    case{i}.enc.hidden.{j}  encoder hidden states, j = 0..L: j < L is the
                            input to encoder layer j (0 = embeddings), L is
                            the final, normalized encoder output
    case{i}.dec.hidden.{j}  the same for the decoder, run once over the
                            generated sequence with the true previous tokens
                            (teacher forcing), so no KV cache is involved
    case{i}.dec.logits      logits for the first LOGIT_STEPS decoder positions

Generation is greedy (no beams, no sampling), up to MAX_NEW_TOKENS. The script
checks that the teacher-forced argmax reproduces greedy generation, which is
what a cache-free decoder has to match before a KV cache is added.
"""
import json
import sys

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5Attention

MAX_NEW_TOKENS = 64
LOGIT_STEPS = 4

TOKENIZER_CASES = [
    "",
    "Hello",
    "translate English to German: The house is wonderful.",
    "Total revenue,\"$1,234.5\",(12.3)%",
    "tabs\tand\nnewlines\r\nand   multiple    spaces",
    "  leading and trailing  ",
    "naïve café résumé — “quotes” … ™ © ® ° ± §",
    "non breaking thin　ideographic space",
    "ＦＵＬＬＷＩＤＴＨ ｔｅｘｔ ①②③ ﬁ ﬂ ㎏ ½",
    "decomposed é and composed é",
    "<extra_id_0> </s> <pad> <unk> literal specials",
    "emoji 😀 and 中文 and Привет",
    "a" * 300,
    "Ünïcödé ÀÉÎÕÜ ß œ æ ø å",
]

MODEL_CASES = [
    "translate English to German: The house is wonderful.",
    "translate English to French: The quarterly results exceeded expectations, and the board raised the dividend.",
    "summarize: Net sales increased 8% to $94.9 billion, driven by record Services revenue and strong iPhone demand, while gross margin expanded to 46.3%. The company returned over $29 billion to shareholders during the quarter.",
    ",2024,2023\nNet sales,\"$383,285\",\"$394,328\"\nCost of sales,\"214,137\",\"223,546\"\nGross margin,\"169,148\",\"170,782\"",
    "summarize: " + " ".join(["Revenue grew in every segment, margins expanded, and operating expenses rose with headcount."] * 30),  # > 512 tokens: truncated
]


def main():
    if len(sys.argv) != 3:
        sys.exit(f"usage: python {sys.argv[0]} <model_dir> <out_prefix>")
    model_dir, out = sys.argv[1], sys.argv[2]
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir).eval()

    def encode(text):
        return tok(text, truncation=True, max_length=512, return_tensors="pt")["input_ids"]

    tokens = []
    for text in TOKENIZER_CASES:
        ids = encode(text)[0].tolist()
        tokens.append({"text": text, "ids": ids, "decoded": tok.decode(ids, skip_special_tokens=True)})
    with open(f"{out}_tokens.json", "w") as f:
        json.dump(tokens, f, ensure_ascii=False, indent=1)

    plain = AutoTokenizer.from_pretrained(model_dir)
    plain.backend_tokenizer.normalizer = None
    nonorm = []
    for text in TOKENIZER_CASES:
        ids = plain(text, truncation=True, max_length=512)["input_ids"]
        nonorm.append({"text": text, "ids": ids, "decoded": plain.decode(ids, skip_special_tokens=True)})
    with open(f"{out}_tokens_nonorm.json", "w") as f:
        json.dump(nonorm, f, ensure_ascii=False, indent=1)

    cfg = model.config
    positions = torch.arange(-1024, 1025)
    buckets = {
        "num_buckets": cfg.relative_attention_num_buckets,
        "max_distance": cfg.relative_attention_max_distance,
        "first_position": -1024,
        "bidirectional": T5Attention._relative_position_bucket(positions, True, cfg.relative_attention_num_buckets, cfg.relative_attention_max_distance).tolist(),
        "unidirectional": T5Attention._relative_position_bucket(positions, False, cfg.relative_attention_num_buckets, cfg.relative_attention_max_distance).tolist(),
    }
    with open(f"{out}_buckets.json", "w") as f:
        json.dump(buckets, f)

    tensors, cases = {}, []
    for i, text in enumerate(MODEL_CASES):
        input_ids = encode(text)
        mask = torch.ones_like(input_ids)
        with torch.no_grad():
            generated = model.generate(input_ids=input_ids, attention_mask=mask, max_new_tokens=MAX_NEW_TOKENS, num_beams=1, do_sample=False)[0]
            full = model(input_ids=input_ids, attention_mask=mask, decoder_input_ids=generated[:-1].unsqueeze(0), output_hidden_states=True)
        if not torch.equal(full.logits[0].argmax(-1), generated[1:]):
            sys.exit(f"case {i}: teacher-forced argmax does not reproduce greedy generation")
        for j, h in enumerate(full.encoder_hidden_states):
            tensors[f"case{i}.enc.hidden.{j}"] = h[0].clone()
        for j, h in enumerate(full.decoder_hidden_states):
            tensors[f"case{i}.dec.hidden.{j}"] = h[0].clone()
        tensors[f"case{i}.dec.logits"] = full.logits[0, :LOGIT_STEPS].clone()
        output = tok.decode(generated, skip_special_tokens=True)
        cases.append({"text": text, "input_ids": input_ids[0].tolist(), "output_ids": generated.tolist(), "output": output})
        print(f"case {i}: {input_ids.shape[1]:3d} in, {len(generated) - 1:2d} out: {output!r}")

    save_file(tensors, f"{out}_model.safetensors")
    with open(f"{out}_model.json", "w") as f:
        json.dump(cases, f, ensure_ascii=False, indent=1)
    print(f"wrote {len(tokens)} tokenizer cases and {len(cases)} model cases to {out}_*")


if __name__ == "__main__":
    main()
