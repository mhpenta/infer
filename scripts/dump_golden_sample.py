"""
Dump the token distributions transformers' sampling produces, so the Go
sample package can be checked against them.

    .venv/bin/python scripts/dump_golden_sample.py

For each case, applies the logits processors model.generate() would, in its
order (RepetitionPenaltyLogitsProcessor, then TemperatureLogitsWarper,
TopKLogitsWarper and TopPLogitsWarper, as GenerationMixin._get_logits_processor
in transformers 5.21.0 builds them, each only when its setting is active), then
softmax. Writes testdata/golden/sample.json: per case the settings, the
sequence so far (for the repetition penalty), the logits (inline, or a row of
a tensor in minicpm5_1b_model.safetensors: real MiniCPM5 logits) and the
tokens left with nonzero probability, with their probabilities.
"""
import json
import os

import torch
from safetensors.torch import load_file
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
GOLDEN = os.path.join(ROOT, "testdata", "golden")


def processors(temperature, top_k, top_p, repetition_penalty):
    p = LogitsProcessorList()
    if repetition_penalty != 1.0:
        p.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
    if temperature != 1.0:
        p.append(TemperatureLogitsWarper(temperature))
    if top_k != 0:
        p.append(TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=1))
    if top_p < 1.0:
        p.append(TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=1))
    return p


def main():
    gen = torch.Generator().manual_seed(7)
    real = load_file(os.path.join(GOLDEN, "minicpm5_1b_model.safetensors"))
    cases = []

    def add(settings, seq, logits, source):
        scores = processors(**settings)(torch.tensor([seq]), logits.clone().unsqueeze(0))[0]
        probs = scores.softmax(-1)
        kept = torch.nonzero(probs > 0).flatten()
        order = kept[torch.argsort(probs[kept], descending=True, stable=True)]
        cases.append({**settings, "sequence": seq, **source,
                      "tokens": order.tolist(), "probs": probs[order].double().tolist()})

    # Synthetic logits over a 512-token vocabulary: a spread of values with a
    # few strong candidates, so the cutoffs fall in different places.
    vocab = 512
    synthetic = []
    for i in range(6):
        x = torch.randn(vocab, generator=gen) * (1 + i)
        x[torch.randint(0, vocab, (5,), generator=gen)] += 8
        synthetic.append(x.float())
    seq = torch.randint(0, vocab, (40,), generator=gen).tolist() + [7, 7, 7]
    settings = [
        dict(temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0),
        dict(temperature=0.7, top_k=0, top_p=1.0, repetition_penalty=1.0),
        dict(temperature=1.0, top_k=50, top_p=1.0, repetition_penalty=1.0),
        dict(temperature=1.0, top_k=0, top_p=0.9, repetition_penalty=1.0),
        dict(temperature=0.6, top_k=20, top_p=0.95, repetition_penalty=1.0),
        dict(temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.3),
        dict(temperature=0.9, top_k=50, top_p=0.95, repetition_penalty=1.1),
        dict(temperature=1.5, top_k=1, top_p=1.0, repetition_penalty=1.0),
        dict(temperature=0.3, top_k=0, top_p=0.5, repetition_penalty=2.0),
    ]
    for i, s in enumerate(settings):
        x = synthetic[i % len(synthetic)]
        add(s, seq, x, {"logits": x.tolist()})

    # Real next-token logits from MiniCPM5-1B (the full 130,560 tokens), with
    # the sequence they follow. Without truncation every token keeps
    # some probability, so these use top-k or top-p.
    model = json.load(open(os.path.join(GOLDEN, "minicpm5_1b_model.json")))
    for name, row, case in [("case0.logits", 0, 0), ("case1.logits", 0, 1), ("case1.logits", 1, 1)]:
        x = real[name][row].float()
        c = model["cases"][case]
        prompt = c["input_ids"] + c["output_ids"][:row]  # row r follows r generated tokens
        for s in [
            dict(temperature=1.0, top_k=50, top_p=0.95, repetition_penalty=1.0),
            dict(temperature=0.9, top_k=50, top_p=0.95, repetition_penalty=1.0),
            dict(temperature=1.0, top_k=0, top_p=0.95, repetition_penalty=1.0),
            dict(temperature=0.7, top_k=0, top_p=0.8, repetition_penalty=1.2),
        ]:
            add(s, prompt, x, {"tensor": name, "row": row})

    with open(os.path.join(GOLDEN, "sample.json"), "w") as f:
        json.dump(cases, f)
    print(f"wrote {len(cases)} cases to testdata/golden/sample.json")


if __name__ == "__main__":
    main()
