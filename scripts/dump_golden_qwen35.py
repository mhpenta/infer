"""
Dump reference outputs for a Qwen3.5 sequence classifier, OpenJev's NLI
cross-encoder (Qwen3_5ForSequenceClassification), so the Go implementation can
be checked layer by layer and decision by decision.

    .venv/bin/python scripts/dump_golden_qwen35.py <model_dir> <out_prefix>
    .venv/bin/python scripts/dump_golden_qwen35.py testdata/models/openjev-0.8b testdata/golden/openjev_0.8b

--model-only writes just <out_prefix>_model.*, for a larger checkpoint that
shares another's tokenizer, with hidden states for case 0 only, to keep the
repository small.

    .venv/bin/python scripts/dump_golden_qwen35.py --model-only testdata/models/openjev-4b testdata/golden/openjev_4b

The model runs in float32 (bf16 checkpoints are widened exactly), with eager
attention; its linear-attention layers run transformers' chunked gated delta
rule. Writes:

  <out_prefix>_tokens.json         tokenizer edge cases, as dump_golden_llama.py
  <out_prefix>_tokens_random.json  writes them
  <out_prefix>_model.json          per case: premise, hypothesis, the text and
                                   its ids, and the 3-way probabilities; the
                                   decision cases (state, question, options,
                                   rubric, and each option's text, ids and
                                   P(entailment), normalized over the options)
  <out_prefix>_model.safetensors
    rope.inv_freq                  the rotary embedding's inverse frequencies
    case{i}.logits                 the score head's 3 logits at the last token
    case{i}.hidden.{j}             hidden states, j = 0..L: j < L is the input
                                   to layer j (0 = embeddings), L is the final,
                                   normed output (HIDDEN_CASES only)
    decision{i}.logits             [options, 3]: each option's logits

Texts are "Premise: {premise}\\nHypothesis: {hypothesis}" (config.nli_template)
encoded without special tokens; the head reads the last token. Decisions
follow OpenJev's code/openjev_decide.py: each option becomes the hypothesis
'The answer to "{question}" is {option}: {rubric}', and its P(entailment) is
normalized over the options.
"""
import json
import sys

import torch
from safetensors.torch import save_file
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from dump_golden_llama import LONG_TEXT, RANDOM_CASES, TOKENIZER_CASES, random_texts

HIDDEN_CASES = {0, 2}
ENTAILMENT = 1  # config.label2id["entailment"]

# (premise, hypothesis): case 2 is past the gated delta rule's 64-token chunk,
# case 3 is about 1,700 tokens.
NLI_CASES = [
    ("A man is playing a guitar on stage.", "Someone is making music."),
    ("The meeting was moved from Tuesday to Thursday.", "The meeting is on Tuesday."),
    (
        "Net sales increased 8% to $94.9 billion, driven by record Services revenue and strong iPhone demand, "
        "while gross margin expanded to 46.3%. Operating expenses rose with headcount and research spending. "
        "The company returned over $29 billion to shareholders through dividends and buybacks during the quarter, "
        "and guided to low single-digit growth for the December quarter.",
        "The company's gross margin decreased during the quarter.",
    ),
    (LONG_TEXT, "The company returned money to shareholders."),
    ("naïve café — “quotes” 日本語 😀\n\nline", "  The text mentions Japanese.  "),
]

TEMPLATE = 'The answer to "{instr}" is {label}: {crit}'
DECISIONS = [
    {
        "state": "Policy: refunds require a receipt and purchase within 30 days. A customer bought 12 days ago "
                 "but has no receipt. Issue a refund.",
        "instructions": "Under the stated policy, is the requested action permitted?",
        "options": ["no", "yes"],
        "rubric": {"no": "A condition is missing.", "yes": "Every condition holds."},
    },
    {
        "state": "I was charged twice for the same order and nobody answers my emails. I want my money back now.",
        "instructions": "Which product area is the message about?",
        "options": ["fees & charges", "card", "refund & dispute", "transfer"],
        "rubric": {},
    },
    {
        "state": {"ticket": 4512, "customer": "Dana", "message": "The app crashes when I open settings.", "plan": "pro"},
        "instructions": "How urgent is this ticket?",
        "options": ["low", "medium", "high"],
        "rubric": {"low": "Cosmetic or a question.", "medium": "A feature is broken but there is a workaround.",
                   "high": "Data loss, an outage, or no workaround."},
    },
]


def main():
    args = sys.argv[1:]
    model_only = args[:1] == ["--model-only"]
    if model_only:
        args = args[1:]
    if len(args) != 2:
        sys.exit(f"usage: python {sys.argv[0]} [--model-only] <model_dir> <out_prefix>")
    model_dir, out = args
    hidden_cases = {0} if model_only else HIDDEN_CASES
    tok = AutoTokenizer.from_pretrained(model_dir)

    def token_cases(texts):
        cases = []
        for text in texts:
            plain = tok(text, add_special_tokens=False)["input_ids"]
            cases.append({
                "text": text,
                "ids": plain,
                "ids_with_special": tok(text)["input_ids"],
                "decoded": tok.decode(plain, skip_special_tokens=False),
                "decoded_skip_special": tok.decode(plain, skip_special_tokens=True),
            })
        return cases

    tokens = []
    if not model_only:
        tokens = token_cases(TOKENIZER_CASES + ["हिन्दी and q́x", "naïve ক্ষ ำ", "à́b ́x"])
        with open(f"{out}_tokens.json", "w") as f:
            json.dump(tokens, f, ensure_ascii=False, indent=1)
        with open(f"{out}_tokens_random.json", "w") as f:
            json.dump(token_cases(random_texts(RANDOM_CASES)), f, ensure_ascii=False, indent=1)

    model = AutoModelForSequenceClassification.from_pretrained(model_dir, dtype=torch.float32, attn_implementation="eager").eval()
    template = model.config.nli_template
    text_config = model.config.get_text_config()
    rotary = model.model.language_model.rotary_emb
    tensors = {"rope.inv_freq": rotary.inv_freq.float().clone()}

    def encode(premise, hypothesis):
        text = template.format(premise=premise.strip(), hypothesis=hypothesis.strip())
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if ids != tok(text)["input_ids"]:
            sys.exit("the tokenizer adds special tokens; the cases assume it does not")
        if text_config.pad_token_id in ids:
            sys.exit("a case contains the pad token, which moves the pooled position")
        return text, ids

    def run(ids, hidden):
        with torch.no_grad():
            r = model(input_ids=torch.tensor([ids]), output_hidden_states=hidden)
        return r.logits[0], r.hidden_states

    cases = []
    for i, (premise, hypothesis) in enumerate(NLI_CASES):
        text, ids = encode(premise, hypothesis)
        logits, hidden = run(ids, i in hidden_cases)
        tensors[f"case{i}.logits"] = logits.clone()
        if i in hidden_cases:
            if len(hidden) != text_config.num_hidden_layers + 1:
                sys.exit(f"case {i}: {len(hidden)} hidden states, want layers + 1")
            for j, h in enumerate(hidden):
                tensors[f"case{i}.hidden.{j}"] = h[0].clone()
        probs = torch.softmax(logits, -1)
        cases.append({"premise": premise, "hypothesis": hypothesis, "text": text, "input_ids": ids, "probs": probs.tolist()})
        print(f"case {i}: {len(ids):5d} tokens, probs {[round(p, 4) for p in probs.tolist()]}")

    decisions = []
    for i, d in enumerate(DECISIONS):
        state = d["state"] if isinstance(d["state"], str) else json.dumps(d["state"], ensure_ascii=False)
        options = []
        logits = []
        for o in d["options"]:
            hypothesis = TEMPLATE.format(instr=d["instructions"].strip(), label=o, crit=str(d["rubric"].get(o, o)))
            text, ids = encode(state, hypothesis)
            l, _ = run(ids, False)
            logits.append(l)
            options.append({"option": o, "hypothesis": hypothesis, "text": text, "input_ids": ids})
        logits = torch.stack(logits)
        p = torch.softmax(logits, -1)[:, ENTAILMENT]
        p = p / max(float(p.sum()), 1e-9)
        for o, x in zip(options, p.tolist()):
            o["p"] = x
        tensors[f"decision{i}.logits"] = logits.clone()
        decisions.append({**d, "state_text": state, "options": options})
        print(f"decision {i}: {[(o['option'], round(o['p'], 4)) for o in options]}")

    save_file(tensors, f"{out}_model.safetensors")
    with open(f"{out}_model.json", "w") as f:
        json.dump({"labels": model.config.id2label, "cases": cases, "decisions": decisions}, f, ensure_ascii=False, indent=1)
    print(f"wrote {len(tokens)} tokenizer cases, {len(cases)} model cases and {len(decisions)} decisions to {out}_*")


if __name__ == "__main__":
    main()
