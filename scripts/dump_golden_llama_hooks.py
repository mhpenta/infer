"""
Dump what a Llama-family decoder's forward hooks see, and what it generates
with edited weights, so package llama's Capture and edited checkpoints can be
checked against transformers. Run after dump_golden_llama.py, whose
<out_prefix>_model.json gives the cases:

    .venv/bin/python scripts/dump_golden_llama_hooks.py <model_dir> <out_prefix>
    .venv/bin/python scripts/dump_golden_llama_hooks.py testdata/models/minicpm5-1b testdata/golden/minicpm5_1b
    .venv/bin/python scripts/dump_golden_llama_hooks.py testdata/models/qwen3-1.7b testdata/golden/qwen3_1.7b

The model runs in float32 with eager attention, as in dump_golden_llama.py.
Hooks on the layers in layers() read, where {p} is the point:

  attention   the output of self_attn.o_proj
  swiglu      the input to mlp.down_proj (a forward pre-hook)
  mlp         the output of mlp.down_proj

Writes:

  <out_prefix>_hooks.json    the layers, the cases each mode covers, and each
                             edit: the tensors it scales and by what, and the
                             tokens generated
  <out_prefix>_hooks.safetensors
    case{i}.whole.{l}.{p}    the prompt + generated[:-1] run whole, without a
                             cache (WHOLE_CASES)
    case{i}.cached.{l}.{p}   the same rows as generate() ran them, the prompt
                             and then a token at a time with a KV cache, which
                             the script checks it did (CACHED_CASES)
    edit{e}.topk.values      with the edit applied: the TOP_K largest logits
    edit{e}.topk.indices     per position of prompt + generated[:-1], and
                             their token ids, descending

The script also checks that each layer's input plus what the hooks saw it add
is the next layer's input, so the hooks read the layer's contributions.
"""
import json
import sys

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

TOP_K = 64
WHOLE_CASES = [0, 1]
CACHED_CASES = [1]
EDIT_CASE = 2

# Each edit scales the named tensors of one layer, by factors exact in bf16,
# and must change the tokens generated, which a smaller edit may not do.
EDIT_LAYER = 3
EDITS = [
    ("o_proj", {"self_attn.o_proj.weight": -1}),
    ("gate_proj", {"mlp.gate_proj.weight": -1}),  # one tensor of a stacked pair
    ("gate_up", {"mlp.gate_proj.weight": -1, "mlp.up_proj.weight": -1}),
]


def layers(model):
    """The first, a middle and the last layer."""
    n = model.config.num_hidden_layers
    return [0, n // 2, n - 1]


class Hooks:
    """Collects what the hooks see, per (layer, point), in call order."""

    def __init__(self, model):
        self.seen = {}
        self.handles = []
        for l in layers(model):
            layer = model.model.layers[l]
            self.handles += [
                layer.self_attn.o_proj.register_forward_hook(self.output(l, "attention")),
                layer.mlp.down_proj.register_forward_pre_hook(self.input(l, "swiglu")),
                layer.mlp.down_proj.register_forward_hook(self.output(l, "mlp")),
            ]

    def add(self, l, point, value):
        self.seen.setdefault((l, point), []).append(value[0].detach().clone())

    def output(self, l, point):
        return lambda _module, _args, out: self.add(l, point, out)

    def input(self, l, point):
        return lambda _module, args: self.add(l, point, args[0])

    def close(self):
        for h in self.handles:
            h.remove()


def check_contributions(model, seq, seen):
    """Each layer's input + attention + mlp must be the next layer's input."""
    with torch.no_grad():
        hidden = model(input_ids=seq, output_hidden_states=True, use_cache=False).hidden_states
    for l in layers(model):
        if l + 1 == model.config.num_hidden_layers:
            continue  # the last hidden state is normalized
        total = hidden[l][0] + seen[l, "attention"][0] + seen[l, "mlp"][0]
        scale = hidden[l + 1][0].abs().max()
        if (total - hidden[l + 1][0]).abs().max() > 1e-5 * scale:
            sys.exit(f"layer {l}: input + attention + mlp is not the next layer's input")


def main():
    if len(sys.argv) != 3:
        sys.exit(f"usage: python {sys.argv[0]} <model_dir> <out_prefix>")
    model_dir, out = sys.argv[1:]
    with open(f"{out}_model.json") as f:
        golden = json.load(f)
    cases, eos = golden["cases"], golden["eos_token_ids"]
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32, attn_implementation="eager").eval()
    tensors = {}

    def generate(ids, n):
        ids = torch.tensor([ids])
        with torch.no_grad():
            return model.generate(
                input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=n,
                do_sample=False, temperature=None, top_p=None, top_k=None, eos_token_id=eos, pad_token_id=eos[0],
            )[0]

    for i in WHOLE_CASES:
        c = cases[i]
        seq = torch.tensor([c["input_ids"] + c["output_ids"][:-1]])
        hooks = Hooks(model)
        with torch.no_grad():
            model(input_ids=seq, use_cache=False)
        hooks.close()
        check_contributions(model, seq, hooks.seen)
        for (l, point), values in hooks.seen.items():
            if len(values) != 1 or values[0].shape[0] != seq.shape[1]:
                sys.exit(f"case {i}: layer {l} {point}: hooks saw {[v.shape for v in values]}")
            tensors[f"case{i}.whole.{l}.{point}"] = values[0]

    for i in CACHED_CASES:
        c = cases[i]
        n = len(c["input_ids"])
        hooks = Hooks(model)
        new = generate(c["input_ids"], len(c["output_ids"]))[n:].tolist()
        hooks.close()
        if new != c["output_ids"]:
            sys.exit(f"case {i}: generated {new}, the goldens say {c['output_ids']}")
        for (l, point), values in hooks.seen.items():
            rows = [v.shape[0] for v in values]
            if rows != [n] + [1] * (len(new) - 1):
                sys.exit(f"case {i}: layer {l} {point}: passes of {rows} rows, so no cache was used")
            tensors[f"case{i}.cached.{l}.{point}"] = torch.cat(values)

    edits = []
    prompt = cases[EDIT_CASE]["input_ids"]
    n_new = len(cases[EDIT_CASE]["output_ids"])
    for e, (name, scales) in enumerate(EDITS):
        params = {f"model.layers.{EDIT_LAYER}.{t}": s for t, s in scales.items()}
        state = dict(model.named_parameters())
        saved = {t: state[t].data.clone() for t in params}
        for t, s in params.items():
            state[t].data *= s
        seq = generate(prompt, n_new)
        new = seq[len(prompt):].tolist()
        with torch.no_grad():
            logits = model(input_ids=seq[:-1].unsqueeze(0)).logits[0]
        top = logits.topk(TOP_K, dim=-1)
        tensors[f"edit{e}.topk.values"] = top.values.clone()
        tensors[f"edit{e}.topk.indices"] = top.indices.to(torch.int32).clone()
        for t in params:
            state[t].data = saved[t]
        edits.append({"name": name, "scales": params, "output_ids": new})
        if new == cases[EDIT_CASE]["output_ids"]:
            sys.exit(f"edit {name} does not change the tokens generated")
        print(f"edit {name}: {new[:8]}...")

    save_file(tensors, f"{out}_hooks.safetensors")
    with open(f"{out}_hooks.json", "w") as f:
        json.dump({"layers": layers(model), "whole_cases": WHOLE_CASES, "cached_cases": CACHED_CASES,
                   "edit_case": EDIT_CASE, "edits": edits}, f, indent=1)
    print(f"wrote {len(tensors)} tensors to {out}_hooks.safetensors")


if __name__ == "__main__":
    main()
