"""
Dump reference outputs for Inkling 2 Mini's text decoder
(Inkling2LoopForConditionalGeneration), so the Go implementation can be
checked layer by layer.

It needs transformers with inkling2_loop (5.24 or main at the time of
writing), which converts the original checkpoint on load: its NVFP4 and MXFP8
experts are dequantized, its tensors renamed and its message queries merged.

    python scripts/dump_golden_inkling2.py tiny testdata/golden

writes a tiny random checkpoint in the original layout (the same tensor
names, dtypes and expert formats, with every feature exercised: windows and
relative extents shorter than the inputs, active L2 clips and log scaling,
NVFP4 and MXFP8 experts, full-attention layers inside the loop core) to
testdata/golden/inkling2_tiny/, loads it with transformers, and writes:

  inkling2_tiny_model.json         the input ids and greedily generated tokens
  inkling2_tiny_model.safetensors
    hidden.{j}                     the input to the j-th layer visit, in the
                                   order layers run (core layers once per
                                   pass), then the final normed output
    logits                         [ids, unpadded vocab], softcapped
    generate.logits                [new tokens, unpadded vocab], each step's

    python scripts/dump_golden_inkling2.py tokens <model_dir> <out_prefix>

writes the tokenizer's goldens, which need only the tokenizer files:
<out_prefix>_tokens.json and _tokens_random.json (as dump_golden_llama.py
writes them) and <out_prefix>_chat.json, conversations rendered by the chat
template with their ids.

    python scripts/dump_golden_inkling2.py model <model_dir> <out_prefix>

does the same for the real checkpoint on a few chat prompts. The model runs
in float32 with its experts dequantized, about 110 GB: run it on a machine
with the memory, then copy <out_prefix>_model.* back.
"""
import json
import math
import os
import sys

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer, Inkling2LoopForConditionalGeneration

from dump_golden_llama import RANDOM_CASES, TOKENIZER_CASES, random_texts

TINY = dict(
    hidden=64, layers=6, block=2, core=(2, 4), passes=3, gate_rank=8,
    heads=4, kv_heads=2, head_dim=16, d_rel=4, rel_extent=6, window=8,
    experts=8, top_k=2, ff=32, vocab=300, unpadded=290, kernel=4,
    full_layers=(1, 3, 5), mxfp8_layers=(4, 5),
)


def tiny_config(t):
    full = set(t["full_layers"])
    text = {
        "model_type": "inkling2_loop_text",
        "hidden_size": t["hidden"],
        "num_hidden_layers": t["layers"],
        "vocab_size": t["vocab"],
        "unpadded_vocab_size": t["unpadded"],
        "num_attention_heads": t["heads"],
        "num_key_value_heads": t["kv_heads"],
        "head_dim": t["head_dim"],
        "swa_num_attention_heads": t["heads"],
        "swa_num_key_value_heads": t["kv_heads"],
        "swa_head_dim": t["head_dim"],
        "sliding_window_size": t["window"],
        "local_layer_ids": [i for i in range(t["layers"]) if i not in full],
        # transformers reads layer_types, Go local_layer_ids.
        "layer_types": ["hybrid" if i in full else "hybrid_sliding" for i in range(t["layers"])],
        "d_rel": t["d_rel"],
        "rel_extent": t["rel_extent"],
        "q_bias": False,
        "o_bias": False,
        "log_scaling_n_floor": 4,
        "log_scaling_alpha": 0.1,
        "rms_norm_eps": 1e-6,
        "use_value_norm_no_gamma": True,
        "qk_l2_clip_threshold": 3.0,
        "rel_l2_clip_threshold": 1.0,
        "sconv_kernel_size": t["kernel"],
        "final_logit_softcapping": 5.0,
        "n_layers_per_attn_res_block": t["block"],
        "attn_res_normalize_values": True,
        "n_routed_experts": t["experts"],
        "num_experts_per_tok": t["top_k"],
        "n_shared_experts": 0,
        "intermediate_size": t["ff"],
        "mlp_gate_softcap": 1.0,
        "mlp_up_softcap": 2.0,
        "route_scale": 2.0,
        "use_gate_bias": True,
        "model_max_length": 4096,
        "loop_config": {
            "core_start": t["core"][0], "core_end": t["core"][1], "num_passes": t["passes"],
            "gate_rank": t["gate_rank"], "is_varloop": False,
        },
    }
    vision = {
        "model_type": "inkling2_vision", "text_hidden_size": t["hidden"], "hidden_size": 32,
        "num_hidden_layers": 1, "num_attention_heads": 2, "projection_hidden_dim": 64,
    }
    return {
        "architectures": ["Inkling2LoopForConditionalGeneration"],
        "model_type": "inkling2_loop",
        "dtype": "float32",
        "eos_token_id": t["unpadded"] - 1,
        "text_config": text,
        "vision_config": vision,
    }


def tiny_weights(t, g):
    """Random weights under the original checkpoint's names and dtypes."""
    h, ff, ne, k = t["hidden"], t["ff"], t["experts"], t["kernel"]
    nh, nkv, hd, dr = t["heads"], t["kv_heads"], t["head_dim"], t["d_rel"]
    bf16 = torch.bfloat16

    def randn(*shape, std=1.0, dtype=bf16):
        return (torch.randn(*shape, generator=g) * std).to(dtype)

    def linear(out, inp, gain=1.0):
        return randn(out, inp, std=gain / math.sqrt(inp))

    def norm(n):
        return (1 + 0.2 * torch.randn(n, generator=g)).to(bf16)

    def nvfp4(rows, cols):
        codes = torch.randint(0, 256, (ne, rows, cols // 2), generator=g, dtype=torch.uint8)
        scale = (0.5 + 1.5 * torch.rand(ne, rows, cols // 16, generator=g)).to(torch.float8_e4m3fn)
        amax = (0.6 / math.sqrt(cols)) * 448 * 6 * (0.5 + torch.rand(ne, generator=g)) / 2.5
        den = torch.tensor([448.0 * 6])
        return codes, scale, amax.float(), den

    def mxfp8(rows, cols):
        values = torch.randn(ne, rows, cols, generator=g).to(torch.float8_e4m3fn)
        exps = 127 + torch.randint(-5, -2, (ne, rows, cols // 32), generator=g)
        return values, exps.to(torch.uint8)

    w = {}
    p = "model.llm."
    w[p + "embed.weight"] = randn(t["vocab"], h)
    w[p + "embed_norm.weight"] = norm(h)
    w[p + "norm.weight"] = norm(h)
    w[p + "unembed.weight"] = linear(t["vocab"], h, 4.0)
    w[p + "attn_res.output_query"] = randn(h, dtype=torch.float32)
    nl = t["core"][1] - t["core"][0]
    w[p + "middle_cycle_cross_pass_queries"] = randn(2 * nl, h, dtype=torch.float32)
    w[p + "middle_cycle_cross_pass_kv_queries"] = randn(2 * nl, h, dtype=torch.float32)
    for i in range(t["layers"]):
        full = i in t["full_layers"]
        lp, a = f"{p}layers.{i}.", f"{p}layers.{i}.attn."
        w[a + "wq_du.weight"] = linear(nh * hd, h)
        w[a + "gate_attn.weight"] = linear(nh * hd, h)
        w[a + "wr_du.weight"] = linear(nh * dr, h)
        w[a + "attn_sink_proj.weight"] = linear(nh, h)
        w[a + "wk_dv.weight"] = linear(nkv * hd, h)
        w[a + "wv_dv.weight"] = linear(nkv * hd, h)
        w[a + "wo_ud.weight"] = linear(h, nh * hd)
        w[a + "q_l2_clip.scale"] = norm(nh * hd).view(nh, hd)
        w[a + "rel_l2_clip.scale"] = norm(nh * dr).view(nh, dr)
        extent = t["rel_extent"] if full else t["window"]
        w[a + "rel_logits_proj.proj"] = randn(dr, extent)
        if full:
            for kv in "kv":
                w[a + f"{kv}_sconv.static_w"] = randn(k, nkv * hd, std=0.3)
                w[a + f"{kv}_sconv.weight_proj"] = linear(nkv * k, h, 0.3)
            for n in "qkv":
                w[lp + f"{n}_input_norm.weight"] = norm(h)
        else:
            w[a + "k_sconv.weight"] = randn(nkv * hd, 1, k, std=0.3)
            w[a + "v_sconv.weight"] = randn(nkv * hd, 1, k, std=0.3)
            w[lp + "attn_norm.weight"] = norm(h)
        w[lp + "attn_sconv.weight"] = randn(h, 1, k, std=0.3)
        w[lp + "mlp_sconv.weight"] = randn(h, 1, k, std=0.3)
        w[lp + "mlp_norm.weight"] = norm(h)
        if i > 0:
            shape = (3, h) if full else (h,)
            w[f"{p}attn_res.attn_queries.{i - 1}"] = randn(*shape, dtype=torch.float32)
        w[f"{p}attn_res.mlp_queries.{i}"] = randn(h, dtype=torch.float32)
        w[lp + "mlp.gate.weight"] = linear(ne, h, 2.0)
        w[lp + "mlp.gate.bias"] = randn(ne, std=0.3, dtype=torch.float32)
        e = lp + "mlp.experts."
        for name, rows, cols in (("w13_weight", 2 * ff, h), ("w2_weight", h, ff)):
            if i in t["mxfp8_layers"]:
                w[e + name], w[e + name + "_scale"] = mxfp8(rows, cols)
            else:
                (w[e + name], w[e + name + "_scale"], w[e + name + "_amax"],
                 w[e + name + "_outer_scale_denominator"]) = nvfp4(rows, cols)
        if t["core"][0] <= i < t["core"][1]:
            for c in ("attn", "k", "v", "mlp"):
                cp = f"{lp}controllers.{c}."
                w[cp + "down_weight"] = linear(t["gate_rank"], 2 * h)
                w[cp + "forget_weight"] = linear(h, t["gate_rank"])
                w[cp + "gain_weight"] = linear(h, t["gate_rank"])
    return {name: v.contiguous() for name, v in w.items()}


def load(model_dir):
    model, info = Inkling2LoopForConditionalGeneration.from_pretrained(
        model_dir, dtype=torch.float32, attn_implementation="eager", output_loading_info=True)
    missing = [k for k in info["missing_keys"] if "vision" not in k]
    if missing or info["unexpected_keys"] or info.get("mismatched_keys"):
        sys.exit(f"loading {model_dir}: missing {missing}, unexpected {info['unexpected_keys']}, "
                 f"mismatched {info.get('mismatched_keys')}")
    return model.eval()


def record(model, ids, max_new):
    """Every layer visit's input, the final output, the logits, and greedy
    generation with the cache."""
    lm = model.model.language_model
    inputs = []
    hooks = [layer.register_forward_pre_hook(lambda _m, args: inputs.append(args[0][0].clone()))
             for layer in lm.layers]
    final = []
    hooks.append(lm.norm.register_forward_hook(lambda _m, _a, out: final.append(out[0].clone())))
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids])).logits[0]
    for hk in hooks:
        hk.remove()
    out = {f"hidden.{j}": x for j, x in enumerate(inputs + final)}
    out["logits"] = logits
    with torch.no_grad():
        gen = model.generate(torch.tensor([ids]), max_new_tokens=max_new, do_sample=False,
                             output_logits=True, return_dict_in_generate=True)
    tokens = gen.sequences[0, len(ids):].tolist()
    out["generate.logits"] = torch.cat([l.float() for l in gen.logits])
    return out, tokens


def tiny(out_dir):
    t = TINY
    g = torch.Generator().manual_seed(20261121)
    ckpt = os.path.join(out_dir, "inkling2_tiny")
    os.makedirs(ckpt, exist_ok=True)
    with open(os.path.join(ckpt, "config.json"), "w") as f:
        json.dump(tiny_config(t), f, indent=2)
    save_file(tiny_weights(t, g), os.path.join(ckpt, "model.safetensors"), metadata={"format": "pt"})

    model = load(ckpt)
    ids = torch.randint(0, t["unpadded"], (21,), generator=g).tolist()
    tensors, tokens = record(model, ids, 6)
    save_file({k: v.contiguous() for k, v in tensors.items()},
              os.path.join(out_dir, "inkling2_tiny_model.safetensors"))
    with open(os.path.join(out_dir, "inkling2_tiny_model.json"), "w") as f:
        json.dump({"input_ids": ids, "generated": tokens}, f, indent=1)


WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather in a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "units": {"type": "string",
                       "enum": ["c", "f"]}}, "required": ["city"]},
    },
}

CHAT_CASES = [
    {"messages": [{"role": "user", "content": "Hello"}], "options": {}},
    {"messages": [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Hi there"}],
     "options": {"reasoning_effort": "none"}},
    {"messages": [{"role": "user", "content": "What is 2+2?"},
                  {"role": "assistant", "reasoning_content": "Simple sum.", "content": "4"},
                  {"role": "user", "content": "And times 3?"}],
     "options": {"reasoning_effort": "low"}},
    {"messages": [{"role": "user", "content": "Rate this: \u00e9t\u00e9 \u4f60\u597d"}], "options": {"reasoning_effort": 0.5}},
    {"messages": [{"role": "user", "content": "x"}], "options": {"reasoning_effort": 0}},
    {"messages": [{"role": "system", "content": "Only a system message."}], "options": {"reasoning_effort": "max"}},
    {"messages": [{"role": "user", "content": "Weather in Paris?"},
                  {"role": "assistant", "content": "", "tool_calls": [
                      {"id": "c1", "type": "function",
                       "function": {"name": "get_weather", "arguments": {"units": "c", "city": "Paris"}}}]},
                  {"role": "tool", "name": "get_weather", "content": "{\"temp\": 18}"},
                  {"role": "assistant", "content": "It is 18\u00b0C in Paris."}],
     "options": {"tools": [WEATHER], "reasoning_effort": "medium"}},
]


def tokens(model_dir, out):
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

    with open(out + "_tokens.json", "w") as f:
        json.dump(token_cases(TOKENIZER_CASES), f, ensure_ascii=False, indent=1)
    with open(out + "_tokens_random.json", "w") as f:
        json.dump(token_cases(random_texts(RANDOM_CASES)), f, ensure_ascii=False, indent=1)
    chats = []
    for c in CHAT_CASES:
        text = tok.apply_chat_template(c["messages"], tokenize=False, **c["options"])
        chats.append({**c, "text": text, "ids": tok(text, add_special_tokens=False)["input_ids"]})
    with open(out + "_chat.json", "w") as f:
        json.dump(chats, f, ensure_ascii=False, indent=1)


PROMPTS = [
    [{"role": "user", "content": "What is 17 * 23? Answer with just the number."}],
    [{"role": "system", "content": "You are terse."},
     {"role": "user", "content": "Name the largest planet in the solar system."}],
]


def real(model_dir, out_prefix):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = load(model_dir)
    cases, tensors = [], {}
    for i, messages in enumerate(PROMPTS):
        text = tok.apply_chat_template(messages, tokenize=False, reasoning_effort="none")
        ids = tok(text, add_special_tokens=False)["input_ids"]
        out, tokens = record(model, ids, 16)
        tensors.update({f"case{i}.{k}": v.contiguous() for k, v in out.items()})
        cases.append({"messages": messages, "text": text, "input_ids": ids, "generated": tokens})
    save_file(tensors, out_prefix + "_model.safetensors")
    with open(out_prefix + "_model.json", "w") as f:
        json.dump({"cases": cases}, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    torch.set_num_threads(os.cpu_count())
    if len(sys.argv) == 3 and sys.argv[1] == "tiny":
        tiny(sys.argv[2])
    elif len(sys.argv) == 4 and sys.argv[1] == "tokens":
        tokens(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 4 and sys.argv[1] == "model":
        real(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)
