"""
Dump reference outputs for a Llama-family decoder (MiniCPM5, Qwen3, Llama),
so the Go implementation can be checked token for token and layer by layer.

    .venv/bin/python scripts/dump_golden_llama.py <model_dir> <out_prefix>
    .venv/bin/python scripts/dump_golden_llama.py testdata/models/minicpm5-1b testdata/golden/minicpm5_1b
    .venv/bin/python scripts/dump_golden_llama.py --chat-only testdata/models/minicpm5-2b testdata/golden/minicpm5_2b

--chat-only writes just <out_prefix>_chat.json, for a model that shares
another's tokenizer but has its own chat template; it loads no weights.
--tools writes just <out_prefix>_chat_tools.json: TOOL_CHAT_CASES, chat
cases with tool definitions, assistant tool calls and tool messages.

    .venv/bin/python scripts/dump_golden_llama.py --tools testdata/models/qwen3-1.7b testdata/golden/qwen3_1.7b

The model runs in float32 (bf16 checkpoints are widened exactly), with eager
attention. Writes:

  <out_prefix>_tokens.json   tokenizer edge cases: text, ids without and with
                             the tokenizer's default special tokens, and the
                             ids decoded back (keeping and skipping specials)
  <out_prefix>_tokens_random.json
                             the same for RANDOM_CASES seeded random strings
                             mixing scripts, Unicode spaces and digits,
                             combining marks, contractions and fragments of
                             special tokens
  <out_prefix>_chat.json     chat-template cases: messages, template options,
                             rendered text and its ids
  <out_prefix>_model.json    per case: prompt ids, greedy generated ids (new
                             tokens only), decoded output
  <out_prefix>_model.safetensors
    rope.inv_freq            the rotary embedding's inverse frequencies
    case{i}.topk.values      per position of prompt + generated[:-1] (teacher
    case{i}.topk.indices     forcing, no KV cache): the TOP_K largest logits
                             and their token ids, descending
    case{i}.logits           full logits for the last prompt position and the
                             first FULL_LOGIT_STEPS - 1 generated positions
                             (FULL_CASES only)
    case{i}.hidden.{j}       hidden states, j = 0..L: j < L is the input to
                             layer j (0 = embeddings), L is the final, normed
                             output (HIDDEN_CASES only)

Generation is greedy (no sampling) up to the case's token limit or an end
token. The script checks that the teacher-forced argmax reproduces greedy
generation, which is what a cache-free decoder has to match before a KV
cache is added.
"""
import json
import random
import sys

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

TOP_K = 64
FULL_LOGIT_STEPS = 2
HIDDEN_CASES = {0, 1}
FULL_CASES = {0, 1}

TOKENIZER_CASES = [
    "",
    "Hello",
    " Hello world",
    "The quick brown fox jumps over the lazy dog.",
    "I'm sure they'll say it's John's, we'd've known. I'M SURE THEY'LL",
    "1 12 123 1234 12345 1234567890 3.14159 $1,234,567.89 (12.3)% 2026-10-26",
    "Total revenue,\"$1,234.5\",(12.3)%",
    "tabs\tand\nnewlines\r\nand   multiple    spaces",
    "  leading and trailing  ",
    "line one\n\n\nline four\n \n  \n",
    "trailing newline\n",
    "naïve café résumé — “quotes” … ™ © ® ° ± §",
    "decomposed é and composed é",
    "non breaking space, thin space, ideographic　space",
    "emoji 😀👍🏽 and 中文 and 日本語 and Привет and العربية",
    "func main() {\n\tfmt.Println(\"hi\") // comment\n}\n",
    "<s> </s> <unk> literal specials",
    "<|im_start|>user\nhi<|im_end|>\n<think>\n\n</think>",
    "hello<s>world</s>again",
    "a" * 300,
    "!!! ??? ... --- *** ### @@@",
    "x" + " " * 20 + "y",
]

CHAT_CASES = [
    {"messages": [{"role": "user", "content": "What is 17 * 23?"}], "options": {"add_generation_prompt": True}},
    {"messages": [{"role": "user", "content": "What is 17 * 23?"}], "options": {"add_generation_prompt": True, "enable_thinking": False}},
    {"messages": [{"role": "user", "content": "What is 17 * 23?"}], "options": {"add_generation_prompt": True, "enable_thinking": True}},
    {
        "messages": [
            {"role": "system", "content": "You are a concise financial analyst."},
            {"role": "user", "content": "Summarize: revenue rose 8% to $94.9B."},
            {"role": "assistant", "content": "Revenue grew 8% to $94.9 billion."},
            {"role": "user", "content": "And margins?"},
        ],
        "options": {"add_generation_prompt": True, "enable_thinking": False},
    },
    {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello! How can I help?"}], "options": {"add_generation_prompt": False}},
    # Reasoning in assistant turns: MiniCPM5-1B drops it before the last user
    # query and keeps it after; MiniCPM5-2B keeps it everywhere and gives turns
    # without it an empty <think> block.
    {
        "messages": [
            {"role": "user", "content": "2+2?"},
            {"role": "assistant", "content": "<think>\nsimple sum\n</think>\n\n4"},
            {"role": "user", "content": "times 3?"},
            {"role": "assistant", "content": "<think>\n\n4*3\n\n</think>\n\n\n12"},
        ],
        "options": {"add_generation_prompt": False},
    },
    {
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "answer", "reasoning_content": "\nfrom the field\n"},
        ],
        "options": {"add_generation_prompt": False},
    },
    {
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "pre<think>a</think>mid<think>b</think>\nend"},  # several tags
            {"role": "assistant", "content": "no tags at all"},
            {"role": "assistant", "content": "</think>only a closing tag"},
            {"role": "assistant", "content": "\n<think>unclosed"},
        ],
        "options": {"add_generation_prompt": True, "enable_thinking": True},
    },
    # A second system message is an ordinary turn; a user turn that is only a
    # tool response does not count as the last query.
    {
        "messages": [
            {"role": "system", "content": "first"},
            {"role": "user", "content": "hello"},
            {"role": "system", "content": "second"},
            {"role": "assistant", "content": "<think>r1</think>a1"},
            {"role": "user", "content": "<tool_response>\n42\n</tool_response>"},
            {"role": "assistant", "content": "<think>r2</think>a2"},
        ],
        "options": {"add_generation_prompt": True, "enable_thinking": False},
    },
    {"messages": [{"role": "assistant", "content": "<think>x</think>no user at all"}], "options": {"add_generation_prompt": False}},
    {"messages": [{"role": "user", "content": ""}], "options": {"add_generation_prompt": True}},
    {"messages": [{"role": "user", "content": "\n  naïve 日本語 😀\n\n"}, {"role": "assistant", "content": "\n\nleading newlines"}], "options": {"add_generation_prompt": True, "enable_thinking": False}},
]

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. \"Paris\""},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}
STOCK_TOOL = {
    "type": "function",
    "function": {
        "name": "get_stock_price",
        "description": "Latest price for a ticker — in USD.\nUse uppercase tickers & <exchange> codes.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "days": {"type": "integer", "minimum": 1, "maximum": 365, "default": 30},
                "threshold": {"type": "number", "default": 0.00001},
                "adjusted": {"type": "boolean", "default": True},
                "note": {"type": ["string", "null"], "default": None},
            },
            "required": ["ticker"],
        },
    },
}
# Escapes and numbers as json.dumps writes them: control characters, quotes,
# backslashes, U+2028, non-ASCII kept, and floats in Python's repr.
EDGE_TOOL = {
    "name": "echo",
    "description": "Tab\there, quote \" backslash \\ control \u0001\u001f line sep \u2028 é 日本 😀 </tool_call>",
    "parameters": {"type": "object", "properties": {}, "examples": [
        1.5, 100.0, 1e16, 1e15, 1e-7, 0.0001, -0.0, 3.141592653589793, 0.1 + 0.2, 2.5e-10, 1e22, 123456789.0,
        12345678901234567890, -7, 0, True, False, None, "", [], {},
    ]},
}

TOOL_CHAT_CASES = [
    {"tools": [WEATHER_TOOL, STOCK_TOOL], "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
     "options": {"add_generation_prompt": True}},
    {"tools": [WEATHER_TOOL], "messages": [{"role": "user", "content": "Weather in Oslo?"}],
     "options": {"add_generation_prompt": True, "enable_thinking": False}},
    {"tools": [STOCK_TOOL], "messages": [
        {"role": "system", "content": "You are a careful financial assistant."},
        {"role": "user", "content": "Price of AAPL?"},
    ], "options": {"add_generation_prompt": True}},
    # A full exchange: call, response, answer, follow-up.
    {"tools": [WEATHER_TOOL], "messages": [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris", "unit": "celsius"}}}]},
        {"role": "tool", "content": '{"temperature": 18, "condition": "cloudy"}'},
        {"role": "assistant", "content": "It's 18°C and cloudy in Paris."},
        {"role": "user", "content": "And in Lyon?"},
    ], "options": {"add_generation_prompt": True, "enable_thinking": False}},
    # Text before two calls (string and nested arguments), reasoning, and two
    # tool responses in a row, then the model's turn.
    {"tools": [WEATHER_TOOL, STOCK_TOOL], "messages": [
        {"role": "user", "content": "Compare AAPL and MSFT."},
        {"role": "assistant", "content": "Let me look both up.", "reasoning_content": "\nTwo tickers, two calls.\n", "tool_calls": [
            {"type": "function", "function": {"name": "get_stock_price", "arguments": '{"ticker": "AAPL"}'}},
            {"type": "function", "function": {"name": "get_stock_price", "arguments": {
                "ticker": "MSFT", "days": 7, "threshold": 0.5, "adjusted": False, "note": None,
                "tags": ["tech", "large cap"], "range": {"from": "2026-01-01", "to": "2026-10-27"}, "label": "Microsoft — ü"}}},
        ]},
        {"role": "tool", "content": '{"price": 231.4}'},
        {"role": "tool", "content": '{"price": 512.07}'},
    ], "options": {"add_generation_prompt": True}},
    # A call without the "function" wrapper; the last message is the call.
    {"tools": [EDGE_TOOL], "messages": [
        {"role": "user", "content": "Echo hi."},
        {"role": "assistant", "content": "<think>\nI should call echo.\n</think>\n\n", "tool_calls": [
            {"name": "echo", "arguments": {"text": "hi", "times": 2.0}}]},
    ], "options": {"add_generation_prompt": False}},
    {"tools": [EDGE_TOOL, WEATHER_TOOL], "messages": [{"role": "user", "content": "Hello"}],
     "options": {"add_generation_prompt": True}},
    # Tool responses before the last user query; the earlier assistant turn
    # loses its reasoning.
    {"tools": [WEATHER_TOOL], "messages": [
        {"role": "user", "content": "Weather in Rome?"},
        {"role": "assistant", "content": "<think>call it</think>", "tool_calls": [
            {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Rome"}}}]},
        {"role": "tool", "content": "sunny, 25"},
        {"role": "assistant", "content": "Sunny and 25°C."},
        {"role": "user", "content": "Thanks!"},
    ], "options": {"add_generation_prompt": True}},
]

LONG_TEXT = " ".join(
    [
        "Net sales increased 8% to $94.9 billion, driven by record Services revenue and strong iPhone demand,",
        "while gross margin expanded to 46.3%. Operating expenses rose with headcount and research spending.",
        "The company returned over $29 billion to shareholders through dividends and buybacks during the quarter.",
    ]
    * 40
)

# Model cases: ("raw", text) is encoded without special tokens; ("chat", messages)
# goes through the chat template with thinking disabled.
MODEL_CASES = [
    ("raw", "The capital of France is", 8),  # after the tokenizer's BOS token, if it has one
    ("chat", [{"role": "user", "content": "What is 17 * 23? Answer with just the number."}], 16),
    ("chat", [
        {"role": "system", "content": "You are a concise financial analyst."},
        {"role": "user", "content": "Summarize in one sentence: Net sales increased 8% to $94.9 billion, driven by record Services revenue and strong iPhone demand, while gross margin expanded to 46.3%."},
    ], 48),
    ("chat", [{"role": "user", "content": "Summarize in one sentence:\n\n" + LONG_TEXT}], 24),  # > 2000 tokens
]


RANDOM_CASES = 400
RANDOM_ALPHABET = [
    "abc", "XYZ", "hello", "World", "THE", "i", "a",  # ASCII words
    "0", "7", "42", "123", "98765",  # ASCII digits
    "'s", "'t", "'re", "'ve", "'m", "'ll", "'d", "'S", "'LL", "'x", "'", "\u017f",  # contractions, long s
    " ", "  ", "   ", "\t", "\n", "\r\n", "\r", "\n\n", " \n", "\u00a0", "\u2009", "\u3000", "\u200b", "\u2028", "\x0b", "\x85",
    ".", ",", "!?", "...", "--", "$", "%", "(", ")", "\"", "#", "@", "_", "~", "`", "^", "|", "\\", "<", ">", "/",
    "é", "e\u0301", "\u0301", "ñ", "ß", "Ω", "ж", "Привет", "日本", "中文字", "한국어", "مرحبا", "שלום", "हिन्दी", "ไทย",
    "٣٤", "Ⅻ", "²", "½", "①", "１２",  # digits from other scripts
    "😀", "👍🏽", "👨\u200d👩\u200d👧", "🇺🇸", "✓", "™", "€", "\ufeff",
    "<s>", "</s>", "<unk>", "<think>", "</think>", "<|im_start|>", "<|im_end|>", "<unused_token_5>", "<|im_", "<th", "s>",
]


def random_texts(n, seed=1234):
    r = random.Random(seed)
    return ["".join(r.choice(RANDOM_ALPHABET) for _ in range(r.randint(1, 24))) for _ in range(n)]


def ids_of(encoded):
    """apply_chat_template returns a list or a BatchEncoding depending on the version."""
    if hasattr(encoded, "keys"):
        encoded = encoded["input_ids"]
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return list(encoded)


def main():
    args = sys.argv[1:]
    mode = args[0] if args[:1] in (["--chat-only"], ["--tools"]) else ""
    if mode:
        args = args[1:]
    if len(args) != 2:
        sys.exit(f"usage: python {sys.argv[0]} [--chat-only | --tools] <model_dir> <out_prefix>")
    model_dir, out = args
    chat_only = mode == "--chat-only"
    tok = AutoTokenizer.from_pretrained(model_dir)
    if mode == "--tools":
        cases = []
        for c in TOOL_CHAT_CASES:
            text = tok.apply_chat_template(c["messages"], tools=c["tools"], tokenize=False, **c["options"])
            cases.append({**c, "text": text, "ids": tok(text, add_special_tokens=False)["input_ids"]})
        with open(f"{out}_chat_tools.json", "w") as f:
            json.dump(cases, f, ensure_ascii=False, indent=1)
        print(f"wrote {len(cases)} tool chat cases to {out}_chat_tools.json")
        return

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

    chats = []
    for c in CHAT_CASES:
        text = tok.apply_chat_template(c["messages"], tokenize=False, **c["options"])
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if ids != ids_of(tok.apply_chat_template(c["messages"], tokenize=True, **c["options"])):
            sys.exit("chat template: encoding the rendered text differs from tokenize=True")
        chats.append({**c, "text": text, "ids": ids})
    with open(f"{out}_chat.json", "w") as f:
        json.dump(chats, f, ensure_ascii=False, indent=1)
    if chat_only:
        print(f"wrote {len(chats)} chat cases to {out}_chat.json")
        return

    tokens = token_cases(TOKENIZER_CASES)
    with open(f"{out}_tokens.json", "w") as f:
        json.dump(tokens, f, ensure_ascii=False, indent=1)
    with open(f"{out}_tokens_random.json", "w") as f:
        json.dump(token_cases(random_texts(RANDOM_CASES)), f, ensure_ascii=False, indent=1)

    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32, attn_implementation="eager").eval()
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    tensors = {"rope.inv_freq": model.model.rotary_emb.inv_freq.float().clone()}
    cases = []
    for i, (kind, prompt, max_new) in enumerate(MODEL_CASES):
        if kind == "raw":
            text = (tok.bos_token or "") + prompt
        else:
            text = tok.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        input_ids = torch.tensor([tok(text, add_special_tokens=False)["input_ids"]])
        n = input_ids.shape[1]
        with torch.no_grad():
            seq = model.generate(
                input_ids=input_ids, attention_mask=torch.ones_like(input_ids), max_new_tokens=max_new,
                do_sample=False, temperature=None, top_p=None, top_k=None, eos_token_id=eos, pad_token_id=eos[0],
            )[0]
            new = seq[n:]
            full = model(input_ids=seq[:-1].unsqueeze(0), output_hidden_states=i in HIDDEN_CASES)
        logits = full.logits[0]
        if not torch.equal(logits[n - 1:].argmax(-1), new):
            sys.exit(f"case {i}: teacher-forced argmax does not reproduce greedy generation")
        top = logits.topk(TOP_K, dim=-1)
        tensors[f"case{i}.topk.values"] = top.values.clone()
        tensors[f"case{i}.topk.indices"] = top.indices.to(torch.int32).clone()
        if i in FULL_CASES:
            tensors[f"case{i}.logits"] = logits[n - 1:n - 1 + FULL_LOGIT_STEPS].clone()
        if i in HIDDEN_CASES:
            if len(full.hidden_states) != model.config.num_hidden_layers + 1:
                sys.exit(f"case {i}: {len(full.hidden_states)} hidden states, want layers + 1")
            for j, h in enumerate(full.hidden_states):
                tensors[f"case{i}.hidden.{j}"] = h[0].clone()
        output = tok.decode(new, skip_special_tokens=True)
        cases.append({"kind": kind, "text": text, "input_ids": input_ids[0].tolist(), "output_ids": new.tolist(), "output": output})
        print(f"case {i}: {n:4d} in, {len(new):2d} out: {output!r}")

    save_file(tensors, f"{out}_model.safetensors")
    with open(f"{out}_model.json", "w") as f:
        json.dump({"eos_token_ids": eos, "top_k": TOP_K, "cases": cases}, f, ensure_ascii=False, indent=1)
    print(f"wrote {len(tokens)} tokenizer cases, {len(chats)} chat cases and {len(cases)} model cases to {out}_*")


if __name__ == "__main__":
    main()
