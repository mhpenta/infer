// Package chat renders conversations into the text a chat model expects,
// reproducing the model's Jinja chat template (as transformers'
// apply_chat_template runs it) in Go.
//
// Templates are recognized by the SHA-256 of their exact source: each one
// supported here is a translation of one template, checked against
// transformers' output, and an unknown template is an error rather than a
// guess. The rendered text is meant for a tokenizer's raw encoding (no added
// post-processor tokens), as the template writes its own.
package chat

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Message is one turn of a conversation.
type Message struct {
	Role    string // "system", "user", "assistant" or "tool" (a tool's response)
	Content string

	// Reasoning is an assistant turn's thinking (reasoning_content). When it
	// is empty, reasoning written in Content as <think>…</think> is split out
	// instead, as the templates do for messages without reasoning_content.
	Reasoning string

	// ToolCalls are the functions an assistant turn calls.
	ToolCalls []ToolCall
}

// ToolCall is one function call by the model.
type ToolCall struct {
	Name string

	// Arguments is a JSON value: normally an object, which templates write
	// as transformers' tojson does (Python's json.dumps), or a JSON string,
	// which they write verbatim, as for string arguments in Python.
	Arguments json.RawMessage
}

// Thinking selects enable_thinking for the generation prompt.
type Thinking int

const (
	ThinkingDefault Thinking = iota // enable_thinking left undefined
	ThinkingOff                     // enable_thinking=False
	ThinkingOn                      // enable_thinking=True
)

// Options are the template's keyword arguments.
type Options struct {
	// AddGenerationPrompt ends the text with the start of an assistant turn.
	AddGenerationPrompt bool
	Thinking            Thinking

	// Tools are the functions the model may call, each a JSON object as
	// transformers' tools argument takes them, such as {"type": "function",
	// "function": {"name": ..., "description": ..., "parameters": {...}}}.
	Tools []json.RawMessage
}

// Template renders conversations for one model family.
type Template struct {
	name       string
	render     func(messages []Message, opts Options) (string, error)
	parseCalls func(reply string) (string, []ToolCall, error) // nil without tool support
}

// Name identifies the template, for logs.
func (t Template) Name() string { return t.name }

// ParseToolCalls splits a reply (the text after any reasoning) into its
// message and the tool calls it makes, in the template's format. It fails
// for a template without tool support, and for a call it cannot read.
func (t Template) ParseToolCalls(reply string) (content string, calls []ToolCall, err error) {
	if t.parseCalls == nil {
		return "", nil, fmt.Errorf("chat: the %s template does not support tools", t.name)
	}
	return t.parseCalls(reply)
}

// Render returns the conversation as the template writes it.
func (t Template) Render(messages []Message, opts Options) (string, error) {
	if len(messages) == 0 {
		return "", errors.New("chat: empty conversation")
	}
	return t.render(messages, opts)
}

// templates maps the SHA-256 of a chat template's source to its Go version.
var templates = map[string]Template{
	// openbmb/MiniCPM5-1B.
	"7451a05cf1e28a79d97d7c0bc951028c0b1915119bf9046acd06a0e3d931f47c": MiniCPM5,
	// openbmb/MiniCPM5-2B.
	"bc1f78ef8a86b6bdf64a056cf89dd0ac506db4f198ab363d85168e2500ab44dd": MiniCPM5For2B,
	// Qwen/Qwen3-1.7B (in tokenizer_config.json).
	"a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8": Qwen3,
}

// Load returns the template of the model in dir: chat_template.jinja, or
// else tokenizer_config.json's chat_template, as transformers reads them.
func Load(dir string) (Template, error) {
	src, err := os.ReadFile(filepath.Join(dir, "chat_template.jinja"))
	if errors.Is(err, os.ErrNotExist) {
		src, err = embeddedTemplate(dir)
	}
	if err != nil {
		return Template{}, fmt.Errorf("chat: %w", err)
	}
	sum := sha256.Sum256(src)
	t, ok := templates[hex.EncodeToString(sum[:])]
	if !ok {
		return Template{}, fmt.Errorf("chat: the chat template in %s (sha256 %x) is not supported", dir, sum)
	}
	return t, nil
}

func embeddedTemplate(dir string) ([]byte, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "tokenizer_config.json"))
	if err != nil {
		return nil, err
	}
	var tc struct {
		ChatTemplate json.RawMessage `json:"chat_template"`
	}
	if err := json.Unmarshal(raw, &tc); err != nil {
		return nil, fmt.Errorf("parse tokenizer_config.json: %w", err)
	}
	var s string
	if err := json.Unmarshal(tc.ChatTemplate, &s); err != nil {
		return nil, errors.New("tokenizer_config.json has no single chat_template")
	}
	return []byte(s), nil
}

// MiniCPM5 is MiniCPM5-1B's template (ChatML with "<s>" first and <think>
// reasoning), without tools: tool definitions, tool calls and "tool"
// messages are not supported. Assistant turns keep their reasoning only
// after the last user query.
var MiniCPM5 = Template{name: "minicpm5", render: renderAs(styleMiniCPM5)}

// MiniCPM5For2B is MiniCPM5-2B's template: MiniCPM5's, except that every
// assistant turn keeps its reasoning, and one with none (and no <think> tags
// in its answer) gets an empty <think> block.
var MiniCPM5For2B = Template{name: "minicpm5-2b", render: renderAs(styleMiniCPM5For2B)}

// Qwen3 is Qwen3's template. Without tools it is MiniCPM5-1B's with three
// differences: no "<s>"; after the last user query, the final assistant turn
// always gets a <think> block, even an empty one; and Thinking: ThinkingOn
// adds nothing to the generation prompt, since the model opens <think>
// itself (ThinkingOff still closes an empty block). With tools, their
// definitions open the system turn; assistant tool calls are written as
// <tool_call> blocks of JSON, and runs of "tool" messages as one user turn
// of <tool_response> blocks.
var Qwen3 = Template{name: "qwen3", render: renderAs(styleQwen3), parseCalls: parseQwen3Calls}

// style is one of the ChatML templates renderChatML reproduces.
type style int

const (
	styleMiniCPM5 style = iota
	styleMiniCPM5For2B
	styleQwen3
)

func renderAs(s style) func([]Message, Options) (string, error) {
	return func(messages []Message, opts Options) (string, error) {
		return renderChatML(messages, opts, s)
	}
}

func renderChatML(messages []Message, opts Options, s style) (string, error) {
	tools := s == styleQwen3 // whether the style renders tools
	for i, m := range messages {
		switch {
		case m.Role == "tool" || len(m.ToolCalls) > 0:
			if !tools {
				return "", fmt.Errorf("chat: message %d: this template's tool format is not supported", i)
			}
		case m.Role != "system" && m.Role != "user" && m.Role != "assistant":
			return "", fmt.Errorf("chat: message %d has role %q; only system, user, assistant and tool are supported", i, m.Role)
		}
	}
	if len(opts.Tools) > 0 && !tools {
		return "", errors.New("chat: this template's tool format is not supported")
	}
	var b strings.Builder
	if s != styleQwen3 {
		b.WriteString("<s>")
	}
	switch {
	case len(opts.Tools) > 0:
		b.WriteString("<|im_start|>system\n")
		if messages[0].Role == "system" {
			b.WriteString(messages[0].Content + "\n\n")
		}
		b.WriteString("# Tools\n\nYou may call one or more functions to assist with the user query.\n\n" +
			"You are provided with function signatures within <tools></tools> XML tags:\n<tools>")
		for i, tool := range opts.Tools {
			js, err := pythonJSON(tool)
			if err != nil {
				return "", fmt.Errorf("chat: tool %d: %w", i, err)
			}
			b.WriteString("\n" + js)
		}
		b.WriteString("\n</tools>\n\nFor each function call, return a json object with function name and arguments " +
			"within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n")
	case messages[0].Role == "system":
		b.WriteString("<|im_start|>system\n" + messages[0].Content + "<|im_end|>\n")
	}
	// The last user query: the last user turn that is not just a tool
	// response. Some styles keep reasoning only after it.
	lastQuery := len(messages) - 1
	for i := len(messages) - 1; i >= 0; i-- {
		if c := messages[i].Content; messages[i].Role == "user" &&
			!(strings.HasPrefix(c, "<tool_response>") && strings.HasSuffix(c, "</tool_response>")) {
			lastQuery = i
			break
		}
	}
	for i, m := range messages {
		switch {
		case m.Role == "user" || (m.Role == "system" && i > 0):
			b.WriteString("<|im_start|>" + m.Role + "\n" + m.Content + "<|im_end|>\n")
		case m.Role == "assistant":
			reasoning, content := splitReasoning(m)
			var think bool // whether the turn gets a <think> block
			switch s {
			case styleMiniCPM5:
				think = i > lastQuery && reasoning != ""
			case styleMiniCPM5For2B:
				think = reasoning != "" || (!strings.Contains(content, "<think>") && !strings.Contains(content, "</think>"))
			case styleQwen3:
				think = i > lastQuery && (i == len(messages)-1 || reasoning != "")
			}
			b.WriteString("<|im_start|>assistant\n")
			if think {
				b.WriteString("<think>\n" + strings.Trim(reasoning, "\n") + "\n</think>\n\n" + strings.TrimLeft(content, "\n"))
			} else {
				b.WriteString(content)
			}
			for k, call := range m.ToolCalls {
				if k > 0 || content != "" {
					b.WriteString("\n")
				}
				args, err := callArguments(call.Arguments)
				if err != nil {
					return "", fmt.Errorf("chat: message %d, tool call %d: %w", i, k, err)
				}
				b.WriteString("<tool_call>\n{\"name\": \"" + call.Name + "\", \"arguments\": " + args + "}\n</tool_call>")
			}
			b.WriteString("<|im_end|>\n")
		case m.Role == "tool":
			if i == 0 || messages[i-1].Role != "tool" {
				b.WriteString("<|im_start|>user")
			}
			b.WriteString("\n<tool_response>\n" + m.Content + "\n</tool_response>")
			if i == len(messages)-1 || messages[i+1].Role != "tool" {
				b.WriteString("<|im_end|>\n")
			}
		}
	}
	if opts.AddGenerationPrompt {
		b.WriteString("<|im_start|>assistant\n")
		switch {
		case opts.Thinking == ThinkingOff:
			b.WriteString("<think>\n\n</think>\n\n")
		case opts.Thinking == ThinkingOn && s != styleQwen3:
			b.WriteString("<think>\n")
		}
	}
	return b.String(), nil
}

// callArguments writes a tool call's arguments: a JSON string verbatim, any
// other value as tojson writes it.
func callArguments(raw json.RawMessage) (string, error) {
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s, nil
	}
	return pythonJSON(raw)
}

// parseQwen3Calls reads Qwen3's tool calls: <tool_call> blocks, each holding
// a JSON object {"name": ..., "arguments": {...}}. The content is the reply
// without them, trimmed.
func parseQwen3Calls(reply string) (string, []ToolCall, error) {
	const open, closing = "<tool_call>", "</tool_call>"
	var content strings.Builder
	var calls []ToolCall
	rest := reply
	for {
		before, after, found := strings.Cut(rest, open)
		content.WriteString(before)
		if !found {
			break
		}
		body, tail, closed := strings.Cut(after, closing)
		if !closed {
			return "", nil, fmt.Errorf("chat: tool call %d is not closed with %s", len(calls), closing)
		}
		var call struct {
			Name      *string         `json:"name"`
			Arguments json.RawMessage `json:"arguments"`
		}
		if err := json.Unmarshal([]byte(body), &call); err != nil {
			return "", nil, fmt.Errorf("chat: tool call %d: %w", len(calls), err)
		}
		if call.Name == nil {
			return "", nil, fmt.Errorf("chat: tool call %d has no name", len(calls))
		}
		if call.Arguments == nil {
			call.Arguments = json.RawMessage("{}")
		}
		calls = append(calls, ToolCall{Name: *call.Name, Arguments: call.Arguments})
		rest = tail
	}
	return strings.TrimSpace(content.String()), calls, nil
}

// splitReasoning returns an assistant message's reasoning and answer: its
// Reasoning field, or else, when Content has a </think>, the text after the
// last <think> before the first </think>, and the text after the last
// </think>, as the template's Python string methods compute them.
func splitReasoning(m Message) (reasoning, content string) {
	if m.Reasoning != "" || !strings.Contains(m.Content, "</think>") {
		return m.Reasoning, m.Content
	}
	before, _, _ := strings.Cut(m.Content, "</think>")
	before = strings.TrimRight(before, "\n")
	if i := strings.LastIndex(before, "<think>"); i >= 0 {
		before = before[i+len("<think>"):]
	}
	after := m.Content[strings.LastIndex(m.Content, "</think>")+len("</think>"):]
	return strings.TrimLeft(before, "\n"), strings.TrimLeft(after, "\n")
}
