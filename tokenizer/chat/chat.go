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
	Role    string // "system", "user" or "assistant"
	Content string

	// Reasoning is an assistant turn's thinking (reasoning_content). When it
	// is empty, reasoning written in Content as <think>…</think> is split out
	// instead, as the templates do for messages without reasoning_content.
	Reasoning string
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
}

// Template renders conversations for one model family.
type Template struct {
	name   string
	render func(messages []Message, opts Options) (string, error)
}

// Name identifies the template, for logs.
func (t Template) Name() string { return t.name }

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
var MiniCPM5 = Template{name: "minicpm5", render: func(messages []Message, opts Options) (string, error) {
	return renderMiniCPM5(messages, opts, false)
}}

// MiniCPM5For2B is MiniCPM5-2B's template: MiniCPM5's, except that every
// assistant turn keeps its reasoning, and one with none (and no <think> tags
// in its answer) gets an empty <think> block.
var MiniCPM5For2B = Template{name: "minicpm5-2b", render: func(messages []Message, opts Options) (string, error) {
	return renderMiniCPM5(messages, opts, true)
}}

func renderMiniCPM5(messages []Message, opts Options, allReasoning bool) (string, error) {
	for i, m := range messages {
		if m.Role != "system" && m.Role != "user" && m.Role != "assistant" {
			return "", fmt.Errorf("chat: message %d has role %q; minicpm5 supports system, user and assistant", i, m.Role)
		}
	}
	var b strings.Builder
	b.WriteString("<s>")
	if messages[0].Role == "system" {
		b.WriteString("<|im_start|>system\n" + messages[0].Content + "<|im_end|>\n")
	}
	// MiniCPM5-1B keeps reasoning only after the last user query: the last
	// user turn that is not just a tool response.
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
			b.WriteString("<|im_start|>assistant\n")
			switch {
			case reasoning != "" && (allReasoning || i > lastQuery):
				b.WriteString("<think>\n" + strings.Trim(reasoning, "\n") + "\n</think>\n\n" + strings.TrimLeft(content, "\n"))
			case allReasoning && !strings.Contains(content, "<think>") && !strings.Contains(content, "</think>"):
				b.WriteString("<think>\n\n</think>\n\n" + strings.TrimLeft(content, "\n"))
			default:
				b.WriteString(content)
			}
			b.WriteString("<|im_end|>\n")
		}
	}
	if opts.AddGenerationPrompt {
		b.WriteString("<|im_start|>assistant\n")
		switch opts.Thinking {
		case ThinkingOff:
			b.WriteString("<think>\n\n</think>\n\n")
		case ThinkingOn:
			b.WriteString("<think>\n")
		}
	}
	return b.String(), nil
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
