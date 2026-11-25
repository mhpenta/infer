package chat

import (
	"encoding/json"
	"slices"
	"testing"

	"github.com/mhpenta/infer/internal/golden"
	"github.com/mhpenta/infer/tokenizer/bpe"
)

type inkling2Case struct {
	Messages []struct {
		Role             string `json:"role"`
		Content          string `json:"content"`
		ReasoningContent string `json:"reasoning_content"`
		Name             string `json:"name"`
		ToolCalls        []struct {
			Function struct {
				Name      string          `json:"name"`
				Arguments json.RawMessage `json:"arguments"`
			} `json:"function"`
		} `json:"tool_calls"`
	} `json:"messages"`
	Options struct {
		Tools           []json.RawMessage `json:"tools"`
		ReasoningEffort json.RawMessage   `json:"reasoning_effort"`
	} `json:"options"`
	Text string `json:"text"`
	IDs  []int  `json:"ids"`
}

// TestInkling2MatchesPython renders Inkling 2's golden conversations, with
// reasoning efforts, reasoning, tools and tool calls, and checks the text and
// its tokens against transformers' apply_chat_template.
func TestInkling2MatchesPython(t *testing.T) {
	dir := golden.ModelDir(t, "INFER_INKLING2_DIR", "inkling2-mini")
	tmpl, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	if tmpl.Name() != "inkling2" {
		t.Fatalf("Load recognized %q, want inkling2", tmpl.Name())
	}
	tok, err := bpe.LoadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	var cases []inkling2Case
	golden.JSON(t, "inkling2_mini_chat.json", &cases)
	for i, c := range cases {
		var messages []Message
		for _, m := range c.Messages {
			msg := Message{Role: m.Role, Content: m.Content, Reasoning: m.ReasoningContent, Name: m.Name}
			for _, call := range m.ToolCalls {
				msg.ToolCalls = append(msg.ToolCalls, ToolCall{Name: call.Function.Name, Arguments: call.Function.Arguments})
			}
			messages = append(messages, msg)
		}
		opts := Options{Tools: c.Options.Tools}
		if e := c.Options.ReasoningEffort; e != nil {
			var s string
			if json.Unmarshal(e, &s) != nil {
				s = string(e) // a number
			}
			opts.ReasoningEffort = s
		}
		text, err := tmpl.Render(messages, opts)
		if err != nil {
			t.Fatalf("case %d: %v", i, err)
		}
		if text != c.Text {
			t.Errorf("case %d:\n got %q\nwant %q", i, text, c.Text)
			continue
		}
		if ids := tok.EncodeRaw(text); !slices.Equal(ids, c.IDs) {
			t.Errorf("case %d: ids\n got %v\nwant %v", i, ids, c.IDs)
		}
	}
}

func TestInkling2Effort(t *testing.T) {
	for effort, want := range map[string]string{"": "0.9", "none": "0", "low": "0.2", " max ": "0.99", "0.5": "0.5", "0": "0", "1e-05": "1e-05"} {
		got, err := Inkling2.Render([]Message{{Role: "user", Content: "x"}}, Options{ReasoningEffort: effort})
		if err != nil {
			t.Fatalf("effort %q: %v", effort, err)
		}
		if w := "<|message_system|><|content_text|>Thinking effort level: " + want + "<|end_message|>"; got[:len(w)] != w {
			t.Errorf("effort %q: %q", effort, got)
		}
	}
	for _, effort := range []string{"huge", "1", "-0.1"} {
		if _, err := Inkling2.Render([]Message{{Role: "user", Content: "x"}}, Options{ReasoningEffort: effort}); err == nil {
			t.Errorf("effort %q was accepted", effort)
		}
	}
}

func TestInkling2ParseToolCalls(t *testing.T) {
	reply := "<|message_model|><|content_thinking|>Need weather.<|end_message|>" +
		"<|message_model|><|content_text|>Checking.<|end_message|>" +
		`<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"Paris"}}<|end_message|>` +
		"<|content_model_end_sampling|>"
	content, calls, err := Inkling2.ParseToolCalls(reply)
	if err != nil {
		t.Fatal(err)
	}
	if content != "Checking." || len(calls) != 1 || calls[0].Name != "get_weather" || string(calls[0].Arguments) != `{"city":"Paris"}` {
		t.Errorf("got %q, %+v", content, calls)
	}
}
