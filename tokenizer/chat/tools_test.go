package chat

import (
	"encoding/json"
	"slices"
	"strings"
	"testing"

	"github.com/mhpenta/infer/internal/golden"
	"github.com/mhpenta/infer/tokenizer/bpe"
)

// toolChatCase is a golden conversation with tools.
type toolChatCase struct {
	Tools    []json.RawMessage `json:"tools"`
	Messages []struct {
		Role             string `json:"role"`
		Content          string `json:"content"`
		ReasoningContent string `json:"reasoning_content"`
		ToolCalls        []struct {
			Name      string          `json:"name"`
			Arguments json.RawMessage `json:"arguments"`
			Function  *struct {
				Name      string          `json:"name"`
				Arguments json.RawMessage `json:"arguments"`
			} `json:"function"`
		} `json:"tool_calls"`
	} `json:"messages"`
	Options struct {
		AddGenerationPrompt bool  `json:"add_generation_prompt"`
		EnableThinking      *bool `json:"enable_thinking"`
	} `json:"options"`
	Text string `json:"text"`
	IDs  []int  `json:"ids"`
}

// TestQwen3ToolsMatchPython renders conversations with tool definitions,
// assistant tool calls (with and without the "function" wrapper, with
// string and object arguments) and tool responses, and checks the text and
// its tokens against transformers' apply_chat_template.
func TestQwen3ToolsMatchPython(t *testing.T) {
	dir := golden.ModelDir(t, "INFER_QWEN3_DIR", "qwen3-1.7b")
	tmpl, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	tok, err := bpe.LoadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	var cases []toolChatCase
	golden.JSON(t, "qwen3_1.7b_chat_tools.json", &cases)
	for i, c := range cases {
		var messages []Message
		for _, m := range c.Messages {
			msg := Message{Role: m.Role, Content: m.Content, Reasoning: m.ReasoningContent}
			for _, call := range m.ToolCalls {
				if f := call.Function; f != nil {
					msg.ToolCalls = append(msg.ToolCalls, ToolCall{Name: f.Name, Arguments: f.Arguments})
				} else {
					msg.ToolCalls = append(msg.ToolCalls, ToolCall{Name: call.Name, Arguments: call.Arguments})
				}
			}
			messages = append(messages, msg)
		}
		opts := Options{AddGenerationPrompt: c.Options.AddGenerationPrompt, Tools: c.Tools}
		if e := c.Options.EnableThinking; e != nil {
			opts.Thinking = ThinkingOff
			if *e {
				opts.Thinking = ThinkingOn
			}
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
			t.Errorf("case %d: ids differ", i)
		}
	}
}

// TestPythonJSON checks the tojson writer against Python's
// json.dumps(json.loads(s), ensure_ascii=False).
func TestPythonJSON(t *testing.T) {
	tests := []struct{ in, want string }{
		{"1.5", "1.5"},
		{"100.0", "100.0"},
		{"100", "100"},
		{"1e16", "1e+16"},
		{"1E16", "1e+16"},
		{"1e15", "1000000000000000.0"},
		{"1e-7", "1e-07"},
		{"0.0001", "0.0001"},
		{"0.00001", "1e-05"},
		{"-0.0", "-0.0"},
		{"-0", "0"},
		{"0", "0"},
		{"0.0", "0.0"},
		{"3.141592653589793", "3.141592653589793"},
		{"0.30000000000000004", "0.30000000000000004"},
		{"2.5e-10", "2.5e-10"},
		{"1e22", "1e+22"},
		{"123456789.0", "123456789.0"},
		{"12345678901234567890", "12345678901234567890"},
		{"-7", "-7"},
		{"1e308", "1e+308"},
		{"5e-324", "5e-324"},
		{"1.0e+2", "100.0"},
		{"12.50", "12.5"},
		{"-1.25e-3", "-0.00125"},
		{`"tab\tq\"b\\ \u0001\u001f\u007f \u2028 é 😀 <&>"`, `"tab\tq\"b\\ \u0001\u001f` + "\x7f \u2028" + ` é 😀 <&>"`},
		{`{"b": 1, "a": [true, false, null, {}, []]}`, `{"b": 1, "a": [true, false, null, {}, []]}`},
		{"[ 1 , 2 ]", "[1, 2]"},
	}
	for _, tt := range tests {
		got, err := pythonJSON(json.RawMessage(tt.in))
		if err != nil {
			t.Errorf("pythonJSON(%s): %v", tt.in, err)
			continue
		}
		if got != tt.want {
			t.Errorf("pythonJSON(%s) = %s, want %s", tt.in, got, tt.want)
		}
	}
	for _, bad := range []string{"", "{", "1 2", `{"a": }`} {
		if _, err := pythonJSON(json.RawMessage(bad)); err == nil {
			t.Errorf("pythonJSON(%q) succeeded", bad)
		}
	}
}

// TestParseToolCalls reads Qwen3's <tool_call> blocks back, including the
// ones Render writes.
func TestParseToolCalls(t *testing.T) {
	reply := "Let me check.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\", \"unit\": \"celsius\"}}\n</tool_call>\n" +
		"<tool_call>\n{\"name\": \"get_time\", \"arguments\": {}}\n</tool_call>"
	content, calls, err := Qwen3.ParseToolCalls(reply)
	if err != nil {
		t.Fatal(err)
	}
	if content != "Let me check." || len(calls) != 2 || calls[0].Name != "get_weather" || calls[1].Name != "get_time" {
		t.Fatalf("content %q, calls %+v", content, calls)
	}
	var args struct{ City, Unit string }
	if err := json.Unmarshal(calls[0].Arguments, &args); err != nil || args.City != "Paris" || args.Unit != "celsius" {
		t.Errorf("arguments %s: %+v, %v", calls[0].Arguments, args, err)
	}

	// Render a call, then parse the assistant turn back.
	text, err := Qwen3.Render([]Message{
		{Role: "user", Content: "q"},
		{Role: "assistant", Content: "ok", ToolCalls: []ToolCall{{Name: "f", Arguments: json.RawMessage(`{"x": [1, 2.5, "é"]}`)}}},
	}, Options{})
	if err != nil {
		t.Fatal(err)
	}
	turn := text[strings.LastIndex(text, "</think>\n\n")+len("</think>\n\n") : strings.LastIndex(text, "<|im_end|>")]
	content, calls, err = Qwen3.ParseToolCalls(turn)
	if err != nil || content != "ok" || len(calls) != 1 || calls[0].Name != "f" || string(calls[0].Arguments) != `{"x": [1, 2.5, "é"]}` {
		t.Errorf("round trip of %q: %q, %+v, %v", turn, content, calls, err)
	}

	if content, calls, err := Qwen3.ParseToolCalls("just text "); err != nil || content != "just text" || calls != nil {
		t.Errorf("plain reply: %q, %v, %v", content, calls, err)
	}
	for _, bad := range []string{
		"<tool_call>\n{\"name\": \"f\", \"arguments\": {}}", // not closed
		"<tool_call>\n{\"name\": \"f\", \"arguments\": {</tool_call>",
		"<tool_call>\n{\"arguments\": {}}\n</tool_call>", // no name
	} {
		if _, _, err := Qwen3.ParseToolCalls(bad); err == nil {
			t.Errorf("ParseToolCalls(%q) succeeded", bad)
		}
	}
	if _, _, err := MiniCPM5.ParseToolCalls(reply); err == nil {
		t.Error("MiniCPM5 parsed tool calls")
	}
}

func TestRenderRejectsTools(t *testing.T) {
	tool := json.RawMessage(`{"type": "function", "function": {"name": "f"}}`)
	for _, tmpl := range []Template{MiniCPM5, MiniCPM5For2B} {
		if _, err := tmpl.Render([]Message{{Role: "user", Content: "q"}}, Options{Tools: []json.RawMessage{tool}}); err == nil {
			t.Errorf("%s rendered tools", tmpl.Name())
		}
		if _, err := tmpl.Render([]Message{{Role: "tool", Content: "42"}}, Options{}); err == nil {
			t.Errorf("%s rendered a tool message", tmpl.Name())
		}
	}
	if _, err := Qwen3.Render([]Message{{Role: "user", Content: "q"}}, Options{Tools: []json.RawMessage{json.RawMessage("{")}}); err == nil {
		t.Error("Qwen3 rendered a tool that is not JSON")
	}
}
