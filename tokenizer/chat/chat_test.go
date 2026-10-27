package chat

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/mhpenta/infer/internal/golden"
	"github.com/mhpenta/infer/tokenizer/bpe"
)

type chatCase struct {
	Messages []struct {
		Role             string `json:"role"`
		Content          string `json:"content"`
		ReasoningContent string `json:"reasoning_content"`
	} `json:"messages"`
	Options struct {
		AddGenerationPrompt bool  `json:"add_generation_prompt"`
		EnableThinking      *bool `json:"enable_thinking"`
	} `json:"options"`
	Text string `json:"text"`
	IDs  []int  `json:"ids"`
}

// TestMiniCPM5MatchesPython renders every golden conversation and checks the
// text, and its tokens, against transformers' apply_chat_template, for each
// MiniCPM5 template.
func TestMiniCPM5MatchesPython(t *testing.T) {
	for _, tc := range []struct{ envVar, model, golden, template string }{
		{"INFER_MINICPM5_DIR", "minicpm5-1b", "minicpm5_1b_chat.json", "minicpm5"},
		{"INFER_MINICPM5_2B_DIR", "minicpm5-2b", "minicpm5_2b_chat.json", "minicpm5-2b"},
	} {
		t.Run(tc.model, func(t *testing.T) {
			dir := golden.ModelDir(t, tc.envVar, tc.model)
			tmpl, err := Load(dir)
			if err != nil {
				t.Fatal(err)
			}
			if tmpl.Name() != tc.template {
				t.Fatalf("Load recognized %q, want %q", tmpl.Name(), tc.template)
			}
			tok, err := bpe.LoadDir(dir)
			if err != nil {
				t.Fatal(err)
			}
			var cases []chatCase
			golden.JSON(t, tc.golden, &cases)
			checkRendering(t, tmpl, tok, cases)
		})
	}
}

func checkRendering(t *testing.T, tmpl Template, tok *bpe.Tokenizer, cases []chatCase) {
	t.Helper()
	for i, c := range cases {
		var messages []Message
		for _, m := range c.Messages {
			messages = append(messages, Message{Role: m.Role, Content: m.Content, Reasoning: m.ReasoningContent})
		}
		opts := Options{AddGenerationPrompt: c.Options.AddGenerationPrompt}
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
			t.Errorf("case %d: ids\n got %v\nwant %v", i, ids, c.IDs)
		}
	}
}

func TestRenderRejects(t *testing.T) {
	if _, err := MiniCPM5.Render(nil, Options{}); err == nil {
		t.Error("rendered an empty conversation")
	}
	if _, err := MiniCPM5.Render([]Message{{Role: "tool", Content: "42"}}, Options{}); err == nil {
		t.Error("rendered a tool message")
	}
}

func TestLoadRejectsUnknownTemplate(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "chat_template.jinja"), []byte("{{ messages }}"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(dir); err == nil || !strings.Contains(err.Error(), "not supported") {
		t.Errorf("Load of an unknown template: %v", err)
	}
	if _, err := Load(t.TempDir()); err == nil {
		t.Error("Load succeeded without a template")
	}
}
