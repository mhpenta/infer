package hf

import (
	"strings"
	"testing"
)

func TestAddedSplit(t *testing.T) {
	added, err := NewAdded([]AddedToken{
		{ID: 1, Content: "<s>"},
		{ID: 2, Content: "<s>x"}, // longer, at the same position
		{ID: 3, Content: "<t>", Normalized: true},
		{ID: 4, Content: "s><t", Normalized: true}, // overlaps a raw token; raw tokens go first
	})
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct{ in, want string }{
		{"", ""},
		{"plain", "plain"},
		{"a<s>b", "a [1] b"},
		{"a<s>xb", "a [2] b"},
		{"<s><t><s>", "[1] [3] [1]"},
		{"<t>", "[3]"},
		{"s><t>", "[4] >"},
		{"<s><t", "[1] <t"},
	}
	for _, tt := range tests {
		var parts []string
		for _, s := range added.Split(tt.in, strings.ToLower) {
			if s.Token != nil {
				parts = append(parts, "["+string(rune('0'+s.Token.ID))+"]")
			} else {
				parts = append(parts, s.Text)
			}
		}
		if got := strings.Join(parts, " "); got != tt.want {
			t.Errorf("Split(%q) = %q, want %q", tt.in, got, tt.want)
		}
	}
	if _, err := NewAdded([]AddedToken{{ID: 1, Content: "x", LStrip: true}}); err == nil {
		t.Error("NewAdded accepted lstrip")
	}
}
