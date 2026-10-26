// Package golden loads the reference outputs written by
// scripts/dump_golden_bert.py and locates model weights for tests.
package golden

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// Case is one text and the ids the Python tokenizer produced for it.
type Case struct {
	Text string `json:"text"`
	IDs  []int  `json:"ids"`
}

// Root returns the repository root, found from this source file.
func Root() string {
	_, file, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(file), "..", "..")
}

// Path returns a path under testdata/golden.
func Path(name string) string {
	return filepath.Join(Root(), "testdata", "golden", name)
}

// Cases reads a JSON list of cases from testdata/golden.
func Cases(t testing.TB, name string) []Case {
	t.Helper()
	raw, err := os.ReadFile(Path(name))
	if err != nil {
		t.Fatalf("read golden cases: %v", err)
	}
	var cases []Case
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatalf("parse golden cases %s: %v", name, err)
	}
	return cases
}

// ModelDir returns the model directory named by envVar, or else
// testdata/models/<name>. Weights are not checked in, so the test is skipped
// when neither exists.
func ModelDir(t testing.TB, envVar, name string) string {
	t.Helper()
	dir := os.Getenv(envVar)
	if dir == "" {
		dir = filepath.Join(Root(), "testdata", "models", name)
	}
	if _, err := os.Stat(filepath.Join(dir, "config.json")); err != nil {
		t.Skipf("model %s not found at %s; set %s or link it into testdata/models", name, dir, envVar)
	}
	return dir
}

// JSON reads testdata/golden/<name> into v.
func JSON(t testing.TB, name string, v any) {
	t.Helper()
	raw, err := os.ReadFile(Path(name))
	if err != nil {
		t.Fatalf("read golden %s: %v", name, err)
	}
	if err := json.Unmarshal(raw, v); err != nil {
		t.Fatalf("parse golden %s: %v", name, err)
	}
}
