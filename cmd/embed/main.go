// Command embed prints mean-pooled BERT embeddings, matching
// sentence-transformers' all-MiniLM-L6-v2 without its Normalize step.
//
//	go run ./cmd/embed <model_dir> "some text" ["more text" ...]
//
// The model directory needs config.json, model.safetensors, vocab.txt and
// tokenizer_config.json. Each embedding is printed as one JSON array per line.
package main

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/mhpenta/infer/model/bert"
	"github.com/mhpenta/infer/pool"
	"github.com/mhpenta/infer/tokenizer/wordpiece"
)

func main() {
	if len(os.Args) < 3 {
		fmt.Fprintf(os.Stderr, "usage: %s <model_dir> <text> [more text...]\n", os.Args[0])
		os.Exit(2)
	}
	if err := run(os.Args[1], os.Args[2:]); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(dir string, texts []string) error {
	tok, err := wordpiece.LoadDir(dir)
	if err != nil {
		return err
	}
	model, err := bert.Load(dir)
	if err != nil {
		return err
	}
	defer model.Close()

	enc := json.NewEncoder(os.Stdout)
	for _, text := range texts {
		hidden, err := model.Forward(tok.Encode(text))
		if err != nil {
			return fmt.Errorf("embed %q: %w", text, err)
		}
		if err := enc.Encode(pool.Mean(hidden)); err != nil {
			return err
		}
	}
	return nil
}
