// Command decide answers typed questions about a state with an OpenJev
// checkpoint (see package openjev), on the CPU.
//
//	go run ./cmd/decide <model_dir> <questions.json> < state.txt
//
// questions.json is a list of questions, each with instructions, options and
// an optional rubric:
//
//	[{"instructions": "Is the refund permitted?", "options": ["no", "yes"],
//	  "rubric": {"no": "A condition is missing.", "yes": "Every condition holds."}}]
//
// The state is read from standard input. Each answer is printed as one JSON
// object per line: the question's options with their probabilities, and the
// most probable option.
package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/mhpenta/infer/openjev"
)

func main() {
	if len(os.Args) != 3 {
		fmt.Fprintf(os.Stderr, "usage: %s <model_dir> <questions.json> < state.txt\n", os.Args[0])
		os.Exit(2)
	}
	if err := run(os.Args[1], os.Args[2]); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(dir, questionsFile string) error {
	raw, err := os.ReadFile(questionsFile)
	if err != nil {
		return err
	}
	var questions []struct {
		Instructions string            `json:"instructions"`
		Options      []string          `json:"options"`
		Rubric       map[string]string `json:"rubric"`
	}
	if err := json.Unmarshal(raw, &questions); err != nil {
		return fmt.Errorf("parse %s: %w", questionsFile, err)
	}
	state, err := io.ReadAll(os.Stdin)
	if err != nil {
		return err
	}
	model, err := openjev.Load(dir)
	if err != nil {
		return err
	}
	defer model.Close()

	qs := make([]openjev.Question, len(questions))
	for i, q := range questions {
		qs[i] = openjev.Question{Instructions: q.Instructions, Options: q.Options, Rubric: q.Rubric}
	}
	start := time.Now()
	answers, err := model.Decide(string(state), qs)
	if err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "decided %d questions in %v\n", len(qs), time.Since(start).Round(time.Millisecond))

	enc := json.NewEncoder(os.Stdout)
	for i, a := range answers {
		probs := map[string]float64{}
		for j, o := range qs[i].Options {
			probs[o] = a[j]
		}
		if err := enc.Encode(map[string]any{
			"instructions":  qs[i].Instructions,
			"answer":        qs[i].Options[a.Best()],
			"probabilities": probs,
		}); err != nil {
			return err
		}
	}
	return nil
}
