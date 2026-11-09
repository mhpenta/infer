package llama

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/mhpenta/infer/accel"
	"github.com/mhpenta/infer/internal/half"
	"github.com/mhpenta/infer/ops"
	"github.com/mhpenta/infer/safetensors"
	"github.com/mhpenta/infer/sample"
	"github.com/mhpenta/infer/tensor"
)

// loadTiny loads writeTinyModel's model on the CPU.
func loadTiny(t *testing.T) *Model {
	t.Helper()
	m, err := LoadWith(writeTinyModel(t), Options{Device: accel.CPU})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { m.Close() })
	return m
}

// TestLastTokenNeverRuns counts the forward passes of a generation that
// ends by reaching maxNew, not an end token: the prompt's, then one per
// token but the last, whose logits nothing would read. A Session's cache
// then holds exactly those positions.
func TestLastTokenNeverRuns(t *testing.T) {
	m := loadTiny(t)
	m.Config.EOS = nil // so that only maxNew ends a generation
	prompt := []int{2, 3, 5, 7}
	for _, maxNew := range []int{0, 1, 2, 5} {
		var passes []int // rows of each forward pass
		capture := Capture{Layers: []int{0}, Points: []Point{MLPOut}, Fn: func(a Activation) {
			start := 0 // the prompt's pass, then a position per token
			if len(passes) > 0 {
				start = len(prompt) + len(passes) - 1
			}
			if a.Start != start {
				t.Errorf("maxNew %d: pass %d starts at position %d, want %d", maxNew, len(passes), a.Start, start)
			}
			passes = append(passes, a.Values.Rows())
		}}
		var out []int
		if err := m.GenerateCapture(prompt, maxNew, nil, capture, func(token int, _ []float32) bool {
			out = append(out, token)
			return true
		}); err != nil {
			t.Fatal(err)
		}
		want := []int{len(prompt)}
		for range max(maxNew-1, 0) {
			want = append(want, 1)
		}
		if len(out) != maxNew || !slices.Equal(passes, want) {
			t.Errorf("maxNew %d: %d tokens from forward passes of %v rows, want passes of %v", maxNew, len(out), passes, want)
		}

		s := m.NewSession()
		if err := s.GenerateWith(prompt, maxNew, nil, func(int, []float32) bool { return true }); err != nil {
			t.Fatal(err)
		}
		next := append(slices.Clone(prompt), out...)
		next = append(next, 11)
		if err := s.GenerateWith(next, 1, nil, func(int, []float32) bool { return true }); err != nil {
			t.Fatal(err)
		}
		if want := len(prompt) + max(maxNew-1, 0); s.Reused() != want {
			t.Errorf("maxNew %d: the next round reused %d positions, want %d", maxNew, s.Reused(), want)
		}
		s.Close()
	}
}

// TestGenerateRejectsNegativeMaxNew checks every way to generate.
func TestGenerateRejectsNegativeMaxNew(t *testing.T) {
	m := loadTiny(t)
	ids := []int{2, 3}
	fn := func(int, []float32) bool { return true }
	s := m.NewSession()
	defer s.Close()
	for name, err := range map[string]error{
		"GenerateWith":         m.GenerateWith(ids, -1, nil, fn),
		"GenerateCapture":      m.GenerateCapture(ids, -1, nil, Capture{Points: []Point{MLPOut}, Fn: func(Activation) {}}, fn),
		"Session.GenerateWith": s.GenerateWith(ids, -1, nil, fn),
	} {
		if err == nil {
			t.Errorf("%s with a negative maxNew succeeded", name)
		}
	}
	if _, err := m.Generate(ids, -1); err == nil {
		t.Error("Generate with a negative maxNew succeeded")
	}
}

// TestGenerateReportsSamplerError generates with a temperature so small
// that dividing the logits by it overflows, leaving the sampler nothing
// finite to choose from.
func TestGenerateReportsSamplerError(t *testing.T) {
	m := loadTiny(t)
	s, err := sample.New(sample.Config{Temperature: 1e-40}, 1)
	if err != nil {
		t.Fatal(err)
	}
	err = m.GenerateWith([]int{2, 3}, 4, s, func(int, []float32) bool { return true })
	if !errors.Is(err, sample.ErrNotFinite) {
		t.Errorf("GenerateWith = %v, want sample.ErrNotFinite", err)
	}
}

// TestHalfPanelsOnlyWithAKernel loads an fp16 weight as the loader packs it
// with and without a SIMD kernel for fp16 panels: fp16 panels with one,
// float32 panels without, and the same product either way.
func TestHalfPanelsOnlyWithAKernel(t *testing.T) {
	const out, in = 3, 8
	header, _ := json.Marshal(map[string]any{
		"w": map[string]any{"dtype": "F16", "shape": []int{out, in}, "data_offsets": []int{0, 2 * out * in}},
	})
	file := append(binary.LittleEndian.AppendUint64(nil, uint64(len(header))), header...)
	for i := range out * in {
		h, ok := half.FromFloat32Exact(float32(i-7) / 8)
		if !ok {
			t.Fatal("the test's weights are not exact in fp16")
		}
		file = binary.LittleEndian.AppendUint16(file, h)
	}
	path := filepath.Join(t.TempDir(), "w.safetensors")
	if err := os.WriteFile(path, file, 0o644); err != nil {
		t.Fatal(err)
	}
	f, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	x := tensor.New(2, in)
	for i := range x.Data {
		x.Data[i] = float32(i%5) - 2
	}
	saved := useHalfPanels
	defer func() { useHalfPanels = saved }()
	var products [][]float32
	for _, kernel := range []bool{false, true} {
		useHalfPanels = func() bool { return kernel }
		l := loader{files: map[string]*safetensors.File{"w": f}}
		p := l.linear("w", out, in)
		if l.err != nil {
			t.Fatal(l.err)
		}
		if p.Half() != kernel {
			t.Errorf("with an fp16 kernel %v: fp16 panels %v", kernel, p.Half())
		}
		products = append(products, ops.MatMulPacked(x, p).Data)
	}
	if !bitsEqual(products[0], products[1]) {
		t.Errorf("float32 panels give %v, fp16 panels %v", products[0], products[1])
	}
}
