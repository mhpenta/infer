package llama

import (
	"encoding/json"
	"fmt"
	"maps"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/mhpenta/infer/accel"
	"github.com/mhpenta/infer/internal/golden"
	"github.com/mhpenta/infer/safetensors"
	"github.com/mhpenta/infer/tensor"
)

// baseWeights reads a checkpoint's index, from tensor name to shard.
func baseWeights(t testing.TB, base string) map[string]string {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(base, "model.safetensors.index.json"))
	if err != nil {
		t.Fatal(err)
	}
	var index struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	if err := json.Unmarshal(raw, &index); err != nil {
		t.Fatal(err)
	}
	return index.WeightMap
}

// editedDir lays out a checkpoint that edits base without copying it: every
// file but the weights symlinked from base, base's shards symlinked as
// base-N.safetensors, delta.safetensors holding the named tensors in dtype,
// each multiplied by its scale (1 if it has none), and an index sending those
// to the delta and all others to the links. No file may be called
// model.safetensors: openWeights prefers it and would ignore the index, and
// with it the delta (TestEditedModelSafetensorsWins).
func editedDir(t testing.TB, base, dtype string, scales map[string]float32, names ...string) string {
	t.Helper()
	dir := t.TempDir()
	entries, err := os.ReadDir(base)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range entries {
		if name := e.Name(); !strings.HasSuffix(name, ".safetensors") && name != "model.safetensors.index.json" {
			if err := os.Symlink(filepath.Join(base, name), filepath.Join(dir, name)); err != nil {
				t.Fatal(err)
			}
		}
	}
	weights := baseWeights(t, base)
	links := map[string]string{}
	for i, shard := range slices.Compact(slices.Sorted(maps.Values(weights))) {
		links[shard] = fmt.Sprintf("base-%d.safetensors", i)
		if err := os.Symlink(filepath.Join(base, shard), filepath.Join(dir, links[shard])); err != nil {
			t.Fatal(err)
		}
	}
	var delta []safetensors.Output
	for _, name := range names {
		scale, ok := scales[name]
		if !ok {
			scale = 1
		}
		delta = append(delta, scaled(t, filepath.Join(base, weights[name]), name, dtype, scale))
	}
	if err := safetensors.Write(filepath.Join(dir, "delta.safetensors"), delta, nil); err != nil {
		t.Fatal(err)
	}
	index := map[string]string{}
	for name, shard := range weights {
		index[name] = links[shard]
		if slices.Contains(names, name) {
			index[name] = "delta.safetensors"
		}
	}
	raw, err := json.Marshal(map[string]any{"metadata": map[string]any{}, "weight_map": index})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), raw, 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

// run is a generation's tokens and each step's logits.
type run struct {
	tokens []int
	logits [][]float32
}

func generateRun(t testing.TB, m *Model, prompt []int, n int) run {
	t.Helper()
	var r run
	if err := m.GenerateFunc(prompt, n, func(token int, logits []float32) bool {
		r.tokens = append(r.tokens, token)
		r.logits = append(r.logits, slices.Clone(logits))
		return true
	}); err != nil {
		t.Fatal(err)
	}
	return r
}

// diff returns how many values differ in any bit and the largest difference
// relative to the row's largest value.
func (r run) diff(o run) (int, float64) {
	bits, worst := 0, 0.0
	for s := range min(len(r.logits), len(o.logits)) {
		peak := 0.0
		for _, v := range o.logits[s] {
			peak = max(peak, math.Abs(float64(v)))
		}
		for i, v := range r.logits[s] {
			if math.Float32bits(v) != math.Float32bits(o.logits[s][i]) {
				bits++
			}
			worst = max(worst, math.Abs(float64(v)-float64(o.logits[s][i]))/peak)
		}
	}
	return bits, worst
}

// hooksGolden is scripts/dump_golden_llama_hooks.py's description of its
// tensors.
type hooksGolden struct {
	Layers      []int `json:"layers"`
	WholeCases  []int `json:"whole_cases"`
	CachedCases []int `json:"cached_cases"`
	EditCase    int   `json:"edit_case"`
	Edits       []struct {
		Name      string             `json:"name"`
		Scales    map[string]float32 `json:"scales"`
		OutputIDs []int              `json:"output_ids"`
	} `json:"edits"`
}

// hooks loads g's hook goldens.
func (g goldenModel) hooks(t testing.TB) (hooksGolden, *safetensors.File) {
	t.Helper()
	var h hooksGolden
	golden.JSON(t, g.prefix+"_hooks.json", &h)
	f, err := safetensors.Open(golden.Path(g.prefix + "_hooks.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { f.Close() })
	return h, f
}

// baseRuns generates c with the base checkpoint on each device, once.
func baseRuns(t *testing.T, base string, c modelCase, devices []string) map[string]run {
	t.Helper()
	runs := map[string]run{}
	for _, device := range devices {
		m, err := LoadWith(base, Options{Device: device})
		if err != nil {
			t.Fatal(err)
		}
		runs[device] = generateRun(t, m, c.InputIDs, len(c.OutputIDs))
		if !slices.Equal(runs[device].tokens, c.OutputIDs) {
			t.Fatalf("the base model on %s generated %v, want %v", device, runs[device].tokens, c.OutputIDs)
		}
		m.Close()
	}
	return runs
}

// heldTypes returns how the model holds layer 3's o_proj, its gate and up
// stack and its q, k and v stack: "bf16" or "f32", whichever device it is on.
func heldTypes(m *Model) (out, gateUp, qkv string) {
	name := func(bf16 bool) string {
		if bf16 {
			return "bf16"
		}
		return "f32"
	}
	if g := m.gpu; g != nil {
		l := g.layers[3]
		return l.out.Type.String(), l.gateUp.Type.String(), l.qkv.Type.String()
	}
	l := m.layers[3]
	return name(l.out.BF16()), name(l.gateUp.BF16()), name(l.qkv.BF16())
}

// TestEditedUnchanged loads deltas that hold base tensors' own values, with
// the layout of editedDir, on every device. The tokens must be the base
// model's. This is also the record of what each dtype does: where it runs
// (Auto included), how the model holds the weights, and how the logits
// compare with the base model's on the same device.
//
//   - A BF16 delta changes nothing: Metal runs it, and logits are bit-identical
//     to the base model's on both devices.
//   - An F32 delta runs on both devices, and Auto stays on Metal. Each layer
//     it covers runs on its float32 kernels, whose logits agree with the bf16
//     kernels' (on Apple silicon, bit for bit). A delta covering only part of
//     a stack (gate without up, q without k and v) makes the whole stack
//     float32, on the GPU as on the CPU.
func TestEditedUnchanged(t *testing.T) {
	base := golden.ModelDir(t, "INFER_MINICPM5_DIR", "minicpm5-1b")
	_, cases, _ := miniCPM.load(t)
	miniCPM.release()
	c := cases[2]
	devices := testDevices(t)
	haveMetal := len(devices) > 1
	runs := baseRuns(t, base, c, devices)

	p := "model.layers.3."
	o, gate, up, q, k, v := p+"self_attn.o_proj.weight", p+"mlp.gate_proj.weight", p+"mlp.up_proj.weight",
		p+"self_attn.q_proj.weight", p+"self_attn.k_proj.weight", p+"self_attn.v_proj.weight"
	every := []string{o, gate, up, q, k, v, p + "mlp.down_proj.weight", p + "input_layernorm.weight",
		"lm_head.weight", "model.embed_tokens.weight", "model.norm.weight"}
	for _, tt := range []struct {
		name, dtype string
		names       []string
		held        [3]string // how the CPU and Metal hold layer 3's o_proj, gate|up and q|k|v
		long        bool      // writes over a gigabyte: skipped with -short
	}{
		{"bf16_o_proj", "BF16", []string{o}, [3]string{"bf16", "bf16", "bf16"}, false},
		{"bf16_gate_of_stack", "BF16", []string{gate}, [3]string{"bf16", "bf16", "bf16"}, false},
		{"bf16_every_kind", "BF16", every, [3]string{"bf16", "bf16", "bf16"}, false},
		{"f32_o_proj", "F32", []string{o}, [3]string{"f32", "bf16", "bf16"}, false},
		{"f32_gate_up", "F32", []string{gate, up}, [3]string{"bf16", "f32", "bf16"}, false},
		{"f32_qkv", "F32", []string{q, k, v}, [3]string{"bf16", "bf16", "f32"}, false},
		{"f32_every_kind", "F32", every, [3]string{"f32", "f32", "f32"}, true},
		{"f32_gate_of_stack", "F32", []string{gate}, [3]string{"bf16", "f32", "bf16"}, false},
		{"f32_up_of_stack", "F32", []string{up}, [3]string{"bf16", "f32", "bf16"}, false},
		{"f32_q_of_stack", "F32", []string{q}, [3]string{"bf16", "bf16", "f32"}, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if tt.long && testing.Short() {
				t.Skip("a large delta skipped with -short")
			}
			dir := editedDir(t, base, tt.dtype, nil, tt.names...)
			for _, device := range append(slices.Clone(devices), accel.Auto) {
				m, err := LoadWith(dir, Options{Device: device})
				if err != nil {
					t.Fatalf("%s: %v", device, err)
				}
				defer m.Close()
				wantOn, cmpDevice := device, device
				if device == accel.Auto {
					wantOn, cmpDevice = accel.CPU, accel.CPU
					if haveMetal {
						wantOn, cmpDevice = accel.Metal, accel.Metal
					}
				}
				if !strings.HasPrefix(m.Device(), wantOn) {
					t.Errorf("%s: runs on %s, want %s", device, m.Device(), wantOn)
				}
				out, gateUp, qkv := heldTypes(m)
				if held := [3]string{out, gateUp, qkv}; held != tt.held {
					t.Errorf("%s: holds o_proj, gate|up and q|k|v in %v, want %v", device, held, tt.held)
				}
				got := generateRun(t, m, c.InputIDs, len(c.OutputIDs))
				if !slices.Equal(got.tokens, c.OutputIDs) {
					t.Errorf("%s: generated %v, want %v", device, got.tokens, c.OutputIDs)
				}
				bits, worst := got.diff(runs[cmpDevice])
				t.Logf("%-5s on %-22s o_proj %-4s gate|up %-4s q|k|v %-4s: %d logits differ from the base model's on %s, by at most %.3g of a row's range",
					device, m.Device(), out, gateUp, qkv, bits, cmpDevice, worst)
				if worst > relTol {
					t.Errorf("%s: logits differ from the base model's by %.3g of a row's range", device, worst)
				}
				if tt.dtype == "BF16" && bits != 0 {
					t.Errorf("%s: a BF16 delta of unchanged values changes %d logits", device, bits)
				}
			}
		})
	}
}

// TestEditedMatchesPython loads the deliberate edits of
// scripts/dump_golden_llama_hooks.py through the layout of editedDir, on each
// device and in both dtypes, for a checkpoint with one shard and one with two
// and tied embeddings. The tokens and logits must be what transformers gets
// from the same edit in memory, and not what the base model gets.
func TestEditedMatchesPython(t *testing.T) {
	for _, g := range goldenModels {
		t.Run(g.dir, func(t *testing.T) {
			base := golden.ModelDir(t, g.envVar, g.dir)
			h, want := g.hooks(t)
			_, cases, _ := g.load(t)
			g.release()
			c := cases[h.EditCase]
			for e, edit := range h.Edits {
				names := slices.Sorted(maps.Keys(edit.Scales))
				for _, dtype := range []string{"BF16", "F32"} {
					t.Run(edit.Name+"/"+dtype, func(t *testing.T) {
						dir := editedDir(t, base, dtype, edit.Scales, names...)
						values, indices := topKEdit(t, want, e, len(c.InputIDs)+len(edit.OutputIDs)-1)
						for _, device := range testDevices(t) {
							m, err := LoadWith(dir, Options{Device: device})
							if err != nil {
								t.Fatalf("%s: %v", device, err)
							}
							defer m.Close()
							var got []int
							worst := 0.0
							err = m.GenerateFunc(c.InputIDs, len(edit.OutputIDs), func(token int, logits []float32) bool {
								p := len(c.InputIDs) - 1 + len(got)
								worst = max(worst, checkTopK(t, p, logits, values[p], indices[p], relTol))
								got = append(got, token)
								return true
							})
							if err != nil {
								t.Fatal(err)
							}
							if !slices.Equal(got, edit.OutputIDs) {
								t.Errorf("%s: generated %v, transformers %v", device, got, edit.OutputIDs)
							}
							if slices.Equal(got, c.OutputIDs) {
								t.Errorf("%s: the edit did not change the tokens", device)
							}
							t.Logf("%-5s: largest step difference from transformers %.2g of a row's range", device, worst)
						}
					})
				}
			}
		})
	}
}

// scaled reads a tensor and scales it, to store as dtype.
func scaled(t testing.TB, shard, name, dtype string, scale float32) safetensors.Output {
	t.Helper()
	f, err := safetensors.Open(shard)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	v, err := f.Tensor(name)
	if err != nil {
		t.Fatal(err)
	}
	data := slices.Clone(v.Data)
	for i := range data {
		data[i] *= scale
	}
	return safetensors.Output{Name: name, DType: dtype, Tensor: tensor.FromData(data, v.Shape...)}
}

// topKEdit reads an edit's golden top-k logits and token ids, per position.
func topKEdit(t *testing.T, want *safetensors.File, e, positions int) ([][]float32, [][]int) {
	t.Helper()
	v, err := want.Tensor(fmt.Sprintf("edit%d.topk.values", e))
	if err != nil {
		t.Fatal(err)
	}
	idx, shape, err := want.Ints(fmt.Sprintf("edit%d.topk.indices", e))
	if err != nil {
		t.Fatal(err)
	}
	if v.Rows() != positions || shape[0] != positions {
		t.Fatalf("golden top-k has %d positions, want %d", v.Rows(), positions)
	}
	k := v.Cols()
	values, indices := make([][]float32, positions), make([][]int, positions)
	for p := range positions {
		values[p], indices[p] = v.Row(p), idx[p*k:(p+1)*k]
	}
	return values, indices
}

// TestEditedModelSafetensorsWins records why editedDir must not name a shard
// model.safetensors: openWeights prefers that file and never reads the index,
// so a delta beside it is silently ignored.
func TestEditedModelSafetensorsWins(t *testing.T) {
	base := writeTinyModel(t)
	dir := t.TempDir()
	if err := os.Symlink(filepath.Join(base, "config.json"), filepath.Join(dir, "config.json")); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(filepath.Join(base, "model.safetensors"), filepath.Join(dir, "model.safetensors")); err != nil {
		t.Fatal(err)
	}
	name := "model.layers.0.self_attn.o_proj.weight"
	delta := scaled(t, filepath.Join(base, "model.safetensors"), name, "F32", 2)
	if err := safetensors.Write(filepath.Join(dir, "delta.safetensors"), []safetensors.Output{delta}, nil); err != nil {
		t.Fatal(err)
	}
	raw, err := json.Marshal(map[string]any{"weight_map": map[string]string{name: "delta.safetensors"}})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), raw, 0o644); err != nil {
		t.Fatal(err)
	}
	output := func(dir string) []float32 {
		m, err := LoadWith(dir, Options{Device: accel.CPU})
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		return slices.Clone(forwardAll(m, []int{1, 2, 3}).Data)
	}
	if !bitsEqual(output(base), output(dir)) {
		t.Error("the delta took effect beside a model.safetensors; openWeights no longer prefers that file")
	}
	// Under another name, the index applies, and the delta with it, once the
	// index lists every tensor.
	if err := os.Remove(filepath.Join(dir, "model.safetensors")); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(filepath.Join(base, "model.safetensors"), filepath.Join(dir, "base-0.safetensors")); err != nil {
		t.Fatal(err)
	}
	f, err := safetensors.Open(filepath.Join(base, "model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	index := map[string]string{}
	for _, n := range f.Names() {
		index[n] = "base-0.safetensors"
	}
	index[name] = "delta.safetensors"
	if raw, err = json.Marshal(map[string]any{"weight_map": index}); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), raw, 0o644); err != nil {
		t.Fatal(err)
	}
	if bitsEqual(output(base), output(dir)) {
		t.Error("the delta did not change the output")
	}
}
