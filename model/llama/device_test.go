package llama

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/mhpenta/infer/accel"
	"github.com/mhpenta/infer/internal/golden"
)

// TestDevice checks how LoadWith chooses a device: Options.Device, then
// $INFER_DEVICE, then Auto, which takes Metal when it is available, for
// bf16 and int8 weights.
func TestDevice(t *testing.T) {
	dir := golden.ModelDir(t, "INFER_MINICPM5_DIR", "minicpm5-1b")
	haveMetal := len(testDevices(t)) > 1
	if _, err := LoadWith(dir, Options{Device: "tpu"}); err == nil || !strings.Contains(err.Error(), `"tpu"`) {
		t.Errorf("an unknown device: %v", err)
	}
	t.Setenv(accel.EnvVar, accel.CPU)
	if got, err := loadDevice(dir, Options{}); err != nil || got != accel.CPU {
		t.Errorf("%s=cpu: loaded on %q, %v", accel.EnvVar, got, err)
	}
	t.Setenv(accel.EnvVar, "")
	want := accel.CPU
	if haveMetal {
		want = accel.Metal + " ("
	}
	for _, int8 := range []bool{false, true} {
		if got, err := loadDevice(dir, Options{Int8: int8}); err != nil || !strings.HasPrefix(got, want) {
			t.Errorf("the default, int8 %v: loaded on %q, %v; want %s…", int8, got, err, want)
		}
	}
	if !haveMetal {
		if _, err := loadDevice(dir, Options{Device: accel.Metal}); err == nil {
			t.Error("Metal loaded without a Metal device")
		}
	}
}

// TestDeviceFallback loads a tiny model with heads of 16 values, which the
// Metal kernels do not take: Auto must fall back to the CPU, and asking for
// Metal must fail and say why.
func TestDeviceFallback(t *testing.T) {
	dir := writeTinyModel(t)
	t.Setenv(accel.EnvVar, "")
	for _, int8 := range []bool{false, true} {
		if got, err := loadDevice(dir, Options{Int8: int8}); err != nil || got != accel.CPU {
			t.Errorf("Auto, int8 %v: loaded on %q, %v; want the CPU", int8, got, err)
		}
	}
	_, err := loadDevice(dir, Options{Device: accel.Metal})
	if err == nil || !strings.Contains(err.Error(), "cannot run on Metal") {
		t.Errorf("Metal with heads of 16: %v", err)
	}
	if !strings.Contains(err.Error(), "head dimension") {
		t.Errorf("the error does not say why: %v", err)
	}
}

// loadDevice loads dir with opts and returns where it runs.
func loadDevice(dir string, opts Options) (string, error) {
	m, err := LoadWith(dir, opts)
	if err != nil {
		return "", err
	}
	defer m.Close()
	return m.Device(), nil
}

// writeTinyModel writes a one-layer Llama model with float32 weights and
// heads of 16 values to a temporary directory.
func writeTinyModel(t *testing.T) string {
	t.Helper()
	const hidden, ff, vocab, heads, kvHeads, headDim = 32, 64, 16, 2, 1, 16
	dir := t.TempDir()
	config, _ := json.Marshal(map[string]any{
		"model_type": "llama", "hidden_size": hidden, "intermediate_size": ff, "num_hidden_layers": 1,
		"num_attention_heads": heads, "num_key_value_heads": kvHeads, "head_dim": headDim, "vocab_size": vocab,
		"rms_norm_eps": 1e-6, "rope_theta": 10000, "tie_word_embeddings": true, "eos_token_id": 1,
	})
	p := "model.layers.0."
	shapes := map[string][]int{
		p + "input_layernorm.weight":          {hidden},
		p + "post_attention_layernorm.weight": {hidden},
		p + "self_attn.q_proj.weight":         {heads * headDim, hidden},
		p + "self_attn.k_proj.weight":         {kvHeads * headDim, hidden},
		p + "self_attn.v_proj.weight":         {kvHeads * headDim, hidden},
		p + "self_attn.o_proj.weight":         {hidden, heads * headDim},
		p + "mlp.gate_proj.weight":            {ff, hidden},
		p + "mlp.up_proj.weight":              {ff, hidden},
		p + "mlp.down_proj.weight":            {hidden, ff},
		"model.norm.weight":                   {hidden},
		"model.embed_tokens.weight":           {vocab, hidden},
	}
	header := map[string]any{}
	var data []byte
	for name, shape := range shapes {
		n := 1
		for _, d := range shape {
			n *= d
		}
		header[name] = map[string]any{"dtype": "F32", "shape": shape, "data_offsets": []int{len(data), len(data) + 4*n}}
		for i := range n {
			data = binary.LittleEndian.AppendUint32(data, math.Float32bits(float32(i%7)/70))
		}
	}
	h, _ := json.Marshal(header)
	file := append(binary.LittleEndian.AppendUint64(nil, uint64(len(h))), h...)
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), append(file, data...), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "config.json"), config, 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}
