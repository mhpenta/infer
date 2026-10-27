package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// writeFile builds a safetensors file from a header and raw data region.
func writeFile(t *testing.T, header map[string]any, data []byte) string {
	t.Helper()
	h, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	buf := binary.LittleEndian.AppendUint64(nil, uint64(len(h)))
	buf = append(append(buf, h...), data...)
	path := filepath.Join(t.TempDir(), "model.safetensors")
	if err := os.WriteFile(path, buf, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestTensorDTypes(t *testing.T) {
	var data []byte
	for _, v := range []float32{1.5, -2, 0.25, 3e-8} {
		data = binary.LittleEndian.AppendUint32(data, math.Float32bits(v))
	}
	// F16: 1.0, -2.0, 65504 (max), 2^-24 (smallest subnormal)
	for _, h := range []uint16{0x3c00, 0xc000, 0x7bff, 0x0001} {
		data = binary.LittleEndian.AppendUint16(data, h)
	}
	// BF16: 1.0, -0.5 (the top 16 bits of their float32 encodings)
	for _, v := range []float32{1, -0.5} {
		data = binary.LittleEndian.AppendUint16(data, uint16(math.Float32bits(v)>>16))
	}
	path := writeFile(t, map[string]any{
		"__metadata__": map[string]string{"format": "pt"},
		"f32":          map[string]any{"dtype": "F32", "shape": []int{2, 2}, "data_offsets": []int{0, 16}},
		"f16":          map[string]any{"dtype": "F16", "shape": []int{4}, "data_offsets": []int{16, 24}},
		"bf16":         map[string]any{"dtype": "BF16", "shape": []int{2}, "data_offsets": []int{24, 28}},
	}, data)

	f, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if got := f.Names(); !slices.Equal(got, []string{"bf16", "f16", "f32"}) {
		t.Errorf("Names() = %v", got)
	}

	tests := []struct {
		name  string
		shape []int
		want  []float32
	}{
		{"f32", []int{2, 2}, []float32{1.5, -2, 0.25, 3e-8}},
		{"f16", []int{4}, []float32{1, -2, 65504, float32(math.Ldexp(1, -24))}},
		{"bf16", []int{2}, []float32{1, -0.5}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := f.Tensor(tt.name)
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(got.Shape, tt.shape) || !slices.Equal(got.Data, tt.want) {
				t.Errorf("got %v %v, want %v %v", got.Shape, got.Data, tt.shape, tt.want)
			}
		})
	}
}

func TestOpenRejectsCorruptFiles(t *testing.T) {
	tests := []struct {
		name    string
		header  map[string]any
		data    []byte
		wantErr string
	}{
		{
			name:    "offsets past end",
			header:  map[string]any{"x": map[string]any{"dtype": "F32", "shape": []int{2}, "data_offsets": []int{0, 8}}},
			data:    make([]byte, 4),
			wantErr: "outside data",
		},
		{
			name:    "shape and offsets disagree",
			header:  map[string]any{"x": map[string]any{"dtype": "F32", "shape": []int{3}, "data_offsets": []int{0, 8}}},
			data:    make([]byte, 8),
			wantErr: "needs 12 bytes",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := Open(writeFile(t, tt.header, tt.data))
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Errorf("Open error = %v, want one containing %q", err, tt.wantErr)
			}
		})
	}

	path := filepath.Join(t.TempDir(), "short.safetensors")
	if err := os.WriteFile(path, []byte{1, 2, 3}, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Open(path); err == nil {
		t.Error("Open of a 3-byte file succeeded")
	}
}

func TestInts(t *testing.T) {
	var data []byte
	for _, v := range []int32{7, -3, 130559} {
		data = binary.LittleEndian.AppendUint32(data, uint32(v))
	}
	for _, v := range []int64{-1 << 40, 5} {
		data = binary.LittleEndian.AppendUint64(data, uint64(v))
	}
	path := writeFile(t, map[string]any{
		"i32": map[string]any{"dtype": "I32", "shape": []int{3}, "data_offsets": []int{0, 12}},
		"i64": map[string]any{"dtype": "I64", "shape": []int{1, 2}, "data_offsets": []int{12, 28}},
	}, data)
	f, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	for _, tt := range []struct {
		name  string
		want  []int
		shape []int
	}{{"i32", []int{7, -3, 130559}, []int{3}}, {"i64", []int{-1 << 40, 5}, []int{1, 2}}} {
		got, shape, err := f.Ints(tt.name)
		if err != nil || !slices.Equal(got, tt.want) || !slices.Equal(shape, tt.shape) {
			t.Errorf("Ints(%q) = %v %v %v, want %v %v", tt.name, got, shape, err, tt.want, tt.shape)
		}
	}
	if _, _, err := f.Ints("missing"); err == nil {
		t.Error("Ints of a missing tensor succeeded")
	}
}
