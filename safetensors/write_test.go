package safetensors

import (
	"cmp"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"unsafe"

	"github.com/mhpenta/infer/internal/golden"
	"github.com/mhpenta/infer/tensor"
)

// bf16Exact are values bf16 holds exactly, specials included.
var bf16Exact = []float32{0, float32(math.Copysign(0, -1)), 1, -0.5, 3.5, float32(math.Inf(1)), float32(math.Inf(-1)),
	math.Float32frombits(0x00010000), math.Float32frombits(0x7f7f0000), math.Float32frombits(0xff7f0000)}

// writerTensors are the tensors the round trip and Python tests write: names
// out of file order, dtypes interleaved, and a scalar and an empty tensor.
func writerTensors() []Output {
	return []Output{
		{"z.first.bf16", "BF16", tensor.FromData(slices.Clone(bf16Exact), 2, 5)},
		{"a.f32", "F32", tensor.FromData([]float32{1.5, -2, 0.25, 3e-8, 1e30, float32(math.Inf(-1))}, 3, 2)},
		{"m.scalar", "F32", tensor.FromData([]float32{7})},
		{"b.empty", "BF16", tensor.New(0, 3)},
		{"c.odd <&> \"name\" é", "BF16", tensor.FromData([]float32{2, -4, 8}, 3)},
		{"b.after.bf16", "F32", tensor.FromData([]float32{1, 2, 3, 4}, 4)},
	}
}

func TestWriteRoundTrip(t *testing.T) {
	tensors := writerTensors()
	path := filepath.Join(t.TempDir(), "x.safetensors")
	if err := Write(path, tensors, map[string]string{"note": "edited"}); err != nil {
		t.Fatal(err)
	}
	f, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if len(f.Names()) != len(tensors) {
		t.Errorf("Names() = %v", f.Names())
	}
	for _, want := range tensors {
		got, err := f.Tensor(want.Name)
		if err != nil {
			t.Fatal(err)
		}
		if f.DType(want.Name) != want.DType || !slices.Equal(got.Shape, want.Tensor.Shape) || !sameBits(got.Data, want.Tensor.Data) {
			t.Errorf("%s: read back %s %v %v, want %s %v %v", want.Name, f.DType(want.Name), got.Shape, got.Data,
				want.DType, want.Tensor.Shape, want.Tensor.Data)
		}
		// F32 data sits at an aligned address, so the reader's view is zero-copy.
		if raw, err := f.Bytes(want.Name); err != nil {
			t.Error(err)
		} else if want.DType == "F32" && len(raw) > 0 && &got.Data[0] != (*float32)(unsafePointer(raw)) {
			t.Errorf("%s: the reader copied F32 data instead of viewing it", want.Name)
		}
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if perm := info.Mode().Perm(); perm != 0o644 {
		t.Errorf("mode %v, want 0644", perm)
	}
	// The layout the format's reference implementation requires.
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	n := binary.LittleEndian.Uint64(raw)
	if (8+n)%8 != 0 {
		t.Errorf("data starts at %d, not 8-byte aligned", 8+n)
	}
	var header map[string]json.RawMessage
	if err := json.Unmarshal(raw[8:8+n], &header); err != nil {
		t.Fatal(err)
	}
	var meta map[string]string
	if err := json.Unmarshal(header["__metadata__"], &meta); err != nil || meta["format"] != "pt" || meta["note"] != "edited" {
		t.Errorf("metadata %v, %v", meta, err)
	}
	type span struct {
		Offsets [2]int64 `json:"data_offsets"`
	}
	var spans [][2]int64
	for name, msg := range header {
		if name != "__metadata__" {
			var s span
			if err := json.Unmarshal(msg, &s); err != nil {
				t.Fatal(err)
			}
			spans = append(spans, s.Offsets)
		}
	}
	slices.SortFunc(spans, func(a, b [2]int64) int { return cmp.Or(cmp.Compare(a[1], b[1]), cmp.Compare(a[0], b[0])) }) // an empty tensor spans nothing
	var end int64
	for _, s := range spans {
		if s[0] != end {
			t.Errorf("a tensor starts at %d, after one ending at %d", s[0], end)
		}
		end = s[1]
	}
	if end != int64(len(raw))-8-int64(n) {
		t.Errorf("tensors end at %d of %d bytes of data", end, int64(len(raw))-8-int64(n))
	}
}

// TestWriteBF16Rounding checks rounding against the values
// scripts/check_safetensors_write.py checks against PyTorch.
func TestWriteBF16Rounding(t *testing.T) {
	for _, tt := range []struct {
		in   uint32
		want uint16
	}{
		{0x3f800000, 0x3f80}, // 1 is exact
		{0x3f807fff, 0x3f80}, // below the tie
		{0x3f808000, 0x3f80}, // a tie goes to the even neighbor below
		{0x3f818000, 0x3f82}, // and above
		{0x3f808001, 0x3f81}, // above the tie
		{0xbf808001, 0xbf81}, // the sign is kept
		{0x3f7fffff, 0x3f80}, // a carry reaches the exponent
		{0x7f7fffff, 0x7f80}, // the largest float32 rounds to infinity
		{0xff7fffff, 0xff80},
		{0x7f800000, 0x7f80}, // infinities
		{0xff800000, 0xff80},
		{0x7f800001, 0x7fc0}, // a NaN that truncation would make infinity
		{0xffc00000, 0x7fc0},
		{0x00008000, 0x0000}, // subnormals round like everything else
		{0x00018000, 0x0002},
		{0x80000000, 0x8000}, // -0
	} {
		if got := bf16FromFloat32(math.Float32frombits(tt.in)); got != tt.want {
			t.Errorf("%#08x: got %#04x, want %#04x", tt.in, got, tt.want)
		}
	}
}

func TestWriteRejects(t *testing.T) {
	one := tensor.New(1)
	for _, tt := range []struct {
		name    string
		tensors []Output
		want    string
	}{
		{"reserved name", []Output{{"__metadata__", "F32", one}}, "cannot be called"},
		{"empty name", []Output{{"", "F32", one}}, "cannot be called"},
		{"duplicate", []Output{{"x", "F32", one}, {"x", "BF16", one}}, "two tensors"},
		{"nil", []Output{{"x", "F32", nil}}, "is nil"},
		{"dtype", []Output{{"x", "F16", one}}, "cannot write dtype"},
		{"shape", []Output{{"x", "F32", &tensor.Tensor{Shape: []int{2}, Data: []float32{1}}}}, "needs 2 values"},
		{"negative", []Output{{"x", "F32", &tensor.Tensor{Shape: []int{-1, -1}, Data: []float32{1}}}}, "has shape"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "x.safetensors")
			err := Write(path, tt.tensors, nil)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Errorf("error %v, want one containing %q", err, tt.want)
			}
			if _, err := os.Stat(path); err == nil {
				t.Error("a rejected write left a file")
			}
		})
	}
}

// TestWriteLeavesTargets checks that writing never touches what path pointed
// to, a base checkpoint's shard behind a symlink or a file still mapped, and
// that a failed write leaves no temporary file.
func TestWriteLeavesTargets(t *testing.T) {
	dir := t.TempDir()
	base := filepath.Join(dir, "base.safetensors")
	old := []Output{{"x", "F32", tensor.FromData([]float32{1, 2}, 2)}}
	if err := Write(base, old, nil); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(dir, "link.safetensors")
	if err := os.Symlink(base, link); err != nil {
		t.Fatal(err)
	}
	mapped, err := Open(link)
	if err != nil {
		t.Fatal(err)
	}
	defer mapped.Close()
	if err := Write(link, []Output{{"x", "F32", tensor.FromData([]float32{3, 4}, 2)}}, nil); err != nil {
		t.Fatal(err)
	}
	if info, err := os.Lstat(link); err != nil || info.Mode()&os.ModeSymlink != 0 {
		t.Errorf("the symlink was not replaced: %v, %v", info, err)
	}
	for _, read := range []func() (*File, error){func() (*File, error) { return Open(base) }, func() (*File, error) { return mapped, nil }} {
		f, err := read()
		if err != nil {
			t.Fatal(err)
		}
		if got, err := f.Tensor("x"); err != nil || !slices.Equal(got.Data, []float32{1, 2}) {
			t.Errorf("the old file now holds %v, %v", got, err)
		}
		if f != mapped {
			f.Close()
		}
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 2 {
		t.Errorf("%d files after writing, want base and link: %v", len(entries), entries)
	}
	if err := Write(filepath.Join(dir, "missing", "x.safetensors"), old, nil); err == nil {
		t.Error("writing into a missing directory succeeded")
	}
}

// TestWriteMatchesPython writes tensors, has
// scripts/check_safetensors_write.py load them with the Python safetensors
// library and compare the file byte for byte with save_file's for the same
// values, and checks rounding against PyTorch's. It needs the reference
// environment in .venv, and skips without it.
func TestWriteMatchesPython(t *testing.T) {
	python := filepath.Join(golden.Root(), ".venv", "bin", "python")
	if _, err := os.Stat(python); err != nil {
		t.Skipf("no reference environment at %s", python)
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "written.safetensors")
	if err := Write(path, writerTensors(), nil); err != nil {
		t.Fatal(err)
	}
	// Values bf16 cannot hold, to compare with PyTorch's rounding: every
	// value in the rounding test, and a sweep across ties and exponents.
	inexact := []float32{}
	for _, bits := range []uint32{0x3f807fff, 0x3f808000, 0x3f818000, 0x3f808001, 0xbf808001, 0x3f7fffff, 0x7f7fffff,
		0xff7fffff, 0x7f800001, 0xffc00000, 0x00008000, 0x00018000, 0x80000000, 0x00000001, 0x00007fff} {
		inexact = append(inexact, math.Float32frombits(bits))
	}
	for i := range 1 << 16 {
		inexact = append(inexact, math.Float32frombits(uint32(i)*0x9e3779b1))
		inexact = append(inexact, math.Float32frombits(uint32(i)<<16|0x8000))
	}
	rounded := filepath.Join(dir, "rounded.safetensors")
	if err := Write(rounded, []Output{
		{"as_f32", "F32", tensor.FromData(inexact, len(inexact))},
		{"as_bf16", "BF16", tensor.FromData(inexact, len(inexact))},
	}, nil); err != nil {
		t.Fatal(err)
	}
	out, err := exec.Command(python, filepath.Join(golden.Root(), "scripts", "check_safetensors_write.py"), path, rounded).CombinedOutput()
	if err != nil {
		t.Fatalf("%v\n%s", err, out)
	}
	t.Logf("%s", out)
}

func sameBits(a, b []float32) bool {
	return slices.EqualFunc(a, b, func(x, y float32) bool { return math.Float32bits(x) == math.Float32bits(y) })
}

func unsafePointer(b []byte) unsafe.Pointer { return unsafe.Pointer(&b[0]) }
