package safetensors

import (
	"bufio"
	"bytes"
	"cmp"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"maps"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"github.com/mhpenta/infer/tensor"
)

// maxHeader is the largest header the reference implementation accepts.
const maxHeader = 100_000_000

// Output is a tensor to write: its values, as float32, and how to store them,
// "F32" or "BF16".
type Output struct {
	Name   string
	DType  string
	Tensor *tensor.Tensor
}

// Write writes tensors to path as the Python library's save_file does, byte
// for byte for the same tensors: the header holds metadata first, then the
// tensors in file order, padded with spaces to 8 bytes, and the data runs
// without gaps from the widest dtype to the narrowest, by name within a
// dtype, so every value is aligned once the file is mapped. Metadata gets
// "format": "pt" unless it sets a format, as transformers' save_pretrained
// writes it.
//
// BF16 values that bf16 cannot hold exactly round to nearest, ties to even,
// as PyTorch's .to(torch.bfloat16) rounds them (the largest finite values
// round to infinity, and every NaN becomes 0x7fc0). Values read from a BF16
// tensor are written back unchanged.
//
// The file appears at path only once it is complete, by renaming a temporary
// file in the same directory, so a failure leaves whatever was there, a file
// mapped by a reader stays intact, and a symlink at path is replaced rather
// than written through.
func Write(path string, tensors []Output, metadata map[string]string) (err error) {
	header, ordered, err := buildHeader(tensors, metadata)
	if err != nil {
		return fmt.Errorf("safetensors: write %s: %w", path, err)
	}
	f, err := os.CreateTemp(filepath.Dir(path), ".safetensors-*")
	if err != nil {
		return fmt.Errorf("safetensors: write %s: %w", path, err)
	}
	defer func() {
		if err != nil {
			f.Close()
			os.Remove(f.Name())
			err = fmt.Errorf("safetensors: write %s: %w", path, err)
		}
	}()
	w := bufio.NewWriterSize(f, 1<<20)
	if _, err := w.Write(header); err != nil {
		return err
	}
	for _, t := range ordered {
		if err := writeValues(w, t); err != nil {
			return err
		}
	}
	if err := w.Flush(); err != nil {
		return err
	}
	if err := f.Chmod(0o644); err != nil { // CreateTemp makes it 0600
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(f.Name(), path)
}

// buildHeader checks tensors and returns the length-prefixed, padded header
// and the tensors in file order.
func buildHeader(tensors []Output, metadata map[string]string) ([]byte, []Output, error) {
	seen := make(map[string]bool, len(tensors))
	for _, t := range tensors {
		switch {
		case t.Name == "" || t.Name == "__metadata__":
			return nil, nil, fmt.Errorf("a tensor cannot be called %q", t.Name)
		case seen[t.Name]:
			return nil, nil, fmt.Errorf("two tensors are called %q", t.Name)
		case t.Tensor == nil:
			return nil, nil, fmt.Errorf("tensor %q is nil", t.Name)
		case t.DType != "F32" && t.DType != "BF16":
			return nil, nil, fmt.Errorf("tensor %q: cannot write dtype %q (want F32 or BF16)", t.Name, t.DType)
		case slices.ContainsFunc(t.Tensor.Shape, func(d int) bool { return d < 0 }):
			return nil, nil, fmt.Errorf("tensor %q has shape %v", t.Name, t.Tensor.Shape)
		case numel(t.Tensor.Shape) != len(t.Tensor.Data):
			return nil, nil, fmt.Errorf("tensor %q: shape %v needs %d values, have %d", t.Name, t.Tensor.Shape, numel(t.Tensor.Shape), len(t.Tensor.Data))
		}
		seen[t.Name] = true
	}
	ordered := slices.Clone(tensors)
	slices.SortStableFunc(ordered, func(a, b Output) int {
		sa, _ := dtypeSize(a.DType)
		sb, _ := dtypeSize(b.DType)
		return cmp.Or(cmp.Compare(sb, sa), cmp.Compare(a.Name, b.Name))
	})

	// The order of a JSON object matters to the format's reference output
	// but not to encoding/json, so this writes the object itself.
	meta := map[string]string{"format": "pt"}
	maps.Copy(meta, metadata)
	var b bytes.Buffer
	b.WriteString(`{"__metadata__":{`)
	for i, k := range slices.Sorted(maps.Keys(meta)) {
		if i > 0 {
			b.WriteByte(',')
		}
		fmt.Fprintf(&b, "%s:%s", quote(k), quote(meta[k]))
	}
	b.WriteByte('}')
	var off int64
	for _, t := range ordered {
		size, _ := dtypeSize(t.DType)
		end := off + int64(size)*int64(len(t.Tensor.Data))
		dims := make([]string, len(t.Tensor.Shape)) // a scalar's shape is [], never null
		for i, d := range t.Tensor.Shape {
			dims[i] = strconv.Itoa(d)
		}
		fmt.Fprintf(&b, `,%s:{"dtype":%q,"shape":[%s],"data_offsets":[%d,%d]}`, quote(t.Name), t.DType, strings.Join(dims, ","), off, end)
		off = end
	}
	b.WriteByte('}')
	for b.Len()%8 != 0 {
		b.WriteByte(' ')
	}
	if b.Len() > maxHeader {
		return nil, nil, fmt.Errorf("a header of %d bytes is over the format's limit of %d", b.Len(), maxHeader)
	}
	return append(binary.LittleEndian.AppendUint64(nil, uint64(b.Len())), b.Bytes()...), ordered, nil
}

// quote returns s as a JSON string, leaving <, > and & as they are, as the
// reference implementation does.
func quote(s string) string {
	var b bytes.Buffer
	e := json.NewEncoder(&b)
	e.SetEscapeHTML(false)
	if err := e.Encode(s); err != nil {
		panic(err) // a string always encodes
	}
	return string(bytes.TrimRight(b.Bytes(), "\n"))
}

// writeValues writes t's values, little-endian, in its dtype.
func writeValues(w *bufio.Writer, t Output) error {
	var buf [4]byte
	for _, v := range t.Tensor.Data {
		n := 2
		if t.DType == "F32" {
			binary.LittleEndian.PutUint32(buf[:], math.Float32bits(v))
			n = 4
		} else {
			binary.LittleEndian.PutUint16(buf[:], bf16FromFloat32(v))
		}
		if _, err := w.Write(buf[:n]); err != nil {
			return err
		}
	}
	return nil
}

// bf16FromFloat32 returns f as bf16 bits, the top half of its float32 bits
// rounded to nearest, ties to even. A carry out of the mantissa makes the
// next exponent, or infinity, as it should.
func bf16FromFloat32(f float32) uint16 {
	if f != f {
		return 0x7fc0
	}
	bits := math.Float32bits(f)
	hi, lo := bits>>16, bits&0xffff
	if lo > 0x8000 || lo == 0x8000 && hi&1 == 1 {
		hi++
	}
	return uint16(hi)
}
