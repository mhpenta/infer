// Package safetensors reads model weights stored in the safetensors format.
//
// The format is an 8-byte little-endian header length, a JSON header mapping
// tensor names to dtype, shape and byte offsets, then the raw tensor bytes.
// Files are memory-mapped read-only: F32 tensors are returned as zero-copy
// views into the mapping, and F16/BF16 tensors are upcast to new float32
// slices. Views stay valid until Close, and writing to one faults.
package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"slices"
	"unsafe"

	"github.com/mhpenta/infer/internal/half"
	"github.com/mhpenta/infer/tensor"
)

// File is an open safetensors file.
type File struct {
	data    []byte // the tensor byte region, after the header
	entries map[string]entry
	release func() error
}

type entry struct {
	DType   string   `json:"dtype"`
	Shape   []int    `json:"shape"`
	Offsets [2]int64 `json:"data_offsets"`
}

// Open maps path and parses its header.
func Open(path string) (*File, error) {
	if !littleEndian() {
		return nil, fmt.Errorf("safetensors: big-endian hosts are not supported")
	}
	raw, release, err := mapFile(path)
	if err != nil {
		return nil, fmt.Errorf("safetensors: open %s: %w", path, err)
	}
	f, err := parse(raw)
	if err != nil {
		release()
		return nil, fmt.Errorf("safetensors: %s: %w", path, err)
	}
	f.release = release
	return f, nil
}

func parse(raw []byte) (*File, error) {
	if len(raw) < 8 {
		return nil, fmt.Errorf("file is %d bytes, too short for a header", len(raw))
	}
	headerLen := binary.LittleEndian.Uint64(raw[:8])
	if headerLen > uint64(len(raw)-8) {
		return nil, fmt.Errorf("header length %d exceeds file size %d", headerLen, len(raw))
	}
	var header map[string]json.RawMessage
	if err := json.Unmarshal(raw[8:8+headerLen], &header); err != nil {
		return nil, fmt.Errorf("parse header: %w", err)
	}
	data := raw[8+headerLen:]
	entries := make(map[string]entry, len(header))
	for name, msg := range header {
		if name == "__metadata__" {
			continue
		}
		var e entry
		if err := json.Unmarshal(msg, &e); err != nil {
			return nil, fmt.Errorf("tensor %q: %w", name, err)
		}
		size, ok := dtypeSize(e.DType)
		if !ok {
			// Kept so Names lists it; Tensor reports the unsupported dtype.
			entries[name] = e
			continue
		}
		start, end := e.Offsets[0], e.Offsets[1]
		if start < 0 || end < start || end > int64(len(data)) {
			return nil, fmt.Errorf("tensor %q: offsets %v outside data of %d bytes", name, e.Offsets, len(data))
		}
		if want := int64(numel(e.Shape) * size); end-start != want {
			return nil, fmt.Errorf("tensor %q: %s %v needs %d bytes, offsets give %d", name, e.DType, e.Shape, want, end-start)
		}
		entries[name] = e
	}
	return &File{data: data, entries: entries}, nil
}

// Names returns the tensor names in sorted order.
func (f *File) Names() []string {
	names := make([]string, 0, len(f.entries))
	for name := range f.entries {
		names = append(names, name)
	}
	slices.Sort(names)
	return names
}

// DType returns the stored dtype of the named tensor ("F32", "F16", ...), or
// "" if there is no such tensor.
func (f *File) DType(name string) string {
	return f.entries[name].DType
}

// Has reports whether the file contains a tensor called name.
func (f *File) Has(name string) bool {
	_, ok := f.entries[name]
	return ok
}

// Tensor returns the named tensor as float32. F32 data is a read-only view
// into the mapping; F16 and BF16 are converted into a new slice.
func (f *File) Tensor(name string) (*tensor.Tensor, error) {
	e, ok := f.entries[name]
	if !ok {
		return nil, fmt.Errorf("safetensors: no tensor %q", name)
	}
	b := f.data[e.Offsets[0]:e.Offsets[1]]
	n := numel(e.Shape)
	shape := slices.Clone(e.Shape)
	switch e.DType {
	case "F32":
		if n == 0 {
			return tensor.New(shape...), nil
		}
		if uintptr(unsafe.Pointer(&b[0]))%4 == 0 {
			return tensor.FromData(unsafe.Slice((*float32)(unsafe.Pointer(&b[0])), n), shape...), nil
		}
		out := make([]float32, n)
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
		}
		return tensor.FromData(out, shape...), nil
	case "F16":
		out := make([]float32, n)
		for i := range out {
			out[i] = half.ToFloat32(binary.LittleEndian.Uint16(b[2*i:]))
		}
		return tensor.FromData(out, shape...), nil
	case "BF16":
		out := make([]float32, n)
		if n > 0 && littleEndian() && uintptr(unsafe.Pointer(&b[0]))%2 == 0 {
			// A bf16 value is the top half of a float32: widen in place.
			bits := unsafe.Slice((*uint32)(unsafe.Pointer(&out[0])), n)
			for i, h := range unsafe.Slice((*uint16)(unsafe.Pointer(&b[0])), n) {
				bits[i] = uint32(h) << 16
			}
			return tensor.FromData(out, shape...), nil
		}
		for i := range out {
			out[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(b[2*i:])) << 16)
		}
		return tensor.FromData(out, shape...), nil
	default:
		return nil, fmt.Errorf("safetensors: tensor %q has unsupported dtype %s", name, e.DType)
	}
}

// Bytes returns the named tensor's stored bytes, little-endian in its dtype
// (see DType), as a read-only view into the mapping, valid until Close.
func (f *File) Bytes(name string) ([]byte, error) {
	e, ok := f.entries[name]
	if !ok {
		return nil, fmt.Errorf("safetensors: no tensor %q", name)
	}
	return f.data[e.Offsets[0]:e.Offsets[1]], nil
}

// Shape returns the named tensor's shape, or nil if there is no such tensor.
func (f *File) Shape(name string) []int {
	return slices.Clone(f.entries[name].Shape)
}

// Ints returns an integer tensor (I32 or I64), converted to int, and its
// shape.
func (f *File) Ints(name string) ([]int, []int, error) {
	e, ok := f.entries[name]
	if !ok {
		return nil, nil, fmt.Errorf("safetensors: no tensor %q", name)
	}
	b := f.data[e.Offsets[0]:e.Offsets[1]]
	out := make([]int, numel(e.Shape))
	switch e.DType {
	case "I32":
		for i := range out {
			out[i] = int(int32(binary.LittleEndian.Uint32(b[4*i:])))
		}
	case "I64":
		for i := range out {
			out[i] = int(int64(binary.LittleEndian.Uint64(b[8*i:])))
		}
	default:
		return nil, nil, fmt.Errorf("safetensors: tensor %q is %s, not an integer tensor", name, e.DType)
	}
	return out, slices.Clone(e.Shape), nil
}

// Close unmaps the file. Tensors returned as views must not be used afterwards.
func (f *File) Close() error {
	if f.release == nil {
		return nil
	}
	err := f.release()
	f.release = nil
	return err
}

func dtypeSize(dtype string) (int, bool) {
	switch dtype {
	case "F32", "I32", "U32":
		return 4, true
	case "F16", "BF16", "I16", "U16":
		return 2, true
	case "F64", "I64", "U64":
		return 8, true
	case "I8", "U8", "BOOL":
		return 1, true
	}
	return 0, false
}

func numel(shape []int) int {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return n
}

func littleEndian() bool {
	x := uint16(1)
	return *(*byte)(unsafe.Pointer(&x)) == 1
}
