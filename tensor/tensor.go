// Package tensor holds the dense float32 tensor used throughout infer.
//
// It is deliberately small: inference only, no autograd, no strides. Data is
// always contiguous and row-major, so a [rows, cols] tensor stores row i at
// Data[i*cols : (i+1)*cols].
package tensor

import "fmt"

// Tensor is a dense, row-major float32 array.
type Tensor struct {
	Shape []int
	Data  []float32
}

// New returns a zeroed tensor with the given shape.
func New(shape ...int) *Tensor {
	return &Tensor{Shape: shape, Data: make([]float32, numel(shape))}
}

// FromData wraps data without copying. It panics if len(data) does not match
// the shape, since that is a programming error rather than bad input.
func FromData(data []float32, shape ...int) *Tensor {
	if n := numel(shape); n != len(data) {
		panic(fmt.Sprintf("tensor: shape %v needs %d values, got %d", shape, n, len(data)))
	}
	return &Tensor{Shape: shape, Data: data}
}

// Size returns the number of elements.
func (t *Tensor) Size() int { return len(t.Data) }

// Rows returns the first dimension of a 2D tensor.
func (t *Tensor) Rows() int { return t.Shape[0] }

// Cols returns the second dimension of a 2D tensor.
func (t *Tensor) Cols() int { return t.Shape[1] }

// Row returns row i of a 2D tensor as a slice sharing the tensor's storage.
func (t *Tensor) Row(i int) []float32 {
	cols := t.Shape[1]
	return t.Data[i*cols : (i+1)*cols]
}

func numel(shape []int) int {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return n
}
