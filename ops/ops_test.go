package ops

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/mhpenta/infer/tensor"
)

func near(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-5 }

func nearAll(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("got %d values, want %d", len(got), len(want))
	}
	for i := range got {
		if !near(got[i], want[i]) {
			t.Fatalf("got %v, want %v", got, want)
		}
	}
}

func TestLinear(t *testing.T) {
	x := tensor.FromData([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	w := tensor.FromData([]float32{1, 0, -1, 0.5, 0.5, 0.5}, 2, 3)
	b := tensor.FromData([]float32{10, -10}, 2)
	y := Linear(x, Pack(w), b)
	// row 0: [1-3, 3] + b; row 1: [4-6, 7.5] + b
	nearAll(t, y.Data, []float32{8, -7, 8, -2.5})

	// Wide enough to split across workers.
	in, out := 5, 200
	x = tensor.New(3, in)
	w = tensor.New(out, in)
	for i := range x.Data {
		x.Data[i] = float32(i%7) - 3
	}
	for i := range w.Data {
		w.Data[i] = float32(i%5) * 0.25
	}
	y = Linear(x, Pack(w), nil)
	for i := range 3 {
		for j := range out {
			var want float32
			for k := range in {
				want += x.Row(i)[k] * w.Row(j)[k]
			}
			if !near(y.Row(i)[j], want) {
				t.Fatalf("y[%d][%d] = %g, want %g", i, j, y.Row(i)[j], want)
			}
		}
	}
}

func TestLayerNorm(t *testing.T) {
	x := tensor.FromData([]float32{1, 2, 3, 4}, 1, 4)
	LayerNorm(x, tensor.FromData([]float32{1, 1, 2, 2}, 4), tensor.FromData([]float32{0, 0, 0, 1}, 4), 0)
	s := float32(1 / math.Sqrt(1.25)) // variance of {1,2,3,4} is 1.25
	nearAll(t, x.Data, []float32{-1.5 * s, -0.5 * s, 2 * 0.5 * s, 2*1.5*s + 1})
}

func TestAddLayerNorm(t *testing.T) {
	x := tensor.FromData([]float32{0, 1, 2, 3, 10, 20, 30, 40}, 2, 4)
	res := tensor.FromData([]float32{1, 1, 1, 1, -9, -18, -27, -36}, 2, 4)
	w, b := tensor.FromData([]float32{1, 1, 2, 2}, 4), tensor.FromData([]float32{0, 0, 0, 1}, 4)
	AddLayerNorm(x, res, w, b, 0)
	// Both rows become {1, 2, 3, 4} before normalizing.
	s := float32(1 / math.Sqrt(1.25))
	want := []float32{-1.5 * s, -0.5 * s, 2 * 0.5 * s, 2*1.5*s + 1}
	nearAll(t, x.Data, append(want, want...))
}

func TestGELU(t *testing.T) {
	x := []float32{-3, -1, 0, 1, 3}
	GELU(x)
	nearAll(t, x, []float32{-0.0040496, -0.1586553, 0, 0.8413447, 2.9959502})
}

func TestSoftmax(t *testing.T) {
	tests := []struct {
		name string
		in   []float32
		want []float32
	}{
		{"uniform", []float32{2, 2, 2, 2}, []float32{0.25, 0.25, 0.25, 0.25}},
		{"two", []float32{0, float32(math.Log(3))}, []float32{0.25, 0.75}},
		{"large values stay finite", []float32{1000, 1000}, []float32{0.5, 0.5}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			Softmax(tt.in)
			nearAll(t, tt.in, tt.want)
		})
	}
}

func TestAttention(t *testing.T) {
	// Two heads of width 1. Head 0's query only matches key 0; head 1's
	// scores tie, so it averages the values.
	q := tensor.FromData([]float32{100, 0}, 1, 2)
	k := tensor.FromData([]float32{1, 1, 0, 1}, 2, 2)
	v := tensor.FromData([]float32{5, 2, 7, 4}, 2, 2)
	out := Attention(q, k, v, 2, 1)
	nearAll(t, out.Data, []float32{5, 3})
}

func TestGather(t *testing.T) {
	table := tensor.FromData([]float32{0, 1, 10, 11, 20, 21}, 3, 2)
	out := Gather(table, []int{2, 0, 2})
	nearAll(t, out.Data, []float32{20, 21, 0, 1, 20, 21})
}

func TestSwiGLUMatchesReference(t *testing.T) {
	r := rand.New(rand.NewPCG(81, 82))
	const rows, ff = 3, 37
	gu := tensor.FromData(randomSlice(r, rows*2*ff), rows, 2*ff)
	for i := range gu.Data {
		gu.Data[i] *= 6 // reach saturated parts of the sigmoid
	}
	out := tensor.New(rows, ff)
	SwiGLUInto(out, gu)
	for i := range rows {
		for j := range ff {
			g, u := float64(gu.Data[i*2*ff+j]), float64(gu.Data[i*2*ff+ff+j])
			want := g / (1 + math.Exp(-g)) * u
			if got := float64(out.Data[i*ff+j]); math.Abs(got-want) > 2e-6*(1+math.Abs(want)) {
				t.Fatalf("out[%d][%d] = %g, want %g", i, j, got, want)
			}
		}
	}
}

func TestSiLUEdges(t *testing.T) {
	x := []float32{0, 1, -1, 100, -100, float32(math.Inf(1)), float32(math.Inf(-1))}
	SiLU(x)
	want := []float32{0, 0.7310586, -0.26894143, 100, 0, float32(math.Inf(1)), 0}
	for i := range x {
		if math.Abs(float64(x[i]-want[i])) > 1e-6 && x[i] != want[i] {
			t.Errorf("SiLU: x[%d] = %g, want %g", i, x[i], want[i])
		}
	}
}
