package llama

import (
	"fmt"
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/mhpenta/infer/tensor"
)

var allPoints = []Point{AttentionOut, SwiGLUOut, MLPOut}

// The largest differences allowed in a row of an activation, from PyTorch's or
// between devices, relative to the row's largest value. Short sequences stay
// within 4.6e-5.
//
// On Qwen3's 2,461-token prompt a few rows go further: 58 of 2,602 in layer
// 14's feed-forward are past 1e-4 between devices, and one, position 1415, is
// 6e-3 apart in attention. That is float32 rounding, not a device's error.
// Against a float64 run of transformers, that row is 7e-4 off on the CPU and
// 5.3e-3 on the GPU, and PyTorch's own float32 run, 5.3e-3 from the CPU, is
// at least 4.6e-3 off too. Attention there mixes in the first token's values of
// about 14,600 (see qwen3 in llama_test.go), whose rounding of about 1e-6 is
// 0.018 here, beside a row whose largest value is 3. The median row stays
// within 7e-6 of float64 on either device, and medianTol holds every row set to
// that order.
const (
	captureTol = 1e-4
	longTol    = 1e-2
	medianTol  = 2e-5
)

// compareRows checks got against want row by row, and returns the largest
// difference relative to a row's largest value. A contribution can be small
// beside the layer's input it was computed from, which sets the rounding, so
// rows are compared separately: Qwen3's first position is thousands of times
// larger than the rest and would hide them.
func compareRows(t *testing.T, name string, got, want *tensor.Tensor, tol float64) float64 {
	t.Helper()
	if !slices.Equal(got.Shape, want.Shape) {
		t.Fatalf("%s: shape %v, want %v", name, got.Shape, want.Shape)
	}
	worst := 0.0
	rels := make([]float64, want.Rows())
	for r := range want.Rows() {
		var peak, diff float64
		for j, v := range want.Row(r) {
			peak = max(peak, math.Abs(float64(v)))
			diff = max(diff, math.Abs(float64(got.Row(r)[j])-float64(v)))
		}
		rels[r] = diff / peak
		if rels[r] > worst || math.IsNaN(rels[r]) {
			worst = rels[r]
		}
	}
	if !(worst <= tol) {
		t.Errorf("%s: a row differs by %.3g of its range, want <= %g", name, worst, tol)
	}
	slices.Sort(rels)
	if median := rels[len(rels)/2]; median > medianTol {
		t.Errorf("%s: the median row differs by %.3g of its range, want <= %g", name, median, medianTol)
	}
	return worst
}

// TestCaptureMatchesPython compares captures with what transformers' forward
// hooks see, for whole sequences and for generation with a KV cache and a
// Session, on each model and device.
func TestCaptureMatchesPython(t *testing.T) {
	for _, g := range goldenModels {
		for _, device := range testDevices(t) {
			g := g.on(device)
			t.Run(g.key(), func(t *testing.T) {
				defer g.release()
				m, cases, _ := g.load(t)
				h, want := g.hooks(t)
				check := func(t *testing.T, name string, rec *Recording, skip int) float64 {
					t.Helper()
					worst := 0.0
					for _, layer := range h.Layers {
						for _, p := range allPoints {
							key := fmt.Sprintf("%s.%d.%s", name, layer, p)
							ref, err := want.Tensor(key)
							if err != nil {
								t.Fatal(err)
							}
							got, ok := rec.Get(layer, p)
							if !ok {
								t.Fatalf("%s: nothing captured", key)
							}
							if got.Start != skip {
								t.Fatalf("%s: starts at position %d, want %d", key, got.Start, skip)
							}
							cols := ref.Cols()
							tail := tensor.FromData(ref.Data[skip*cols:], ref.Rows()-skip, cols)
							worst = max(worst, compareRows(t, key, got.Values, tail, captureTol))
						}
					}
					return worst
				}
				for _, i := range h.WholeCases {
					var rec Recording
					if err := m.Capture(cases[i].sequence(), Capture{Layers: h.Layers, Points: allPoints, Fn: rec.Record}); err != nil {
						t.Fatal(err)
					}
					t.Logf("case %d whole: largest row difference from PyTorch %.2g", i, check(t, fmt.Sprintf("case%d.whole", i), &rec, 0))
				}
				for _, i := range h.CachedCases {
					c := cases[i]
					var rec Recording
					passes := map[int]int{}
					var got []int
					err := m.GenerateCapture(c.InputIDs, len(c.OutputIDs), nil, Capture{Layers: h.Layers, Points: allPoints, Fn: func(a Activation) {
						passes[a.Values.Rows()]++
						rec.Record(a)
					}}, func(token int, _ []float32) bool {
						got = append(got, token)
						return true
					})
					if err != nil {
						t.Fatal(err)
					}
					if !slices.Equal(got, c.OutputIDs) {
						t.Fatalf("case %d: generated %v, want %v", i, got, c.OutputIDs)
					}
					selected := len(h.Layers) * len(allPoints)
					if passes[len(c.InputIDs)] != selected || passes[1] != selected*(len(c.OutputIDs)-1) {
						t.Errorf("case %d: passes of each size %v; the cache was not used as generate uses it", i, passes)
					}
					t.Logf("case %d cached: largest row difference from PyTorch %.2g", i, check(t, fmt.Sprintf("case%d.cached", i), &rec, 0))

					// A Session reports only what its cache does not reuse.
					s := m.NewSession()
					defer s.Close()
					half := len(c.InputIDs) / 2
					if err := s.GenerateWith(c.InputIDs[:half], 1, nil, func(int, []float32) bool { return true }); err != nil {
						t.Fatal(err)
					}
					var srec Recording
					if err := s.SetCapture(&Capture{Layers: h.Layers, Points: allPoints, Fn: srec.Record}); err != nil {
						t.Fatal(err)
					}
					if err := s.GenerateWith(c.InputIDs, len(c.OutputIDs), nil, func(int, []float32) bool { return true }); err != nil {
						t.Fatal(err)
					}
					if s.Reused() < half {
						t.Errorf("case %d: the session reused %d positions, want at least %d", i, s.Reused(), half)
					}
					t.Logf("case %d session, %d reused: largest row difference %.2g", i, s.Reused(), check(t, fmt.Sprintf("case%d.cached", i), &srec, s.Reused()))
					// And nothing once capture is off.
					if err := s.SetCapture(nil); err != nil {
						t.Fatal(err)
					}
					before, _ := srec.Get(h.Layers[0], MLPOut)
					if err := s.GenerateWith(c.InputIDs, 1, nil, func(int, []float32) bool { return true }); err != nil {
						t.Fatal(err)
					}
					after, _ := srec.Get(h.Layers[0], MLPOut)
					if !bitsEqual(before.Values.Data, after.Values.Data) {
						t.Error("a session captured after SetCapture(nil)")
					}
				}
			})
		}
	}
}

// TestCaptureDevicesAgree compares every layer's activations between the CPU
// and the GPU: for a whole sequence (the long case crosses the GPU's chunks of
// gpuChunk positions, so its rows arrive in several passes), and from
// generation.
func TestCaptureDevicesAgree(t *testing.T) {
	devices := testDevices(t)
	if len(devices) < 2 {
		t.Skip("needs Metal")
	}
	for _, g := range goldenModels {
		t.Run(g.dir, func(t *testing.T) {
			_, cases, _ := g.load(t)
			n := g.on(devices[0])
			g.release()
			record := func(device string, c modelCase, layers []int) (whole, generated *Recording) {
				g := n.on(device)
				defer g.release()
				m, _, _ := g.load(t)
				whole, generated = &Recording{}, &Recording{}
				if err := m.Capture(c.sequence(), Capture{Layers: layers, Points: allPoints, Fn: whole.Record}); err != nil {
					t.Fatal(err)
				}
				if len(c.InputIDs) > gpuChunk {
					return whole, generated // the chunks are what the long prompt is for; generating it takes minutes
				}
				if err := m.GenerateCapture(c.InputIDs, len(c.OutputIDs), nil, Capture{Layers: layers, Points: allPoints, Fn: generated.Record},
					func(int, []float32) bool { return true }); err != nil {
					t.Fatal(err)
				}
				return whole, generated
			}
			for _, i := range []int{2, longCase} {
				if i == longCase && testing.Short() {
					continue
				}
				var layers []int // nil is all of them
				tol := captureTol
				if i == longCase {
					layers = []int{0, n.layersOf(t) / 2}
					tol = longTol
				}
				cw, cg := record(devices[0], cases[i], layers)
				mw, mg := record(devices[1], cases[i], layers)
				for name, pair := range map[string][2]*Recording{"whole": {cw, mw}, "generated": {cg, mg}} {
					worst, compared := 0.0, 0
					for layer := range n.layersOf(t) {
						for _, p := range allPoints {
							a, ok := pair[0].Get(layer, p)
							b, okb := pair[1].Get(layer, p)
							if ok != okb {
								t.Fatalf("case %d %s layer %d %v: captured on one device only", i, name, layer, p)
							}
							if ok {
								worst = max(worst, compareRows(t, fmt.Sprintf("case %d %s layer %d %v", i, name, layer, p), b.Values, a.Values, tol))
								compared += a.Values.Rows()
							}
						}
					}
					if compared == 0 {
						if name == "generated" && i == longCase {
							continue
						}
						t.Fatalf("case %d %s: nothing compared", i, name)
					}
					t.Logf("case %d %s: %d rows, largest row difference between devices %.2g", i, name, compared, worst)
				}
			}
		})
	}
}

// layersOf returns the model's number of layers, from its config.
func (g goldenModel) layersOf(t *testing.T) int {
	t.Helper()
	m, _, _ := g.on(g.onDevice()).load(t)
	return m.Config.NumHiddenLayers
}

// TestCaptureSelects checks that only what is selected arrives, once, in the
// documented order and shape, whatever the order of the selection.
func TestCaptureSelects(t *testing.T) {
	for _, device := range testDevices(t) {
		g := miniCPM.on(device)
		t.Run(g.key(), func(t *testing.T) {
			defer g.release()
			m, cases, _ := g.load(t)
			ids := cases[0].sequence()
			c := m.Config
			type seen struct {
				layer int
				point Point
			}
			var got []seen
			err := m.Capture(ids, Capture{Layers: []int{5, 2, 5}, Points: []Point{MLPOut, AttentionOut, MLPOut}, Fn: func(a Activation) {
				got = append(got, seen{a.Layer, a.Point})
				if a.Start != 0 || !slices.Equal(a.Values.Shape, []int{len(ids), c.HiddenSize}) {
					t.Errorf("layer %d %v: start %d, shape %v", a.Layer, a.Point, a.Start, a.Values.Shape)
				}
			}})
			if err != nil {
				t.Fatal(err)
			}
			want := []seen{{2, AttentionOut}, {2, MLPOut}, {5, AttentionOut}, {5, MLPOut}}
			if !slices.Equal(got, want) {
				t.Errorf("delivered %v, want %v", got, want)
			}
			got = nil
			if err := m.Capture(ids, Capture{Points: []Point{SwiGLUOut}, Fn: func(a Activation) {
				got = append(got, seen{a.Layer, a.Point})
				if !slices.Equal(a.Values.Shape, []int{len(ids), c.IntermediateSize}) {
					t.Errorf("layer %d %v: shape %v", a.Layer, a.Point, a.Values.Shape)
				}
			}}); err != nil {
				t.Fatal(err)
			}
			if len(got) != c.NumHiddenLayers {
				t.Errorf("nil Layers delivered %d layers, want %d", len(got), c.NumHiddenLayers)
			}
			fn := func(Activation) {}
			for _, bad := range []struct {
				c    Capture
				want string
			}{
				{Capture{Points: allPoints}, "Fn"},
				{Capture{Fn: fn}, "Point"},
				{Capture{Points: allPoints, Layers: []int{}, Fn: fn}, "empty"},
				{Capture{Points: allPoints, Layers: []int{c.NumHiddenLayers}, Fn: fn}, "no layer"},
				{Capture{Points: allPoints, Layers: []int{-1}, Fn: fn}, "no layer"},
				{Capture{Points: []Point{numPoints}, Fn: fn}, "no capture point"},
			} {
				if err := m.Capture(ids, bad.c); err == nil || !strings.Contains(err.Error(), bad.want) {
					t.Errorf("Capture(%+v): %v, want an error about %q", bad.c, err, bad.want)
				}
				if err := m.GenerateCapture(ids, 1, nil, bad.c, func(int, []float32) bool { return true }); err == nil {
					t.Errorf("GenerateCapture(%+v) succeeded", bad.c)
				}
				if err := m.NewSession().SetCapture(&bad.c); err == nil {
					t.Errorf("SetCapture(%+v) succeeded", bad.c)
				}
			}
		})
	}
}

// TestCaptureLeavesOutput checks that capturing changes nothing, bit for bit,
// whatever Fn does to what it is shown, and that the contributions captured
// are what takes each layer's input to the next's.
func TestCaptureLeavesOutput(t *testing.T) {
	for _, device := range testDevices(t) {
		g := miniCPM.on(device)
		t.Run(g.key(), func(t *testing.T) {
			defer g.release()
			m, cases, _ := g.load(t)
			c := cases[2]
			want := generateRun(t, m, c.InputIDs, len(c.OutputIDs))
			for _, tt := range []struct {
				name string
				fn   func(Activation)
			}{
				{"reading", func(Activation) {}},
				{"writing", func(a Activation) {
					for i := range a.Values.Data {
						a.Values.Data[i] = 1e9
					}
				}},
			} {
				var r run
				err := m.GenerateCapture(c.InputIDs, len(c.OutputIDs), nil, Capture{Points: allPoints, Fn: tt.fn}, func(token int, logits []float32) bool {
					r.tokens = append(r.tokens, token)
					r.logits = append(r.logits, slices.Clone(logits))
					return true
				})
				if err != nil {
					t.Fatal(err)
				}
				if !slices.Equal(r.tokens, want.tokens) {
					t.Errorf("%s: generated %v, want %v", tt.name, r.tokens, want.tokens)
				}
				if bits, worst := r.diff(want); bits != 0 {
					t.Errorf("%s: %d logits differ, by up to %.3g of a row's range", tt.name, bits, worst)
				}
			}

			ids := cases[1].sequence()
			states, err := m.HiddenStates(ids)
			if err != nil {
				t.Fatal(err)
			}
			var rec Recording
			if err := m.Capture(ids, Capture{Points: []Point{AttentionOut, MLPOut}, Fn: rec.Record}); err != nil {
				t.Fatal(err)
			}
			for layer := range m.Config.NumHiddenLayers - 1 { // the last hidden state is normalized
				attn, _ := rec.Get(layer, AttentionOut)
				mlp, _ := rec.Get(layer, MLPOut)
				sum := slices.Clone(states[layer].Data)
				for i := range sum {
					sum[i] = sum[i] + attn.Values.Data[i]
					sum[i] = sum[i] + mlp.Values.Data[i]
				}
				if !bitsEqual(sum, states[layer+1].Data) {
					t.Errorf("layer %d: input + attention + mlp is not the next layer's input", layer)
				}
			}
		})
	}
}

func TestRecording(t *testing.T) {
	rows := func(start int, v ...float32) Activation {
		return Activation{Layer: 1, Point: MLPOut, Start: start, Values: tensor.FromData(v, len(v)/2, 2)}
	}
	var r Recording
	if _, ok := r.Get(1, MLPOut); ok {
		t.Error("an empty Recording has rows")
	}
	r.Record(rows(4, 1, 2, 3, 4))
	r.Record(rows(6, 5, 6))
	got, _ := r.Get(1, MLPOut)
	if got.Start != 4 || !slices.Equal(got.Values.Data, []float32{1, 2, 3, 4, 5, 6}) {
		t.Errorf("assembled %+v", got)
	}
	got.Values.Data[0] = 100 // the caller's to keep
	r.Record(rows(5, 7, 8))  // a rewind replaces from there on
	got, _ = r.Get(1, MLPOut)
	if got.Start != 4 || !slices.Equal(got.Values.Data, []float32{1, 2, 7, 8}) {
		t.Errorf("after rewinding: %+v", got)
	}
	r.Record(rows(0, 9, 9)) // to before the start, it starts over
	got, _ = r.Get(1, MLPOut)
	if got.Start != 0 || !slices.Equal(got.Values.Data, []float32{9, 9}) {
		t.Errorf("after rewinding past the start: %+v", got)
	}
	defer func() {
		if recover() == nil {
			t.Error("a gap in positions did not panic")
		}
	}()
	r.Record(rows(5, 1, 1))
}
