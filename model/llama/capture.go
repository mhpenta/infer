package llama

import (
	"errors"
	"fmt"
	"slices"

	"github.com/mhpenta/infer/sample"
	"github.com/mhpenta/infer/tensor"
)

// Point is a place in a layer's forward pass where a Capture reads values.
// HiddenStates exposes what enters each layer, which mixes these.
type Point uint8

const (
	// AttentionOut is what attention adds to the layer's input: the output
	// of o_proj, [rows, hidden].
	AttentionOut Point = iota
	// SwiGLUOut is SiLU(gate_proj(x))·up_proj(x), the input to down_proj,
	// [rows, intermediate].
	SwiGLUOut
	// MLPOut is what the feed-forward adds to the layer's input: the output
	// of down_proj, [rows, hidden].
	MLPOut
	numPoints
)

func (p Point) String() string {
	if p < numPoints {
		return [...]string{"attention", "swiglu", "mlp"}[p]
	}
	return fmt.Sprintf("Point(%d)", uint8(p))
}

// Activation is the values at one Point of one layer for the rows one forward
// pass ran.
type Activation struct {
	Layer int
	Point Point
	// Start is the sequence position of the first row: row r is position
	// Start+r.
	Start int
	// Values is the model's own scratch memory, [rows, cols] with cols as
	// Point says. It is valid until the function it was passed to returns, so
	// copy what you keep. The pass has already used the values, so writing to
	// them changes nothing that follows.
	Values *tensor.Tensor
}

// Capture selects activations to read during a forward pass. Nothing is
// copied: Fn sees each selection in place, and memory is only set aside for
// what is selected (on the CPU, none; on the GPU, rows times the widths
// selected). Without a Capture, a forward pass only checks a pointer per
// layer, and with one it computes the same values, bit for bit.
type Capture struct {
	// Layers selects layers by index; nil selects every layer.
	Layers []int
	// Points selects what to read from each of them, at least one.
	Points []Point
	// Fn receives the selection: for each forward pass, in order of layer and
	// then of Point, each selection once, on the goroutine that runs the
	// model. Prompts on the GPU run in several passes when they are long
	// (gpuChunk), so use Start, not the order of calls, to place rows.
	Fn func(Activation)
}

// capturePlan is a checked Capture.
type capturePlan struct {
	fn    func(Activation)
	masks []uint8 // per layer, a bit per Point
	width int     // the values one row takes across the whole selection
}

func (m *Model) plan(c Capture) (*capturePlan, error) {
	switch {
	case c.Fn == nil:
		return nil, errors.New("llama: a Capture needs a Fn")
	case len(c.Points) == 0:
		return nil, errors.New("llama: a Capture needs a Point")
	case c.Layers != nil && len(c.Layers) == 0:
		return nil, errors.New("llama: a Capture's Layers is empty; nil selects every layer")
	}
	var mask uint8
	for _, p := range c.Points {
		if p >= numPoints {
			return nil, fmt.Errorf("llama: no capture point %v", p)
		}
		mask |= 1 << p
	}
	plan := &capturePlan{fn: c.Fn, masks: make([]uint8, m.Config.NumHiddenLayers)}
	if c.Layers == nil {
		for i := range plan.masks {
			plan.masks[i] = mask
		}
	}
	for _, i := range c.Layers {
		if i < 0 || i >= len(plan.masks) {
			return nil, fmt.Errorf("llama: no layer %d of %d", i, len(plan.masks))
		}
		plan.masks[i] = mask
	}
	for _, layer := range plan.masks {
		for _, p := range capturePoints {
			if layer&(1<<p) != 0 {
				plan.width += m.Config.captureCols(p)
			}
		}
	}
	return plan, nil
}

// report calls fn with values if the plan selects layer's point. Both devices
// report after the pass has used the values.
func (p *capturePlan) report(layer int, point Point, start int, values *tensor.Tensor) {
	if p.masks[layer]&(1<<point) != 0 {
		p.fn(Activation{Layer: layer, Point: point, Start: start, Values: values})
	}
}

// capturePoints are the points in the order a Capture delivers them.
var capturePoints = [...]Point{AttentionOut, SwiGLUOut, MLPOut}

func (c Config) captureCols(p Point) int {
	if p == SwiGLUOut {
		return c.IntermediateSize
	}
	return c.HiddenSize
}

// Capture runs the whole sequence ids (no cache reuse, as HiddenStates) and
// calls c.Fn with what it selects. Collect it with a Recording to keep it.
func (m *Model) Capture(ids []int, c Capture) error {
	if err := m.checkIDs(ids); err != nil {
		return err
	}
	plan, err := m.plan(c)
	if err != nil {
		return err
	}
	st := m.newState(len(ids))
	defer st.release()
	st.capture = plan
	_, err = st.forward(ids, nil)
	return err
}

// GenerateCapture is GenerateWith also calling c.Fn as generation runs the KV
// cache's forward passes: the prompt's positions, before fn sees the first
// token, then each token's single position before fn sees the next. The last
// token generated never runs, so maxNew tokens give len(ids)+maxNew-1 rows.
func (m *Model) GenerateCapture(ids []int, maxNew int, s *sample.Sampler, c Capture, fn func(token int, logits []float32) bool) error {
	if err := m.checkPrompt(ids, maxNew); err != nil {
		return err
	}
	plan, err := m.plan(c)
	if err != nil {
		return err
	}
	st := m.newState(len(ids) + maxNew)
	defer st.release()
	st.capture = plan
	return st.generate(ids, maxNew, s, fn)
}

// SetCapture makes later calls to GenerateWith call c.Fn as GenerateCapture
// does, or stop doing so with a nil c. Only positions that run are captured:
// those the cache reuses (Reused) are not, so the first row of a call is at
// position Reused.
func (s *Session) SetCapture(c *Capture) error {
	if c == nil {
		s.st.capture = nil
		return nil
	}
	plan, err := s.m.plan(*c)
	if err != nil {
		return err
	}
	s.st.capture = plan
	return nil
}

// Recording keeps copies of what a Capture selects, assembled across forward
// passes. Use its Record as the Capture's Fn:
//
//	var rec llama.Recording
//	err := m.Capture(ids, llama.Capture{Layers: []int{3}, Points: []llama.Point{llama.MLPOut}, Fn: rec.Record})
//	mlp, ok := rec.Get(3, llama.MLPOut)
//
// A position that runs again, as when a Session rewinds, replaces what the
// Recording held from that position on.
type Recording struct {
	entries map[recordingKey]*recorded
}

type recordingKey struct {
	layer int
	point Point
}

type recorded struct {
	start, cols int
	data        []float32
}

// Record adds a's rows.
func (r *Recording) Record(a Activation) {
	if r.entries == nil {
		r.entries = map[recordingKey]*recorded{}
	}
	key := recordingKey{a.Layer, a.Point}
	e := r.entries[key]
	cols := a.Values.Cols()
	switch {
	case e == nil || a.Start < e.start:
		e = &recorded{start: a.Start, cols: cols}
		r.entries[key] = e
	case a.Start > e.start+len(e.data)/cols:
		panic(fmt.Sprintf("llama.Recording: layer %d's %v jumps from position %d to %d", a.Layer, a.Point, e.start+len(e.data)/cols, a.Start))
	default:
		e.data = e.data[:(a.Start-e.start)*cols]
	}
	e.data = append(e.data, a.Values.Data...)
}

// Get returns the rows recorded for a layer's point, which the caller may
// keep, and whether there are any.
func (r *Recording) Get(layer int, point Point) (Activation, bool) {
	e := r.entries[recordingKey{layer, point}]
	if e == nil {
		return Activation{}, false
	}
	return Activation{Layer: layer, Point: point, Start: e.start,
		Values: tensor.FromData(slices.Clone(e.data), len(e.data)/e.cols, e.cols)}, true
}
