package inkling2

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"

	"github.com/mhpenta/infer/ops"
	"github.com/mhpenta/infer/safetensors"
	"github.com/mhpenta/infer/tensor"
)

// openWeights opens model.safetensors, or every shard listed in
// model.safetensors.index.json, and maps each tensor name to its file.
func openWeights(dir string) ([]*safetensors.File, map[string]*safetensors.File, error) {
	byName := map[string]*safetensors.File{}
	single := filepath.Join(dir, "model.safetensors")
	if _, err := os.Stat(single); err == nil {
		f, err := safetensors.Open(single)
		if err != nil {
			return nil, nil, err
		}
		for _, name := range f.Names() {
			byName[name] = f
		}
		return []*safetensors.File{f}, byName, nil
	}
	raw, err := os.ReadFile(filepath.Join(dir, "model.safetensors.index.json"))
	if err != nil {
		return nil, nil, fmt.Errorf("no model.safetensors or index: %w", err)
	}
	var index struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	if err := json.Unmarshal(raw, &index); err != nil {
		return nil, nil, fmt.Errorf("parse model.safetensors.index.json: %w", err)
	}
	var files []*safetensors.File
	shards := map[string]*safetensors.File{}
	names := make([]string, 0, len(index.WeightMap))
	for name := range index.WeightMap {
		names = append(names, name)
	}
	slices.Sort(names)
	for _, name := range names {
		shard := index.WeightMap[name]
		f, ok := shards[shard]
		if !ok {
			if f, err = safetensors.Open(filepath.Join(dir, filepath.Base(shard))); err != nil {
				for _, open := range files {
					open.Close()
				}
				return nil, nil, err
			}
			shards[shard] = f
			files = append(files, f)
		}
		byName[name] = f
	}
	return files, byName, nil
}

// loader reads tensors by name, remembering the first error.
type loader struct {
	files map[string]*safetensors.File
	err   error
}

func (l *loader) fail(format string, args ...any) {
	if l.err == nil {
		l.err = fmt.Errorf(format, args...)
	}
}

// entry returns the file holding name, checking its dtype and shape.
func (l *loader) entry(name string, dtypes []string, shape ...int) *safetensors.File {
	if l.err != nil {
		return nil
	}
	f, ok := l.files[name]
	if !ok {
		l.fail("tensor %s is missing", name)
		return nil
	}
	if !slices.Contains(dtypes, f.DType(name)) {
		l.fail("tensor %s is %s, want one of %v", name, f.DType(name), dtypes)
		return nil
	}
	if got := f.Shape(name); !slices.Equal(got, shape) {
		l.fail("tensor %s has shape %v, want %v", name, got, shape)
		return nil
	}
	return f
}

// tensor returns a float tensor as float32.
func (l *loader) tensor(name string, shape ...int) *tensor.Tensor {
	f := l.entry(name, []string{"F32", "BF16", "F16"}, shape...)
	if f == nil {
		return nil
	}
	t, err := f.Tensor(name)
	if err != nil {
		l.fail("%w", err)
		return nil
	}
	return t
}

// floats returns a float tensor's values, copied out of the mapping.
func (l *loader) floats(name string, shape ...int) []float32 {
	t := l.tensor(name, shape...)
	if t == nil {
		return nil
	}
	return slices.Clone(t.Data)
}

// bytes returns a tensor's stored bytes, a view into the mapping.
func (l *loader) bytes(name string, dtype string, shape ...int) []byte {
	f := l.entry(name, []string{dtype}, shape...)
	if f == nil {
		return nil
	}
	b, err := f.Bytes(name)
	if err != nil {
		l.fail("%w", err)
		return nil
	}
	return b
}

// pack packs a linear weight [out, in] as bf16 panels when it is stored as
// bf16, float32 otherwise.
func (l *loader) pack(w *tensor.Tensor, bf16 bool) *ops.Packed {
	if l.err != nil {
		return nil
	}
	if !bf16 {
		return ops.Pack(w)
	}
	p, err := ops.PackBF16(w)
	if err != nil {
		l.fail("%w", err)
	}
	return p
}

// stacked loads linear layers over the same input, outs[i] outputs each,
// as one, so one product computes all of them.
func (l *loader) stacked(in int, names []string, outs []int) *ops.Packed {
	var rows [][]float32
	bf16 := true
	for i, name := range names {
		t := l.tensor(name, outs[i], in)
		if l.err != nil {
			return nil
		}
		rows = append(rows, t.Data)
		bf16 = bf16 && l.files[name].DType(name) == "BF16"
	}
	w := slices.Concat(rows...)
	return l.pack(tensor.FromData(w, len(w)/in, in), bf16)
}

// linear loads one nn.Linear weight [out, in].
func (l *loader) linear(name string, out, in int) *ops.Packed {
	return l.stacked(in, []string{name}, []int{out})
}

// conv loads a depthwise convolution weight stored [channels, 1, kernel].
func (l *loader) conv(name string, channels, kernel int) []float32 {
	return l.floats(name, channels, 1, kernel)
}

// experts loads a layer's routed experts, stored NVFP4 (with amax) or MXFP8:
// w13, every expert's gate and up projections with their rows interleaved
// (gate j, then up j), and w2, the down projections.
func (l *loader) experts(p string, c Config) (gateUp, down []*ops.Microscaled) {
	h, ff, n := c.HiddenSize, c.IntermediateSize, c.NRoutedExperts
	load := func(name string, rows, cols int) []*ops.Microscaled {
		if l.err != nil {
			return nil
		}
		f, ok := l.files[name]
		if !ok {
			l.fail("tensor %s is missing", name)
			return nil
		}
		out := make([]*ops.Microscaled, n)
		if f.DType(name) == "U8" { // NVFP4: two codes per byte
			codes := l.bytes(name, "U8", n, rows, cols/2)
			scales := l.bytes(name+"_scale", "F8_E4M3", n, rows, cols/16)
			amax := l.floats(name+"_amax", n)
			den := l.floats(name+"_outer_scale_denominator", 1)
			if l.err != nil {
				return nil
			}
			for e := range n {
				size, ssize := rows*cols/2, rows*cols/16
				// transformers: amax.float() / denominator.float().
				w, err := ops.NewNVFP4(rows, cols, codes[e*size:(e+1)*size], scales[e*ssize:(e+1)*ssize], amax[e]/den[0])
				if err != nil {
					l.fail("%s: %w", name, err)
					return nil
				}
				out[e] = w
			}
			return out
		}
		values := l.bytes(name, "F8_E4M3", n, rows, cols)
		scales := l.bytes(name+"_scale", "U8", n, rows, cols/32)
		if l.err != nil {
			return nil
		}
		for e := range n {
			size, ssize := rows*cols, rows*cols/32
			w, err := ops.NewMXFP8(rows, cols, values[e*size:(e+1)*size], scales[e*ssize:(e+1)*ssize])
			if err != nil {
				l.fail("%s: %w", name, err)
				return nil
			}
			out[e] = w
		}
		return out
	}
	return load(p+"w13_weight", 2*ff, h), load(p+"w2_weight", h, ff)
}

// bf16Rows returns the stored bytes of the first rows of a bf16 matrix
// [≥rows, cols], a view into the mapping.
func (l *loader) bf16Rows(name string, rows, cols int) []byte {
	if l.err != nil {
		return nil
	}
	f, ok := l.files[name]
	switch {
	case !ok:
		l.fail("tensor %s is missing", name)
		return nil
	case f.DType(name) != "BF16":
		l.fail("tensor %s is %s, want BF16", name, f.DType(name))
		return nil
	}
	if shape := f.Shape(name); len(shape) != 2 || shape[0] < rows || shape[1] != cols {
		l.fail("tensor %s has shape %v, want [≥%d, %d]", name, shape, rows, cols)
		return nil
	}
	b, err := f.Bytes(name)
	if err != nil {
		l.fail("%w", err)
		return nil
	}
	return b[:2*rows*cols]
}

// bf16Row writes row r of bf16 bytes [*, len(dst)] to dst as float32.
func bf16Row(dst []float32, b []byte, r int) {
	b = b[2*r*len(dst):]
	for j := range dst {
		dst[j] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(b[2*j:])) << 16)
	}
}

// newModel loads the weights of a checkpoint as Thinking Machines ship it,
// under model.llm.
func newModel(c Config, byName map[string]*safetensors.File) (*Model, error) {
	if _, ok := byName["model.llm.embed.weight"]; !ok {
		return nil, fmt.Errorf("no model.llm.embed.weight: not an original Inkling 2 checkpoint")
	}
	l := &loader{files: byName}
	h, k := c.HiddenSize, c.SconvKernelSize
	m := &Model{Config: c}
	const base = "model.llm."

	m.embed = l.bf16Rows(base+"embed.weight", c.VocabSize, h)
	m.embedNorm = l.floats(base+"embed_norm.weight", h)
	m.norm = l.floats(base+"norm.weight", h)
	m.finalQuery = l.floats(base+"attn_res.output_query", h)

	for i := range c.NumHiddenLayers {
		p := fmt.Sprintf("%slayers.%d.", base, i)
		heads, kvHeads, hd := c.heads(i)
		kv := kvHeads * hd
		ly := layer{
			index:   i,
			sliding: c.sliding(i),
			loop:    c.inLoop(i),
			heads:   heads, kvHeads: kvHeads, headDim: hd,
			relExtent: c.relExtent(i),
		}
		a := p + "attn."
		ly.q = l.stacked(h,
			[]string{a + "wq_du.weight", a + "gate_attn.weight", a + "wr_du.weight", a + "attn_sink_proj.weight"},
			[]int{heads * hd, heads * hd, heads * c.DRel, heads})
		ly.qScale = l.floats(a+"q_l2_clip.scale", heads, hd)
		ly.relScale = l.floats(a+"rel_l2_clip.scale", heads, c.DRel)
		ly.relProj = l.floats(a+"rel_logits_proj.proj", c.DRel, ly.relExtent)
		ly.out = l.linear(a+"wo_ud.weight", h, heads*hd)
		if ly.sliding {
			ly.k = l.linear(a+"wk_dv.weight", kv, h)
			ly.v = l.linear(a+"wv_dv.weight", kv, h)
			ly.kConv = l.conv(a+"k_sconv.weight", kv, k)
			ly.vConv = l.conv(a+"v_sconv.weight", kv, k)
			ly.inNorm = l.floats(p+"attn_norm.weight", h)
		} else {
			// The dynamic convolutions' per-token taps come from the same
			// inputs as the keys and values: one product each.
			ly.k = l.stacked(h, []string{a + "wk_dv.weight", a + "k_sconv.weight_proj"}, []int{kv, kvHeads * k})
			ly.v = l.stacked(h, []string{a + "wv_dv.weight", a + "v_sconv.weight_proj"}, []int{kv, kvHeads * k})
			ly.kConv = l.floats(a+"k_sconv.static_w", k, kv)
			ly.vConv = l.floats(a+"v_sconv.static_w", k, kv)
			ly.qNorm = l.floats(p+"q_input_norm.weight", h)
			ly.kNorm = l.floats(p+"k_input_norm.weight", h)
			ly.vNorm = l.floats(p+"v_input_norm.weight", h)
		}
		ly.postAttnConv = l.conv(p+"attn_sconv.weight", h, k)
		ly.postMLPConv = l.conv(p+"mlp_sconv.weight", h, k)
		ly.mlpNorm = l.floats(p+"mlp_norm.weight", h)

		// attn_queries.j belongs to layer j+1: layer 0 has none.
		if i > 0 {
			name := fmt.Sprintf("%sattn_res.attn_queries.%d", base, i-1)
			if ly.sliding {
				ly.attnQueries = l.floats(name, h)
			} else {
				ly.attnQueries = l.floats(name, 3, h)
			}
		}
		ly.mlpQuery = l.floats(fmt.Sprintf("%sattn_res.mlp_queries.%d", base, i), h)

		ly.router = l.linear(p+"mlp.gate.weight", c.NRoutedExperts, h)
		ly.routerBias = l.floats(p+"mlp.gate.bias", c.NRoutedExperts)
		ly.gateUp, ly.down = l.experts(p+"mlp.experts.", c)

		if ly.loop {
			r := c.Loop.GateRank
			for j, name := range []string{"attn", "k", "v", "mlp"} {
				g := p + "controllers." + name + "."
				ly.gates[j] = messageGate{
					down:   l.linear(g+"down_weight", r, 2*h),
					forget: l.linear(g+"forget_weight", h, r),
					gain:   l.linear(g+"gain_weight", h, r),
				}
			}
		}
		m.layers = append(m.layers, ly)
	}

	// The message queries are stored as two [2·loop layers, hidden]
	// tensors: each loop layer's query and MLP queries, then its key and
	// value queries. messageQueries holds query, key, value, MLP per layer.
	nl := c.loopLayers()
	qm := l.floats(base+"middle_cycle_cross_pass_queries", 2*nl, h)
	kv := l.floats(base+"middle_cycle_cross_pass_kv_queries", 2*nl, h)
	if l.err == nil {
		for j := range nl {
			row := func(t []float32, r int) []float32 { return t[r*h : (r+1)*h] }
			m.messageQueries = append(m.messageQueries, slices.Concat(row(qm, 2*j), row(kv, 2*j), row(kv, 2*j+1), row(qm, 2*j+1)))
		}
	}

	// The unembedding is stored padded to vocab_size; only the first
	// unpadded_vocab_size rows are real.
	if un := l.bf16Rows(base+"unembed.weight", c.UnpaddedVocabSize, h); un != nil {
		w := tensor.New(c.UnpaddedVocabSize, h)
		for r := range c.UnpaddedVocabSize {
			bf16Row(w.Row(r), un, r)
		}
		m.unembed = l.pack(w, true)
	}
	if l.err != nil {
		return nil, l.err
	}
	return m, nil
}
