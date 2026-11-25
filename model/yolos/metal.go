package yolos

import (
	"errors"
	"fmt"
	"math"
	"sync"

	"github.com/mhpenta/infer/accel"
	"github.com/mhpenta/infer/metal"
	"github.com/mhpenta/infer/model/internal/gpu"
	"github.com/mhpenta/infer/safetensors"
	"github.com/mhpenta/infer/tensor"
	"github.com/mhpenta/infer/vision/postprocess"
)

// The Metal path runs the entire YOLOS detector (patch projection, encoder
// layers, final LayerNorm, and both 3-layer ReLU MLP heads) in a single
// command buffer. Interpolated positional embeddings are computed on CPU once
// per patch grid (gridH, gridW) and cached as GPU buffers.

type gpuModel struct {
	gpu     *accel.GPU
	buffers []*metal.Buffer

	// CPU copies of small embedding tables for host-side assembly / interpolation.
	clsToken   []float32 // [hidden]
	detTokens  []float32 // [numDet * hidden]
	basePos    []float32 // [baseSeq * hidden]
	baseMidPos []float32 // [(numLayers-1) * baseSeq * hidden] when UseMidPositionEmbeddings

	patchProj metal.Weights
	layers    []gpuLayer
	finalLN   gpuNorm
	classHead [3]metal.Weights
	boxHead   [3]metal.Weights

	mu        sync.Mutex
	posCache  map[[2]int]gpuPosEntry
	workspace []*gpuWork
}

type gpuPosEntry struct {
	pos *metal.Buffer
	mid *metal.Buffer // [(numLayers-1), seq, hidden] contiguous, nil when unused
}

type gpuNorm struct{ w, b metal.Ref }

type gpuLayer struct {
	lnBefore             gpuNorm
	qkv, attnOut         metal.Weights
	lnAfter              gpuNorm
	intermediate, output metal.Weights
}

func openGPU(c Config, weights *safetensors.File) (*gpuModel, error) {
	headDim := c.HiddenSize / c.NumAttentionHeads
	patchDim := c.NumChannels * c.PatchSize * c.PatchSize
	if c.HiddenSize%4 != 0 || c.IntermediateSize%4 != 0 || patchDim%4 != 0 || headDim%32 != 0 || headDim > 256 {
		return nil, fmt.Errorf("the Metal kernels need hidden, intermediate and patch sizes that are multiples of 4 and a head dimension that is a multiple of 32 up to 256 (have %d, %d, %d, %d)",
			c.HiddenSize, c.IntermediateSize, patchDim, headDim)
	}
	g, err := accel.OpenMetal()
	if err != nil {
		return nil, err
	}
	files := func(name string) *safetensors.File {
		if weights.Has(name) {
			return weights
		}
		return nil
	}
	dry := gpu.NewLoader(files, "", nil, false)
	loadGPU(dry, c)
	if dry.Err != nil {
		return nil, dry.Err
	}
	if err := gpu.Fits(g.Device, dry.Bytes); err != nil {
		return nil, err
	}

	// Read small embedding tensors kept on host for per-resolution interpolation
	// and token assembly into the shared GPU input buffer.
	h := c.HiddenSize
	baseH, baseW := c.baseGrid()
	baseSeq := 1 + baseH*baseW + c.NumDetectionTokens
	clsT, err := readChecked(weights, "vit.embeddings.cls_token", 1, 1, h)
	if err != nil {
		return nil, err
	}
	detT, err := readChecked(weights, "vit.embeddings.detection_tokens", 1, c.NumDetectionTokens, h)
	if err != nil {
		return nil, err
	}
	posT, err := readChecked(weights, "vit.embeddings.position_embeddings", 1, baseSeq, h)
	if err != nil {
		return nil, err
	}
	var midT *tensor.Tensor
	if c.UseMidPositionEmbeddings && c.NumHiddenLayers > 1 {
		midT, err = readChecked(weights, midPosTensorName(weights), c.NumHiddenLayers-1, 1, baseSeq, h)
		if err != nil {
			return nil, err
		}
	}

	l := gpu.NewLoader(files, "", g.Device, false)
	m := loadGPU(l, c)
	if l.Err != nil {
		gpu.Release(l.Buffers)
		return nil, l.Err
	}
	m.gpu, m.buffers = g, l.Buffers
	m.clsToken = append([]float32(nil), clsT.Data...)
	m.detTokens = append([]float32(nil), detT.Data...)
	m.basePos = append([]float32(nil), posT.Data...)
	if midT != nil {
		m.baseMidPos = append([]float32(nil), midT.Data...)
	}
	m.posCache = make(map[[2]int]gpuPosEntry)
	return m, nil
}

func readChecked(f *safetensors.File, name string, shape ...int) (*tensor.Tensor, error) {
	t, err := f.Tensor(name)
	if err != nil {
		return nil, err
	}
	if !equalShape(t.Shape, shape) {
		return nil, fmt.Errorf("tensor %s has shape %v, want %v", name, t.Shape, shape)
	}
	return t, nil
}

func loadGPU(l *gpu.Loader, c Config) *gpuModel {
	h, ff := c.HiddenSize, c.IntermediateSize
	norm := func(name string) gpuNorm {
		return gpuNorm{l.Vector(name+".weight", h), l.Vector(name+".bias", h)}
	}
	linear := func(in, out int, names ...string) metal.Weights {
		var weights, biases []string
		var outs []int
		for _, name := range names {
			weights = append(weights, name+".weight")
			biases = append(biases, name+".bias")
			outs = append(outs, out)
		}
		return l.Linear(in, outs, weights, biases)
	}
	m := &gpuModel{
		patchProj: l.Conv2dAsLinear(
			"vit.embeddings.patch_embeddings.projection.weight",
			"vit.embeddings.patch_embeddings.projection.bias",
			h, c.NumChannels, c.PatchSize, c.PatchSize,
		),
		finalLN: norm("vit.layernorm"),
	}
	for i := range c.NumHiddenLayers {
		p := fmt.Sprintf("vit.encoder.layer.%d.", i)
		m.layers = append(m.layers, gpuLayer{
			lnBefore:     norm(p + "layernorm_before"),
			qkv:          linear(h, h, p+"attention.attention.query", p+"attention.attention.key", p+"attention.attention.value"),
			attnOut:      linear(h, h, p+"attention.output.dense"),
			lnAfter:      norm(p + "layernorm_after"),
			intermediate: linear(h, ff, p+"intermediate.dense"),
			output:       linear(ff, h, p+"output.dense"),
		})
	}
	m.classHead = [3]metal.Weights{
		linear(h, h, "class_labels_classifier.layers.0"),
		linear(h, h, "class_labels_classifier.layers.1"),
		linear(h, c.NumLabels()+1, "class_labels_classifier.layers.2"),
	}
	m.boxHead = [3]metal.Weights{
		linear(h, h, "bbox_predictor.layers.0"),
		linear(h, h, "bbox_predictor.layers.1"),
		linear(h, 4, "bbox_predictor.layers.2"),
	}
	return m
}

func (g *gpuModel) release() {
	gpu.Release(g.buffers)
	g.buffers = nil
	g.mu.Lock()
	defer g.mu.Unlock()
	for _, e := range g.posCache {
		if e.pos != nil {
			e.pos.Release()
		}
		if e.mid != nil {
			e.mid.Release()
		}
	}
	g.posCache = nil
	for _, w := range g.workspace {
		w.release()
	}
	g.workspace = nil
}

// posEmbedBuf returns cached GPU buffers holding the interpolated initial and
// (when UseMidPositionEmbeddings is true) mid-layer positional embeddings.
func (g *gpuModel) posEmbedBuf(c Config, gridH, gridW int) (gpuPosEntry, error) {
	key := [2]int{gridH, gridW}
	g.mu.Lock()
	if e, ok := g.posCache[key]; ok {
		g.mu.Unlock()
		return e, nil
	}
	g.mu.Unlock()

	baseH, baseW := c.baseGrid()
	baseSeq := 1 + baseH*baseW + c.NumDetectionTokens
	newSeq := 1 + gridH*gridW + c.NumDetectionTokens
	h := c.HiddenSize

	data := interpolatePosEmbed(g.basePos, baseH, baseW, gridH, gridW, c.NumDetectionTokens, h)
	posBuf, err := g.gpu.Device.NewBuffer(4 * len(data))
	if err != nil {
		return gpuPosEntry{}, err
	}
	copy(metal.Slice[float32](posBuf), data)

	var midBuf *metal.Buffer
	if len(g.baseMidPos) > 0 {
		depth := c.NumHiddenLayers - 1
		midBuf, err = g.gpu.Device.NewBuffer(4 * depth * newSeq * h)
		if err != nil {
			posBuf.Release()
			return gpuPosEntry{}, err
		}
		dst := metal.Slice[float32](midBuf)
		srcStride := baseSeq * h
		dstStride := newSeq * h
		for i := range depth {
			md := interpolatePosEmbed(g.baseMidPos[i*srcStride:(i+1)*srcStride], baseH, baseW, gridH, gridW, c.NumDetectionTokens, h)
			copy(dst[i*dstStride:(i+1)*dstStride], md)
		}
	}

	entry := gpuPosEntry{pos: posBuf, mid: midBuf}
	g.mu.Lock()
	if existing, ok := g.posCache[key]; ok {
		g.mu.Unlock()
		posBuf.Release()
		if midBuf != nil {
			midBuf.Release()
		}
		return existing, nil
	}
	g.posCache[key] = entry
	g.mu.Unlock()
	return entry, nil
}

type gpuWork struct {
	rows, slots int
	patches     *metal.Buffer
	states      *metal.Buffer // slots * rows * hidden
	norm, qkv   *metal.Buffer
	ctx, attn   *metal.Buffer
	ffn         *metal.Buffer
	finalNorm   *metal.Buffer
	mlp0, mlp1  *metal.Buffer
	logits      *metal.Buffer
	boxes       *metal.Buffer
	cache       *metal.KVCache
}

func (w *gpuWork) release() {
	for _, b := range []*metal.Buffer{
		w.patches, w.states, w.norm, w.qkv, w.ctx, w.attn, w.ffn,
		w.finalNorm, w.mlp0, w.mlp1, w.logits, w.boxes,
	} {
		if b != nil {
			b.Release()
		}
	}
	if w.cache != nil {
		w.cache.Release()
	}
}

func (g *gpuModel) work(c Config, rows int, keepAll bool) (*gpuWork, error) {
	needSlots := 2
	if keepAll {
		needSlots = len(g.layers) + 1
	}
	g.mu.Lock()
	for i, w := range g.workspace {
		if w.rows >= rows && w.slots >= needSlots {
			g.workspace = append(g.workspace[:i], g.workspace[i+1:]...)
			g.mu.Unlock()
			return w, nil
		}
	}
	outgrown := g.workspace
	g.workspace = nil
	g.mu.Unlock()
	for _, w := range outgrown {
		w.release()
	}

	dev, h, ff, det := g.gpu.Device, c.HiddenSize, c.IntermediateSize, c.NumDetectionTokens
	patchDim := c.NumChannels * c.PatchSize * c.PatchSize
	maxPatches := max(rows-1-det, 1)

	w := &gpuWork{rows: rows, slots: needSlots}
	var errs []error
	alloc := func(floats int) *metal.Buffer {
		b, err := dev.NewBuffer(4 * floats)
		errs = append(errs, err)
		return b
	}
	w.patches = alloc(maxPatches * patchDim)
	w.states = alloc(needSlots * rows * h)
	w.norm = alloc(rows * h)
	w.qkv = alloc(rows * 3 * h)
	w.ctx = alloc(rows * h)
	w.attn = alloc(rows * h)
	w.ffn = alloc(rows * ff)
	w.finalNorm = alloc(rows * h)
	w.mlp0 = alloc(det * h)
	w.mlp1 = alloc(det * h)
	w.logits = alloc(det * (c.NumLabels() + 1))
	w.boxes = alloc(det * 4)
	var err error
	w.cache, err = dev.NewKVCache(c.NumAttentionHeads, h/c.NumAttentionHeads, rows)
	if err = errors.Join(append(errs, err)...); err != nil {
		w.release()
		return nil, err
	}
	return w, nil
}

func (g *gpuModel) done(w *gpuWork) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.workspace = append(g.workspace, w)
}

func (g *gpuModel) run(c Config, patches *tensor.Tensor, gridH, gridW int, keepAll bool) (*Output, error) {
	k := g.gpu.Kernels
	h, ff, det := c.HiddenSize, c.IntermediateSize, c.NumDetectionTokens
	numPatches := gridH * gridW
	seq := 1 + numPatches + det
	patchDim := c.NumChannels * c.PatchSize * c.PatchSize

	posEntry, err := g.posEmbedBuf(c, gridH, gridW)
	if err != nil {
		return nil, err
	}
	w, err := g.work(c, seq, keepAll)
	if err != nil {
		return nil, err
	}
	defer g.done(w)

	// Upload patches and pre-populate CLS (row 0) and detection tokens (tail)
	// in slot 0 before encoding the single command buffer.
	copy(metal.Slice[float32](w.patches), patches.Data)
	slot0 := metal.Slice[float32](w.states)
	copy(slot0[:h], g.clsToken)
	copy(slot0[(1+numPatches)*h:seq*h], g.detTokens)

	slot := func(i int) metal.Ref {
		idx := i
		if !keepAll {
			idx = i & 1
		}
		return w.states.At(4 * idx * seq * h)
	}

	eps := float32(c.LayerNormEps)
	attend := metal.AttendArgs{
		Heads: c.NumAttentionHeads,
		Scale: float32(1 / math.Sqrt(float64(h/c.NumAttentionHeads))),
		Keys:  seq,
	}
	norm, qkv, ctx, attn, ffn := w.norm.At(0), w.qkv.At(0), w.ctx.At(0), w.attn.At(0), w.ffn.At(0)
	finalNorm := w.finalNorm.At(0)
	mlp0, mlp1 := w.mlp0.At(0), w.mlp1.At(0)

	var keepPatchBuf *metal.Buffer
	if keepAll {
		keepPatchBuf, err = g.gpu.Device.NewBuffer(4 * numPatches * h)
		if err != nil {
			return nil, err
		}
		defer keepPatchBuf.Release()
	}

	cb := g.gpu.Device.NewCommandBuffer()

	// Patch projection writes directly into rows 1..numPatches of slot 0.
	x0 := slot(0)
	patchRef := metal.Ref{Buf: x0.Buf, Off: x0.Off + 4*h}
	k.Linear(cb, patchRef, h, w.patches.At(0), patchDim, numPatches, g.patchProj)
	if keepPatchBuf != nil {
		k.Linear(cb, keepPatchBuf.At(0), h, w.patches.At(0), patchDim, numPatches, g.patchProj)
	}

	k.Add(cb, x0, posEntry.pos.At(0), seq*h)

	for i, l := range g.layers {
		x, next := slot(i), slot(i+1)
		if keepAll && posEntry.mid != nil && i > 0 {
			// In keepAll mode, slot(i) preserves the raw YolosLayer output
			// matching transformers' output_hidden_states, while finalNorm
			// carries slot(i) + mid_position_embeddings[i-1] into layer i.
			x = finalNorm
		}
		// Pre-LN self-attention + residual into attn.
		k.LayerNormInto(cb, norm, x, l.lnBefore.w, l.lnBefore.b, seq, h, eps)
		k.Linear(cb, qkv, 3*h, norm, h, seq, l.qkv)
		k.KVAppend(cb, w.cache, w.qkv.At(4*h), w.qkv.At(8*h), 3*h, seq, 0)
		k.Attend(cb, ctx, h, qkv, 3*h, w.cache, seq, attend)
		k.Linear(cb, attn, h, ctx, h, seq, l.attnOut)
		k.Add(cb, attn, x, seq*h)

		// Pre-LN GELU FFN + residual into next.
		k.LayerNormInto(cb, norm, attn, l.lnAfter.w, l.lnAfter.b, seq, h, eps)
		k.Linear(cb, ffn, ff, norm, h, seq, l.intermediate)
		k.GELU(cb, ffn, seq*ff)
		k.Linear(cb, next, h, ffn, ff, seq, l.output)
		k.Add(cb, next, attn, seq*h)

		if posEntry.mid != nil && i < len(g.layers)-1 {
			if keepAll {
				k.Scale(cb, finalNorm, next, seq*h, 1)
				k.Add(cb, finalNorm, posEntry.mid.At(4*i*seq*h), seq*h)
			} else {
				k.Add(cb, next, posEntry.mid.At(4*i*seq*h), seq*h)
			}
		}
	}

	lastX := slot(len(g.layers))
	k.LayerNormInto(cb, finalNorm, lastX, g.finalLN.w, g.finalLN.b, seq, h, eps)

	// Detection token features: last det rows of finalNorm.
	// Byte offset 4*(seq-det)*h is a multiple of 16 because h is a multiple of 4.
	detFeat := w.finalNorm.At(4 * (seq - det) * h)

	// Class head (3-layer ReLU MLP).
	k.Linear(cb, mlp0, h, detFeat, h, det, g.classHead[0])
	k.ReLU(cb, mlp0, det*h)
	k.Linear(cb, mlp1, h, mlp0, h, det, g.classHead[1])
	k.ReLU(cb, mlp1, det*h)
	k.Linear(cb, w.logits.At(0), c.NumLabels()+1, mlp1, h, det, g.classHead[2])

	// Box head (3-layer ReLU MLP).
	k.Linear(cb, mlp0, h, detFeat, h, det, g.boxHead[0])
	k.ReLU(cb, mlp0, det*h)
	k.Linear(cb, mlp1, h, mlp0, h, det, g.boxHead[1])
	k.ReLU(cb, mlp1, det*h)
	k.Linear(cb, w.boxes.At(0), 4, mlp1, h, det, g.boxHead[2])

	if err := cb.Wait(); err != nil {
		return nil, err
	}

	out := &Output{
		Logits:    tensor.New(det, c.NumLabels()+1),
		PredBoxes: tensor.New(det, 4),
	}
	copy(out.Logits.Data, metal.Slice[float32](w.logits)[:det*(c.NumLabels()+1)])
	copy(out.PredBoxes.Data, metal.Slice[float32](w.boxes)[:det*4])
	postprocess.Sigmoid(out.PredBoxes.Data)

	if keepAll {
		out.PatchEmbed = tensor.New(numPatches, h)
		copy(out.PatchEmbed.Data, metal.Slice[float32](keepPatchBuf)[:numPatches*h])
		out.PosEmbed = tensor.New(seq, h)
		copy(out.PosEmbed.Data, metal.Slice[float32](posEntry.pos)[:seq*h])
		statesSlice := metal.Slice[float32](w.states)
		for i := 0; i <= len(g.layers); i++ {
			t := tensor.New(seq, h)
			copy(t.Data, statesSlice[i*seq*h:(i+1)*seq*h])
			out.HiddenStates = append(out.HiddenStates, t)
		}
		out.LastHiddenState = tensor.New(seq, h)
		copy(out.LastHiddenState.Data, metal.Slice[float32](w.finalNorm)[:seq*h])
	}
	return out, nil
}
