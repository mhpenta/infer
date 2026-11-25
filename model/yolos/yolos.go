// Package yolos implements the YOLOS (You Only Look At One Sequence) Vision
// Transformer object detector for inference, matching Hugging Face's
// YolosForObjectDetection layer by layer on CPU and Apple silicon Metal.
//
// Architecture: 16x16 non-overlapping patch projection (Conv2d reshaped as a
// linear layer), prepended CLS token and appended learnable detection tokens,
// 2D bicubic-interpolated positional embeddings, Pre-LN bidirectional ViT
// encoder layers with exact-erf GELU, final LayerNorm, and two 3-layer ReLU
// MLP heads predicting class logits and sigmoid-normalized (cx, cy, w, h)
// bounding boxes.
package yolos

import (
	"encoding/json"
	"fmt"
	"image"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"sync"

	"github.com/mhpenta/infer/accel"
	"github.com/mhpenta/infer/ops"
	"github.com/mhpenta/infer/safetensors"
	"github.com/mhpenta/infer/tensor"
	"github.com/mhpenta/infer/vision/postprocess"
	"github.com/mhpenta/infer/vision/preprocess"
)

// Config is the subset of a Hugging Face YOLOS config.json the detector needs.
type Config struct {
	HiddenSize               int               `json:"hidden_size"`
	NumHiddenLayers          int               `json:"num_hidden_layers"`
	NumAttentionHeads        int               `json:"num_attention_heads"`
	IntermediateSize         int               `json:"intermediate_size"`
	HiddenAct                string            `json:"hidden_act"`
	LayerNormEps             float64           `json:"layer_norm_eps"`
	ImageSize                [2]int            `json:"image_size"`
	PatchSize                int               `json:"patch_size"`
	NumChannels              int               `json:"num_channels"`
	NumDetectionTokens       int               `json:"num_detection_tokens"`
	QKVBias                  bool              `json:"qkv_bias"`
	UseMidPositionEmbeddings bool              `json:"use_mid_position_embeddings"`
	ID2Label                 map[string]string `json:"id2label"`
}

// NumLabels returns the number of object classes (excluding the trailing
// "no-object" background logit).
func (c Config) NumLabels() int {
	if len(c.ID2Label) > 0 {
		return len(c.ID2Label)
	}
	return 91
}

// Labels returns ID2Label keyed by integer class ID.
func (c Config) Labels() map[int]string {
	out := make(map[int]string, len(c.ID2Label))
	for k, v := range c.ID2Label {
		if id, err := strconv.Atoi(k); err == nil {
			out[id] = v
		}
	}
	return out
}

func (c Config) baseGrid() (h, w int) {
	return c.ImageSize[0] / c.PatchSize, c.ImageSize[1] / c.PatchSize
}

func (c Config) validate() error {
	switch {
	case c.HiddenSize <= 0 || c.NumHiddenLayers <= 0 || c.NumAttentionHeads <= 0 || c.IntermediateSize <= 0:
		return fmt.Errorf("hidden_size, num_hidden_layers, num_attention_heads and intermediate_size must be positive")
	case c.HiddenSize%c.NumAttentionHeads != 0:
		return fmt.Errorf("hidden_size %d is not divisible by %d heads", c.HiddenSize, c.NumAttentionHeads)
	case c.HiddenAct != "gelu":
		return fmt.Errorf("hidden_act %q is not supported (want gelu)", c.HiddenAct)
	case c.NumChannels != 3:
		return fmt.Errorf("num_channels %d is not supported (want 3)", c.NumChannels)
	case c.PatchSize <= 0:
		return fmt.Errorf("patch_size must be positive, got %d", c.PatchSize)
	case c.ImageSize[0] < c.PatchSize || c.ImageSize[1] < c.PatchSize:
		return fmt.Errorf("image_size %v smaller than patch_size %d", c.ImageSize, c.PatchSize)
	case c.NumDetectionTokens <= 0:
		return fmt.Errorf("num_detection_tokens must be positive, got %d", c.NumDetectionTokens)
	case !c.QKVBias:
		return fmt.Errorf("qkv_bias=false is not supported")
	}
	return nil
}

// Options control how LoadWith prepares a model.
type Options struct {
	// Device is where the model runs: accel.CPU, accel.Metal, or accel.Auto.
	// Empty means $INFER_DEVICE, or Auto when that is unset too.
	Device string
}

// Output holds the raw outputs of a YOLOS forward pass.
type Output struct {
	// Logits is [num_detection_tokens, num_labels+1]; the last column is the
	// "no-object" background class.
	Logits *tensor.Tensor
	// PredBoxes is [num_detection_tokens, 4] in sigmoid-normalized
	// (center_x, center_y, width, height) coordinates.
	PredBoxes *tensor.Tensor
	// HiddenStates is populated only by HiddenStatesPixels: [0] is the
	// embedding output (patches + tokens + pos_embed), [1..NumHiddenLayers]
	// are the outputs of each encoder layer before vit.layernorm.
	HiddenStates []*tensor.Tensor
	// LastHiddenState is populated only by HiddenStatesPixels: the output of
	// vit.layernorm over the final encoder layer.
	LastHiddenState *tensor.Tensor
	// PatchEmbed and PosEmbed are populated only by HiddenStatesPixels for
	// layer-by-layer parity checks.
	PatchEmbed *tensor.Tensor
	PosEmbed   *tensor.Tensor
}

// Model is a loaded YOLOS object detector. It is safe for concurrent use.
type Model struct {
	Config     Config
	Preprocess preprocess.Config

	labels map[int]string

	clsToken  *tensor.Tensor // [1, hidden]
	detTokens *tensor.Tensor // [numDet, hidden]
	posEmb    *tensor.Tensor // [1 + baseH*baseW + numDet, hidden]
	midPosEmb *tensor.Tensor // [numLayers-1, baseSeq, hidden] when UseMidPositionEmbeddings
	patchProj linear         // [hidden, 3*patch*patch]
	layers    []layer
	finalLN   layerNorm
	classHead [3]linear
	boxHead   [3]linear

	weights *safetensors.File

	// posCache caches CPU bicubic-interpolated position embeddings by [gridH, gridW].
	posMu    sync.Mutex
	posCache map[[2]int]posEntry

	workspaces sync.Pool

	gpu    *gpuModel
	device string
}

type posEntry struct {
	pos *tensor.Tensor
	mid []*tensor.Tensor // len == numLayers-1 when UseMidPositionEmbeddings
}

type linear struct {
	weight *ops.Packed
	bias   *tensor.Tensor
}

type layerNorm struct{ weight, bias *tensor.Tensor }

type layer struct {
	lnBefore             layerNorm
	qkv                  linear
	attnOut              linear
	lnAfter              layerNorm
	intermediate, output linear
}

// Load reads config.json, preprocessor_config.json (if present) and
// model.safetensors from dir.
func Load(dir string) (*Model, error) {
	return LoadWith(dir, Options{})
}

// LoadWith is Load with options.
func LoadWith(dir string, opts Options) (*Model, error) {
	device, err := accel.Resolve(opts.Device)
	if err != nil {
		return nil, fmt.Errorf("yolos: %w", err)
	}
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("yolos: %w", err)
	}
	var config Config
	if err := json.Unmarshal(raw, &config); err != nil {
		return nil, fmt.Errorf("yolos: parse config.json: %w", err)
	}
	if err := config.validate(); err != nil {
		return nil, fmt.Errorf("yolos: %w", err)
	}
	preCfg := preprocess.DefaultYOLOSConfig()
	if _, statErr := os.Stat(filepath.Join(dir, "preprocessor_config.json")); statErr == nil {
		if preCfg, err = preprocess.LoadConfig(dir); err != nil {
			return nil, fmt.Errorf("yolos: %w", err)
		}
		if preCfg.ModSize == 0 {
			preCfg.ModSize = config.PatchSize
		}
	}

	weights, err := safetensors.Open(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		return nil, fmt.Errorf("yolos: %w", err)
	}

	var m *Model
	if device != accel.CPU {
		var g *gpuModel
		if g, err = openGPU(config, weights); err == nil {
			m = &Model{
				Config:     config,
				Preprocess: preCfg,
				labels:     config.Labels(),
				weights:    weights,
				gpu:        g,
				device:     fmt.Sprintf("%s (%s)", accel.Metal, g.gpu.Device.Name()),
			}
		} else if device == accel.Metal {
			err = fmt.Errorf("cannot run on Metal: %w", err)
		} else {
			err = nil // Auto: fall back to CPU
		}
	}
	if m == nil && err == nil {
		m, err = newModel(config, preCfg, weights)
	}
	if err != nil {
		weights.Close()
		return nil, fmt.Errorf("yolos: %w", err)
	}
	return m, nil
}

func newModel(config Config, preCfg preprocess.Config, weights *safetensors.File) (*Model, error) {
	l := loader{weights: weights}
	h, ff := config.HiddenSize, config.IntermediateSize
	baseH, baseW := config.baseGrid()
	baseSeq := 1 + baseH*baseW + config.NumDetectionTokens
	patchDim := config.NumChannels * config.PatchSize * config.PatchSize

	clsRaw := l.tensor("vit.embeddings.cls_token", 1, 1, h)
	detRaw := l.tensor("vit.embeddings.detection_tokens", 1, config.NumDetectionTokens, h)
	posRaw := l.tensor("vit.embeddings.position_embeddings", 1, baseSeq, h)
	var midRaw *tensor.Tensor
	if config.UseMidPositionEmbeddings && config.NumHiddenLayers > 1 {
		midName := midPosTensorName(weights)
		midRaw = l.tensor(midName, config.NumHiddenLayers-1, 1, baseSeq, h)
	}
	projW := l.tensor("vit.embeddings.patch_embeddings.projection.weight", h, config.NumChannels, config.PatchSize, config.PatchSize)
	projB := l.tensor("vit.embeddings.patch_embeddings.projection.bias", h)

	var patchProj linear
	if l.err == nil {
		patchProj = linear{
			weight: ops.Pack(tensor.FromData(projW.Data, h, patchDim)),
			bias:   projB,
		}
	}

	m := &Model{
		Config:     config,
		Preprocess: preCfg,
		labels:     config.Labels(),
		patchProj:  patchProj,
		finalLN:    l.layerNorm("vit.layernorm", h),
		weights:    weights,
		posCache:   make(map[[2]int]posEntry),
		device:     accel.CPU,
	}
	if l.err == nil {
		m.clsToken = tensor.FromData(clsRaw.Data, 1, h)
		m.detTokens = tensor.FromData(detRaw.Data, config.NumDetectionTokens, h)
		m.posEmb = tensor.FromData(posRaw.Data, baseSeq, h)
		if midRaw != nil {
			m.midPosEmb = tensor.FromData(midRaw.Data, config.NumHiddenLayers-1, baseSeq, h)
		}
	}
	for i := range config.NumHiddenLayers {
		p := fmt.Sprintf("vit.encoder.layer.%d.", i)
		m.layers = append(m.layers, layer{
			lnBefore:     l.layerNorm(p+"layernorm_before", h),
			qkv:          l.stacked(h, h, p+"attention.attention.query", p+"attention.attention.key", p+"attention.attention.value"),
			attnOut:      l.linear(p+"attention.output.dense", h, h),
			lnAfter:      l.layerNorm(p+"layernorm_after", h),
			intermediate: l.linear(p+"intermediate.dense", ff, h),
			output:       l.linear(p+"output.dense", h, ff),
		})
	}
	m.classHead = [3]linear{
		l.linear("class_labels_classifier.layers.0", h, h),
		l.linear("class_labels_classifier.layers.1", h, h),
		l.linear("class_labels_classifier.layers.2", config.NumLabels()+1, h),
	}
	m.boxHead = [3]linear{
		l.linear("bbox_predictor.layers.0", h, h),
		l.linear("bbox_predictor.layers.1", h, h),
		l.linear("bbox_predictor.layers.2", 4, h),
	}
	if l.err != nil {
		return nil, l.err
	}
	return m, nil
}

// Device returns where the model runs: "cpu", or "metal (" and the GPU's name ")".
func (m *Model) Device() string { return m.device }

// Close releases the memory-mapped weights and any GPU memory.
func (m *Model) Close() error {
	if m.gpu != nil {
		m.gpu.release()
	}
	return m.weights.Close()
}

// Detect preprocesses img using m.Preprocess, runs inference, and decodes
// detections with score > scoreThreshold (normalized [0, 1] corner boxes,
// clipped to [0, 1]).
func (m *Model) Detect(img image.Image, scoreThreshold float32) ([]postprocess.Detection, error) {
	return m.DetectWith(img, m.Preprocess, postprocess.Options{
		ScoreThreshold: scoreThreshold,
		Labels:         m.labels,
		ClipBoxes:      true,
	})
}

// DetectWith runs preprocessing with preCfg, model inference, and
// postprocessing with postOpts. If postOpts.Labels is nil, m's config labels
// are used.
func (m *Model) DetectWith(img image.Image, preCfg preprocess.Config, postOpts postprocess.Options) ([]postprocess.Detection, error) {
	pre, err := preprocess.Process(img, preCfg)
	if err != nil {
		return nil, fmt.Errorf("yolos: %w", err)
	}
	out, err := m.ForwardPixels(pre.Pixels)
	if err != nil {
		return nil, err
	}
	if postOpts.Labels == nil {
		postOpts.Labels = m.labels
	}
	return postprocess.DecodeDETR(out.Logits, out.PredBoxes, postOpts)
}

// ForwardPixels runs YOLOS on a preprocessed [3, H, W] CHW float32 tensor and
// returns class logits and predicted boxes.
func (m *Model) ForwardPixels(pixels *tensor.Tensor) (*Output, error) {
	return m.run(pixels, false)
}

// HiddenStatesPixels is ForwardPixels while also capturing PatchEmbed,
// PosEmbed, every encoder layer's HiddenStates, and LastHiddenState.
func (m *Model) HiddenStatesPixels(pixels *tensor.Tensor) (*Output, error) {
	return m.run(pixels, true)
}

func (m *Model) run(pixels *tensor.Tensor, keepAll bool) (*Output, error) {
	if pixels == nil {
		return nil, fmt.Errorf("yolos: nil pixel tensor")
	}
	patches, gridH, gridW, err := preprocess.ExtractPatches(pixels, m.Config.PatchSize)
	if err != nil {
		return nil, fmt.Errorf("yolos: %w", err)
	}
	if m.gpu != nil {
		return m.gpu.run(m.Config, patches, gridH, gridW, keepAll)
	}
	return m.runCPU(patches, gridH, gridW, keepAll), nil
}

// interpolatePosEmbed returns the [1 + gridH*gridW + numDet, hidden]
// position embeddings for a (gridH, gridW) patch grid, matching Hugging Face's
// InterpolateInitialPositionEmbeddings.
func interpolatePosEmbed(basePos []float32, baseH, baseW, gridH, gridW, numDet, h int) []float32 {
	basePatches := baseH * baseW
	newPatches := gridH * gridW
	newSeq := 1 + newPatches + numDet

	out := make([]float32, newSeq*h)
	// CLS token position embedding (row 0).
	copy(out[:h], basePos[:h])
	// Interpolated spatial patch position embeddings (rows 1..newPatches).
	patchSrc := basePos[h : (1+basePatches)*h]
	patchDst := preprocess.BicubicInterpolate2D(patchSrc, baseH, baseW, gridH, gridW, h)
	copy(out[h:(1+newPatches)*h], patchDst)
	// Detection token position embeddings (last numDet rows).
	copy(out[(1+newPatches)*h:], basePos[(1+basePatches)*h:])
	return out
}

func midPosTensorName(weights *safetensors.File) string {
	if weights.Has("vit.encoder.mid_position_embeddings") {
		return "vit.encoder.mid_position_embeddings"
	}
	return "vit.encoder.interpolation.mid_position_embeddings"
}

func (m *Model) posEmbedFor(gridH, gridW int) posEntry {
	key := [2]int{gridH, gridW}
	m.posMu.Lock()
	if e, ok := m.posCache[key]; ok {
		m.posMu.Unlock()
		return e
	}
	m.posMu.Unlock()

	baseH, baseW := m.Config.baseGrid()
	baseSeq := 1 + baseH*baseW + m.Config.NumDetectionTokens
	newSeq := 1 + gridH*gridW + m.Config.NumDetectionTokens
	h := m.Config.HiddenSize

	data := interpolatePosEmbed(m.posEmb.Data, baseH, baseW, gridH, gridW, m.Config.NumDetectionTokens, h)
	e := posEntry{pos: tensor.FromData(data, newSeq, h)}
	if m.midPosEmb != nil {
		depth := m.Config.NumHiddenLayers - 1
		e.mid = make([]*tensor.Tensor, depth)
		stride := baseSeq * h
		for i := range depth {
			md := interpolatePosEmbed(m.midPosEmb.Data[i*stride:(i+1)*stride], baseH, baseW, gridH, gridW, m.Config.NumDetectionTokens, h)
			e.mid[i] = tensor.FromData(md, newSeq, h)
		}
	}

	m.posMu.Lock()
	m.posCache[key] = e
	m.posMu.Unlock()
	return e
}

type workspace struct {
	buf                                      []float32
	x, norm, qkv, context, attn, hidden, out *tensor.Tensor
	mlp0, mlp1                               *tensor.Tensor
}

func (m *Model) workspace(seq int) *workspace {
	h, ff, det := m.Config.HiddenSize, m.Config.IntermediateSize, m.Config.NumDetectionTokens
	ws, _ := m.workspaces.Get().(*workspace)
	if ws == nil {
		ws = &workspace{}
	}
	need := seq*(8*h+ff) + 2*det*h
	if cap(ws.buf) < need {
		ws.buf = make([]float32, need)
	}
	rest := ws.buf
	take := func(rows, cols int) *tensor.Tensor {
		n := rows * cols
		t := tensor.FromData(rest[:n:n], rows, cols)
		rest = rest[n:]
		return t
	}
	ws.x = take(seq, h)
	ws.norm = take(seq, h)
	ws.qkv = take(seq, 3*h)
	ws.context = take(seq, h)
	ws.attn = take(seq, h)
	ws.out = take(seq, h)
	ws.hidden = take(seq, ff)
	ws.mlp0 = take(det, h)
	ws.mlp1 = take(det, h)
	return ws
}

func (m *Model) runCPU(patches *tensor.Tensor, gridH, gridW int, keepAll bool) *Output {
	h := m.Config.HiddenSize
	numPatches := gridH * gridW
	detN := m.Config.NumDetectionTokens
	seq := 1 + numPatches + detN

	ws := m.workspace(seq)
	defer m.workspaces.Put(ws)

	// Assemble [CLS, patch_proj(patches), det_tokens] into ws.x.
	copy(ws.x.Row(0), m.clsToken.Data)
	patchOut := tensor.FromData(ws.x.Data[h:(1+numPatches)*h], numPatches, h)
	ops.LinearInto(patchOut, patches, m.patchProj.weight, m.patchProj.bias)
	copy(ws.x.Data[(1+numPatches)*h:], m.detTokens.Data)

	var out Output
	if keepAll {
		out.PatchEmbed = clone(patchOut)
	}

	pe := m.posEmbedFor(gridH, gridW)
	if keepAll {
		out.PosEmbed = clone(pe.pos)
	}
	ops.Add(ws.x.Data, pe.pos.Data)

	if keepAll {
		out.HiddenStates = append(out.HiddenStates, clone(ws.x))
	}

	headDim := h / m.Config.NumAttentionHeads
	scale := float32(1 / math.Sqrt(float64(headDim)))
	eps := m.Config.LayerNormEps

	for i, l := range m.layers {
		// Pre-LN self-attention + residual.
		ops.LayerNormInto(ws.norm, ws.x, l.lnBefore.weight, l.lnBefore.bias, eps)
		ops.LinearInto(ws.qkv, ws.norm, l.qkv.weight, l.qkv.bias)
		ops.AttentionQKVInto(ws.context, ws.qkv, m.Config.NumAttentionHeads, scale)
		ops.LinearInto(ws.attn, ws.context, l.attnOut.weight, l.attnOut.bias)
		ops.Add(ws.x.Data, ws.attn.Data)

		// Pre-LN GELU feed-forward + residual.
		ops.LayerNormInto(ws.norm, ws.x, l.lnAfter.weight, l.lnAfter.bias, eps)
		ops.LinearInto(ws.hidden, ws.norm, l.intermediate.weight, l.intermediate.bias)
		ops.GELU(ws.hidden.Data)
		ops.LinearInto(ws.out, ws.hidden, l.output.weight, l.output.bias)
		ops.Add(ws.x.Data, ws.out.Data)

		if keepAll {
			out.HiddenStates = append(out.HiddenStates, clone(ws.x))
		}
		if i < len(pe.mid) {
			ops.Add(ws.x.Data, pe.mid[i].Data)
		}
	}

	// Final LayerNorm (out-of-place into ws.norm so ws.x isn't needed anymore).
	ops.LayerNormInto(ws.norm, ws.x, m.finalLN.weight, m.finalLN.bias, eps)
	if keepAll {
		out.LastHiddenState = clone(ws.norm)
	}

	// Slice the last detN rows for the two 3-layer ReLU MLP prediction heads.
	detFeat := tensor.FromData(ws.norm.Data[(seq-detN)*h:seq*h], detN, h)

	out.Logits = tensor.New(detN, m.Config.NumLabels()+1)
	runMLP(out.Logits, detFeat, m.classHead, ws.mlp0, ws.mlp1)

	out.PredBoxes = tensor.New(detN, 4)
	runMLP(out.PredBoxes, detFeat, m.boxHead, ws.mlp0, ws.mlp1)
	postprocess.Sigmoid(out.PredBoxes.Data)

	return &out
}

func runMLP(dst, src *tensor.Tensor, layers [3]linear, h0, h1 *tensor.Tensor) {
	ops.LinearInto(h0, src, layers[0].weight, layers[0].bias)
	ops.ReLU(h0.Data)
	ops.LinearInto(h1, h0, layers[1].weight, layers[1].bias)
	ops.ReLU(h1.Data)
	ops.LinearInto(dst, h1, layers[2].weight, layers[2].bias)
}

func clone(t *tensor.Tensor) *tensor.Tensor {
	c := tensor.New(t.Shape...)
	copy(c.Data, t.Data)
	return c
}

type loader struct {
	weights *safetensors.File
	err     error
}

func (l *loader) tensor(name string, shape ...int) *tensor.Tensor {
	if l.err != nil {
		return nil
	}
	t, err := l.weights.Tensor(name)
	if err != nil {
		l.err = err
		return nil
	}
	if !equalShape(t.Shape, shape) {
		l.err = fmt.Errorf("tensor %s has shape %v, want %v", name, t.Shape, shape)
		return nil
	}
	return t
}

func (l *loader) linear(name string, out, in int) linear {
	w := l.tensor(name+".weight", out, in)
	b := l.tensor(name+".bias", out)
	if l.err != nil {
		return linear{}
	}
	return linear{weight: ops.Pack(w), bias: b}
}

func (l *loader) stacked(out, in int, names ...string) linear {
	w := make([]float32, 0, len(names)*out*in)
	b := make([]float32, 0, len(names)*out)
	for _, name := range names {
		wt, bt := l.tensor(name+".weight", out, in), l.tensor(name+".bias", out)
		if l.err != nil {
			return linear{}
		}
		w, b = append(w, wt.Data...), append(b, bt.Data...)
	}
	n := len(names) * out
	return linear{weight: ops.Pack(tensor.FromData(w, n, in)), bias: tensor.FromData(b, n)}
}

func (l *loader) layerNorm(name string, size int) layerNorm {
	return layerNorm{weight: l.tensor(name+".weight", size), bias: l.tensor(name+".bias", size)}
}

func equalShape(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
