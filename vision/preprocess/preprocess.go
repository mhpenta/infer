// Package preprocess prepares Go images for vision transformer models:
// RGB extraction, antialiased bilinear resizing matching Hugging Face's
// YolosImageProcessor, rescaling, channel normalization, non-overlapping patch
// extraction, and PyTorch-compatible 2D bicubic interpolation.
package preprocess

import (
	"encoding/json"
	"fmt"
	"image"
	"math"
	"os"
	"path/filepath"

	"github.com/mhpenta/infer/tensor"
)

// DefaultImageNetMean and DefaultImageNetStd are the standard ImageNet RGB
// normalization constants used by ViT and YOLOS checkpoints.
var (
	DefaultImageNetMean = [3]float32{0.485, 0.456, 0.406}
	DefaultImageNetStd  = [3]float32{0.229, 0.224, 0.225}
)

// Size specifies target resize dimensions, matching Hugging Face's
// preprocessor_config.json "size" object.
type Size struct {
	ShortestEdge int `json:"shortest_edge"`
	LongestEdge  int `json:"longest_edge"`
	Height       int `json:"height"`
	Width        int `json:"width"`
}

// Config configures image preprocessing. Use LoadConfig to read a Hugging Face
// preprocessor_config.json or DefaultYOLOSConfig for standard defaults.
type Config struct {
	DoResize      bool       `json:"do_resize"`
	DoRescale     bool       `json:"do_rescale"`
	RescaleFactor float32    `json:"rescale_factor"`
	DoNormalize   bool       `json:"do_normalize"`
	ImageMean     [3]float32 `json:"image_mean"`
	ImageStd      [3]float32 `json:"image_std"`
	Size          Size       `json:"size"`
	// ModSize rounds resized height and width down to multiples of ModSize
	// when > 0 (16 for YOLOS so images divide cleanly into patches).
	ModSize int `json:"mod_size,omitempty"`
}

// DefaultYOLOSConfig returns the standard YolosImageProcessor defaults.
func DefaultYOLOSConfig() Config {
	return Config{
		DoResize:      true,
		DoRescale:     true,
		RescaleFactor: 1.0 / 255.0,
		DoNormalize:   true,
		ImageMean:     DefaultImageNetMean,
		ImageStd:      DefaultImageNetStd,
		Size:          Size{ShortestEdge: 512, LongestEdge: 1333},
		ModSize:       16,
	}
}

// LoadConfig reads preprocessor_config.json from dir, filling Hugging Face's
// defaults for omitted fields (such as do_rescale=true and rescale_factor=1/255).
func LoadConfig(dir string) (Config, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "preprocessor_config.json"))
	if err != nil {
		return Config{}, fmt.Errorf("preprocess: %w", err)
	}
	cfg := DefaultYOLOSConfig()
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return Config{}, fmt.Errorf("preprocess: parse preprocessor_config.json: %w", err)
	}
	return cfg, nil
}

// Result holds a preprocessed image tensor and its geometry metadata.
type Result struct {
	// Pixels is the [3, Height, Width] CHW float32 tensor.
	Pixels *tensor.Tensor
	// OrigWidth and OrigHeight are the source image dimensions before resizing.
	OrigWidth, OrigHeight int
	// Width and Height are the processed dimensions of Pixels.
	Width, Height int
}

// TargetSize computes the resized (height, width) matching Hugging Face's
// get_size_with_aspect_ratio_yolos.
func TargetSize(origH, origW int, size Size, modSize int) (int, int) {
	if size.Height > 0 && size.Width > 0 {
		oh, ow := size.Height, size.Width
		if modSize > 0 {
			oh -= oh % modSize
			ow -= ow % modSize
		}
		return oh, ow
	}
	if size.ShortestEdge <= 0 {
		return origH, origW
	}
	target := size.ShortestEdge
	maxSize := size.LongestEdge
	hasRaw := false
	var rawSize float64
	if maxSize > 0 {
		minOrig := float64(min(origH, origW))
		maxOrig := float64(max(origH, origW))
		if maxOrig/minOrig*float64(target) > float64(maxSize) {
			rawSize = float64(maxSize) * minOrig / maxOrig
			target = int(math.Round(rawSize))
			hasRaw = true
		}
	}
	var oh, ow int
	switch {
	case origW < origH:
		ow = target
		if hasRaw {
			oh = int(rawSize * float64(origH) / float64(origW))
		} else {
			oh = int(float64(target) * float64(origH) / float64(origW))
		}
	case (origH <= origW && origH == target) || (origW <= origH && origW == target):
		oh, ow = origH, origW
	default:
		oh = target
		if hasRaw {
			ow = int(rawSize * float64(origW) / float64(origH))
		} else {
			ow = int(float64(target) * float64(origW) / float64(origH))
		}
	}
	if modSize > 0 {
		ow -= ow % modSize
		oh -= oh % modSize
	}
	return oh, ow
}

// Process converts img to RGB, optionally resizes it with antialiased bilinear
// resampling, rescales, and normalizes it into a [3, H, W] CHW float32 tensor.
func Process(img image.Image, cfg Config) (*Result, error) {
	if img == nil {
		return nil, fmt.Errorf("preprocess: nil image")
	}
	b := img.Bounds()
	origW, origH := b.Dx(), b.Dy()
	if origW <= 0 || origH <= 0 {
		return nil, fmt.Errorf("preprocess: empty image %dx%d", origW, origH)
	}
	rgb := extractRGB(img)

	dstH, dstW := origH, origW
	if cfg.DoResize {
		dstH, dstW = TargetSize(origH, origW, cfg.Size, cfg.ModSize)
		if dstH <= 0 || dstW <= 0 {
			return nil, fmt.Errorf("preprocess: invalid target size %dx%d for source %dx%d", dstW, dstH, origW, origH)
		}
		if dstH != origH || dstW != origW {
			rgb = resizeBilinearAA(rgb, origH, origW, dstH, dstW)
		}
	}

	pixels := tensor.New(3, dstH, dstW)
	plane := dstH * dstW
	scale := float32(1)
	if cfg.DoRescale {
		scale = cfg.RescaleFactor
		if scale == 0 {
			scale = 1.0 / 255.0
		}
	}
	var invStd [3]float32
	for c := range 3 {
		if cfg.DoNormalize {
			if cfg.ImageStd[c] == 0 {
				return nil, fmt.Errorf("preprocess: zero image_std[%d]", c)
			}
			invStd[c] = 1.0 / cfg.ImageStd[c]
		}
	}

	dstR := pixels.Data[0:plane]
	dstG := pixels.Data[plane : 2*plane]
	dstB := pixels.Data[2*plane : 3*plane]
	if cfg.DoNormalize {
		m0, m1, m2 := cfg.ImageMean[0], cfg.ImageMean[1], cfg.ImageMean[2]
		s0, s1, s2 := invStd[0], invStd[1], invStd[2]
		for i := range plane {
			p := 3 * i
			dstR[i] = (float32(rgb[p])*scale - m0) * s0
			dstG[i] = (float32(rgb[p+1])*scale - m1) * s1
			dstB[i] = (float32(rgb[p+2])*scale - m2) * s2
		}
	} else {
		for i := range plane {
			p := 3 * i
			dstR[i] = float32(rgb[p]) * scale
			dstG[i] = float32(rgb[p+1]) * scale
			dstB[i] = float32(rgb[p+2]) * scale
		}
	}

	return &Result{
		Pixels:     pixels,
		OrigWidth:  origW,
		OrigHeight: origH,
		Width:      dstW,
		Height:     dstH,
	}, nil
}

// extractRGB returns tightly packed [H, W, 3] 8-bit RGB bytes.
func extractRGB(img image.Image) []uint8 {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	out := make([]uint8, h*w*3)
	switch src := img.(type) {
	case *image.RGBA:
		for y := range h {
			row := src.Pix[(b.Min.Y+y-src.Rect.Min.Y)*src.Stride:]
			base := y * w * 3
			x0 := (b.Min.X - src.Rect.Min.X) * 4
			for x := range w {
				si, di := x0+x*4, base+x*3
				out[di] = row[si]
				out[di+1] = row[si+1]
				out[di+2] = row[si+2]
			}
		}
	case *image.NRGBA:
		for y := range h {
			row := src.Pix[(b.Min.Y+y-src.Rect.Min.Y)*src.Stride:]
			base := y * w * 3
			x0 := (b.Min.X - src.Rect.Min.X) * 4
			for x := range w {
				si, di := x0+x*4, base+x*3
				out[di] = row[si]
				out[di+1] = row[si+1]
				out[di+2] = row[si+2]
			}
		}
	case *image.YCbCr:
		for y := range h {
			base := y * w * 3
			sy := b.Min.Y + y
			for x := range w {
				sx := b.Min.X + x
				yi := src.YOffset(sx, sy)
				ci := src.COffset(sx, sy)
				r, g, bl := ycbcrToRGB(src.Y[yi], src.Cb[ci], src.Cr[ci])
				di := base + x*3
				out[di], out[di+1], out[di+2] = r, g, bl
			}
		}
	default:
		for y := range h {
			base := y * w * 3
			sy := b.Min.Y + y
			for x := range w {
				r, g, bl, _ := img.At(b.Min.X+x, sy).RGBA()
				di := base + x*3
				out[di] = uint8(r >> 8)
				out[di+1] = uint8(g >> 8)
				out[di+2] = uint8(bl >> 8)
			}
		}
	}
	return out
}

func ycbcrToRGB(y, cb, cr uint8) (uint8, uint8, uint8) {
	yy1 := int32(y) * 0x10101
	cb1 := int32(cb) - 128
	cr1 := int32(cr) - 128
	r := (yy1 + 91881*cr1) >> 16
	g := (yy1 - 22554*cb1 - 46802*cr1) >> 16
	b := (yy1 + 116130*cb1) >> 16
	return clampU8(r), clampU8(g), clampU8(b)
}

func clampU8(v int32) uint8 {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v)
}

// filter1D holds precomputed tap indices and weights for a 1D separable filter.
type filter1D struct {
	xmin    []int
	ksize   []int
	offsets []int
	weights []float32
}

// buildBilinearAA builds a PIL / Torchvision-compatible antialiased triangle
// filter mapping srcLen pixels to dstLen pixels.
func buildBilinearAA(srcLen, dstLen int) filter1D {
	scale := float64(srcLen) / float64(dstLen)
	support := scale
	if support < 1.0 {
		support = 1.0
	}
	invScale := 1.0 / support

	xmin := make([]int, dstLen)
	ksize := make([]int, dstLen)
	offsets := make([]int, dstLen)
	maxTaps := int(math.Ceil(support))*2 + 2
	weights := make([]float32, 0, dstLen*maxTaps)

	for out := range dstLen {
		center := (float64(out) + 0.5) * scale
		lo := max(int(math.Floor(center-support+0.5)), 0)
		hi := min(int(math.Floor(center+support+0.5)), srcLen)
		if hi <= lo {
			lo = min(max(int(math.Floor(center)), 0), srcLen-1)
			hi = lo + 1
		}
		xmin[out] = lo
		ksize[out] = hi - lo
		offsets[out] = len(weights)

		var total float64
		start := len(weights)
		for i := lo; i < hi; i++ {
			d := math.Abs((float64(i) + 0.5 - center) * invScale)
			w := 0.0
			if d < 1.0 {
				w = 1.0 - d
			}
			weights = append(weights, float32(w))
			total += w
		}
		if total > 0 {
			invTotal := float32(1.0 / total)
			for i := start; i < len(weights); i++ {
				weights[i] *= invTotal
			}
		} else if len(weights) > start {
			weights[start] = 1
		}
	}
	return filter1D{xmin: xmin, ksize: ksize, offsets: offsets, weights: weights}
}

// resizeBilinearAA resizes a tightly packed [srcH, srcW, 3] RGB byte buffer to
// [dstH, dstW, 3] using separable antialiased bilinear filtering.
func resizeBilinearAA(src []uint8, srcH, srcW, dstH, dstW int) []uint8 {
	fx := buildBilinearAA(srcW, dstW)
	fy := buildBilinearAA(srcH, dstH)

	// Horizontal pass: [srcH, srcW, 3] -> [srcH, dstW, 3] float32.
	tmp := make([]float32, srcH*dstW*3)
	for y := range srcH {
		sRow := src[y*srcW*3:]
		tRow := tmp[y*dstW*3:]
		for x := range dstW {
			lo, ks, off := fx.xmin[x], fx.ksize[x], fx.offsets[x]
			var r, g, b float32
			for k := range ks {
				w := fx.weights[off+k]
				si := (lo + k) * 3
				r += w * float32(sRow[si])
				g += w * float32(sRow[si+1])
				b += w * float32(sRow[si+2])
			}
			di := x * 3
			tRow[di], tRow[di+1], tRow[di+2] = r, g, b
		}
	}

	// Vertical pass: [srcH, dstW, 3] float32 -> [dstH, dstW, 3] uint8.
	dst := make([]uint8, dstH*dstW*3)
	rowStride := dstW * 3
	for y := range dstH {
		lo, ks, off := fy.xmin[y], fy.ksize[y], fy.offsets[y]
		dRow := dst[y*rowStride:]
		for x := range rowStride {
			var acc float32
			for k := range ks {
				acc += fy.weights[off+k] * tmp[(lo+k)*rowStride+x]
			}
			v := int32(math.Round(float64(acc)))
			dRow[x] = clampU8(v)
		}
	}
	return dst
}

// ExtractPatches flattens non-overlapping patchSize×patchSize patches from a
// [3, H, W] CHW tensor into a [gridH*gridW, 3*patchSize*patchSize] 2D tensor.
// Each row stores one patch in [C, pH, pW] order, matching a PyTorch Conv2d
// weight [outCh, 3, patchSize, patchSize] viewed as [outCh, 3*patchSize*patchSize].
func ExtractPatches(pixels *tensor.Tensor, patchSize int) (patches *tensor.Tensor, gridH, gridW int, err error) {
	if len(pixels.Shape) != 3 || pixels.Shape[0] != 3 {
		return nil, 0, 0, fmt.Errorf("preprocess.ExtractPatches: want [3, H, W], got %v", pixels.Shape)
	}
	if patchSize <= 0 {
		return nil, 0, 0, fmt.Errorf("preprocess.ExtractPatches: invalid patch size %d", patchSize)
	}
	h, w := pixels.Shape[1], pixels.Shape[2]
	gridH, gridW = h/patchSize, w/patchSize
	if gridH == 0 || gridW == 0 {
		return nil, 0, 0, fmt.Errorf("preprocess.ExtractPatches: image %dx%d smaller than patch size %d", w, h, patchSize)
	}
	patchArea := patchSize * patchSize
	patchDim := 3 * patchArea
	out := tensor.New(gridH*gridW, patchDim)
	plane := h * w
	for py := range gridH {
		y0 := py * patchSize
		for px := range gridW {
			x0 := px * patchSize
			row := out.Row(py*gridW + px)
			for c := range 3 {
				cBase := c * plane
				rBase := c * patchArea
				for ky := range patchSize {
					srcOff := cBase + (y0+ky)*w + x0
					copy(row[rBase+ky*patchSize:rBase+(ky+1)*patchSize], pixels.Data[srcOff:srcOff+patchSize])
				}
			}
		}
	}
	return out, gridH, gridW, nil
}

// BicubicInterpolate2D resamples a [srcH*srcW, cols] spatial feature grid to
// [dstH*dstW, cols], matching PyTorch's
// F.interpolate(..., size=(dstH, dstW), mode="bicubic", align_corners=False)
// (Keys cubic kernel with a = -0.75).
func BicubicInterpolate2D(src []float32, srcH, srcW, dstH, dstW, cols int) []float32 {
	if srcH == dstH && srcW == dstW {
		out := make([]float32, len(src))
		copy(out, src)
		return out
	}
	// Precompute 4-tap horizontal and vertical indices & weights.
	hx, hwx := bicubicWeights(srcW, dstW)
	hy, hwy := bicubicWeights(srcH, dstH)

	// Horizontal pass: [srcH, dstW, cols].
	tmp := make([]float32, srcH*dstW*cols)
	for sy := range srcH {
		sRowBase := sy * srcW * cols
		tRowBase := sy * dstW * cols
		for dx := range dstW {
			ix := hx[dx*4 : dx*4+4]
			wx := hwx[dx*4 : dx*4+4]
			s0 := src[sRowBase+ix[0]*cols:]
			s1 := src[sRowBase+ix[1]*cols:]
			s2 := src[sRowBase+ix[2]*cols:]
			s3 := src[sRowBase+ix[3]*cols:]
			dst := tmp[tRowBase+dx*cols : tRowBase+(dx+1)*cols]
			w0, w1, w2, w3 := wx[0], wx[1], wx[2], wx[3]
			for c := range cols {
				dst[c] = float32(w0*float64(s0[c]) + w1*float64(s1[c]) + w2*float64(s2[c]) + w3*float64(s3[c]))
			}
		}
	}

	// Vertical pass: [dstH, dstW, cols].
	out := make([]float32, dstH*dstW*cols)
	rowStride := dstW * cols
	for dy := range dstH {
		iy := hy[dy*4 : dy*4+4]
		wy := hwy[dy*4 : dy*4+4]
		t0 := tmp[iy[0]*rowStride:]
		t1 := tmp[iy[1]*rowStride:]
		t2 := tmp[iy[2]*rowStride:]
		t3 := tmp[iy[3]*rowStride:]
		dst := out[dy*rowStride : (dy+1)*rowStride]
		w0, w1, w2, w3 := wy[0], wy[1], wy[2], wy[3]
		for i := range rowStride {
			dst[i] = float32(w0*float64(t0[i]) + w1*float64(t1[i]) + w2*float64(t2[i]) + w3*float64(t3[i]))
		}
	}
	return out
}

// bicubicWeights computes PyTorch's align_corners=False bicubic (a = -0.75)
// clamped indices and 4 tap weights for every output position.
func bicubicWeights(srcLen, dstLen int) (idx []int, weights []float64) {
	const a = -0.75
	scale := float64(srcLen) / float64(dstLen)
	idx = make([]int, dstLen*4)
	weights = make([]float64, dstLen*4)
	clamp := func(v int) int {
		if v < 0 {
			return 0
		}
		if v >= srcLen {
			return srcLen - 1
		}
		return v
	}
	conv1 := func(x float64) float64 {
		return ((a+2.0)*x-(a+3.0))*x*x + 1.0
	}
	conv2 := func(x float64) float64 {
		return ((a*x-5.0*a)*x+8.0*a)*x - 4.0*a
	}
	for out := range dstLen {
		realX := (float64(out)+0.5)*scale - 0.5
		inX := int(math.Floor(realX))
		t := realX - float64(inX)

		base := out * 4
		idx[base] = clamp(inX - 1)
		idx[base+1] = clamp(inX)
		idx[base+2] = clamp(inX + 1)
		idx[base+3] = clamp(inX + 2)

		weights[base] = conv2(t + 1.0)
		weights[base+1] = conv1(t)
		weights[base+2] = conv1(1.0 - t)
		weights[base+3] = conv2(2.0 - t)
	}
	return idx, weights
}
