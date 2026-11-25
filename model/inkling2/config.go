package inkling2

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
)

// Config is the text model's configuration: config.json's text_config, with
// its loop_config, and the end token from the top level.
type Config struct {
	ModelType         string `json:"model_type"`
	HiddenSize        int    `json:"hidden_size"`
	NumHiddenLayers   int    `json:"num_hidden_layers"`
	VocabSize         int    `json:"vocab_size"`
	UnpaddedVocabSize int    `json:"unpadded_vocab_size"`

	// Full-attention layers.
	NumAttentionHeads int `json:"num_attention_heads"`
	NumKeyValueHeads  int `json:"num_key_value_heads"`
	HeadDim           int `json:"head_dim"`
	// Sliding-window layers, whose ids LocalLayerIDs lists.
	SWANumAttentionHeads int   `json:"swa_num_attention_heads"`
	SWANumKeyValueHeads  int   `json:"swa_num_key_value_heads"`
	SWAHeadDim           int   `json:"swa_head_dim"`
	SlidingWindowSize    int   `json:"sliding_window_size"`
	LocalLayerIDs        []int `json:"local_layer_ids"`

	QBias bool `json:"q_bias"`
	OBias bool `json:"o_bias"`

	// The relative position bias: each head's d_rel relative states mix a
	// learned bank of profiles over rel_extent backward distances.
	DRel               int     `json:"d_rel"`
	RelExtent          int     `json:"rel_extent"`
	QKL2ClipThreshold  float64 `json:"qk_l2_clip_threshold"`
	RelL2ClipThreshold float64 `json:"rel_l2_clip_threshold"`
	LogScalingNFloor   int     `json:"log_scaling_n_floor"`
	LogScalingAlpha    float64 `json:"log_scaling_alpha"`

	RMSNormEps          float64 `json:"rms_norm_eps"`
	UseValueNormNoGamma bool    `json:"use_value_norm_no_gamma"`
	SconvKernelSize     int     `json:"sconv_kernel_size"`

	// Attention residuals over blocks of layers.
	NLayersPerAttnResBlock int  `json:"n_layers_per_attn_res_block"`
	AttnResNormalizeValues bool `json:"attn_res_normalize_values"`

	// The routed experts.
	NRoutedExperts   int     `json:"n_routed_experts"`
	NumExpertsPerTok int     `json:"num_experts_per_tok"`
	NSharedExperts   int     `json:"n_shared_experts"`
	IntermediateSize int     `json:"intermediate_size"`
	MLPGateSoftcap   float64 `json:"mlp_gate_softcap"`
	MLPUpSoftcap     float64 `json:"mlp_up_softcap"`
	RouteScale       float64 `json:"route_scale"`
	UseGateBias      bool    `json:"use_gate_bias"`

	FinalLogitSoftcapping float64 `json:"final_logit_softcapping"`

	Loop LoopConfig `json:"loop_config"`

	// EOS is the end token (config.json's top-level eos_token_id).
	EOS int `json:"-"`
}

// LoopConfig is the looped core: layers [CoreStart, CoreEnd) run NumPasses
// times with shared weights, each pass after the first reading messages
// from the one before through rank-GateRank gates.
type LoopConfig struct {
	CoreStart int  `json:"core_start"`
	CoreEnd   int  `json:"core_end"`
	NumPasses int  `json:"num_passes"`
	GateRank  int  `json:"gate_rank"`
	IsVarloop bool `json:"is_varloop"`
}

// sliding reports whether layer i is a sliding-window layer.
func (c Config) sliding(i int) bool { return slices.Contains(c.LocalLayerIDs, i) }

// inLoop reports whether layer i is in the looped core.
func (c Config) inLoop(i int) bool { return c.Loop.CoreStart <= i && i < c.Loop.CoreEnd }

// boundary reports whether layer i starts a block of the attention
// residuals: its input joins the block states.
func (c Config) boundary(i int) bool { return i%c.NLayersPerAttnResBlock == 0 }

// loopLayers is the number of layers in the looped core.
func (c Config) loopLayers() int { return c.Loop.CoreEnd - c.Loop.CoreStart }

// Visits is how many layers each position runs through: the layers outside
// the core once, the core NumPasses times.
func (c Config) Visits() int { return c.NumHiddenLayers + (c.Loop.NumPasses-1)*c.loopLayers() }

// heads returns layer i's query heads, key/value heads and head size.
func (c Config) heads(i int) (heads, kvHeads, headDim int) {
	if c.sliding(i) {
		return c.SWANumAttentionHeads, c.SWANumKeyValueHeads, c.SWAHeadDim
	}
	return c.NumAttentionHeads, c.NumKeyValueHeads, c.HeadDim
}

// relExtent returns how many backward distances layer i's relative bias
// covers: the window in sliding layers.
func (c Config) relExtent(i int) int {
	if c.sliding(i) {
		return c.SlidingWindowSize
	}
	return c.RelExtent
}

// logScale returns the full-attention layers' scale of queries and relative
// states at position pos, 1 + α·log(max((pos+1)/floor, 1)), in float32 as
// transformers' Inkling2LoopLogScale computes it.
func (c Config) logScale(pos int) float32 {
	r := float32(pos+1) / float32(c.LogScalingNFloor)
	if r <= 1 {
		return 1
	}
	return 1 + float32(c.LogScalingAlpha)*float32(math.Log(float64(r)))
}

// validate rejects configurations this package does not implement rather
// than running them wrong.
func (c Config) validate() error {
	l := c.Loop
	switch {
	case c.ModelType != "inkling2_loop_text":
		return fmt.Errorf("model_type %q is not inkling2_loop_text", c.ModelType)
	case c.HiddenSize <= 0 || c.NumHiddenLayers <= 0 || c.VocabSize <= 0:
		return errors.New("config is missing sizes")
	case c.UnpaddedVocabSize <= 0 || c.UnpaddedVocabSize > c.VocabSize:
		return fmt.Errorf("unpadded_vocab_size %d of vocab_size %d", c.UnpaddedVocabSize, c.VocabSize)
	case c.QBias || c.OBias:
		return errors.New("attention biases are not supported")
	case !c.UseValueNormNoGamma:
		return errors.New("value norms with a weight are not supported")
	case !c.AttnResNormalizeValues:
		return errors.New("attention residuals over unnormalized values are not supported")
	case c.NSharedExperts != 0:
		return errors.New("shared experts are not supported")
	case !c.UseGateBias:
		return errors.New("routing without a gate bias is not supported")
	case c.NRoutedExperts <= 0 || c.NumExpertsPerTok <= 0 || c.NumExpertsPerTok > c.NRoutedExperts || c.IntermediateSize <= 0:
		return fmt.Errorf("%d of %d experts of %d", c.NumExpertsPerTok, c.NRoutedExperts, c.IntermediateSize)
	case c.SconvKernelSize < 1:
		return fmt.Errorf("sconv_kernel_size %d", c.SconvKernelSize)
	case c.NLayersPerAttnResBlock < 1:
		return fmt.Errorf("n_layers_per_attn_res_block %d", c.NLayersPerAttnResBlock)
	case c.DRel <= 0 || c.RelExtent <= 0 || c.SlidingWindowSize <= 0:
		return errors.New("relative position sizes are missing")
	case c.LogScalingNFloor <= 0:
		return fmt.Errorf("log_scaling_n_floor %d", c.LogScalingNFloor)
	case l.IsVarloop:
		return errors.New("variable loops are not supported")
	case l.NumPasses < 1 || l.GateRank <= 0:
		return fmt.Errorf("%d loop passes of gate rank %d", l.NumPasses, l.GateRank)
	case l.CoreStart <= 0 || l.CoreEnd <= l.CoreStart || l.CoreEnd > c.NumHiddenLayers ||
		l.CoreStart%c.NLayersPerAttnResBlock != 0 || l.CoreEnd%c.NLayersPerAttnResBlock != 0:
		return fmt.Errorf("loop core [%d, %d) is not whole blocks of %d of %d layers", l.CoreStart, l.CoreEnd, c.NLayersPerAttnResBlock, c.NumHiddenLayers)
	}
	for i := range c.NumHiddenLayers {
		heads, kvHeads, hd := c.heads(i)
		if heads <= 0 || kvHeads <= 0 || heads%kvHeads != 0 || hd <= 0 {
			return fmt.Errorf("layer %d: %d heads do not split into %d key/value heads of %d", i, heads, kvHeads, hd)
		}
	}
	for _, id := range c.LocalLayerIDs {
		if id < 0 || id >= c.NumHiddenLayers {
			return fmt.Errorf("local layer id %d of %d layers", id, c.NumHiddenLayers)
		}
	}
	return nil
}

// readConfig reads the text configuration from config.json in dir.
func readConfig(dir string) (Config, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return Config{}, err
	}
	var top struct {
		ModelType  string          `json:"model_type"`
		TextConfig json.RawMessage `json:"text_config"`
		EOS        *int            `json:"eos_token_id"`
	}
	if err := json.Unmarshal(raw, &top); err != nil {
		return Config{}, fmt.Errorf("parse config.json: %w", err)
	}
	if top.ModelType != "inkling2_loop" || len(top.TextConfig) == 0 {
		return Config{}, fmt.Errorf("config.json has model_type %q, not inkling2_loop with a text_config", top.ModelType)
	}
	c := Config{
		// transformers' defaults for fields config.json may leave out.
		RMSNormEps:             1e-6,
		SconvKernelSize:        4,
		NLayersPerAttnResBlock: 6,
		AttnResNormalizeValues: true,
		UseValueNormNoGamma:    true,
		UseGateBias:            true,
		EOS:                    -1,
	}
	if err := json.Unmarshal(top.TextConfig, &c); err != nil {
		return Config{}, fmt.Errorf("parse config.json: %w", err)
	}
	if c.SWANumAttentionHeads == 0 {
		c.SWANumAttentionHeads, c.SWANumKeyValueHeads, c.SWAHeadDim = c.NumAttentionHeads, c.NumKeyValueHeads, c.HeadDim
	}
	if top.EOS != nil {
		c.EOS = *top.EOS
	}
	return c, nil
}
