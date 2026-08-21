// Package modelspec turns a HuggingFace model id into a concrete serving plan.
//
// Given only "Qwen/Qwen3-32B" it fetches the model's metadata and config, works
// out how much memory the weights and KV cache actually need, and decides which
// engine, quantization, tensor-parallel size and context length will fit on a
// given node. This is the arithmetic an operator would otherwise do by hand,
// badly, before discovering the mistake as an out-of-memory crash ten minutes
// into a weight download.
package modelspec

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"time"
)

const hfAPI = "https://huggingface.co"

// Info is what we learn about a model from HuggingFace.
type Info struct {
	ID          string   `json:"id"`
	Params      int64    `json:"params"`
	ParamsLabel string   `json:"params_label"`
	DType       string   `json:"dtype"`
	Gated       bool     `json:"gated"`
	Private     bool     `json:"private"`
	Downloads   int64    `json:"downloads"`
	License     string   `json:"license"`
	Tags        []string `json:"tags"`

	Architecture string `json:"architecture"`
	MaxPositions int    `json:"max_positions"`
	HiddenSize   int    `json:"hidden_size"`
	NumLayers    int    `json:"num_layers"`
	NumHeads     int    `json:"num_heads"`
	NumKVHeads   int    `json:"num_kv_heads"`
	HeadDim      int    `json:"head_dim"`
	VocabSize    int    `json:"vocab_size"`
	IsMoE        bool   `json:"is_moe"`
	// IsMLA marks DeepSeek-style Multi-head Latent Attention, where the KV
	// cache is a compressed latent shared across all heads rather than
	// per-head tensors. It changes how tensor parallelism can be applied.
	IsMLA bool `json:"is_mla"`
	// IsHybrid marks a model that interleaves Mamba-style linear attention
	// with ordinary full attention. It changes two things that matter: only
	// the full-attention layers hold a per-token KV cache, and vLLM cannot
	// store that cache at fp8 for these models.
	IsHybrid bool `json:"is_hybrid"`
	// FullAttentionLayers is how many layers actually keep a KV cache. It
	// equals NumLayers on an ordinary transformer.
	FullAttentionLayers int `json:"full_attention_layers"`
	// PublishedQuant is set when the checkpoint on HuggingFace is already
	// quantized, e.g. "fp8". The download is then that size, not bf16.
	PublishedQuant string `json:"published_quant,omitempty"`
	// HasVision and HasAudio mark extra input towers. Their parameters are
	// included in the total count, but they also need activation memory during
	// encoding that the text-only arithmetic does not model.
	HasVision bool `json:"has_vision"`
	HasAudio  bool `json:"has_audio"`
	// ActiveParams is the per-token compute cost for MoE models. Memory still
	// needs the full parameter count, since every expert must be resident.
	ActiveParams int64 `json:"active_params,omitempty"`
}

// hfModelResponse is the subset of the HF model API we read.
type hfModelResponse struct {
	ModelID   string   `json:"modelId"`
	Downloads int64    `json:"downloads"`
	Tags      []string `json:"tags"`
	Private   bool     `json:"private"`
	// Gated is false, or a string like "auto"/"manual". Its type varies, so it
	// is decoded loosely rather than as a bool.
	Gated       any `json:"gated"`
	Safetensors *struct {
		Total      int64            `json:"total"`
		Parameters map[string]int64 `json:"parameters"`
	} `json:"safetensors"`
}

// hfConfig is the subset of a model's config.json that determines memory use.
type hfConfig struct {
	Architectures []string `json:"architectures"`
	MaxPositions  int      `json:"max_position_embeddings"`
	HiddenSize    int      `json:"hidden_size"`
	NumLayers     int      `json:"num_hidden_layers"`
	NumHeads      int      `json:"num_attention_heads"`
	NumKVHeads    int      `json:"num_key_value_heads"`
	HeadDim       int      `json:"head_dim"`
	VocabSize     int      `json:"vocab_size"`
	TorchDType    string   `json:"torch_dtype"`
	// MoE indicators. Different families name these differently, so several
	// spellings are checked.
	NumExperts       int `json:"num_experts"`
	NumLocalExperts  int `json:"num_local_experts"`
	NumRoutedExperts int `json:"n_routed_experts"`
	// Fields needed to work out how many parameters are active per token,
	// which is what determines generation speed on an MoE model.
	ExpertsPerTok      int `json:"num_experts_per_tok"`
	SharedExperts      int `json:"n_shared_experts"`
	MoEIntermediate    int `json:"moe_intermediate_size"`
	FirstKDenseReplace int `json:"first_k_dense_replace"`
	// QLoRARank is set by DeepSeek-style Multi-head Latent Attention, where the
	// KV cache is a compressed latent rather than per-head tensors.
	QLoRARank  int `json:"q_lora_rank"`
	KVLoRARank int `json:"kv_lora_rank"`
	// LayerTypes names each layer's attention. Hybrid models interleave
	// "linear_attention" (a fixed-size recurrent state) with "full_attention"
	// (a growing KV cache), and only the latter costs memory per token.
	LayerTypes []string `json:"layer_types"`
	// FullAttentionInterval is the shorthand some configs use instead of
	// spelling out layer_types: every Nth layer is full attention.
	FullAttentionInterval int `json:"full_attention_interval"`
	// LinearConvKernelDim and MambaSSMDType appear only on models with
	// Mamba-style linear attention.
	LinearConvKernelDim int    `json:"linear_conv_kernel_dim"`
	MambaSSMDType       string `json:"mamba_ssm_dtype"`
	// QuantizationConfig is present when the published checkpoint is already
	// quantized, which changes both the download size and the usable formats.
	QuantizationConfig *quantConfig `json:"quantization_config"`

	// Multimodal models nest the language model's geometry one level down and
	// leave the top level holding only the wrapper. The key name varies by
	// family, so all the common spellings are read.
	TextConfig     *hfConfig `json:"text_config"`
	LLMConfig      *hfConfig `json:"llm_config"`
	LanguageConfig *hfConfig `json:"language_config"`
	// VisionConfig marks an image or video input tower.
	VisionConfig *struct {
		HiddenSize int `json:"hidden_size"`
		NumLayers  int `json:"num_hidden_layers"`
		ImageSize  int `json:"image_size"`
	} `json:"vision_config"`
	AudioConfig *struct {
		HiddenSize int `json:"hidden_size"`
	} `json:"audio_config"`
}

// languageConfig returns the sub-config holding the text model's geometry.
//
// A multimodal checkpoint's top-level config describes the wrapper, not the
// transformer: reading it directly yields zero layers and zero KV heads, which
// silently produces a plan with no KV cache and a meaningless concurrency
// figure.
func (c *hfConfig) languageConfig() *hfConfig {
	if c.NumLayers > 0 {
		return c
	}
	for _, nested := range []*hfConfig{c.TextConfig, c.LLMConfig, c.LanguageConfig} {
		if nested != nil && nested.NumLayers > 0 {
			return nested
		}
	}
	return c
}

func (c *hfConfig) routedExperts() int {
	for _, n := range []int{c.NumRoutedExperts, c.NumLocalExperts, c.NumExperts} {
		if n > 0 {
			return n
		}
	}
	return 0
}

type Client struct {
	HTTP  *http.Client
	Token string // optional; required for gated or private repos
}

func NewClient(token string) *Client {
	return &Client{
		HTTP:  &http.Client{Timeout: 20 * time.Second},
		Token: token,
	}
}

// Fetch resolves a HuggingFace model id into an Info.
func (c *Client) Fetch(ctx context.Context, id string) (*Info, error) {
	id = strings.Trim(strings.TrimSpace(id), "/")
	if id == "" || strings.Count(id, "/") != 1 {
		return nil, fmt.Errorf("model id must be in owner/name form, got %q", id)
	}

	var meta hfModelResponse
	if err := c.getJSON(ctx, hfAPI+"/api/models/"+id, &meta); err != nil {
		return nil, fmt.Errorf("look up %s: %w", id, err)
	}
	var raw hfConfig
	if err := c.getJSON(ctx, hfAPI+"/"+id+"/raw/main/config.json", &raw); err != nil {
		return nil, fmt.Errorf("read config.json for %s: %w", id, err)
	}
	// Geometry comes from the language model; the wrapper keeps the
	// architecture name and the modality towers.
	cfg := *raw.languageConfig()
	if cfg.QuantizationConfig == nil {
		cfg.QuantizationConfig = raw.QuantizationConfig
	}

	info := &Info{
		ID:           id,
		Downloads:    meta.Downloads,
		Tags:         meta.Tags,
		Private:      meta.Private,
		Gated:        meta.Gated != nil && meta.Gated != false,
		MaxPositions: cfg.MaxPositions,
		HiddenSize:   cfg.HiddenSize,
		NumLayers:    cfg.NumLayers,
		NumHeads:     cfg.NumHeads,
		NumKVHeads:   cfg.NumKVHeads,
		HeadDim:      cfg.HeadDim,
		VocabSize:    cfg.VocabSize,
		DType:        cfg.TorchDType,
	}
	if len(raw.Architectures) > 0 {
		info.Architecture = raw.Architectures[0]
	} else if len(cfg.Architectures) > 0 {
		info.Architecture = cfg.Architectures[0]
	}
	info.HasVision = raw.VisionConfig != nil
	info.HasAudio = raw.AudioConfig != nil
	// Models without grouped-query attention omit num_key_value_heads; it then
	// equals the attention head count, which matters a lot for KV cache size.
	if info.NumKVHeads == 0 {
		info.NumKVHeads = cfg.NumHeads
	}
	if info.HeadDim == 0 && cfg.NumHeads > 0 {
		info.HeadDim = cfg.HiddenSize / cfg.NumHeads
	}
	info.IsMoE = cfg.routedExperts() > 0
	info.IsMLA = cfg.QLoRARank > 0 || cfg.KVLoRARank > 0
	info.FullAttentionLayers, info.IsHybrid = cfg.attentionLayout()
	if cfg.QuantizationConfig != nil {
		info.PublishedQuant = cfg.QuantizationConfig.scheme()
	}

	if meta.Safetensors != nil {
		info.Params = meta.Safetensors.Total
		// The parameters map is keyed by dtype (BF16, F32, ...). The dominant
		// key is the weight precision as published.
		var best int64
		for dt, n := range meta.Safetensors.Parameters {
			if n > best {
				best, info.DType = n, dt
			}
		}
	}
	info.computeActiveParams(&cfg)
	info.ParamsLabel = FormatParams(info.Params)
	return info, nil
}

// computeActiveParams works out how many parameters an MoE model actually
// touches per token.
//
// This matters twice over. Memory must hold every expert, so a 304B model needs
// 304B worth of VRAM. Speed depends only on the experts actually routed to, so
// the same model generates at the pace of a far smaller one. Conflating the two
// produces pricing that is wrong by an order of magnitude, which is exactly
// what happens if you price a 304B MoE as though it were a 304B dense model.
func (i *Info) computeActiveParams(cfg *hfConfig) {
	experts := cfg.routedExperts()
	if experts == 0 || cfg.MoEIntermediate == 0 || cfg.HiddenSize == 0 || i.Params == 0 {
		return
	}
	// Each expert is a gate, up and down projection between hidden size and
	// the MoE intermediate size.
	perExpert := int64(3) * int64(cfg.HiddenSize) * int64(cfg.MoEIntermediate)
	moeLayers := int64(cfg.NumLayers - cfg.FirstKDenseReplace)
	if moeLayers <= 0 {
		moeLayers = int64(cfg.NumLayers)
	}
	totalExpert := int64(experts) * perExpert * moeLayers

	perTok := cfg.ExpertsPerTok + cfg.SharedExperts
	if perTok <= 0 {
		return
	}
	activeExpert := int64(perTok) * perExpert * moeLayers

	// Everything that is not a routed expert runs on every token.
	active := i.Params - (totalExpert - activeExpert)
	if active > 0 && active < i.Params {
		i.ActiveParams = active
	}
}

func (c *Client) getJSON(ctx context.Context, url string, dst any) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	if c.Token != "" {
		req.Header.Set("Authorization", "Bearer "+c.Token)
	}
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	switch resp.StatusCode {
	case http.StatusOK:
	case http.StatusNotFound:
		return fmt.Errorf("not found on HuggingFace (check spelling, or the repo may be private)")
	case http.StatusUnauthorized, http.StatusForbidden:
		return fmt.Errorf("access denied: this repo is gated or private, set a HuggingFace token")
	default:
		return fmt.Errorf("huggingface returned %s", resp.Status)
	}
	return json.NewDecoder(resp.Body).Decode(dst)
}

// FormatParams renders a parameter count the way model names do.
func FormatParams(n int64) string {
	switch {
	case n <= 0:
		return "unknown"
	case n >= 1e12:
		return fmt.Sprintf("%.1fT", float64(n)/1e12)
	case n >= 1e9:
		return fmt.Sprintf("%.1fB", float64(n)/1e9)
	case n >= 1e6:
		return fmt.Sprintf("%.0fM", float64(n)/1e6)
	}
	return fmt.Sprintf("%d", n)
}

// GGUFCandidate is a GGUF conversion of a model, as published by the community.
type GGUFCandidate struct {
	Repo      string `json:"repo"`
	Downloads int64  `json:"downloads"`
	Official  bool   `json:"official"`
	// exact marks a repo named exactly "<model>-GGUF", the conventional name
	// for a straight conversion with nothing else changed.
	exact bool
}

// ResolveGGUF finds GGUF conversions of a model for llama.cpp.
//
// llama.cpp cannot read the original safetensors weights, so a CPU install
// needs a converted repo. These are published by the model owner sometimes and
// by the community usually, which is a step that reliably strands people. The
// owner's own conversion is preferred, then the most-downloaded community one,
// since download count is a decent proxy for "this conversion actually works".
func (c *Client) ResolveGGUF(ctx context.Context, id string) ([]GGUFCandidate, error) {
	owner, name, ok := strings.Cut(id, "/")
	if !ok {
		return nil, fmt.Errorf("model id must be in owner/name form, got %q", id)
	}

	var results []struct {
		ModelID   string `json:"modelId"`
		Downloads int64  `json:"downloads"`
	}
	url := fmt.Sprintf("%s/api/models?search=%s&filter=gguf&sort=downloads&direction=-1&limit=20",
		hfAPI, name)
	if err := c.getJSON(ctx, url, &results); err != nil {
		return nil, err
	}

	// HuggingFace search is fuzzy, so a query for "Qwen3-8B" also returns
	// Qwen3-VL-8B and Qwen3-Embedding-8B, which are entirely different models.
	// Serving one of those instead would be a silent, confusing failure, so
	// anything whose name is not a prefix-extension of the requested model is
	// discarded rather than merely ranked lower.
	seen := map[string]bool{}
	var out []GGUFCandidate
	for _, r := range results {
		if seen[r.ModelID] {
			continue
		}
		seen[r.ModelID] = true

		candOwner, candName, ok := strings.Cut(r.ModelID, "/")
		if !ok || !isGGUFVariantOf(candName, name) {
			continue
		}
		out = append(out, GGUFCandidate{
			Repo:      r.ModelID,
			Downloads: r.Downloads,
			Official:  strings.EqualFold(candOwner, owner),
			exact:     strings.EqualFold(candName, name+"-GGUF"),
		})
	}
	// Exact conversions first, then the owner's own, then by popularity.
	sortCandidates(out)
	if len(out) == 0 {
		return nil, fmt.Errorf("no GGUF conversion found for %s; llama.cpp needs one, "+
			"or use a GPU node with vLLM which reads the original weights", id)
	}
	return out, nil
}

// isGGUFVariantOf reports whether a candidate repo name is a GGUF conversion of
// the requested model, rather than a different model that merely shares words.
//
// The test is that the candidate begins with the model name and the remainder
// is only conversion-related decoration -- "-GGUF", "-gguf", "-i1-GGUF" and so
// on. "Qwen3-8B-GGUF" passes for "Qwen3-8B"; "Qwen3-VL-8B-Instruct-GGUF" does
// not, because it does not start with "Qwen3-8B" at all.
func isGGUFVariantOf(candidate, model string) bool {
	c, m := strings.ToLower(candidate), strings.ToLower(model)
	if !strings.HasPrefix(c, m) {
		return false
	}
	rest := strings.Trim(strings.TrimPrefix(c, m), "-._")
	if rest == "" {
		return true
	}
	// Whatever follows the model name must look like conversion metadata, not
	// a different variant such as "instruct" or "vl".
	for _, part := range strings.FieldsFunc(rest, func(r rune) bool {
		return r == '-' || r == '.' || r == '_'
	}) {
		switch {
		case part == "gguf", part == "ggml":
		case strings.HasPrefix(part, "i1"), strings.HasPrefix(part, "imat"):
		case strings.HasPrefix(part, "q") || strings.HasPrefix(part, "iq"):
		default:
			return false
		}
	}
	return true
}

func sortCandidates(c []GGUFCandidate) {
	for i := 1; i < len(c); i++ {
		for j := i; j > 0; j-- {
			a, b := c[j-1], c[j]
			better := (b.exact && !a.exact) ||
				(b.exact == a.exact && b.Official && !a.Official) ||
				(b.exact == a.exact && b.Official == a.Official && b.Downloads > a.Downloads)
			if !better {
				break
			}
			c[j-1], c[j] = b, a
		}
	}
}

// QuantCandidate is a pre-quantized publication of a model.
type QuantCandidate struct {
	Repo      string `json:"repo"`
	Quant     string `json:"quant"`
	Downloads int64  `json:"downloads"`
	Official  bool   `json:"official"`
}

// quantSuffixes maps the decoration people put on a repo name onto the
// quantization scheme it denotes. Only formats vLLM can serve are listed:
// MLX is Apple-only and GGUF belongs to llama.cpp, so both are absent and are
// rejected as "not a quantization" by the variant test below.
var quantSuffixes = map[string]string{
	"fp8": "fp8", "w8a8": "fp8", "fp8-dynamic": "fp8", "fp8dynamic": "fp8",
	"awq": "awq", "gptq": "gptq", "w4a16": "awq", "int4": "awq", "4bit": "awq",
	"nvfp4": "nvfp4", "fp4": "nvfp4", "mxfp4": "nvfp4",
}

// ResolveQuantized finds pre-quantized publications of a model.
//
// This exists because quantization is a property of a checkpoint, not a runtime
// switch. Telling vLLM "--quantization awq" against a bf16 repository does not
// quantize anything; it looks for an AWQ config that is not there and exits.
// So when a model does not fit at its native precision, the answer is a
// different repository, and the operator should not have to go and find it.
func (c *Client) ResolveQuantized(ctx context.Context, id string) ([]QuantCandidate, error) {
	owner, name, ok := strings.Cut(id, "/")
	if !ok {
		return nil, fmt.Errorf("model id must be in owner/name form, got %q", id)
	}

	var results []struct {
		ModelID   string `json:"modelId"`
		Downloads int64  `json:"downloads"`
	}
	url := fmt.Sprintf("%s/api/models?search=%s&sort=downloads&direction=-1&limit=50", hfAPI, name)
	if err := c.getJSON(ctx, url, &results); err != nil {
		return nil, err
	}

	seen := map[string]bool{}
	var out []QuantCandidate
	for _, r := range results {
		if seen[r.ModelID] {
			continue
		}
		seen[r.ModelID] = true

		candOwner, candName, ok := strings.Cut(r.ModelID, "/")
		if !ok {
			continue
		}
		// mlx-community republishes everything in a format only Apple silicon
		// can read, and its repo names carry no marker saying so.
		if strings.EqualFold(candOwner, "mlx-community") {
			continue
		}
		quant, ok := quantVariantOf(candName, name)
		if !ok {
			continue
		}
		out = append(out, QuantCandidate{
			Repo: r.ModelID, Quant: quant, Downloads: r.Downloads,
			Official: strings.EqualFold(candOwner, owner),
		})
	}

	// The model's own publisher first, then by popularity. An official FP8 repo
	// is worth far more than a more-downloaded community 4-bit one: it is the
	// same weights, quantized by the people who trained them.
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Official != out[j].Official {
			return out[i].Official
		}
		return out[i].Downloads > out[j].Downloads
	})
	if len(out) == 0 {
		return nil, fmt.Errorf("no pre-quantized publication of %s found on HuggingFace", id)
	}
	return out, nil
}

// quantVariantOf reports whether a candidate repo is the requested model
// republished in a smaller format, and which format that is.
//
// The test is deliberately strict. HuggingFace search is fuzzy, and a query for
// "Qwen3.8-27B" returns dozens of abliterated, uncensored and merged
// derivatives. Those are different models with different behaviour; serving one
// in place of the requested model would be a silent substitution. So every
// component after the model name must be recognisably quantization metadata,
// and anything unrecognised rejects the candidate.
func quantVariantOf(candidate, model string) (string, bool) {
	c, m := strings.ToLower(candidate), strings.ToLower(model)
	if !strings.HasPrefix(c, m) {
		return "", false
	}
	rest := strings.Trim(strings.TrimPrefix(c, m), "-._")
	if rest == "" {
		return "", false // the original repo, not a quantization of it
	}
	quant := ""
	for _, part := range strings.FieldsFunc(rest, func(r rune) bool {
		return r == '-' || r == '.' || r == '_'
	}) {
		if q, ok := quantSuffixes[part]; ok {
			// "AWQ-INT4" names one scheme twice; keep the first, which is the
			// more specific kernel name.
			if quant == "" {
				quant = q
			}
			continue
		}
		// Bare precision markers that add nothing and change nothing.
		if part == "quantized" || part == "quant" || part == "vllm" {
			continue
		}
		return "", false
	}
	if quant == "" {
		return "", false
	}
	return quant, true
}

// quantConfig is the quantization block from a checkpoint's config.json.
type quantConfig struct {
	QuantMethod string `json:"quant_method"`
	// Bits is how AWQ and GPTQ record their width, directly on the block.
	Bits int `json:"bits"`
	// ConfigGroups is how compressed-tensors records it: per-target groups,
	// each naming a bit width and whether it is integer or float.
	ConfigGroups map[string]struct {
		Weights *struct {
			NumBits int    `json:"num_bits"`
			Type    string `json:"type"`
		} `json:"weights"`
	} `json:"config_groups"`
}

// scheme reduces a quantization block to the vocabulary the planner sizes and
// selects kernels with.
//
// "compressed-tensors" is a container rather than a format: the same
// quant_method covers 4-bit integer, 8-bit integer and fp8 checkpoints, and
// the width only appears inside config_groups. Treating the wrapper as a
// format meant a 4-bit 27B model was sized at 51.7 GiB instead of 15.5 GiB and
// declared not to fit on a card with room to spare.
func (q *quantConfig) scheme() string {
	method := strings.ToLower(q.QuantMethod)
	if method != "compressed-tensors" && method != "compressed_tensors" {
		if q.Bits == 8 && (method == "awq" || method == "gptq") {
			return "int8"
		}
		return method
	}
	for _, g := range q.ConfigGroups {
		if g.Weights == nil || g.Weights.NumBits == 0 {
			continue
		}
		isFloat := strings.EqualFold(g.Weights.Type, "float")
		switch {
		case g.Weights.NumBits <= 4 && isFloat:
			return "nvfp4"
		case g.Weights.NumBits <= 4:
			return "awq"
		case g.Weights.NumBits == 8 && isFloat:
			return "fp8"
		case g.Weights.NumBits == 8:
			return "int8"
		}
	}
	// Nothing legible inside. Report the wrapper, which the planner sizes
	// conservatively rather than optimistically.
	return "compressed-tensors"
}

// attentionLayout reports how many layers keep a growing KV cache, and whether
// this is a hybrid model.
//
// Qwen3.5 and its relatives interleave Mamba-style linear attention with full
// attention -- layer_types lists them, and full_attention_interval is the
// shorthand. A linear-attention layer holds a fixed-size recurrent state per
// sequence rather than a cache that grows with every token, so counting all
// layers overstates the KV cost by the interleave ratio: fourfold on a model
// with every fourth layer attending fully.
func (c *hfConfig) attentionLayout() (full int, hybrid bool) {
	if len(c.LayerTypes) > 0 {
		for _, t := range c.LayerTypes {
			switch strings.ToLower(t) {
			case "full_attention", "attention":
				full++
			default:
				hybrid = true
			}
		}
		if hybrid {
			return full, true
		}
		return full, false
	}
	if c.FullAttentionInterval > 1 && c.NumLayers > 0 {
		return c.NumLayers / c.FullAttentionInterval, true
	}
	if c.LinearConvKernelDim > 0 || c.MambaSSMDType != "" {
		// Known to be hybrid but the layout is not spelled out; assume every
		// layer caches, which errs towards too little concurrency rather than
		// too much.
		return c.NumLayers, true
	}
	return c.NumLayers, false
}
