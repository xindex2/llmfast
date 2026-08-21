package modelspec

import (
	"fmt"
	"math"
	"strings"
)

// Bytes per parameter for each supported weight format. The int4 and GGUF
// figures are above their nominal bit width because quantized formats also
// store per-group scales and zero points, which are easy to forget and produce
// an out-of-memory crash after the download rather than before it.
const (
	bytesBF16 = 2.0
	bytesFP8  = 1.0
	bytesINT4 = 0.60 // AWQ/GPTQ 4-bit plus scales
	bytesQ4KM = 0.58 // llama.cpp Q4_K_M, ~4.6 bits/weight
	bytesQ8_0 = 1.06 // llama.cpp Q8_0
	// KV cache is kept at fp16 by default in both engines.
	bytesKVElem = 2.0
	// vLLM reserves this share of VRAM by default; the rest is activations,
	// fragmentation and CUDA context.
	defaultGPUMemUtil = 0.90
	// Headroom left for the OS and page cache when planning a CPU node.
	cpuRAMHeadroomBytes = 4 << 30
	// Share of theoretical GPU bandwidth a good kernel actually achieves.
	gpuBandwidthEfficiency = 0.75
	// Single-stream decode ceiling. Past this, per-token launch overhead and
	// the sequential dependency between tokens dominate, not bandwidth.
	maxSingleStreamTPS = 200.0
)

// GPU is one accelerator on a node.
type GPU struct {
	Name      string `json:"name"`
	VRAMBytes int64  `json:"vram_bytes"`
	Index     int    `json:"index"`
}

// Node is an inference host's hardware, as reported by its agent.
type Node struct {
	Name          string `json:"name"`
	GPUs          []GPU  `json:"gpus"`
	CPUCores      int    `json:"cpu_cores"`
	CPUModel      string `json:"cpu_model"`
	RAMBytes      int64  `json:"ram_bytes"`
	DiskFreeBytes int64  `json:"disk_free_bytes"`
	// MemBandwidthGBs is the node's measured or estimated main-memory
	// bandwidth. It is the binding constraint for CPU inference, so an honest
	// number here is what makes the throughput estimate meaningful.
	MemBandwidthGBs float64 `json:"mem_bandwidth_gbs"`
	HasNVMe         bool    `json:"has_nvme"`
	// GPUFraction is the share of each physical GPU this node actually gets.
	// Providers increasingly sell MIG partitions and vGPU slices, where the
	// VRAM figure is only half the story: a 1/2 slice also has roughly half the
	// compute, so throughput scales down with it. 0 or 1 means a whole GPU.
	GPUFraction float64 `json:"gpu_fraction,omitempty"`
}

// gpuShare returns the fraction of each GPU available, defaulting to a whole one.
func (n *Node) gpuShare() float64 {
	if n.GPUFraction <= 0 || n.GPUFraction > 1 {
		return 1
	}
	return n.GPUFraction
}

func (n *Node) HasGPU() bool { return len(n.GPUs) > 0 }

func (n *Node) TotalVRAM() int64 {
	var t int64
	for _, g := range n.GPUs {
		t += g.VRAMBytes
	}
	return t
}

// Plan is a concrete, runnable serving configuration.
type Plan struct {
	Engine       string `json:"engine"`       // "vllm" or "llamacpp"
	Quantization string `json:"quantization"` // fp8, bf16, awq, q4_k_m, q8_0
	// KVCacheDType is the storage format for the KV cache: "auto" (fp16) or
	// "fp8". fp8 halves the per-token cost and so roughly doubles concurrency.
	KVCacheDType   string `json:"kv_cache_dtype"`
	TensorParallel int    `json:"tensor_parallel"`
	MaxModelLen    int    `json:"max_model_len"`
	MaxNumSeqs     int    `json:"max_num_seqs"`

	WeightBytes   int64 `json:"weight_bytes"`
	KVBytesPerTok int64 `json:"kv_bytes_per_token"`
	KVBudgetBytes int64 `json:"kv_budget_bytes"`
	DiskBytes     int64 `json:"disk_bytes"`

	// EstTokensPerSec is a single-stream ceiling, not a promise. On CPU it is
	// memory-bandwidth bound and reasonably predictable; on GPU it is a rough
	// lower bound because batching and kernel quality dominate.
	EstTokensPerSec float64 `json:"est_tokens_per_sec"`

	Fits     bool     `json:"fits"`
	Warnings []string `json:"warnings"`
	Blockers []string `json:"blockers"`

	// Viable reports whether this plan could serve OpenRouter traffic at a
	// competitive throughput, which is a much higher bar than merely fitting.
	Viable        bool   `json:"viable"`
	ViabilityNote string `json:"viability_note"`
}

// quantBytes maps a quantization name to bytes per parameter.
func quantBytes(q string) float64 {
	switch strings.ToLower(q) {
	case "fp32", "float32":
		return 4.0
	case "bf16", "fp16", "float16", "bfloat16":
		return bytesBF16
	case "fp8":
		return bytesFP8
	case "awq", "gptq", "int4", "nvfp4", "fp4":
		return bytesINT4
	case "q4_k_m":
		return bytesQ4KM
	case "q8_0":
		return bytesQ8_0
	}
	return bytesBF16
}

// KVBytesPerToken is the per-token KV cache cost for one sequence.
//
// Two tensors (K and V) per layer, sized by the number of key/value heads
// rather than attention heads: grouped-query attention shrinks this by the
// head ratio, which is why a modern 32B model is far cheaper to serve at long
// context than its parameter count suggests.
func (i *Info) KVBytesPerToken() int64 {
	if i.NumLayers == 0 || i.NumKVHeads == 0 || i.HeadDim == 0 {
		return 0
	}
	return int64(2 * float64(i.NumLayers) * float64(i.NumKVHeads) * float64(i.HeadDim) * bytesKVElem)
}

// PlanFor produces the best plan for running a model on a node.
//
// wantContext of 0 means "use the model's full context". The planner will trim
// it when the KV cache cannot support useful concurrency at that length,
// because advertising a context you can only serve one request at a time is
// worse business than advertising a shorter one you can serve sixty.
func PlanFor(info *Info, node *Node, wantContext int) *Plan {
	if node.HasGPU() {
		return planGPU(info, node, wantContext)
	}
	return planCPU(info, node, wantContext)
}

func planGPU(info *Info, node *Node, wantContext int) *Plan {
	p := &Plan{Engine: "vllm", MaxNumSeqs: 256}

	// Prefer FP8 on hardware that supports it: it halves weight memory at
	// little quality cost, and the freed VRAM becomes KV cache, which is what
	// actually limits concurrency.

	nGPU := len(node.GPUs)
	// Multi-head Latent Attention compresses the KV cache into a single shared
	// latent, so the "KV heads must divide the TP size" rule does not apply:
	// vLLM replicates the latent across ranks instead of sharding it. Applying
	// the rule anyway would force TP=1 on DeepSeek models and make every
	// multi-GPU node look unusable.
	kvHeadsForTP := info.NumKVHeads
	if info.IsMLA {
		kvHeadsForTP = 0
	}
	tp := largestValidTP(nGPU, kvHeadsForTP)
	a := nodeArch(node)
	ladder := precisionLadder(a)

	// Only the GPUs the model is actually sharded across contribute memory.
	// Summing the whole node regardless of TP was wrong in the worst possible
	// direction: it reported a 283 GiB model as fitting on an 8-GPU node at
	// TP=1, where it would in fact have to live inside one 141 GiB card.
	usable := int64(float64(perGPUVRAM(node)*int64(tp)) * defaultGPUMemUtil)
	quant := ""
	for _, candidate := range ladder {
		weights := int64(float64(info.Params) * quantBytes(candidate))
		if usable-weights > 0 {
			quant = candidate
			p.WeightBytes = weights
			p.KVBudgetBytes = usable - weights
			break
		}
	}
	p.TensorParallel = tp
	if quant == "" {
		smallest := ladder[len(ladder)-1]
		p.Quantization = smallest
		p.WeightBytes = int64(float64(info.Params) * quantBytes(smallest))
		reason := fmt.Sprintf(
			"weights need %s even at %s, but only %s is addressable (%d of %d GPU(s) at TP=%d)",
			humanBytes(p.WeightBytes), smallest,
			humanBytes(perGPUVRAM(node)*int64(tp)), tp, nGPU, tp)
		if tp < nGPU {
			reason += fmt.Sprintf("; tensor parallelism is capped at %d because the model has %d KV head(s)",
				tp, info.NumKVHeads)
		}
		if !a.int4 {
			reason += fmt.Sprintf("; %s has no 4-bit kernels, so there is no smaller option", a.name)
		}
		p.Blockers = append(p.Blockers, reason)
		// Nothing can run, so the capacity fields would be meaningless. Leaving
		// the default 256 in place reads as "256 concurrent" next to a blocker.
		p.MaxNumSeqs, p.MaxModelLen, p.KVBudgetBytes, p.DiskBytes = 0, 0, 0, 0
		p.Fits = false
		return p
	}
	p.Quantization = quant

	// Store the KV cache at fp8 where the hardware allows. It halves the
	// per-token cost, and KV cache -- not weights -- is what limits concurrency
	// on every card we have looked at. The trade is a small accuracy loss that
	// grows with context length, so it is worth measuring on long prompts.
	p.KVCacheDType = "auto"
	p.KVBytesPerTok = info.KVBytesPerToken()
	if a.fp8KVCache {
		p.KVCacheDType = "fp8"
		p.KVBytesPerTok /= 2
	}
	p.MaxModelLen = chooseContext(info, wantContext)
	p.DiskBytes = downloadBytes(info, quant)

	if p.KVBytesPerTok == 0 {
		// The geometry could not be read, so concurrency cannot be derived.
		// Leaving the default in place would present a made-up number with the
		// same confidence as a computed one.
		p.MaxNumSeqs = 1
		p.Warnings = append(p.Warnings,
			"could not read this model's attention geometry, so the KV cache and concurrency "+
				"could not be computed; the figures above cover weights only")
	}
	if p.KVBytesPerTok > 0 {
		totalKVTokens := p.KVBudgetBytes / p.KVBytesPerTok
		// Trim context until at least this many sequences fit concurrently. A
		// provider serving one request at a time cannot hold a routing slot.
		// Only the net result is reported: the intermediate halvings are an
		// implementation detail, not something an operator needs to read.
		const minConcurrency = 8
		original := p.MaxModelLen
		for p.MaxModelLen > 8192 && totalKVTokens/int64(p.MaxModelLen) < minConcurrency {
			p.MaxModelLen /= 2
		}
		if p.MaxModelLen != original {
			p.Warnings = append(p.Warnings, fmt.Sprintf(
				"context reduced from %d to %d so at least %d requests fit concurrently; "+
					"more VRAM would let you advertise the full window",
				original, p.MaxModelLen, minConcurrency))
		}
		if seqs := int(totalKVTokens / int64(p.MaxModelLen)); seqs < p.MaxNumSeqs {
			p.MaxNumSeqs = maxInt(1, seqs)
		}
	}

	if p.MaxNumSeqs < 4 {
		p.Warnings = append(p.Warnings,
			"very low concurrency; this GPU is undersized for this model")
	}
	if quant == "awq" {
		msg := "4-bit quantization needed to fit; expect some quality loss and find a prequantized AWQ/GPTQ repo"
		if !a.fp8 {
			msg += fmt.Sprintf(". %s has no hardware FP8, so 4-bit is the only step below full precision", a.name)
		}
		p.Warnings = append(p.Warnings, msg)
	}
	if p.KVCacheDType == "fp8" {
		p.Warnings = append(p.Warnings,
			"KV cache is planned at fp8, which halves its per-token cost and roughly doubles "+
				"concurrency. Accuracy loss is small but grows with context length; measure it "+
				"on long prompts before publishing, and fall back to --kv-cache-dtype auto if "+
				"quality suffers")
	}
	if quant == "fp16" && strings.Contains(strings.ToLower(info.DType), "bf16") {
		p.Warnings = append(p.Warnings, fmt.Sprintf(
			"%s cannot run bfloat16, so this bf16 checkpoint must be narrowed to fp16; "+
				"some models overflow and produce NaNs when converted", a.name))
	}
	if !a.flashAttention2 {
		p.Warnings = append(p.Warnings, fmt.Sprintf(
			"%s predates FlashAttention-2, so vLLM falls back to a slower attention backend; "+
				"expect throughput below what the raw specifications suggest", a.name))
	}
	if strings.HasPrefix(a.name, "Pascal") {
		p.Warnings = append(p.Warnings,
			"Pascal is generations before vLLM's supported baseline; treat any plan on it "+
				"as unproven and verify the model loads at all before paying for a month")
	}
	if a.crippledFP16 {
		p.Warnings = append(p.Warnings,
			"this card runs fp16 at 1/64 the rate of fp32, so the plan above uses fp32 and needs "+
				"twice the memory. llama.cpp with a GGUF build is the practical route on these "+
				"cards, at much lower concurrency than vLLM")
	}
	if !node.HasNVMe {
		p.Warnings = append(p.Warnings,
			"no NVMe detected; weight loading from spinning disk will make restarts slow")
	}
	if info.HasVision || info.HasAudio {
		// The encoder's weights are inside the parameter count, but encoding a
		// batch of images or video frames allocates activation memory on top of
		// the KV cache, and a single video can expand into thousands of tokens.
		p.Warnings = append(p.Warnings,
			"multimodal model: the vision/audio encoder needs activation memory beyond the "+
				"figures above, and media expands into many tokens, so leave headroom and "+
				"measure with real image or video requests before advertising capacity")
	}
	if node.DiskFreeBytes > 0 && node.DiskFreeBytes < p.DiskBytes {
		p.Blockers = append(p.Blockers, fmt.Sprintf(
			"needs about %s of disk for weights, only %s free",
			humanBytes(p.DiskBytes), humanBytes(node.DiskFreeBytes)))
	}
	// System RAM matters on a GPU node too: weights are read from disk into
	// host memory before being moved to the device. Memory-mapped loading makes
	// a shortfall survivable rather than fatal, but a host with a fraction of
	// the model's size will thrash, and on a small container it can be killed
	// outright by the OOM reaper part-way through loading.
	if node.RAMBytes > 0 && node.RAMBytes < p.WeightBytes {
		msg := fmt.Sprintf(
			"host has %s of RAM for %s of weights; loading streams through host memory, "+
				"so expect slow starts and verify it completes before relying on it",
			humanBytes(node.RAMBytes), humanBytes(p.WeightBytes))
		if node.RAMBytes*3 < p.WeightBytes {
			p.Blockers = append(p.Blockers, msg+" — at less than a third of the model size this "+
				"usually fails to load at all")
		} else {
			p.Warnings = append(p.Warnings, msg)
		}
	}

	// Single-stream decode speed, from the same physics as the CPU estimate:
	// bandwidth divided by the bytes read per token. Tensor parallelism adds
	// bandwidth; a fractional slice removes it.
	bw, known := gpuBandwidthGBs(node.GPUs[0].Name)
	effective := bw * float64(p.TensorParallel) * node.gpuShare()
	active := float64(info.Params) * quantBytes(quant)
	if info.IsMoE && info.ActiveParams > 0 {
		active = float64(info.ActiveParams) * quantBytes(quant)
	}
	if active > 0 {
		p.EstTokensPerSec = effective * 1e9 / active * gpuBandwidthEfficiency
		// Below a certain model size the bottleneck stops being bandwidth and
		// becomes kernel launch latency and the sequential dependency between
		// tokens, so the estimate is capped rather than extrapolated.
		if p.EstTokensPerSec > maxSingleStreamTPS {
			p.EstTokensPerSec = maxSingleStreamTPS
		}
	}
	if !known {
		p.Warnings = append(p.Warnings, fmt.Sprintf(
			"memory bandwidth for %q is not in our table, so the speed estimate assumes a "+
				"mid-range %.0f GB/s; check the card's specification before relying on it",
			node.GPUs[0].Name, bw))
	}
	// Where a smaller weight format would be substantially faster, say so. The
	// ladder picks the highest quality that fits, which is the right default for
	// a user of the model -- but a provider is scored on throughput, and reading
	// a third as many bytes per token is a large enough win to be a deliberate
	// choice rather than an accident.
	if idx := indexOf(ladder, quant); idx >= 0 && idx+1 < len(ladder) {
		next := ladder[idx+1]
		if ratio := quantBytes(quant) / quantBytes(next); ratio >= 1.5 {
			p.Warnings = append(p.Warnings, fmt.Sprintf(
				"%s weights read %.1fx more per token than %s would: dropping to %s would raise "+
					"single-stream throughput to roughly %.0f tok/s and free VRAM for more "+
					"concurrency, at some quality cost. Worth measuring both",
				quant, ratio, next, next, p.EstTokensPerSec*ratio))
		}
	}
	if node.gpuShare() < 1 {
		p.Warnings = append(p.Warnings, fmt.Sprintf(
			"this is a %.0f%% slice of a GPU, not a whole one: the VRAM figure is only half the "+
				"story, since the slice also gets roughly that share of the compute and therefore "+
				"of the throughput. Compare the price against a whole card before committing",
			node.gpuShare()*100))
	}
	p.Fits = len(p.Blockers) == 0
	p.Viable = p.Fits && p.MaxNumSeqs >= 4
	if p.Viable {
		p.ViabilityNote = "competitive for OpenRouter traffic"
	} else if p.Fits {
		p.ViabilityNote = "runs, but concurrency is too low to hold a routing slot"
	}
	return p
}

func planCPU(info *Info, node *Node, wantContext int) *Plan {
	// llama.cpp rather than vLLM: vLLM's CPU backend wants AVX-512, and
	// llama.cpp is both faster on older cores and far easier to operate.
	p := &Plan{Engine: "llamacpp", TensorParallel: 1, MaxNumSeqs: 4}

	quant := "q4_k_m"
	p.Quantization = quant
	p.WeightBytes = int64(float64(info.Params) * quantBytes(quant))
	p.KVBytesPerTok = info.KVBytesPerToken()
	p.MaxModelLen = chooseContext(info, wantContext)
	if p.MaxModelLen > 32768 {
		p.MaxModelLen = 32768
		p.Warnings = append(p.Warnings, "context capped at 32k; CPU prefill on long prompts is very slow")
	}
	p.DiskBytes = downloadBytes(info, quant)

	usable := node.RAMBytes - cpuRAMHeadroomBytes
	p.KVBudgetBytes = usable - p.WeightBytes
	if p.KVBudgetBytes <= 0 {
		p.Blockers = append(p.Blockers, fmt.Sprintf(
			"weights need %s at 4-bit but the node has %s of RAM (leaving %s for the OS)",
			humanBytes(p.WeightBytes), humanBytes(node.RAMBytes), humanBytes(cpuRAMHeadroomBytes)))
		p.Fits = false
		return p
	}
	if p.KVBytesPerTok > 0 {
		if seqs := int(p.KVBudgetBytes / p.KVBytesPerTok / int64(p.MaxModelLen)); seqs < p.MaxNumSeqs {
			p.MaxNumSeqs = maxInt(1, seqs)
		}
	}
	if node.DiskFreeBytes > 0 && node.DiskFreeBytes < p.DiskBytes {
		p.Blockers = append(p.Blockers, fmt.Sprintf("needs %s of disk, only %s free",
			humanBytes(p.DiskBytes), humanBytes(node.DiskFreeBytes)))
	}

	// CPU decoding reads every active weight once per token, so throughput is
	// memory bandwidth divided by weight size. This estimate is usually within
	// a factor of about 1.5 in practice.
	active := info.Params
	if info.IsMoE && info.ActiveParams > 0 {
		active = info.ActiveParams
	}
	activeBytes := float64(active) * quantBytes(quant)
	bw := node.MemBandwidthGBs
	if bw <= 0 {
		bw = 40 // conservative default for a DDR4 server
	}
	if activeBytes > 0 {
		p.EstTokensPerSec = (bw * 1e9) / activeBytes * 0.65 // efficiency factor
	}

	if p.MaxNumSeqs > 1 {
		// Worth stating explicitly, because the concurrency figure reads like
		// throughput multiplies with it. On a GPU it roughly does: decoding is
		// compute bound and batching amortises the weight read across streams.
		// On CPU the weight read *is* the bottleneck, so N concurrent requests
		// share one stream's throughput rather than adding to it.
		p.Warnings = append(p.Warnings, fmt.Sprintf(
			"%d concurrent requests share this throughput, they do not multiply it: "+
				"CPU decoding is memory-bandwidth bound, so batching buys far less than it does on a GPU",
			p.MaxNumSeqs))
	}

	p.Fits = len(p.Blockers) == 0
	// OpenRouter deprioritizes any endpoint more than 1.5 standard deviations
	// below the peer median throughput. Peers on GPUs run at 50+ tok/s, so
	// anything far below that will not hold a routing slot regardless of price.
	const competitiveTPS = 25
	p.Viable = p.Fits && p.EstTokensPerSec >= competitiveTPS
	switch {
	case !p.Fits:
		p.ViabilityNote = "does not fit on this node"
	case p.Viable:
		p.ViabilityNote = fmt.Sprintf("about %.0f tok/s; marginal but worth measuring", p.EstTokensPerSec)
	default:
		p.ViabilityNote = fmt.Sprintf(
			"about %.1f tok/s on CPU. GPU providers serve this at 50-100+ tok/s, so OpenRouter would "+
				"deprioritize this endpoint. Fine for testing the pipeline, not for live traffic.",
			p.EstTokensPerSec)
		p.Warnings = append(p.Warnings, "CPU inference is not competitive for live OpenRouter traffic")
	}
	return p
}

// downloadBytes estimates the disk needed for the weights actually fetched.
//
// bf16 and fp8 plans download the original checkpoint at its published
// precision, because vLLM quantizes to fp8 on load. A 4-bit or GGUF plan
// downloads a separate prequantized repository, which is far smaller — using
// the bf16 figure there would overstate the disk requirement threefold and
// could block a plan that comfortably fits.
func downloadBytes(info *Info, quant string) int64 {
	const tokenizerAndConfigOverhead = 2 << 30
	var perParam float64
	// A checkpoint published already quantized downloads at that size. Assuming
	// bf16 would double the reported disk requirement for models like
	// DeepSeek-V4, which ships in fp8.
	if info.PublishedQuant != "" && (quant == "fp8" || quant == "bf16" || quant == "fp16") {
		return int64(float64(info.Params)*quantBytes(info.PublishedQuant)) + tokenizerAndConfigOverhead
	}
	switch strings.ToLower(quant) {
	case "awq", "gptq", "int4", "nvfp4", "fp4":
		perParam = bytesINT4
	case "q4_k_m":
		perParam = bytesQ4KM
	case "q8_0":
		perParam = bytesQ8_0
	default:
		// fp8 included: vLLM quantizes the bf16 checkpoint at load time, so the
		// download is still full size.
		perParam = bytesBF16
	}
	return int64(float64(info.Params)*perParam) + tokenizerAndConfigOverhead
}

// chooseContext picks a starting context length.
func chooseContext(info *Info, want int) int {
	if want > 0 {
		if info.MaxPositions > 0 && want > info.MaxPositions {
			return info.MaxPositions
		}
		return want
	}
	if info.MaxPositions > 0 {
		return info.MaxPositions
	}
	return 8192
}

// largestValidTP returns the biggest tensor-parallel size that divides both the
// GPU count and the KV head count. vLLM rejects a TP size that does not divide
// the KV heads evenly, and the failure message is not obvious.
func largestValidTP(gpus, kvHeads int) int {
	if gpus <= 1 {
		return 1
	}
	for tp := gpus; tp > 1; tp-- {
		if gpus%tp == 0 && (kvHeads == 0 || kvHeads%tp == 0) {
			return tp
		}
	}
	return 1
}

// arch describes the capabilities of a GPU generation that change the plan.
//
// These are hard constraints, not preferences. Planning bf16 on Volta or 4-bit
// on a card without the kernels produces a configuration that fails at load
// time, after the operator has already paid for the machine and waited out a
// weight download.
type arch struct {
	name string
	// bf16 requires Ampere (SM80) or newer. Volta and Turing must use fp16,
	// which matters because most modern checkpoints are published in bf16 and
	// can overflow when narrowed to fp16.
	bf16 bool
	// fp8 requires Ada, Hopper or Blackwell.
	fp8 bool
	// int4 covers AWQ and GPTQ, whose kernels need Turing (SM75) or newer.
	// There is therefore no 4-bit escape hatch on Volta at all.
	int4 bool
	// flashAttention2 needs SM80. Older cards fall back to a slower backend,
	// which costs throughput beyond what the raw specs suggest.
	flashAttention2 bool
	// crippledFP16 marks consumer Pascal (GTX 10-series), where half precision
	// runs at 1/64 the rate of single precision. The datacenter P100 is the
	// exception: it has full-rate fp16. On a crippled part the only workable
	// path is fp32, which doubles the memory a model needs.
	crippledFP16 bool
	// fp8KVCache marks hardware where vLLM can store the KV cache at fp8
	// instead of fp16. It needs compute capability 8.0, so Ampere and newer.
	// This is the single largest lever on concurrency: it halves the per-token
	// KV cost, which roughly doubles how many requests fit in the same VRAM.
	fp8KVCache bool
}

func archOf(name string) arch {
	n := strings.ToUpper(name)
	switch {
	// Pascal (2016). No bf16, no fp8, and no 4-bit kernels either, since AWQ
	// and GPTQ need Turing. fp16 throughput is also poor: on most Pascal parts
	// half precision runs at a fraction of fp32 rate.
	case strings.Contains(n, "P100"):
		// GP100 is the one Pascal part with full-rate fp16.
		return arch{name: "Pascal"}
	case strings.Contains(n, "P40"), strings.Contains(n, "P4 "),
		strings.Contains(n, "GTX 10"), strings.Contains(n, "TITAN X"):
		return arch{name: "Pascal (consumer)", crippledFP16: true}
	case strings.Contains(n, "V100"), strings.Contains(n, "TITAN V"):
		return arch{name: "Volta"}
	case strings.Contains(n, "T4"), strings.Contains(n, "RTX 20"),
		strings.Contains(n, "QUADRO RTX"):
		return arch{name: "Turing", int4: true}
	}
	if gpuHasFP8(name) {
		return arch{name: "Ada or newer", bf16: true, fp8: true, int4: true,
			flashAttention2: true, fp8KVCache: true}
	}
	// Everything else we recognise as a datacenter or workstation part is
	// treated as Ampere: bf16 and 4-bit, no hardware fp8 for weights, but the
	// fp8 KV cache still works because that is storage, not arithmetic.
	return arch{name: "Ampere", bf16: true, int4: true,
		flashAttention2: true, fp8KVCache: true}
}

// nodeArch returns the least capable architecture on the node, since a mixed
// node can only run what all of its cards support.
func nodeArch(node *Node) arch {
	out := arch{name: "unknown", bf16: true, fp8: true, int4: true, flashAttention2: true}
	for i, g := range node.GPUs {
		a := archOf(g.Name)
		if i == 0 {
			out = a
			continue
		}
		out.bf16 = out.bf16 && a.bf16
		out.fp8 = out.fp8 && a.fp8
		out.int4 = out.int4 && a.int4
		out.flashAttention2 = out.flashAttention2 && a.flashAttention2
		if a.name != out.name {
			out.name = "mixed"
		}
	}
	return out
}

// precisionLadder lists the weight formats to try, best first, for a given
// architecture. Each step trades quality for memory.
func precisionLadder(a arch) []string {
	switch {
	case a.fp8:
		// fp8 is near-lossless and halves weight memory, so it is preferred
		// over bf16 outright; the freed VRAM becomes KV cache.
		return []string{"fp8", "awq"}
	case a.bf16 && a.int4:
		return []string{"bf16", "awq"}
	case a.bf16:
		return []string{"bf16"}
	case a.int4:
		return []string{"fp16", "awq"}
	case a.crippledFP16:
		// Half precision is 64x slower than single here, so fp32 is the only
		// usable format -- and it needs twice the memory of fp16.
		return []string{"fp32"}
	}
	// Volta: fp16 only, with no smaller fallback.
	return []string{"fp16"}
}

// supportsFP8 reports whether every GPU on the node has hardware FP8.
//
// Ada Lovelace, Hopper and Blackwell do. Ampere (A100, A40, A6000, A5000 and
// the rest of the A-series) does not: asking vLLM for FP8 there falls back to a
// slow emulated path, so those cards must use bf16 or a 4-bit scheme instead.
//
// The card names are matched loosely because providers spell them
// inconsistently -- "NVIDIA A40", "RTX 4000 Ada", "RTX PRO 4500" and
// "L40S" all appear in the wild for the same silicon families.
func supportsFP8(node *Node) bool {
	for _, g := range node.GPUs {
		if !gpuHasFP8(g.Name) {
			return false
		}
	}
	return len(node.GPUs) > 0
}

func gpuHasFP8(name string) bool {
	n := strings.ToUpper(name)
	// Ampere A-series has no FP8. Check it first, because "RTX A4000" would
	// otherwise be caught by a looser rule below.
	for _, ampere := range []string{"A100", "A40", "A30", "A10", "A6000", "A5500",
		"A5000", "A4500", "A4000", "A2000"} {
		if strings.Contains(n, ampere) {
			return false
		}
	}
	switch {
	// Hopper and Blackwell datacenter parts.
	case strings.Contains(n, "H100"), strings.Contains(n, "H200"), strings.Contains(n, "H800"),
		strings.Contains(n, "B100"), strings.Contains(n, "B200"), strings.Contains(n, "GB200"):
		return true
	// Ada Lovelace: the L-series, the GeForce 40-series, and every "... Ada"
	// workstation card (2000/4000/4500/5000/5880/6000).
	case strings.Contains(n, "L40"), strings.Contains(n, "L4"),
		strings.Contains(n, "ADA"), strings.Contains(n, "4090"):
		return true
	// Blackwell consumer and RTX PRO workstation cards.
	case strings.Contains(n, "5090"), strings.Contains(n, "RTX PRO"):
		return true
	}
	return false
}

// gpuBandwidthGBs returns a GPU's memory bandwidth in GB/s.
//
// Decode speed is bandwidth-bound in exactly the way CPU inference is: every
// generated token reads the active weights once. Bandwidth therefore predicts
// single-stream throughput far better than the card's marketing tier does, and
// the two diverge sharply on parts built for something other than compute --
// the A16 is a virtual-desktop card whose bandwidth is a fifth of an A40's
// despite both being Ampere.
//
// Figures are published specifications. An unrecognised card falls back to a
// deliberately mid-range guess and is flagged, rather than silently assuming
// the best case.
func gpuBandwidthGBs(name string) (float64, bool) {
	n := strings.ToUpper(name)
	table := []struct {
		match string
		bw    float64
	}{
		{"B200", 8000}, {"H200", 4800}, {"H100 SXM", 3350}, {"H100", 2000},
		{"A100-SXM", 2039}, {"A100 80", 2039}, {"A100", 1555},
		{"RTX PRO 6000", 1792}, {"5090", 1792},
		{"6000 ADA", 960}, {"L40", 864}, {"A6000", 768}, {"A40", 696},
		{"A10G", 600}, {"A10", 600}, {"L4", 300},
		// A16 is four small GA107 dies on one board for virtual desktops.
		{"A16", 200}, {"A2", 200},
		{"4090", 1008}, {"3090 TI", 1008}, {"3090", 936}, {"3080 TI", 912},
		{"4080", 717}, {"3080", 760}, {"A5000", 768}, {"A4500", 640},
		{"A4000", 448}, {"4070", 504}, {"3070", 448}, {"2080", 448},
		{"V100", 900}, {"P100", 732}, {"GTX 10", 484},
	}
	for _, e := range table {
		if strings.Contains(n, e.match) {
			return e.bw, true
		}
	}
	return 600, false
}

// perGPUVRAM returns the memory of the smallest GPU on the node. Tensor
// parallelism shards evenly, so a mixed node is limited by its smallest card.
func perGPUVRAM(node *Node) int64 {
	var min int64
	for _, g := range node.GPUs {
		if min == 0 || g.VRAMBytes < min {
			min = g.VRAMBytes
		}
	}
	return min
}

func humanBytes(b int64) string {
	const unit = 1024
	if b < unit {
		return fmt.Sprintf("%d B", b)
	}
	div, exp := int64(unit), 0
	for n := b / unit; n >= unit && exp < 4; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %ciB", float64(b)/float64(div), "KMGTP"[exp])
}

func indexOf(list []string, want string) int {
	for i, v := range list {
		if v == want {
			return i
		}
	}
	return -1
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// SuggestPricing proposes per-token prices in USD as decimal strings.
//
// The shape follows what the market charges: price scales with active
// parameters (the real compute cost), completion costs roughly 3x prompt
// because decoding is memory-bound and cannot be batched as efficiently as
// prefill, and cached prompt tokens are priced at a small fraction since they
// skip prefill entirely. These are a starting point to undercut on, not a
// recommendation to charge.
func SuggestPricing(info *Info) (prompt, completion, cached string) {
	active := float64(info.Params)
	if info.IsMoE && info.ActiveParams > 0 {
		active = float64(info.ActiveParams)
	}
	b := active / 1e9
	// Roughly $0.03 per million prompt tokens per 10B active parameters,
	// with a floor so tiny models are not priced at zero.
	perMPrompt := math.Max(0.02, 0.03*b/10)
	perMCompletion := perMPrompt * 3
	perMCached := perMPrompt * 0.2

	f := func(perMillion float64) string {
		return strings.TrimRight(strings.TrimRight(
			fmt.Sprintf("%.11f", perMillion/1e6), "0"), ".")
	}
	return f(perMPrompt), f(perMCompletion), f(perMCached)
}
