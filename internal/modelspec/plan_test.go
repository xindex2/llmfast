package modelspec

import (
	"strconv"
	"strings"
	"testing"
)

// qwen32B mirrors the real Qwen/Qwen3-32B config, verified against the
// HuggingFace API. Using real numbers means a regression in the sizing maths
// shows up as a wrong answer about actual hardware.
// qwen32BFP8 is the same model as qwen32B, republished at fp8 by its authors.
// Most GPUs worth serving from can only hold a model this size in that form,
// and it is a distinct fixture rather than a flag because that is exactly the
// distinction the planner now draws: quantization is a property of a
// checkpoint, not a setting.
func qwen32BFP8() *Info {
	i := qwen32B()
	i.PublishedQuant = "fp8"
	return i
}

func qwen32B() *Info {
	return &Info{
		ID: "Qwen/Qwen3-32B", Params: 32_762_123_264, DType: "BF16",
		Architecture: "Qwen3ForCausalLM", MaxPositions: 40960,
		// head_dim is 128 as declared in the real config, NOT hidden_size /
		// num_heads (which would give 80). See TestDeclaredHeadDimWins.
		HiddenSize: 5120, NumLayers: 64, NumHeads: 64, NumKVHeads: 8, HeadDim: 128,
	}
}

func gpuNode(name string, vramGB int64, count int) *Node {
	n := &Node{Name: name, CPUCores: 32, RAMBytes: 256 << 30,
		DiskFreeBytes: 4000 << 30, HasNVMe: true}
	for i := 0; i < count; i++ {
		n.GPUs = append(n.GPUs, GPU{Name: name, VRAMBytes: vramGB << 30, Index: i})
	}
	return n
}

func TestKVBytesPerToken(t *testing.T) {
	// 2 tensors x 64 layers x 8 KV heads x 128 head dim x 2 bytes = 262144.
	if got := qwen32B().KVBytesPerToken(); got != 262144 {
		t.Errorf("KV per token = %d, want 262144", got)
	}
	// Grouped-query attention is the whole point: without it the same model
	// would need 8x the KV cache.
	mha := qwen32B()
	mha.NumKVHeads = mha.NumHeads
	if mha.KVBytesPerToken() != 8*262144 {
		t.Errorf("MHA KV per token = %d, want 8x the GQA figure", mha.KVBytesPerToken())
	}
}

// TestDeclaredHeadDimWins guards a real trap. Qwen3-32B declares head_dim 128
// while hidden_size/num_heads gives 80. Deriving it would understate the KV
// cache by 1.6x, and the plan would promise concurrency the GPU cannot deliver
// -- surfacing as an out-of-memory crash under load rather than at startup.
func TestDeclaredHeadDimWins(t *testing.T) {
	info := qwen32B()
	derived := info.HiddenSize / info.NumHeads
	if info.HeadDim == derived {
		t.Fatal("fixture no longer exercises the divergence between declared and derived head_dim")
	}
	if got, wrong := info.KVBytesPerToken(), int64(2*info.NumLayers*info.NumKVHeads*derived*2); got == wrong {
		t.Errorf("KV sizing used the derived head_dim (%d) instead of the declared one (%d)",
			derived, info.HeadDim)
	}
}

func TestKVBytesUnknownGeometry(t *testing.T) {
	// A config we could not parse must report zero rather than a wrong number
	// that would silently produce a confident, incorrect plan.
	if got := (&Info{}).KVBytesPerToken(); got != 0 {
		t.Errorf("KV per token = %d, want 0 for unknown geometry", got)
	}
}

func TestWeightSizing(t *testing.T) {
	info := qwen32BFP8()
	p := PlanFor(info, gpuNode("NVIDIA H100 80GB HBM3", 80, 1), 0)
	// 32.76B params at FP8 is one byte each: about 30.5 GiB.
	wantGiB := 30.5
	gotGiB := float64(p.WeightBytes) / (1 << 30)
	if gotGiB < wantGiB-0.5 || gotGiB > wantGiB+0.5 {
		t.Errorf("FP8 weights = %.1f GiB, want about %.1f", gotGiB, wantGiB)
	}
	if p.Quantization != "fp8" {
		t.Errorf("quantization = %q, want fp8 on H100", p.Quantization)
	}
}

// TestFP8CheckpointsRunFromAmpereUp draws the line between fp8 arithmetic and
// fp8 storage. Only Ada and newer multiply in fp8, but Ampere can still *load*
// an fp8 checkpoint: the Marlin kernel unpacks it to fp16 after reading it, so
// the VRAM and memory-bandwidth savings survive and only the tensor-core
// speed-up is lost. Refusing fp8 on Ampere would rule out the single
// configuration that makes a 27B model serveable on a 48GB A40.
func TestFP8CheckpointsRunFromAmpereUp(t *testing.T) {
	for _, name := range []string{
		"NVIDIA A100-SXM4-80GB", "NVIDIA A40",
		"NVIDIA H100 80GB HBM3", "NVIDIA L40S", "NVIDIA GeForce RTX 4090",
	} {
		p := PlanFor(qwen32BFP8(), gpuNode(name, 80, 1), 0)
		if p.Quantization != "fp8" {
			t.Errorf("%s quantization = %q, want fp8 from the checkpoint", name, p.Quantization)
		}
		if len(p.Blockers) != 0 {
			t.Errorf("%s should run an fp8 checkpoint: %v", name, p.Blockers)
		}
	}

	// On Ampere the operator should be told the speed-up is absent, so a
	// disappointing benchmark is not a surprise.
	p := PlanFor(qwen32BFP8(), gpuNode("NVIDIA A100-SXM4-80GB", 80, 1), 0)
	if !strings.Contains(strings.Join(p.Warnings, " "), "no fp8 arithmetic") {
		t.Errorf("Ampere should warn that fp8 buys memory, not arithmetic: %v", p.Warnings)
	}
	// And on Ada it should not, because there the speed-up is real.
	q := PlanFor(qwen32BFP8(), gpuNode("NVIDIA L40S", 80, 1), 0)
	if strings.Contains(strings.Join(q.Warnings, " "), "no fp8 arithmetic") {
		t.Errorf("Ada has fp8 tensor cores and should not carry that warning: %v", q.Warnings)
	}

	// Turing and older cannot load fp8 at all, and must say so.
	if got := PlanFor(qwen32BFP8(), gpuNode("Tesla T4", 16, 1), 0); got.Fits {
		t.Error("Turing has no fp8 support and must not accept an fp8 checkpoint")
	}
}

// TestFourBitCheckpointFitsWhereFullPrecisionCannot covers the case that used
// to be handled by "stepping down": a 32B model cannot fit a 24GB card at fp8,
// but an AWQ publication of it can. The planner does not perform that step --
// it cannot, since no flag converts weights -- so the operator installs the
// 4-bit repository and the planner sizes it correctly.
func TestFourBitCheckpointFitsWhereFullPrecisionCannot(t *testing.T) {
	fp8 := PlanFor(qwen32BFP8(), gpuNode("NVIDIA GeForce RTX 4090", 24, 1), 0)
	if fp8.Fits {
		t.Error("32B at fp8 is ~30.5 GiB and cannot fit a 24GB card")
	}
	if !fp8.NeedsQuantized {
		t.Error("a 4-bit publication would fit, so the plan should point at one")
	}

	awq := qwen32B()
	awq.PublishedQuant = "awq"
	p := PlanFor(awq, gpuNode("NVIDIA GeForce RTX 4090", 24, 1), 0)
	if p.Quantization != "awq" {
		t.Errorf("quantization = %q, want awq from the checkpoint", p.Quantization)
	}
	if !p.Fits {
		t.Errorf("expected a 4-bit plan to fit on 24GB: %v", p.Blockers)
	}
	// With the fp8 KV cache this configuration is genuinely viable: 4-bit
	// weights plus a halved KV cost leave room for a usable number of
	// concurrent requests on a 24GB card.
	if p.MaxNumSeqs < 4 {
		t.Errorf("concurrency = %d; 4-bit weights with an fp8 KV cache should fit several",
			p.MaxNumSeqs)
	}
}

// TestFitsButNotViable keeps the distinction the planner exists to draw, using
// a configuration that is genuinely marginal rather than one the fp8 KV cache
// has since rescued.
func TestFitsButNotViable(t *testing.T) {
	// Qwen3.8-27B's real geometry on a 32GB card: 26 GiB of fp8 weights leave
	// under 3 GiB of KV budget, and its 262k context floor means the planner
	// cannot trim below 8k to buy concurrency back.
	info := qwen32BFP8()
	info.Params = 27_800_000_000
	info.MaxPositions = 262144
	node := gpuNode("NVIDIA GeForce RTX 5090", 32, 1)
	node.RAMBytes = 128 << 30

	p := PlanFor(info, node, 0)
	if !p.Fits {
		t.Fatalf("26 GiB of fp8 weights should fit in 32GB: %v", p.Blockers)
	}
	if p.Viable {
		t.Errorf("a %d-concurrent plan should not be called viable", p.MaxNumSeqs)
	}
}

func TestBlocksWhenTrulyTooLarge(t *testing.T) {
	// A 1T model cannot be served on one 24GB card at any precision.
	huge := qwen32B()
	huge.Params = 1_000_000_000_000
	p := PlanFor(huge, gpuNode("NVIDIA GeForce RTX 4090", 24, 1), 0)
	if p.Fits {
		t.Error("a 1T model must not be reported as fitting on 24GB")
	}
	if len(p.Blockers) == 0 {
		t.Error("expected a blocker explaining why it does not fit")
	}
}

// TestTensorParallelDividesKVHeads pins a constraint vLLM enforces with an
// error message that does not explain itself.
func TestTensorParallelDividesKVHeads(t *testing.T) {
	cases := []struct{ gpus, kvHeads, wantTP int }{
		{1, 8, 1},
		{2, 8, 2},
		{4, 8, 4},
		{8, 8, 8},
		{8, 4, 4}, // 8 GPUs but only 4 KV heads
		{3, 8, 1}, // 3 does not divide 8
		{4, 6, 2}, // 4 does not divide 6, but 2 does
	}
	for _, c := range cases {
		if got := largestValidTP(c.gpus, c.kvHeads); got != c.wantTP {
			t.Errorf("largestValidTP(%d gpus, %d kv heads) = %d, want %d",
				c.gpus, c.kvHeads, got, c.wantTP)
		}
	}
}

func TestTensorParallelAppliedInPlan(t *testing.T) {
	p := PlanFor(qwen32B(), gpuNode("NVIDIA H100 80GB HBM3", 80, 8), 0)
	if p.TensorParallel != 8 {
		t.Errorf("TP = %d, want 8", p.TensorParallel)
	}
	// More GPUs means more KV budget, so the full context should survive.
	if p.MaxModelLen != 40960 {
		t.Errorf("context = %d, want the full 40960 on 8 GPUs", p.MaxModelLen)
	}
}

func TestContextTrimmedForConcurrency(t *testing.T) {
	p := PlanFor(qwen32BFP8(), gpuNode("NVIDIA L40S", 48, 1), 0)
	if p.MaxModelLen >= 40960 {
		t.Errorf("context = %d, expected a reduction to preserve concurrency", p.MaxModelLen)
	}
	if p.MaxNumSeqs < 8 {
		t.Errorf("concurrency = %d, want at least 8 after trimming", p.MaxNumSeqs)
	}
	// The reduction must be explained exactly once, not once per halving.
	var trims int
	for _, w := range p.Warnings {
		if len(w) > 7 && w[:7] == "context" {
			trims++
		}
	}
	if trims != 1 {
		t.Errorf("got %d context warnings, want exactly 1", trims)
	}
}

func TestRequestedContextIsCappedAtModelMax(t *testing.T) {
	p := PlanFor(qwen32B(), gpuNode("NVIDIA H100 80GB HBM3", 80, 8), 999999)
	if p.MaxModelLen > 40960 {
		t.Errorf("context = %d, must not exceed the model's %d", p.MaxModelLen, 40960)
	}
}

func TestCPUPlanUsesLlamaCpp(t *testing.T) {
	node := &Node{Name: "cpu", CPUCores: 16, RAMBytes: 32 << 30,
		DiskFreeBytes: 1000 << 30, MemBandwidthGBs: 40}
	p := PlanFor(qwen32B(), node, 0)
	if p.Engine != "llamacpp" {
		t.Errorf("engine = %q, want llamacpp on a CPU node", p.Engine)
	}
	if p.Quantization != "q4_k_m" {
		t.Errorf("quantization = %q, want q4_k_m", p.Quantization)
	}
	// 32B at ~0.58 bytes/param is ~19GB, read once per token at 40GB/s with a
	// 0.65 efficiency factor: on the order of 1-2 tok/s.
	if p.EstTokensPerSec > 5 {
		t.Errorf("estimated %.1f tok/s for a 32B on CPU; that is implausibly optimistic", p.EstTokensPerSec)
	}
	if p.Viable {
		t.Error("a 1.4 tok/s endpoint must not be reported as viable for OpenRouter")
	}
	if p.ViabilityNote == "" {
		t.Error("expected an explanation of why it is not viable")
	}
}

func TestCPUPlanBlocksWhenRAMTooSmall(t *testing.T) {
	node := &Node{Name: "small", CPUCores: 4, RAMBytes: 8 << 30,
		DiskFreeBytes: 500 << 30, MemBandwidthGBs: 20}
	p := PlanFor(qwen32B(), node, 0)
	if p.Fits {
		t.Error("a 19GB 4-bit model must not fit in 8GB of RAM")
	}
}

func TestSmallModelIsViableOnCPU(t *testing.T) {
	// A 1.7B at 4-bit is about 1GB, so a 40GB/s node should manage tens of
	// tokens per second. This is the case where the CPU box is genuinely useful.
	small := &Info{Params: 2_000_000_000, NumLayers: 28, NumKVHeads: 8, HeadDim: 128,
		MaxPositions: 32768}
	node := &Node{Name: "cpu", CPUCores: 16, RAMBytes: 32 << 30,
		DiskFreeBytes: 1000 << 30, MemBandwidthGBs: 40}
	p := PlanFor(small, node, 0)
	if !p.Fits {
		t.Fatal("a 2B model must fit in 32GB of RAM")
	}
	if p.EstTokensPerSec < 15 {
		t.Errorf("estimated %.1f tok/s for a 2B on CPU, expected more", p.EstTokensPerSec)
	}
}

func TestMoEUsesActiveParamsForSpeedButTotalForMemory(t *testing.T) {
	moe := &Info{
		Params: 235_000_000_000, ActiveParams: 22_000_000_000, IsMoE: true,
		NumLayers: 94, NumKVHeads: 4, HeadDim: 128, MaxPositions: 32768,
	}
	node := &Node{Name: "cpu", CPUCores: 64, RAMBytes: 512 << 30,
		DiskFreeBytes: 8000 << 30, MemBandwidthGBs: 200}
	p := PlanFor(moe, node, 0)
	// Memory must reflect all 235B parameters: every expert has to be resident.
	totalParams := float64(moe.Params)
	wantWeights := int64(totalParams * bytesQ4KM)
	if p.WeightBytes != wantWeights {
		t.Errorf("weights = %d, want %d (all experts resident)", p.WeightBytes, wantWeights)
	}
	// Speed should reflect only the 22B active per token, so it must be far
	// faster than a dense 235B would be.
	dense := *moe
	dense.IsMoE, dense.ActiveParams = false, 0
	densePlan := PlanFor(&dense, node, 0)
	if p.EstTokensPerSec <= densePlan.EstTokensPerSec {
		t.Errorf("MoE estimate %.1f should exceed dense estimate %.1f",
			p.EstTokensPerSec, densePlan.EstTokensPerSec)
	}
}

func TestSuggestPricingShape(t *testing.T) {
	prompt, completion, cached := SuggestPricing(qwen32B())
	p, err := strconv.ParseFloat(prompt, 64)
	if err != nil {
		t.Fatalf("prompt price %q is not a number: %v", prompt, err)
	}
	c, _ := strconv.ParseFloat(completion, 64)
	ch, _ := strconv.ParseFloat(cached, 64)

	if c <= p {
		t.Errorf("completion (%v) must cost more than prompt (%v)", c, p)
	}
	if ch >= p {
		t.Errorf("cached prompt (%v) must cost less than full prompt (%v)", ch, p)
	}
	// Prices are per single token and must be parseable as the decimal strings
	// OpenRouter requires, not scientific notation.
	for _, s := range []string{prompt, completion, cached} {
		for _, r := range s {
			if r != '.' && (r < '0' || r > '9') {
				t.Errorf("price %q contains %q; must be a plain decimal string", s, r)
			}
		}
	}
	// A 32B should land in a plausible band: $0.05-$0.50 per million prompt.
	if perM := p * 1e6; perM < 0.05 || perM > 0.5 {
		t.Errorf("suggested prompt price $%.3f/M is outside a plausible range", perM)
	}
}

func TestSuggestPricingHasFloor(t *testing.T) {
	tiny := &Info{Params: 100_000_000}
	prompt, _, _ := SuggestPricing(tiny)
	p, _ := strconv.ParseFloat(prompt, 64)
	if p <= 0 {
		t.Error("even a tiny model must not be priced at zero by default")
	}
}

// TestFP8CapabilityTable pins the FP8 story per GPU generation. Getting this
// wrong is expensive in both directions: claiming FP8 on Ampere silently drops
// vLLM onto a slow emulated path, and denying it on Ada wastes half the VRAM
// that would otherwise have become KV cache.
func TestFP8CapabilityTable(t *testing.T) {
	cases := map[string]bool{
		// Ampere: no hardware FP8.
		"NVIDIA A100-SXM4-80GB": false,
		"NVIDIA A40":            false,
		"RTX A6000":             false,
		"RTX A5000":             false,
		"RTX A4500":             false,
		"RTX A4000":             false,
		"NVIDIA A10G":           false,
		// Ada Lovelace: yes, including the workstation "Ada" cards whose names
		// do not contain "L4".
		"NVIDIA L40S":             true,
		"NVIDIA L40":              true,
		"NVIDIA L4":               true,
		"RTX 6000 Ada":            true,
		"RTX 4000 Ada":            true,
		"RTX 2000 Ada":            true,
		"NVIDIA GeForce RTX 4090": true,
		// Hopper and Blackwell.
		"NVIDIA H100 80GB HBM3":   true,
		"NVIDIA H200":             true,
		"NVIDIA B200":             true,
		"NVIDIA GeForce RTX 5090": true,
		"RTX PRO 6000":            true,
		"RTX PRO 4500":            true,
	}
	for name, want := range cases {
		if got := gpuHasFP8(name); got != want {
			t.Errorf("gpuHasFP8(%q) = %v, want %v", name, got, want)
		}
	}
}

// TestAmpereNeverPlansFP8 covers the step-down path, not just the capability
// check: a model too large for bf16 on an A40 must fall to 4-bit rather than
// landing on an FP8 plan the card cannot execute natively.
func TestAmpereNeverPlansFP8(t *testing.T) {
	a40 := gpuNode("NVIDIA A40", 48, 1)
	for _, params := range []int64{8_000_000_000, 32_762_123_264, 70_000_000_000} {
		info := qwen32B()
		info.Params = params
		p := PlanFor(info, a40, 0)
		if p.Quantization == "fp8" {
			t.Errorf("%s params on an A40 planned fp8, which Ampere only emulates",
				FormatParams(params))
		}
	}
}

// TestSmallModelKeepsFullPrecision: when a model fits comfortably at its native
// precision there is nothing to trade away, and the planner must not invent a
// quantization the checkpoint does not have.
func TestSmallModelKeepsFullPrecision(t *testing.T) {
	small := qwen32B()
	small.Params = 8_200_000_000
	p := PlanFor(small, gpuNode("RTX 4000 Ada", 20, 1), 0)
	if p.Quantization != "bf16" {
		t.Errorf("quantization = %q, want the checkpoint's own bf16", p.Quantization)
	}
	if p.QuantFromCheckpoint {
		t.Error("bf16 here is the native precision, not a published quantization")
	}
	// It should still mention that an fp8 publication would be faster, since
	// throughput is what a provider is scored on.
	if !strings.Contains(strings.Join(p.Warnings, " "), "pre-quantized") {
		t.Errorf("expected advice about a smaller checkpoint: %v", p.Warnings)
	}
}

func TestArchDetection(t *testing.T) {
	cases := map[string]string{
		"Tesla V100-SXM2-16GB":  "Volta",
		"Tesla V100-SXM3-32GB":  "Volta",
		"Tesla T4":              "Turing",
		"NVIDIA A100-SXM4-80GB": "Ampere",
		"NVIDIA A40":            "Ampere",
		"NVIDIA L40S":           "Ada or newer",
		"NVIDIA H100 80GB HBM3": "Ada or newer",
		"RTX 4000 Ada":          "Ada or newer",
	}
	for name, want := range cases {
		if got := archOf(name).name; got != want {
			t.Errorf("archOf(%q) = %q, want %q", name, got, want)
		}
	}
}

// TestVoltaHasNoFallback is the constraint that makes V100 a trap at these
// prices: it cannot run bf16 and has no 4-bit kernels, so a model that does not
// fit in fp16 does not fit at all.
func TestVoltaHasNoFallback(t *testing.T) {
	v100 := gpuNode("Tesla V100-SXM3-32GB", 32, 1)
	a := nodeArch(v100)
	if a.bf16 || a.fp8 || a.int4 {
		t.Fatalf("Volta reported unsupported capabilities: %+v", a)
	}
	if a.fp8Weights || a.nvfp4 {
		t.Errorf("Volta cannot load fp8 or nvfp4 checkpoints: %+v", a)
	}
	if got := nativePrecision(a); got != "fp16" {
		t.Errorf("Volta native precision = %q, want fp16", got)
	}
	if got := smallerFormat(a, "fp16"); got != "" {
		t.Errorf("Volta has nowhere to step down to, got %q", got)
	}

	// A 32B in fp16 is ~61 GiB against ~28.8 usable. On other architectures the
	// planner would point at a pre-quantized checkpoint; on Volta there is no
	// such checkpoint it could run, so it must not suggest one.
	p := PlanFor(qwen32B(), v100, 0)
	if p.Fits {
		t.Error("a 32B fp16 model cannot fit on a 32GB V100")
	}
	if p.NeedsQuantized {
		t.Error("Volta cannot run any quantized format, so it must not suggest one")
	}
}

// TestUnquantizedCheckpointNeverGetsAQuantizationFlag is the regression test
// for a launch failure in production. The planner saw a 27B model that could
// not fit on a 48GB A40, stepped down its precision ladder to 4-bit, and
// emitted "--quantization awq" pointed at Qwen's bf16 repository. That flag
// asserts the checkpoint is already AWQ; it does not quantize anything. vLLM
// looked for an AWQ config, did not find one, and exited:
//
//	Value error, Cannot find the config file for awq
//
// The plan must instead say the model does not fit and that a different
// repository is needed.
func TestUnquantizedCheckpointNeverGetsAQuantizationFlag(t *testing.T) {
	info := qwen32B()
	info.Params = 27_000_000_000
	info.PublishedQuant = "" // Qwen/Qwen3.8-27B: full-precision weights

	p := PlanFor(info, gpuNode("NVIDIA A40", 48, 1), 0)
	if p.Fits {
		t.Fatal("27B bf16 is ~50 GiB and cannot fit a 48GB A40")
	}
	if p.Quantization == "awq" || p.Quantization == "gptq" || p.Quantization == "fp8" {
		t.Errorf("plan claims %q from a full-precision checkpoint; vLLM would fail to launch",
			p.Quantization)
	}
	if !p.NeedsQuantized {
		t.Error("a 4-bit or fp8 publication would fit, so the plan should point at one")
	}
}

// TestPublishedQuantIsHonoured covers the other half: when the checkpoint *is*
// quantized, that format is what runs, and its smaller weights are what decide
// whether it fits. Qwen/Qwen3.8-27B-FP8 is the repository that makes this model
// serveable on a single A40.
func TestPublishedQuantIsHonoured(t *testing.T) {
	info := qwen32B()
	info.Params = 27_000_000_000
	info.PublishedQuant = "fp8"

	p := PlanFor(info, gpuNode("NVIDIA A40", 48, 1), 0)
	if p.Quantization != "fp8" {
		t.Errorf("quantization = %q, want fp8 from the checkpoint", p.Quantization)
	}
	if !p.QuantFromCheckpoint {
		t.Error("fp8 came from the checkpoint, not from a choice")
	}
	if !p.Fits {
		t.Fatalf("27B at fp8 is ~25 GiB and must fit a 48GB A40: %v", p.Blockers)
	}
	// Ampere has no fp8 arithmetic but loads fp8 weights via Marlin. Blocking
	// here would rule out the one configuration that works.
	if len(p.Blockers) != 0 {
		t.Errorf("A40 can load fp8 weights via Marlin, unexpected blockers: %v", p.Blockers)
	}
}

// TestNVFP4NeedsBlackwell guards the opposite error: accepting a checkpoint the
// card cannot run. NVFP4 repositories of popular models are common and an A40
// cannot execute them.
func TestNVFP4NeedsBlackwell(t *testing.T) {
	info := qwen32B()
	info.Params = 27_000_000_000
	info.PublishedQuant = "modelopt_fp4"

	p := PlanFor(info, gpuNode("NVIDIA A40", 48, 1), 0)
	if p.Fits {
		t.Error("Ampere has no NVFP4 kernels")
	}
	if len(p.Blockers) == 0 || !strings.Contains(strings.Join(p.Blockers, " "), "Blackwell") {
		t.Errorf("blocker should name Blackwell: %v", p.Blockers)
	}
}

// TestFittingIsNotTheSameAsViable pins the distinction the whole planner exists
// to draw. A 14B at fp16 leaves about 1 GiB of KV cache on a 32GB V100: it
// loads, serves roughly one request at a time, and would lose every routing
// decision at OpenRouter.
func TestFittingIsNotTheSameAsViable(t *testing.T) {
	info := qwen32B()
	info.Params = 14_800_000_000
	p := PlanFor(info, gpuNode("Tesla V100-SXM3-32GB", 32, 1), 0)
	if !p.Fits {
		t.Fatal("a 14B at fp16 does fit in 32GB, just barely")
	}
	if p.Viable {
		t.Errorf("a plan with %d concurrent requests must not be called viable", p.MaxNumSeqs)
	}
	if p.MaxNumSeqs > 3 {
		t.Errorf("concurrency = %d, expected very low given ~1 GiB of KV budget", p.MaxNumSeqs)
	}
}

func TestVoltaWarnsAboutBF16Checkpoint(t *testing.T) {
	info := qwen32B()
	info.Params = 8_200_000_000
	info.DType = "BF16"
	p := PlanFor(info, gpuNode("Tesla V100-SXM3-32GB", 32, 1), 0)
	if p.Quantization != "fp16" {
		t.Errorf("quantization = %q on Volta, want fp16", p.Quantization)
	}
	var sawNarrowing, sawAttention bool
	for _, w := range p.Warnings {
		if strings.Contains(w, "narrowed to fp16") {
			sawNarrowing = true
		}
		if strings.Contains(w, "FlashAttention-2") {
			sawAttention = true
		}
	}
	if !sawNarrowing {
		t.Error("expected a warning about narrowing a bf16 checkpoint to fp16")
	}
	if !sawAttention {
		t.Error("expected a warning that Volta lacks FlashAttention-2")
	}
}

// TestMixedNodeUsesLeastCapableCard: a node is only as capable as its weakest
// GPU, because the model is sharded across all of them.
func TestMixedNodeUsesLeastCapableCard(t *testing.T) {
	mixed := &Node{Name: "mixed", CPUCores: 8, RAMBytes: 64 << 30, HasNVMe: true}
	mixed.GPUs = []GPU{
		{Name: "NVIDIA L40S", VRAMBytes: 48 << 30},
		{Name: "NVIDIA A40", VRAMBytes: 48 << 30},
	}
	if a := nodeArch(mixed); a.fp8 {
		t.Error("a node containing an Ampere card must not be planned for fp8")
	}
}

// TestCPUPlanWarnsBatchingDoesNotScale guards a misreading the output invites.
// A CPU plan reporting "4 concurrent" looks like four times the throughput; it
// is not, because the per-token weight read is the bottleneck and every stream
// contends for the same memory bandwidth.
func TestCPUPlanWarnsBatchingDoesNotScale(t *testing.T) {
	small := &Info{Params: 1_540_000_000, NumLayers: 28, NumKVHeads: 2, HeadDim: 128,
		MaxPositions: 32768, DType: "BF16"}
	node := &Node{Name: "cpu", CPUCores: 16, RAMBytes: 32 << 30,
		DiskFreeBytes: 4000 << 30, MemBandwidthGBs: 45}
	p := PlanFor(small, node, 0)
	if p.MaxNumSeqs <= 1 {
		t.Skip("plan reported no concurrency, nothing to warn about")
	}
	found := false
	for _, w := range p.Warnings {
		if strings.Contains(w, "do not multiply") {
			found = true
		}
	}
	if !found {
		t.Errorf("expected a warning that CPU concurrency does not multiply throughput: %v", p.Warnings)
	}
}

// deepseekV4Flash mirrors deepseek-ai/DeepSeek-V4-Flash-0731, verified against
// its published config: a 304B mixture of experts with roughly 35B active per
// token, Multi-head Latent Attention (1 KV head), and an fp8 checkpoint.
func deepseekV4Flash() *Info {
	return &Info{
		ID: "deepseek-ai/DeepSeek-V4-Flash-0731", Params: 304_200_000_000,
		ActiveParams: 34_800_000_000, IsMoE: true, IsMLA: true,
		PublishedQuant: "fp8", DType: "I8",
		Architecture: "DeepseekV4ForCausalLM", MaxPositions: 1048576,
		HiddenSize: 4096, NumLayers: 43, NumHeads: 64, NumKVHeads: 1, HeadDim: 512,
	}
}

// TestVRAMAccountsForTensorParallel is the most dangerous class of bug the
// planner can have: reporting that a model fits when it does not. Summing every
// GPU on the node regardless of TP claimed a 283 GiB model would run at TP=1,
// where it would actually have to live inside a single card.
func TestVRAMAccountsForTensorParallel(t *testing.T) {
	info := deepseekV4Flash()
	eight := gpuNode("NVIDIA H200", 141, 8)

	// With MLA the KV-head rule must not cap TP, so all 8 GPUs are used.
	p := PlanFor(info, eight, 0)
	if p.TensorParallel != 8 {
		t.Fatalf("TP = %d on an 8-GPU node, want 8 (MLA does not constrain sharding)", p.TensorParallel)
	}
	if !p.Fits {
		t.Fatalf("283 GiB across 8x141GB should fit: %v", p.Blockers)
	}

	// A dense model with 1 KV head genuinely cannot shard, and must then be
	// judged against a single GPU rather than the whole node.
	dense := deepseekV4Flash()
	dense.IsMLA = false
	dp := PlanFor(dense, eight, 0)
	if dp.TensorParallel != 1 {
		t.Fatalf("TP = %d, want 1 when 1 KV head genuinely blocks sharding", dp.TensorParallel)
	}
	if dp.Fits {
		t.Error("a 283 GiB model must not be reported as fitting on one 141GB GPU")
	}
}

func TestBlockerNamesTheAddressableMemory(t *testing.T) {
	p := PlanFor(deepseekV4Flash(), gpuNode("NVIDIA H100 80GB HBM3", 80, 2), 0)
	if p.Fits {
		t.Fatal("a 304B model cannot fit on 2x80GB")
	}
	// The message must state what is actually addressable, not the node total,
	// or the operator will think adding GPUs at the same TP would help.
	if !strings.Contains(p.Blockers[0], "addressable") {
		t.Errorf("blocker should name the addressable memory: %q", p.Blockers[0])
	}
}

// TestMoEPricingUsesActiveParams: pricing a 304B MoE as though every parameter
// ran on every token overstates it by roughly an order of magnitude, which
// would put the suggestion far above what the market actually charges.
func TestMoEPricingUsesActiveParams(t *testing.T) {
	moe := deepseekV4Flash()
	promptActive, _, _ := SuggestPricing(moe)

	dense := deepseekV4Flash()
	dense.IsMoE, dense.ActiveParams = false, 0
	promptDense, _, _ := SuggestPricing(dense)

	a, _ := strconv.ParseFloat(promptActive, 64)
	d, _ := strconv.ParseFloat(promptDense, 64)
	if a >= d {
		t.Errorf("MoE pricing (%v) should be below dense pricing (%v) for the same total size", a, d)
	}
	// The real market for this model is $0.065-0.14 per million prompt tokens.
	// The suggestion should land in the same order of magnitude.
	if perM := a * 1e6; perM < 0.03 || perM > 0.5 {
		t.Errorf("suggested $%.3f/M is not in the same range as the observed market", perM)
	}
}

// TestPrequantizedCheckpointDownloadSize: DeepSeek-V4 ships in fp8, so the
// download is about 285 GiB rather than the 566 GiB a bf16 assumption gives.
func TestPrequantizedCheckpointDownloadSize(t *testing.T) {
	p := PlanFor(deepseekV4Flash(), gpuNode("NVIDIA H200", 141, 8), 0)
	gib := float64(p.DiskBytes) / (1 << 30)
	if gib > 320 {
		t.Errorf("disk = %.0f GiB; an fp8 checkpoint should not be sized as bf16", gib)
	}
	if gib < 260 {
		t.Errorf("disk = %.0f GiB, implausibly small for 304B parameters at fp8", gib)
	}
}

// TestNestedMultimodalConfig covers the config shape used by vision-language
// models, where the transformer's geometry sits under text_config and the top
// level holds only the wrapper. Reading the top level yields zero layers and
// zero KV heads, which produces a plan with no KV cache and a default
// concurrency figure that is pure fiction.
func TestNestedMultimodalConfig(t *testing.T) {
	raw := &hfConfig{
		Architectures: []string{"Qwen3_5ForConditionalGeneration"},
		TextConfig: &hfConfig{
			NumLayers: 64, HiddenSize: 5120, NumHeads: 24, NumKVHeads: 4,
			HeadDim: 256, MaxPositions: 262144,
		},
		VisionConfig: &struct {
			HiddenSize int `json:"hidden_size"`
			NumLayers  int `json:"num_hidden_layers"`
			ImageSize  int `json:"image_size"`
		}{HiddenSize: 1152},
	}
	cfg := raw.languageConfig()
	if cfg.NumLayers != 64 || cfg.NumKVHeads != 4 || cfg.MaxPositions != 262144 {
		t.Fatalf("languageConfig did not resolve the nested geometry: %+v", cfg)
	}

	// A plain text model must be returned unchanged.
	flat := &hfConfig{NumLayers: 36, NumKVHeads: 8}
	if got := flat.languageConfig(); got.NumLayers != 36 {
		t.Errorf("a flat config should be returned as-is, got %+v", got)
	}
}

func TestMultimodalWarnsAboutEncoderMemory(t *testing.T) {
	info := qwen32BFP8()
	info.Params = 27_800_000_000
	info.HasVision = true
	p := PlanFor(info, gpuNode("NVIDIA L40S", 48, 1), 0)
	found := false
	for _, w := range p.Warnings {
		if strings.Contains(w, "multimodal") {
			found = true
		}
	}
	if !found {
		t.Errorf("a vision model should warn about encoder activation memory: %v", p.Warnings)
	}
}

// TestZeroGeometryDoesNotFakeConcurrency: if we could not read the geometry we
// must not report a concurrency number, because the default would be presented
// as though it had been calculated.
func TestZeroGeometryDoesNotFakeConcurrency(t *testing.T) {
	unknown := &Info{ID: "x/y", Params: 27_800_000_000} // no layers, no KV heads
	p := PlanFor(unknown, gpuNode("NVIDIA L40S", 48, 1), 0)
	if p.KVBytesPerTok != 0 {
		t.Fatal("fixture should have unknown geometry")
	}
	if p.MaxNumSeqs > 1 {
		t.Errorf("concurrency = %d with unknown KV geometry; it cannot be computed, "+
			"so it must not be asserted", p.MaxNumSeqs)
	}
}

// TestPascalIsRecognised: P100 and the GTX 10-series predate every acceleration
// vLLM relies on. Classifying them as Ampere would produce a bf16 plan on
// hardware that cannot run bf16 at all.
func TestPascalIsRecognised(t *testing.T) {
	// All Pascal parts lack bf16, fp8 and 4-bit kernels. They differ only in
	// whether fp16 is usable, which TestConsumerPascalFP16IsCrippled covers.
	for _, name := range []string{"Tesla P100-PCIE-16GB", "Tesla P40", "NVIDIA GeForce GTX 1080 Ti"} {
		a := archOf(name)
		if !strings.HasPrefix(a.name, "Pascal") {
			t.Errorf("archOf(%q) = %q, want a Pascal variant", name, a.name)
		}
		if a.bf16 || a.fp8 || a.int4 {
			t.Errorf("%s reported capabilities it does not have: %+v", name, a)
		}
	}
	if got := nativePrecision(archOf("Tesla P100-PCIE-16GB")); got != "fp16" {
		t.Errorf("P100 native precision = %q, want fp16", got)
	}
}

// TestHostRAMIsChecked covers a constraint that is invisible in a VRAM-only
// view: weights are staged through host memory on the way to the device. A
// container advertising 8GB of RAM cannot reliably load a 26 GiB model however
// much VRAM the GPU has.
func TestHostRAMIsChecked(t *testing.T) {
	info := qwen32BFP8()
	info.Params = 27_800_000_000

	tiny := gpuNode("NVIDIA GeForce RTX 5090", 32, 1)
	tiny.RAMBytes = 8 << 30
	p := PlanFor(info, tiny, 0)
	if p.Fits {
		t.Errorf("8 GiB of host RAM should block a %s model from loading",
			humanBytes(p.WeightBytes))
	}

	// Somewhat short of the model size is a warning, not a blocker: mmap makes
	// it slow rather than impossible.
	modest := gpuNode("NVIDIA GeForce RTX 5090", 32, 1)
	modest.RAMBytes = 20 << 30
	q := PlanFor(info, modest, 0)
	found := false
	for _, w := range q.Warnings {
		if strings.Contains(w, "streams through host memory") {
			found = true
		}
	}
	if !found {
		t.Errorf("expected a host-RAM warning, got %v", q.Warnings)
	}
	if !q.Fits {
		t.Error("20 GiB of RAM should be a warning, not a blocker")
	}

	// Plenty of RAM says nothing at all.
	roomy := gpuNode("NVIDIA GeForce RTX 5090", 32, 1)
	roomy.RAMBytes = 128 << 30
	for _, w := range PlanFor(info, roomy, 0).Warnings {
		if strings.Contains(w, "host memory") {
			t.Errorf("should not warn about RAM when there is plenty: %q", w)
		}
	}
}

// TestConsumerPascalFP16IsCrippled separates the GTX 10-series from the P100.
// They share an architecture name but not a usable precision: GP102 runs fp16
// at 1/64 the rate of fp32, so the only workable format is fp32 -- which needs
// twice the memory and turns a marginal plan into an impossible one.
func TestConsumerPascalFP16IsCrippled(t *testing.T) {
	consumer := archOf("NVIDIA GeForce GTX 1080 Ti")
	if !consumer.crippledFP16 {
		t.Error("GTX 1080 Ti should be marked as having crippled fp16")
	}
	if got := nativePrecision(consumer); got != "fp32" {
		t.Errorf("consumer Pascal native precision = %q, want fp32", got)
	}

	// The datacenter P100 has full-rate fp16 and must not be tarred with it.
	p100 := archOf("Tesla P100-PCIE-16GB")
	if p100.crippledFP16 {
		t.Error("P100 has full-rate fp16 and should not be marked crippled")
	}
	if got := nativePrecision(p100); got != "fp16" {
		t.Errorf("P100 native precision = %q, want fp16", got)
	}
}

func TestFP32DoublesTheMemoryRequirement(t *testing.T) {
	info := qwen32B()
	info.Params = 27_800_000_000

	pascal := gpuNode("NVIDIA GeForce GTX 1080 Ti", 11, 2)
	pascal.RAMBytes = 64 << 30
	p := PlanFor(info, pascal, 0)
	if p.Quantization != "fp32" {
		t.Errorf("quantization = %q on consumer Pascal, want fp32", p.Quantization)
	}
	// 27.8B at 4 bytes is about 103 GiB, against 22 GiB of addressable VRAM.
	if p.Fits {
		t.Error("a 103 GiB fp32 model must not fit on 2x11GB")
	}
	if gib := float64(p.WeightBytes) / (1 << 30); gib < 100 || gib > 108 {
		t.Errorf("fp32 weights = %.0f GiB, want about 103", gib)
	}
}

// TestFractionalGPUScalesThroughput covers MIG partitions and vGPU slices,
// which providers increasingly sell. The VRAM figure alone is misleading: a
// half slice also gets roughly half the compute, so a plan that looks identical
// to a whole card on memory delivers half the tokens.
func TestFractionalGPUScalesThroughput(t *testing.T) {
	info := qwen32B()
	info.Params = 27_800_000_000
	info.PublishedQuant = "awq" // 24GB holds this model only in 4-bit

	whole := gpuNode("NVIDIA A40", 24, 1)
	whole.RAMBytes = 60 << 30
	full := PlanFor(info, whole, 0)

	half := gpuNode("NVIDIA A40", 24, 1)
	half.RAMBytes = 60 << 30
	half.GPUFraction = 0.5
	sliced := PlanFor(info, half, 0)

	if sliced.EstTokensPerSec >= full.EstTokensPerSec {
		t.Errorf("a half slice estimated %.0f tok/s against a whole card's %.0f",
			sliced.EstTokensPerSec, full.EstTokensPerSec)
	}
	found := false
	for _, w := range sliced.Warnings {
		if strings.Contains(w, "slice of a GPU") {
			found = true
		}
	}
	if !found {
		t.Errorf("a fractional GPU must say so: %v", sliced.Warnings)
	}
	// An out-of-range or unset fraction means a whole GPU.
	for _, f := range []float64{0, 1, 1.5, -1} {
		n := gpuNode("NVIDIA A40", 24, 1)
		n.RAMBytes = 60 << 30
		n.GPUFraction = f
		if got := PlanFor(info, n, 0).EstTokensPerSec; got != full.EstTokensPerSec {
			t.Errorf("fraction %v gave %.0f tok/s, want the whole-card %.0f", f, got, full.EstTokensPerSec)
		}
	}
}

// TestFP8KVCacheDoublesConcurrency pins the largest concurrency lever available.
// The KV cache, not the weights, is what limits how many requests fit; halving
// its per-token cost roughly doubles the number that do.
func TestFP8KVCacheDoublesConcurrency(t *testing.T) {
	info := qwen32BFP8()
	info.Params = 27_800_000_000

	node := gpuNode("NVIDIA A40", 48, 1)
	node.RAMBytes = 128 << 30
	p := PlanFor(info, node, 8192)
	if p.KVCacheDType != "fp8" {
		t.Fatalf("KV cache dtype = %q on Ampere, want fp8", p.KVCacheDType)
	}
	// The per-token cost must actually be halved, not merely labelled.
	if want := info.KVBytesPerToken() / 2; p.KVBytesPerTok != want {
		t.Errorf("KV per token = %d, want %d (half of fp16)", p.KVBytesPerTok, want)
	}

	// Pascal and Volta predate it and must stay on fp16.
	for _, name := range []string{"Tesla V100-SXM3-32GB", "Tesla P100-PCIE-16GB"} {
		old := gpuNode(name, 32, 1)
		old.RAMBytes = 128 << 30
		small := qwen32B()
		small.Params = 8_200_000_000
		if got := PlanFor(small, old, 0).KVCacheDType; got != "auto" {
			t.Errorf("%s KV dtype = %q, want auto (no fp8 KV before Ampere)", name, got)
		}
	}
}

// TestGPUThroughputFollowsBandwidth replaces a constant with physics. Decode is
// bandwidth-bound, so a card built for virtual desktops is far slower than a
// compute card of the same architecture, and a constant hid that entirely.
func TestGPUThroughputFollowsBandwidth(t *testing.T) {
	if bw, ok := gpuBandwidthGBs("NVIDIA A16"); !ok || bw > 300 {
		t.Errorf("A16 bandwidth = %v (known=%v); it is a VDI card at about 200 GB/s", bw, ok)
	}
	if bw, _ := gpuBandwidthGBs("NVIDIA A40"); bw < 600 {
		t.Errorf("A40 bandwidth = %v, want about 696", bw)
	}
	// An unknown card must be flagged rather than silently assumed fast.
	if _, known := gpuBandwidthGBs("Some Future GPU 9000"); known {
		t.Error("an unrecognised card should not report a known bandwidth")
	}

	info := qwen32B()
	info.Params = 8_200_000_000

	// Same architecture, same quantization, 3.5x the bandwidth.
	fast := gpuNode("NVIDIA A40", 48, 1)
	fast.RAMBytes = 128 << 30
	slow := gpuNode("NVIDIA A16", 48, 1) // VRAM equalised to isolate bandwidth
	slow.RAMBytes = 128 << 30

	fp, sp := PlanFor(info, fast, 0), PlanFor(info, slow, 0)
	if fp.Quantization != sp.Quantization {
		t.Fatalf("test needs matching quantization, got %q vs %q", fp.Quantization, sp.Quantization)
	}
	if fp.EstTokensPerSec <= sp.EstTokensPerSec*2 {
		t.Errorf("A40 estimated %.0f tok/s vs A16 %.0f; bandwidth differs 3.5x so the "+
			"estimates should too", fp.EstTokensPerSec, sp.EstTokensPerSec)
	}
}

// TestSmallModelsAreCappedNotExtrapolated: below a certain size the bottleneck
// stops being bandwidth, so the estimate must not run away.
func TestSmallModelsAreCappedNotExtrapolated(t *testing.T) {
	tiny := qwen32B()
	tiny.Params = 500_000_000
	node := gpuNode("NVIDIA H200", 141, 8)
	node.RAMBytes = 512 << 30
	if got := PlanFor(tiny, node, 0).EstTokensPerSec; got > maxSingleStreamTPS {
		t.Errorf("estimated %.0f tok/s, above the %.0f cap; single-stream decode is "+
			"latency-bound at this size, not bandwidth-bound", got, maxSingleStreamTPS)
	}
}

// TestQuantizationTradeoffIsSurfaced: the ladder picks the best quality that
// fits, which is right for a model's user but is a throughput decision a
// provider should make deliberately.
func TestQuantizationTradeoffIsSurfaced(t *testing.T) {
	info := qwen32B()
	info.Params = 8_200_000_000
	node := gpuNode("NVIDIA A40", 48, 1) // Ampere: bf16 fits, awq would be faster
	node.RAMBytes = 128 << 30

	p := PlanFor(info, node, 0)
	if p.Quantization != "bf16" {
		t.Fatalf("quantization = %q, expected bf16 to fit on 48GB", p.Quantization)
	}
	found := false
	for _, w := range p.Warnings {
		if strings.Contains(w, "more per token") {
			found = true
		}
	}
	if !found {
		t.Errorf("a 3.3x throughput difference should be surfaced: %v", p.Warnings)
	}
}
