package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"sync"
	"sync/atomic"
	"time"
)

func execTimeout(d time.Duration) (context.Context, context.CancelFunc) {
	return context.WithTimeout(context.Background(), d)
}

// Spec is an install request from the gateway: what to serve and how.
type Spec struct {
	// HFID is the HuggingFace repo, e.g. "Qwen/Qwen3-8B".
	HFID string `json:"hf_id"`
	// ServedName is the public model id clients will use. Setting it here means
	// the gateway never has to rewrite the model name in response frames.
	ServedName string `json:"served_name"`

	Engine       string `json:"engine"` // "vllm" or "llamacpp"
	Quantization string `json:"quantization"`
	// KVCacheDType is "auto" (fp16) or "fp8". Storing the KV cache at fp8
	// halves its per-token cost, which is the largest single lever on how many
	// concurrent requests fit in a given amount of VRAM.
	KVCacheDType   string `json:"kv_cache_dtype,omitempty"`
	TensorParallel int    `json:"tensor_parallel"`
	MaxModelLen    int    `json:"max_model_len"`
	MaxNumSeqs     int    `json:"max_num_seqs"`

	// Hybrid marks a model that interleaves Mamba-style linear attention with
	// full attention. vLLM supports neither prefix caching nor an fp8 KV cache
	// over that recurrent state, and rejects both at startup rather than
	// ignoring them, so the flags have to be withheld rather than merely
	// unhelpful.
	Hybrid bool `json:"hybrid,omitempty"`
	// QuantFromCheckpoint says the quantization was read from the checkpoint
	// rather than chosen. vLLM detects it from config.json and picks the right
	// kernel for the hardware, which is more reliable than our name for it.
	QuantFromCheckpoint bool `json:"quant_from_checkpoint,omitempty"`

	// GGUFRepo overrides the auto-resolved GGUF repository for llama.cpp.
	GGUFRepo string `json:"gguf_repo,omitempty"`
	// LocalGGUF is a .gguf file already on this machine. It takes precedence
	// over GGUFRepo: a checkpoint that was converted or fine-tuned locally has
	// no repository to download from.
	LocalGGUF string `json:"local_gguf,omitempty"`
	// ExtraArgs are appended verbatim, for tuning we do not model.
	ExtraArgs []string `json:"extra_args,omitempty"`
}

// Runtime is how this agent was configured to launch engines.
type Runtime struct {
	// EngineHost is the address engines bind to. It defaults to 0.0.0.0, which
	// is right on a single machine where nothing else can reach the port -- and
	// wrong the moment the box is on a public network, because an inference
	// endpoint has no authentication of its own. Anyone who finds the port gets
	// free use of the GPU. On a multi-node deployment this should be the
	// private mesh address, so the gateway can reach it and nobody else can.
	EngineHost string

	// Mode is "native" (binaries on PATH) or "docker".
	Mode string
	// VLLMImage is used in docker mode.
	VLLMImage string
	// HFCacheDir is bind-mounted in docker mode and set as HF_HOME natively, so
	// weights survive restarts instead of being re-downloaded.
	HFCacheDir string
	HFToken    string
}

// BuildCommand renders the exact process invocation for a spec.
//
// It is a pure function so the gateway can show the operator precisely what
// will run before anything is executed, and so the same command can be copied
// and run by hand when debugging.
func BuildCommand(s Spec, rt Runtime, port int) (bin string, args []string, env []string, err error) {
	switch s.Engine {
	case "vllm":
		return buildVLLM(s, rt, port)
	case "freetoken":
		return buildFreeToken(s, rt, port)
	case "llamacpp":
		return buildLlamaCpp(s, rt, port)
	}
	return "", nil, nil, fmt.Errorf("unknown engine %q", s.Engine)
}

func buildVLLM(s Spec, rt Runtime, port int) (string, []string, []string, error) {
	if s.HFID == "" {
		return "", nil, nil, fmt.Errorf("hf_id is required")
	}
	vllmArgs := []string{
		"serve", s.HFID,
		"--served-model-name", s.ServedName,
		"--port", strconv.Itoa(port),
		"--host", rt.engineHost(),
		// Without chunked prefill, one long prompt stalls every other stream on
		// the replica and p99 TTFT collapses under mixed traffic.
		"--enable-chunked-prefill",
	}
	if !s.Hybrid {
		// Prefix caching is the largest available TTFT win and produces the
		// cached_tokens we bill at a discount. A hybrid model's recurrent
		// state cannot be reused across a shared prefix, and asking for it
		// fails at startup.
		vllmArgs = append(vllmArgs, "--enable-prefix-caching")
	}
	if s.TensorParallel > 1 {
		vllmArgs = append(vllmArgs, "--tensor-parallel-size", strconv.Itoa(s.TensorParallel))
	}
	// bf16 is vLLM's default for a bf16 checkpoint; passing it explicitly as a
	// "quantization" is an error, so it is only set when it is a real scheme.
	//
	// A pre-quantized checkpoint describes itself in config.json, and vLLM
	// reads that and selects a kernel appropriate to the hardware -- fp8-marlin
	// on Ampere, native fp8 on Ada and newer. Naming the scheme ourselves can
	// pin it to a kernel the card does not have, so the flag is only passed
	// when we picked the format rather than read it.
	if s.Quantization != "" && s.Quantization != "bf16" && s.Quantization != "fp16" &&
		!s.QuantFromCheckpoint {
		vllmArgs = append(vllmArgs, "--quantization", s.Quantization)
	}
	if s.MaxModelLen > 0 {
		vllmArgs = append(vllmArgs, "--max-model-len", strconv.Itoa(s.MaxModelLen))
	}
	if s.MaxNumSeqs > 0 {
		vllmArgs = append(vllmArgs, "--max-num-seqs", strconv.Itoa(s.MaxNumSeqs))
	}
	// "auto" is vLLM's default, so it is only worth passing the override.
	if s.KVCacheDType != "" && s.KVCacheDType != "auto" {
		vllmArgs = append(vllmArgs, "--kv-cache-dtype", s.KVCacheDType)
	}
	vllmArgs = append(vllmArgs, s.ExtraArgs...)

	env := []string{}
	if rt.HFToken != "" {
		env = append(env, "HF_TOKEN="+rt.HFToken)
	}

	if rt.Mode == "docker" {
		image := rt.VLLMImage
		if image == "" {
			image = "vllm/vllm-openai:latest"
		}
		docker := []string{
			"run", "--rm",
			"--gpus", "all",
			// Tensor parallelism communicates over shared memory between worker
			// processes. Docker's default 64MB /dev/shm crashes it with an
			// error that does not point at the cause.
			"--ipc=host",
			"-p", fmt.Sprintf("%d:%d", port, port),
		}
		if rt.HFCacheDir != "" {
			docker = append(docker, "-v", rt.HFCacheDir+":/root/.cache/huggingface")
		}
		if rt.HFToken != "" {
			docker = append(docker, "-e", "HF_TOKEN="+rt.HFToken)
		}
		docker = append(docker, image)
		docker = append(docker, vllmArgs[1:]...) // the image entrypoint is already "vllm serve"
		return "docker", docker, env, nil
	}

	if rt.HFCacheDir != "" {
		env = append(env, "HF_HOME="+rt.HFCacheDir)
	}
	// Some base images export HF_HUB_ENABLE_HF_TRANSFER=1 without the package
	// that implements it, and any pip operation that removes hf_transfer
	// leaves the variable behind. The download then aborts with "Fast download
	// using 'hf_transfer' is enabled but 'hf_transfer' package is not
	// available" -- and because config.json never arrives, the error the
	// operator actually sees is the far less helpful "Can't load the
	// configuration of <model>". Turning the accelerator off when it is not
	// installed costs some download speed and nothing else.
	if os.Getenv("HF_HUB_ENABLE_HF_TRANSFER") != "" && !hfTransferInstalled() {
		env = append(env, "HF_HUB_ENABLE_HF_TRANSFER=0")
	}
	return "vllm", vllmArgs, env, nil
}

// hfTransferInstalled reports whether the accelerator the environment asks for
// is actually importable. The answer cannot change while the agent runs, so it
// is resolved once.
var hfTransferInstalled = sync.OnceValue(func() bool {
	ctx, cancel := execTimeout(20 * time.Second)
	defer cancel()
	for _, py := range []string{"python3", "python"} {
		if err := exec.CommandContext(ctx, py, "-c", "import hf_transfer").Run(); err == nil {
			return true
		}
	}
	return false
})

func buildLlamaCpp(s Spec, rt Runtime, port int) (string, []string, []string, error) {
	quant := s.Quantization
	if quant == "" {
		quant = "Q4_K_M"
	}
	var source []string
	switch {
	case s.LocalGGUF != "":
		// Already on disk, so nothing to download and nothing to resolve.
		source = []string{"-m", s.LocalGGUF}
	case s.GGUFRepo != "":
		// -hf downloads the GGUF straight from HuggingFace, which is what makes
		// a CPU install genuinely one step.
		source = []string{"-hf", s.GGUFRepo + ":" + quant}
	default:
		return "", nil, nil, fmt.Errorf(
			"llama.cpp needs a GGUF: none was resolved for %q, and no local .gguf was given", s.HFID)
	}
	args := append(source,
		"--port", strconv.Itoa(port),
		"--host", rt.engineHost(),
		"--alias", s.ServedName,
		// Continuous batching, so concurrent requests share the process rather
		// than serialising.
		"--cont-batching",
	)
	if s.MaxModelLen > 0 {
		args = append(args, "-c", strconv.Itoa(s.MaxModelLen))
	}
	if s.MaxNumSeqs > 0 {
		args = append(args, "-np", strconv.Itoa(s.MaxNumSeqs))
	}
	args = append(args, s.ExtraArgs...)

	env := []string{}
	if rt.HFToken != "" {
		env = append(env, "HF_TOKEN="+rt.HFToken)
	}
	if rt.HFCacheDir != "" {
		env = append(env, "LLAMA_CACHE="+rt.HFCacheDir)
	}
	return "llama-server", args, env, nil
}

// EngineAvailable reports whether the engine's binary is installed.
func EngineAvailable(engine string, rt Runtime) bool {
	if rt.Mode == "docker" && engine == "vllm" {
		_, err := exec.LookPath("docker")
		return err == nil
	}
	switch engine {
	case "vllm":
		_, err := exec.LookPath("vllm")
		return err == nil
	case "llamacpp":
		_, err := exec.LookPath("llama-server")
		return err == nil
	case "freetoken":
		_, err := exec.LookPath("ft")
		return err == nil
	}
	return false
}

// supportedArchs caches what the installed vLLM can serve. Empty means "not
// known yet", never "nothing supported".
var supportedArchs atomic.Pointer[[]string]

// SupportedArchs returns the cached architecture list without blocking.
//
// It must not block: this is read while answering /v1/node/info, which the
// gateway polls with a short timeout. Importing vLLM to ask its registry takes
// seconds on a local disk and far longer from a venv on a network filesystem,
// and doing that inside the handler made the whole node look unreachable:
//
//	node gpu-a unreachable: context deadline exceeded
//
// StartArchDetection populates it in the background instead.
func SupportedArchs() []string {
	if p := supportedArchs.Load(); p != nil {
		return *p
	}
	return nil
}

// StartArchDetection asks vLLM for its model registry once, in the background.
// Callers see an empty list until it answers, which they already treat as
// "could not determine" rather than as a refusal.
func StartArchDetection() {
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
		defer cancel()
		const script = `
import json, sys
try:
    from vllm.model_executor.models.registry import ModelRegistry
    sys.stdout.write(json.dumps(sorted(ModelRegistry.get_supported_archs())))
except Exception:
    sys.stdout.write("[]")
`
		for _, py := range []string{"python3", "python"} {
			out, err := exec.CommandContext(ctx, py, "-c", script).Output()
			if err != nil {
				continue
			}
			var archs []string
			if json.Unmarshal(out, &archs) == nil && len(archs) > 0 {
				supportedArchs.Store(&archs)
				return
			}
		}
	}()
}

// buildFreeToken launches FreeToken's OpenAI-compatible server.
//
// FreeToken is a mixture-of-experts engine that treats VRAM and host RAM as one
// pool: hot experts are cached on the GPU and the rest stream from system
// memory, with the split chosen from the machine's measured bandwidth. That
// lets a card hold a model considerably larger than its VRAM, at the cost of
// per-token speed -- so it is worth reaching for only when the alternative is
// not serving the model at all.
//
// It needs an NVIDIA GPU and CUDA 13; there is no CPU-only mode. The planner
// checks both before offering it.
func buildFreeToken(s Spec, rt Runtime, port int) (string, []string, []string, error) {
	if s.HFID == "" {
		return "", nil, nil, fmt.Errorf("hf_id is required")
	}
	args := []string{
		"serve",
		"--model", s.HFID,
		"--host", rt.engineHost(),
		"--port", strconv.Itoa(port),
		"--served-model-name", s.ServedName,
	}
	if s.MaxModelLen > 0 {
		args = append(args, "--max-seq-len-override", strconv.Itoa(s.MaxModelLen))
	}
	// Let it size the expert cache from what it measures. A fixed split is the
	// thing this engine exists to avoid, and a number we picked from a spec
	// sheet would be exactly that.
	args = append(args, "--moe-cache-auto")
	args = append(args, s.ExtraArgs...)

	env := []string{}
	if rt.HFToken != "" {
		env = append(env, "HF_TOKEN="+rt.HFToken)
	}
	if rt.HFCacheDir != "" {
		env = append(env, "HF_HOME="+rt.HFCacheDir)
	}
	if os.Getenv("HF_HUB_ENABLE_HF_TRANSFER") != "" && !hfTransferInstalled() {
		env = append(env, "HF_HUB_ENABLE_HF_TRANSFER=0")
	}
	return "ft", args, env, nil
}

// engineHost is where engines listen. 0.0.0.0 unless told otherwise.
func (rt Runtime) engineHost() string {
	if rt.EngineHost != "" {
		return rt.EngineHost
	}
	return "0.0.0.0"
}
