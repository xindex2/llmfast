package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"sync"
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
	// ExtraArgs are appended verbatim, for tuning we do not model.
	ExtraArgs []string `json:"extra_args,omitempty"`
}

// Runtime is how this agent was configured to launch engines.
type Runtime struct {
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
		"--host", "0.0.0.0",
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
	repo := s.GGUFRepo
	if repo == "" {
		return "", nil, nil, fmt.Errorf("llama.cpp needs a GGUF repository; none was resolved for %q", s.HFID)
	}
	quant := s.Quantization
	if quant == "" {
		quant = "Q4_K_M"
	}
	args := []string{
		// -hf downloads the GGUF straight from HuggingFace, which is what makes
		// a CPU install genuinely one step.
		"-hf", repo + ":" + quant,
		"--port", strconv.Itoa(port),
		"--host", "0.0.0.0",
		"--alias", s.ServedName,
		// Continuous batching, so concurrent requests share the process rather
		// than serialising.
		"--cont-batching",
	}
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
	}
	return false
}

// SupportedArchs asks the installed vLLM which model architectures it can
// serve.
//
// This is the single most useful thing to know before an install, and the
// hardest to infer: a checkpoint's architecture is in its config.json, and
// whether the engine implements it depends on the exact vLLM release. Getting
// it wrong costs five restart attempts and a traceback whose real cause --
// "Transformers does not recognize this architecture" -- is buried dozens of
// frames above the line that gets reported. The registry answers it exactly.
//
// Resolved once: importing vLLM takes several seconds, and the answer cannot
// change while the agent is running.
var SupportedArchs = sync.OnceValue(func() []string {
	ctx, cancel := execTimeout(120 * time.Second)
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
			return archs
		}
	}
	return nil
})
