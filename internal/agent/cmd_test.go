package agent

import (
	"strings"
	"testing"
)

// TestHybridModelsGetNeitherPrefixCachingNorFP8KV covers the launch failure on
// Qwen3.5. It interleaves Mamba-style linear attention with full attention;
// vLLM supports neither prefix caching nor an fp8 KV cache over that recurrent
// state, and rejects both at startup rather than ignoring them. Passing them
// meant the engine could never come up, and the traceback it printed ended in
// a generic re-raise that named none of it.
func TestHybridModelsGetNeitherPrefixCachingNorFP8KV(t *testing.T) {
	_, args, _, err := BuildCommand(Spec{
		HFID: "Qwen/Qwen3.8-27B-FP8", ServedName: "qwen/qwen3.8-27b-fp8",
		Engine: "vllm", Quantization: "fp8", QuantFromCheckpoint: true,
		KVCacheDType: "auto", Hybrid: true, MaxModelLen: 32768, MaxNumSeqs: 24,
	}, Runtime{Mode: "native"}, 18000)
	if err != nil {
		t.Fatal(err)
	}
	line := strings.Join(args, " ")

	if strings.Contains(line, "--enable-prefix-caching") {
		t.Error("prefix caching is not supported over a recurrent state and fails at startup")
	}
	if strings.Contains(line, "--kv-cache-dtype fp8") {
		t.Error("fp8 KV cache is not supported for hybrid models")
	}
	// A checkpoint that describes its own quantization must be left to vLLM,
	// which picks fp8-marlin on Ampere and native fp8 on Ada. Naming it
	// ourselves can pin it to a kernel the card does not have.
	if strings.Contains(line, "--quantization") {
		t.Errorf("quantization came from the checkpoint and should not be forced: %s", line)
	}
	// Chunked prefill is still wanted: without it one long prompt stalls every
	// other stream on the replica.
	if !strings.Contains(line, "--enable-chunked-prefill") {
		t.Error("chunked prefill should still be enabled")
	}
}

// TestOrdinaryModelsKeepEveryOptimisation guards against the fix above being
// applied too widely: on a normal transformer both flags are large wins.
func TestOrdinaryModelsKeepEveryOptimisation(t *testing.T) {
	_, args, _, err := BuildCommand(Spec{
		HFID: "Qwen/Qwen3-32B", ServedName: "qwen/qwen3-32b",
		Engine: "vllm", Quantization: "awq",
		KVCacheDType: "fp8", MaxModelLen: 32768, MaxNumSeqs: 16,
	}, Runtime{Mode: "native"}, 18000)
	if err != nil {
		t.Fatal(err)
	}
	line := strings.Join(args, " ")
	for _, want := range []string{
		"--enable-prefix-caching", "--enable-chunked-prefill",
		"--kv-cache-dtype fp8", "--quantization awq",
	} {
		if !strings.Contains(line, want) {
			t.Errorf("missing %s in: %s", want, line)
		}
	}
}

// TestRootCauseSkipsTheRethrow is the diagnostic half of the same failure.
// A chained Python traceback ends with whatever re-raised last -- for vLLM
// always "Engine core initialization failed. See root cause above." -- while
// the line that explains anything is dozens of frames earlier. Reporting the
// last twenty lines therefore showed the operator the one part of the output
// that carries no information, twice in a row.
func TestRootCauseSkipsTheRethrow(t *testing.T) {
	log := []string{
		"(APIServer pid=95984) INFO 10-02 loading model weights",
		"(EngineCore pid=95991) ValueError: fp8 KV cache is not supported for hybrid models",
		"(APIServer pid=95984)   File \"vllm/v1/engine/utils.py\", line 1253, in wait_for_engine_startup",
		"(APIServer pid=95984)     raise RuntimeError(",
		"(APIServer pid=95984) RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}",
	}
	got := rootCause(log)
	if !strings.Contains(got, "fp8 KV cache is not supported") {
		t.Errorf("rootCause = %q, want the ValueError that actually explains it", got)
	}
	if strings.Contains(got, "See root cause above") {
		t.Error("rootCause returned the re-raise, which is what the old code did")
	}
	// The pid decoration must be stripped so the message reads cleanly.
	if strings.HasPrefix(got, "(") {
		t.Errorf("pid prefix was not stripped: %q", got)
	}
}

// TestRootCausePrefersTheSpecificError: among several exceptions, an
// out-of-memory tells an operator what to change where a bare RuntimeError
// does not.
func TestRootCausePrefersTheSpecificError(t *testing.T) {
	log := []string{
		"RuntimeError: something went wrong",
		"torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB",
	}
	if got := rootCause(log); !strings.Contains(got, "OutOfMemoryError") {
		t.Errorf("rootCause = %q, want the out-of-memory", got)
	}
	// Nothing recognisable must return nothing, so the caller falls back to
	// showing the tail rather than printing a misleading line.
	if got := rootCause([]string{"INFO: starting", "INFO: done"}); got != "" {
		t.Errorf("rootCause = %q, want empty when there is no error line", got)
	}
}
