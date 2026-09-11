package modelspec

import "testing"

// A checkpoint that mixes precisions weighs what its shards weigh, not what a
// single bytes-per-parameter figure predicts.
//
// DeepSeek V4.1 Flash is the case that forced this: quant_method is fp8, but
// expert_dtype is fp4, and HuggingFace reports those experts as 557B parameters
// of "I8" -- the container they are packed into, two weights to the byte. Taken
// at face value the model sizes at 711 GiB; the shards sum to 510 GB. Eight
// H100s hold one and not the other.
func TestWeightBytesPrefersTheMeasurement(t *testing.T) {
	const measured = 510 << 30

	info := &Info{
		Params:          763_200_000_000,
		CheckpointBytes: measured,
		PublishedQuant:  "fp8",
	}

	if got := weightBytes(info, "fp8", true); got != measured {
		t.Errorf("served as published: got %s, want the measured %s",
			humanBytes(got), humanBytes(measured))
	}

	// Re-quantizing to a format nobody has published yet has nothing to
	// measure, so the estimate is still the best available answer.
	want := int64(float64(info.Params) * quantBytes("awq"))
	if got := weightBytes(info, "awq", false); got != want {
		t.Errorf("re-quantized: got %s, want the estimate %s",
			humanBytes(got), humanBytes(want))
	}

	// And a repository that never reported sizes falls back rather than
	// claiming a model weighs nothing.
	unmeasured := &Info{Params: info.Params}
	if got := weightBytes(unmeasured, "fp8", true); got <= 0 {
		t.Errorf("unmeasured checkpoint sized at %d bytes", got)
	}
}

// Sub-byte and integer dtypes must not be silently treated as bf16, which would
// double every figure derived from a local directory's size on disk.
func TestQuantBytesForDType(t *testing.T) {
	for dtype, want := range map[string]float64{
		"F32": 4, "BF16": 2, "F16": 2, "F8_E4M3": 1, "I8": 1, "I4": 0.5,
		"something-new": 2,
	} {
		if got := quantBytesForDType(dtype); got != want {
			t.Errorf("%s: got %v bytes per parameter, want %v", dtype, got, want)
		}
	}
}
