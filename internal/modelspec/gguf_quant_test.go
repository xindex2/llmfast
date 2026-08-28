package modelspec

import "testing"

func TestQuantFromFilename(t *testing.T) {
	cases := map[string]string{
		"Qwen3-0.6B-Q8_0.gguf":                     "q8_0",
		"Qwen3-0.6B-Q4_K_M.gguf":                   "q4_k_m",
		"Qwen3-0.6B-IQ4_XS.gguf":                   "iq4_xs",
		"Qwen3-0.6B-BF16.gguf":                     "bf16",
		"qwen3-30b-a3b-Q4_K_M-00001-of-00003.gguf": "q4_k_m",
		"Qwen3-0.6B.gguf":                          "",
		"README.md":                                "",
		"model.safetensors":                        "",
	}
	for name, want := range cases {
		if got := quantFromFilename(name); got != want {
			t.Errorf("quantFromFilename(%q) = %q, want %q", name, got, want)
		}
	}
}

func TestPickQuantFallsBackToWhatExists(t *testing.T) {
	// The official Qwen3-0.6B-GGUF publishes only Q8_0. Asking it for q4_k_m
	// made llama.cpp download nothing and fail with "no GGUF files found in
	// repository", which reads like the repository is empty.
	only8 := GGUFCandidate{Repo: "Qwen/Qwen3-0.6B-GGUF", Quants: []string{"q8_0"}}
	if got := only8.PickQuant("q4_k_m"); got != "q8_0" {
		t.Errorf("PickQuant = %q, want the q8_0 it actually has", got)
	}
	// An exact match is kept.
	many := GGUFCandidate{Quants: []string{"q8_0", "q4_k_m", "bf16"}}
	if got := many.PickQuant("q4_k_m"); got != "q4_k_m" {
		t.Errorf("PickQuant = %q, want q4_k_m", got)
	}
	// Smaller is preferred when the request is absent, since these run on
	// bandwidth-bound CPUs.
	noQ4 := GGUFCandidate{Quants: []string{"bf16", "q8_0", "q6_k"}}
	if got := noQ4.PickQuant("q4_k_m"); got != "q6_k" {
		t.Errorf("PickQuant = %q, want the smallest available (q6_k)", got)
	}
	// Nothing known: pass the request through and let the engine decide.
	if got := (GGUFCandidate{}).PickQuant("q4_k_m"); got != "q4_k_m" {
		t.Errorf("PickQuant = %q, want the request unchanged", got)
	}
}
