package modelspec

import (
	"context"
	"os"
	"testing"
	"time"
)

// TestLivePlanAgainstHuggingFace checks the planner against the real published
// metadata rather than a fixture, because the failure this guards against was
// a mismatch between what we assumed a repository contained and what it did.
// It needs the network, so it is opt-in:
//
//	LIVE=1 go test ./internal/modelspec -run TestLivePlan -v
func TestLivePlanAgainstHuggingFace(t *testing.T) {
	if os.Getenv("LIVE") == "" {
		t.Skip("needs network; set LIVE=1")
	}
	c := NewClient("")
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	a40 := &Node{Name: "gpu-a", CPUCores: 32, RAMBytes: 46 << 30, DiskFreeBytes: 500 << 30, HasNVMe: true,
		GPUs: []GPU{{Name: "NVIDIA A40", VRAMBytes: 48 << 30}}}

	for _, id := range []string{"Qwen/Qwen3.8-27B", "Qwen/Qwen3.8-27B-FP8"} {
		info, err := c.Fetch(ctx, id)
		if err != nil {
			t.Fatalf("%s: %v", id, err)
		}
		p := PlanFor(info, a40, 32768)
		t.Logf("\n=== %s ===\n  params=%.1fB published_quant=%q\n  plan: quant=%s ctx=%d seqs=%d fits=%v viable=%v\n  weights=%s kv/tok=%d kvbudget=%s\n  blockers=%v\n  needs_quantized=%v",
			id, float64(info.Params)/1e9, info.PublishedQuant,
			p.Quantization, p.MaxModelLen, p.MaxNumSeqs, p.Fits, p.Viable,
			humanBytes(p.WeightBytes), p.KVBytesPerTok, humanBytes(p.KVBudgetBytes),
			p.Blockers, p.NeedsQuantized)
	}

	cands, err := c.ResolveQuantized(ctx, "Qwen/Qwen3.8-27B")
	if err != nil {
		t.Fatalf("resolve: %v", err)
	}
	t.Log("\n=== quantized publications found ===")
	for _, q := range cands {
		t.Logf("  %-45s %-6s official=%v dl=%d", q.Repo, q.Quant, q.Official, q.Downloads)
	}
}
