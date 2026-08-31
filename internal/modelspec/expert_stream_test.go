package modelspec

import "testing"

// TestExpertStreamingOfferedOnlyWhereItHelps: an MoE that overflows VRAM can
// still be served by streaming cold experts from host RAM. A dense model
// cannot -- it reads every weight for every token, so PCIe is not a slower
// path but a hopeless one.
func TestExpertStreamingOfferedOnlyWhereItHelps(t *testing.T) {
	node := gpuNode("NVIDIA A40", 45, 1)
	node.RAMBytes = 128 << 30

	moe := qwen32B()
	moe.Params = 117_000_000_000
	moe.ActiveParams = 5_100_000_000
	moe.IsMoE = true
	moe.PublishedQuant = "awq" // a format Ampere can actually execute
	p := PlanFor(moe, node, 32768)
	if p.Fits {
		t.Fatal("70 GiB of weights cannot fit 45 GiB of VRAM")
	}
	if !p.CanStreamExperts {
		t.Errorf("an MoE that fits VRAM+RAM should offer expert streaming: %v", p.Blockers)
	}

	dense := qwen32B()
	dense.Params = 117_000_000_000
	dense.IsMoE = false
	dense.ActiveParams = 0
	dense.PublishedQuant = "awq"
	if q := PlanFor(dense, node, 32768); q.CanStreamExperts {
		t.Error("a dense model must not be offered expert streaming")
	}

	// Too big even for VRAM plus RAM.
	huge := qwen32B()
	huge.Params = 671_000_000_000
	huge.ActiveParams = 37_000_000_000
	huge.IsMoE = true
	huge.PublishedQuant = "awq"
	if q := PlanFor(huge, node, 32768); q.CanStreamExperts {
		t.Error("400 GiB does not fit VRAM plus 128 GiB of RAM")
	}
}
