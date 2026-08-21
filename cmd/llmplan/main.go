// Command llmplan answers "can this machine serve this model, and how well"
// before any weights are downloaded or any GPU is rented.
//
//	llmplan Qwen/Qwen3-8B --gpu "RTX 4090:24"
//	llmplan Qwen/Qwen3-32B --cpu 16 --ram 32 --bandwidth 40
//	llmplan Qwen/Qwen3-32B --compare
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/llmfast/gateway/internal/modelspec"
)

func main() {
	var (
		gpuSpec   = flag.String("gpu", "", `GPUs as "NAME:VRAM_GB" repeated with commas, e.g. "H100:80,H100:80"`)
		cores     = flag.Int("cpu", 8, "CPU cores (used when no GPU is given)")
		ramGB     = flag.Int("ram", 32, "system RAM in GB")
		diskGB    = flag.Int("disk", 500, "free disk in GB")
		bandwidth = flag.Float64("bandwidth", 0, "memory bandwidth in GB/s; matters only for CPU inference")
		nvme      = flag.Bool("nvme", false, "node has NVMe storage")
		fraction  = flag.Float64("gpu-fraction", 1, "share of each GPU you get, e.g. 0.5 for a 1/2 MIG or vGPU slice")
		ctxLen    = flag.Int("context", 0, "desired context length (0 = the model's maximum)")
		compare   = flag.Bool("compare", false, "compare across a standard set of machines")
	)
	// Go's flag package stops at the first positional argument, so the model id
	// is pulled out first and the remainder re-parsed. This lets the id appear
	// before the flags, which is the order people naturally type.
	flag.Parse()
	args := flag.Args()
	if len(args) < 1 {
		fmt.Fprintln(os.Stderr, "usage: llmplan <huggingface-model-id> [flags]")
		flag.PrintDefaults()
		os.Exit(2)
	}
	id := args[0]
	if err := flag.CommandLine.Parse(args[1:]); err != nil {
		os.Exit(2)
	}

	client := modelspec.NewClient(os.Getenv("HF_TOKEN"))
	info, err := client.Fetch(context.Background(), id)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("\n%s\n", info.ID)
	fmt.Printf("  %s parameters, published as %s", info.ParamsLabel, info.DType)
	if info.IsMoE {
		fmt.Printf(" (mixture of experts)")
	}
	fmt.Printf("\n  %s, %d layers, %d attention heads, %d KV heads\n",
		info.Architecture, info.NumLayers, info.NumHeads, info.NumKVHeads)
	fmt.Printf("  max context %d tokens, KV cache %.0f KB per token\n",
		info.MaxPositions, float64(info.KVBytesPerToken())/1024)
	if info.Gated {
		fmt.Printf("  NOTE: gated repo, needs an accepted licence and HF_TOKEN\n")
	}
	p, c, cached := modelspec.SuggestPricing(info)
	fmt.Printf("  suggested pricing per million tokens: prompt $%.2f, completion $%.2f, cached $%.3f\n\n",
		perM(p), perM(c), perM(cached))

	var nodes []*modelspec.Node
	if *compare {
		nodes = standardNodes()
	} else {
		nodes = []*modelspec.Node{buildNode(*gpuSpec, *cores, *ramGB, *diskGB, *bandwidth, *nvme, *fraction)}
	}

	for _, n := range nodes {
		report(info, n, *ctxLen)
	}
}

func report(info *modelspec.Info, n *modelspec.Node, ctxLen int) {
	plan := modelspec.PlanFor(info, n, ctxLen)

	mark := "NOT VIABLE"
	if plan.Viable {
		mark = "VIABLE"
	} else if plan.Fits {
		mark = "FITS ONLY"
	}
	fmt.Printf("%-34s  %s\n", n.Name, mark)
	kv := ""
	if plan.KVCacheDType == "fp8" {
		kv = ", fp8 KV cache"
	}
	fmt.Printf("  engine %s, %s weights%s, TP=%d, context %d, ~%d concurrent\n",
		plan.Engine, plan.Quantization, kv, plan.TensorParallel, plan.MaxModelLen, plan.MaxNumSeqs)
	fmt.Printf("  weights %s, KV budget %s, disk %s\n",
		gib(plan.WeightBytes), gib(plan.KVBudgetBytes), gib(plan.DiskBytes))
	if plan.EstTokensPerSec > 0 {
		fmt.Printf("  estimated %.0f tok/s single stream\n", plan.EstTokensPerSec)
	}
	if plan.ViabilityNote != "" {
		fmt.Printf("  %s\n", plan.ViabilityNote)
	}
	for _, w := range plan.Warnings {
		fmt.Printf("  warning: %s\n", w)
	}
	for _, b := range plan.Blockers {
		fmt.Printf("  BLOCKER: %s\n", b)
	}
	fmt.Println()
}

func standardNodes() []*modelspec.Node {
	g := func(name string, gb int64, n int) []modelspec.GPU {
		out := make([]modelspec.GPU, n)
		for i := range out {
			out[i] = modelspec.GPU{Name: name, VRAMBytes: gb << 30, Index: i}
		}
		return out
	}
	return []*modelspec.Node{
		{Name: "CPU server, 32GB DDR4", CPUCores: 16, RAMBytes: 32 << 30,
			DiskFreeBytes: 1000 << 30, MemBandwidthGBs: 40},
		{Name: "1x RTX 4090 24GB", GPUs: g("NVIDIA GeForce RTX 4090", 24, 1),
			CPUCores: 16, RAMBytes: 64 << 30, DiskFreeBytes: 1000 << 30, HasNVMe: true},
		{Name: "1x L40S 48GB", GPUs: g("NVIDIA L40S", 48, 1),
			CPUCores: 16, RAMBytes: 128 << 30, DiskFreeBytes: 2000 << 30, HasNVMe: true},
		{Name: "1x H100 80GB", GPUs: g("NVIDIA H100 80GB HBM3", 80, 1),
			CPUCores: 32, RAMBytes: 256 << 30, DiskFreeBytes: 4000 << 30, HasNVMe: true},
		{Name: "2x H100 80GB", GPUs: g("NVIDIA H100 80GB HBM3", 80, 2),
			CPUCores: 64, RAMBytes: 512 << 30, DiskFreeBytes: 8000 << 30, HasNVMe: true},
		{Name: "8x H200 141GB", GPUs: g("NVIDIA H200", 141, 8),
			CPUCores: 128, RAMBytes: 2048 << 30, DiskFreeBytes: 16000 << 30, HasNVMe: true},
	}
}

func buildNode(gpuSpec string, cores, ramGB, diskGB int, bw float64, nvme bool, frac float64) *modelspec.Node {
	n := &modelspec.Node{
		Name: "this machine", CPUCores: cores,
		RAMBytes: int64(ramGB) << 30, DiskFreeBytes: int64(diskGB) << 30,
		MemBandwidthGBs: bw, HasNVMe: nvme, GPUFraction: frac,
	}
	for i, part := range strings.Split(gpuSpec, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		name, vram, ok := strings.Cut(part, ":")
		if !ok {
			fmt.Fprintf(os.Stderr, "bad --gpu value %q, want NAME:VRAM_GB\n", part)
			os.Exit(2)
		}
		gb, err := strconv.Atoi(strings.TrimSpace(vram))
		if err != nil {
			fmt.Fprintf(os.Stderr, "bad VRAM in %q: %v\n", part, err)
			os.Exit(2)
		}
		n.GPUs = append(n.GPUs, modelspec.GPU{Name: strings.TrimSpace(name), VRAMBytes: int64(gb) << 30, Index: i})
	}
	if len(n.GPUs) > 0 {
		n.Name = fmt.Sprintf("%dx %s", len(n.GPUs), n.GPUs[0].Name)
	}
	return n
}

func perM(perToken string) float64 {
	v, _ := strconv.ParseFloat(perToken, 64)
	return v * 1e6
}

func gib(b int64) string {
	if b <= 0 {
		return "n/a"
	}
	return fmt.Sprintf("%.1f GiB", float64(b)/float64(1<<30))
}
