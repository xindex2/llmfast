// Command llmfast-agent runs on each inference node.
//
// It reports the node's hardware to the gateway and launches, supervises and
// stops inference engines on request. Running an agent rather than giving the
// gateway SSH access means the gateway can only perform the operations this
// API exposes -- it cannot run arbitrary commands on a machine holding your
// GPUs and your HuggingFace token.
//
// Bind it to a private interface. The token protects the API, but there is no
// reason for it to be reachable from the internet.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/llmfast/gateway/internal/agent"
	"github.com/llmfast/gateway/internal/modelspec"
)

var version = "dev"

func main() {
	var (
		listen    = flag.String("listen", "0.0.0.0:9900", "control API listen address")
		name      = flag.String("name", "", "node name as the gateway will know it (default: hostname)")
		stateDir  = flag.String("state-dir", "/var/lib/llmfast-agent", "where instance state and logs live")
		hfCache   = flag.String("hf-cache", "", "HuggingFace cache directory (default: <state-dir>/hf)")
		mode      = flag.String("mode", "native", `how to launch engines: "native" or "docker"`)
		vllmImage = flag.String("vllm-image", "vllm/vllm-openai:latest", "container image used in docker mode")
		portBase  = flag.Int("port-base", 18000, "first port to allocate to engine processes")
		bandwidth = flag.Float64("mem-bandwidth", 0, "memory bandwidth in GB/s; overrides estimation, matters for CPU inference")
		showHW    = flag.Bool("hardware", false, "print detected hardware and exit")
	)
	flag.Parse()

	log := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))

	nodeName := *name
	if nodeName == "" {
		nodeName, _ = os.Hostname()
	}
	if *hfCache == "" {
		*hfCache = *stateDir + "/hf"
	}

	node := agent.DetectHardware(nodeName, *stateDir, *bandwidth)

	if *showHW {
		printHardware(node)
		return
	}

	// The token is only read from the environment. A credential passed as a
	// flag is visible in `ps` output to every user on the box.
	token := os.Getenv("LLMFAST_AGENT_TOKEN")
	if token == "" {
		fmt.Fprintln(os.Stderr,
			"error: LLMFAST_AGENT_TOKEN is not set.\n\n"+
				"  Generate one and give the same value to the gateway:\n"+
				"    export LLMFAST_AGENT_TOKEN=$(openssl rand -hex 32)")
		os.Exit(1)
	}

	rt := agent.Runtime{
		Mode:       *mode,
		VLLMImage:  *vllmImage,
		HFCacheDir: *hfCache,
		HFToken:    os.Getenv("HF_TOKEN"),
	}

	sup := agent.NewSupervisor(rt, *stateDir, *portBase,
		func(format string, args ...any) { log.Info(fmt.Sprintf(format, args...)) })

	// Ask vLLM for its model registry in the background; it imports the whole
	// library, which is far too slow to do while answering a request.
	agent.StartArchDetection()

	srv := agent.NewServer(node, sup, rt, token, version)

	printHardware(node)
	var engines []string
	for _, e := range []string{"vllm", "llamacpp"} {
		if agent.EngineAvailable(e, rt) {
			engines = append(engines, e)
		}
	}
	if len(engines) == 0 {
		log.Warn("no inference engine found on PATH; install vllm or llama-server before adding a model",
			"mode", *mode)
	} else {
		log.Info("engines available", "engines", engines, "mode", *mode)
	}

	// Bring back whatever was running when this agent last stopped.
	sup.Restore()

	httpSrv := &http.Server{
		Addr:              *listen,
		Handler:           srv.Handler(),
		ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout:       30 * time.Second,
		WriteTimeout:      60 * time.Second,
	}

	errCh := make(chan error, 1)
	go func() {
		log.Info("agent listening", "addr", *listen, "node", nodeName)
		if err := httpSrv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			errCh <- err
		}
	}()

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	select {
	case err := <-errCh:
		log.Error("listener failed", "err", err)
	case s := <-sig:
		log.Info("shutting down", "signal", s.String())
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	_ = httpSrv.Shutdown(ctx)
	// Engines are stopped so they do not survive as orphans holding GPU memory.
	log.Info("stopping engines")
	sup.StopAll()
}

// printHardware writes the detected inventory in a form an operator can sanity
// check at a glance. Getting this wrong silently is the most likely cause of a
// model being planned for hardware that does not exist.
func printHardware(n modelspec.Node) {
	fmt.Printf("\n  node: %s\n", n.Name)
	if n.CPUModel != "" {
		fmt.Printf("  cpu:  %s (%d cores)\n", n.CPUModel, n.CPUCores)
	} else {
		fmt.Printf("  cpu:  %d cores\n", n.CPUCores)
	}
	fmt.Printf("  ram:  %s\n", gib(n.RAMBytes))
	fmt.Printf("  disk: %s free%s\n", gib(n.DiskFreeBytes), nvmeNote(n.HasNVMe))

	if len(n.GPUs) == 0 {
		fmt.Printf("  gpu:  none detected\n")
		fmt.Printf("        CPU inference is limited to roughly %.0f GB/s of memory bandwidth,\n",
			n.MemBandwidthGBs)
		fmt.Printf("        which caps throughput. Use `llmplan <model>` to see what fits.\n\n")
		return
	}
	var total int64
	for _, g := range n.GPUs {
		fmt.Printf("  gpu %d: %s, %s\n", g.Index, g.Name, gib(g.VRAMBytes))
		total += g.VRAMBytes
	}
	fmt.Printf("  total VRAM: %s across %d GPU(s)\n\n", gib(total), len(n.GPUs))
}

func gib(b int64) string {
	if b <= 0 {
		return "unknown"
	}
	return fmt.Sprintf("%.1f GiB", float64(b)/float64(1<<30))
}

func nvmeNote(nvme bool) string {
	if nvme {
		return " (NVMe)"
	}
	return " (no NVMe detected; weight loading will be slow)"
}
