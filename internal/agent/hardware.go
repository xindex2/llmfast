// Package agent is the node-side daemon that runs on each inference host.
//
// It detects what hardware the box actually has, launches and supervises
// inference engines, and reports both back to the gateway. Keeping this on the
// node means the gateway never needs SSH credentials or the ability to run
// arbitrary commands on the GPU fleet: it can only ask for the operations this
// API exposes.
package agent

import (
	"bufio"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/llmfast/gateway/internal/modelspec"
)

// DetectHardware inspects the machine. Every probe degrades gracefully: a
// missing tool yields a zero field rather than an error, because a partial
// inventory is far more useful than a refusal to report anything.
func DetectHardware(name, dataDir string, bandwidthOverride float64) modelspec.Node {
	n := modelspec.Node{
		Name:            name,
		GPUs:            detectGPUs(),
		CPUCores:        runtime.NumCPU(),
		CPUModel:        detectCPUModel(),
		RAMBytes:        detectRAM(),
		DiskFreeBytes:   detectDiskFree(dataDir),
		MemBandwidthGBs: bandwidthOverride,
		HasNVMe:         detectNVMe(dataDir),
	}
	if n.MemBandwidthGBs <= 0 {
		n.MemBandwidthGBs = estimateBandwidth(n.CPUModel, n.CPUCores)
	}
	// nvidia-smi is fast. Importing torch is not -- seconds locally, far longer
	// from a venv on a network filesystem -- so it is filled in behind us and
	// read back on the next hardware refresh rather than delaying startup.
	if len(n.GPUs) > 0 {
		n.DriverCUDA = detectDriverCUDA()
		n.TorchCUDA = TorchCUDA()
		startTorchProbe()
	}
	return n
}

// detectGPUs shells out to nvidia-smi. Parsing its CSV output is more reliable
// across driver versions than binding to NVML.
func detectGPUs() []modelspec.GPU {
	ctx, cancel := execTimeout(5 * time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "nvidia-smi",
		"--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits").Output()
	if err != nil {
		return nil // no NVIDIA driver, or no GPU
	}
	var gpus []modelspec.GPU
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		parts := strings.Split(line, ",")
		if len(parts) < 3 {
			continue
		}
		idx, _ := strconv.Atoi(strings.TrimSpace(parts[0]))
		// nvidia-smi reports memory in MiB with nounits.
		mib, err := strconv.ParseInt(strings.TrimSpace(parts[2]), 10, 64)
		if err != nil {
			continue
		}
		gpus = append(gpus, modelspec.GPU{
			Index:     idx,
			Name:      strings.TrimSpace(parts[1]),
			VRAMBytes: mib * 1024 * 1024,
		})
	}
	return gpus
}

func detectCPUModel() string {
	switch runtime.GOOS {
	case "linux":
		f, err := os.Open("/proc/cpuinfo")
		if err != nil {
			return ""
		}
		defer f.Close()
		sc := bufio.NewScanner(f)
		for sc.Scan() {
			if name, val, ok := strings.Cut(sc.Text(), ":"); ok &&
				strings.TrimSpace(name) == "model name" {
				return strings.TrimSpace(val)
			}
		}
	case "darwin":
		ctx, cancel := execTimeout(3 * time.Second)
		defer cancel()
		if out, err := exec.CommandContext(ctx, "sysctl", "-n", "machdep.cpu.brand_string").Output(); err == nil {
			return strings.TrimSpace(string(out))
		}
	}
	return ""
}

// detectRAM returns the memory this process may actually use.
//
// /proc/meminfo reports the host's memory even inside a container, so a pod
// limited to 50GB on a 512GB machine reports 512GB. That is not a cosmetic
// error: the planner uses this figure to decide whether a model's weights can
// be staged into host memory at all, and an inflated reading turns that check
// into a rubber stamp. The cgroup limit is the real ceiling, so the smaller of
// the two wins.
func detectRAM() int64 {
	host := hostRAM()
	limit := cgroupMemoryLimit()
	if limit > 0 && (host == 0 || limit < host) {
		return limit
	}
	return host
}

// cgroupMemoryLimit reads the container's memory ceiling, trying cgroup v2
// first and falling back to v1. Zero means unlimited or unreadable.
func cgroupMemoryLimit() int64 {
	for _, path := range []string{
		"/sys/fs/cgroup/memory.max",                   // cgroup v2
		"/sys/fs/cgroup/memory/memory.limit_in_bytes", // cgroup v1
	} {
		if n := parseCgroupMemoryFile(path); n > 0 {
			return n
		}
	}
	return 0
}

// parseCgroupMemoryFile reads one cgroup memory file, returning zero when it is
// absent, unreadable, or says the limit is unlimited.
func parseCgroupMemoryFile(path string) int64 {
	raw, err := os.ReadFile(path)
	if err != nil {
		return 0
	}
	v := strings.TrimSpace(string(raw))
	if v == "max" {
		return 0
	}
	n, err := strconv.ParseInt(v, 10, 64)
	if err != nil || n <= 0 {
		return 0
	}
	// cgroup v1 signals "unlimited" with a value near the maximum int64, which
	// would otherwise look like an enormous but legitimate limit.
	if n > 1<<62 {
		return 0
	}
	return n
}

func hostRAM() int64 {
	switch runtime.GOOS {
	case "linux":
		f, err := os.Open("/proc/meminfo")
		if err != nil {
			return 0
		}
		defer f.Close()
		sc := bufio.NewScanner(f)
		for sc.Scan() {
			if key, val, ok := strings.Cut(sc.Text(), ":"); ok && key == "MemTotal" {
				fields := strings.Fields(val) // "32768000 kB"
				if len(fields) > 0 {
					if kb, err := strconv.ParseInt(fields[0], 10, 64); err == nil {
						return kb * 1024
					}
				}
			}
		}
	case "darwin":
		ctx, cancel := execTimeout(3 * time.Second)
		defer cancel()
		if out, err := exec.CommandContext(ctx, "sysctl", "-n", "hw.memsize").Output(); err == nil {
			if b, err := strconv.ParseInt(strings.TrimSpace(string(out)), 10, 64); err == nil {
				return b
			}
		}
	}
	return 0
}

// detectDiskFree reports free space on the filesystem holding path.
//
// The state directory often does not exist yet on a first run, so it walks up
// to the nearest existing ancestor. Reporting "unknown" would make the planner
// skip its disk check entirely, which is the one that catches a 60GB download
// aimed at a 20GB volume.
func detectDiskFree(path string) int64 {
	if path == "" {
		path = "."
	}
	for i := 0; i < 12; i++ {
		var st syscall.Statfs_t
		if err := syscall.Statfs(path, &st); err == nil {
			free := int64(st.Bavail) * int64(st.Bsize)
			// A pod volume is often a share of a cluster filesystem, and
			// statfs then reports the whole cluster: a RunPod volume backed by
			// MooseFS answers 203 TiB free for a 100GB allocation. Presenting
			// that next to a model's disk requirement is worse than presenting
			// nothing, because the check silently always passes.
			if free > networkShareThreshold {
				return 0
			}
			return free
		}
		parent := filepath.Dir(path)
		if parent == path {
			break
		}
		path = parent
	}
	return 0
}

// detectNVMe reports whether the data directory sits on non-rotational storage.
// It matters more than it sounds: loading 60GB of weights off a spinning disk
// turns a restart into a five-minute outage.
func detectNVMe(path string) bool {
	if runtime.GOOS != "linux" {
		return runtime.GOOS == "darwin" // Macs are all flash
	}
	entries, err := os.ReadDir("/sys/block")
	if err != nil {
		return false
	}
	// If any block device is non-rotational we assume the data directory is on
	// it. Mapping a path to its backing device exactly would mean walking
	// mounts and device-mapper layers for very little extra accuracy.
	for _, e := range entries {
		b, err := os.ReadFile("/sys/block/" + e.Name() + "/queue/rotational")
		if err != nil {
			continue
		}
		if strings.TrimSpace(string(b)) == "0" {
			return true
		}
	}
	return false
}

// estimateBandwidth guesses main-memory bandwidth from the CPU description.
//
// This is the single most important number for CPU inference throughput and
// the hardest to detect without root, so it is a deliberately conservative
// guess that the operator can override in the agent config.
func estimateBandwidth(cpuModel string, cores int) float64 {
	m := strings.ToLower(cpuModel)
	switch {
	case strings.Contains(m, "apple m"):
		return 100 // unified memory, varies widely by tier
	case strings.Contains(m, "epyc"), strings.Contains(m, "xeon platinum"),
		strings.Contains(m, "xeon gold"):
		return 150 // modern 8-12 channel DDR4/DDR5
	case strings.Contains(m, "xeon e5"):
		return 40 // quad-channel DDR4, and often only half the slots populated
	case strings.Contains(m, "xeon"):
		return 60
	}
	if cores >= 32 {
		return 80
	}
	return 40
}

// detectDriverCUDA reads the highest CUDA version this host's driver supports.
//
// nvidia-smi prints it in its header banner rather than exposing it as a
// --query-gpu field, so the banner is what gets parsed.
func detectDriverCUDA() string {
	ctx, cancel := execTimeout(5 * time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "nvidia-smi").Output()
	if err != nil {
		return ""
	}
	m := cudaBanner.FindSubmatch(out)
	if m == nil {
		return ""
	}
	return string(m[1])
}

var cudaBanner = regexp.MustCompile(`CUDA Version:\s*([0-9]+\.[0-9]+)`)

// detectTorchCUDA reports the CUDA version the installed PyTorch was compiled
// against, which is what actually has to be no newer than the driver.
//
// Importing torch is slow -- several seconds -- so this runs once when the
// agent starts its hardware scan, not per install.
func detectTorchCUDA() string {
	ctx, cancel := execTimeout(60 * time.Second)
	defer cancel()
	for _, py := range []string{"python3", "python"} {
		out, err := exec.CommandContext(ctx, py, "-c",
			"import torch,sys; sys.stdout.write(torch.version.cuda or \"\")").Output()
		if err != nil {
			continue
		}
		if v := strings.TrimSpace(string(out)); v != "" {
			return v
		}
	}
	return ""
}

// networkShareThreshold is the point past which a free-space figure is more
// likely to be a shared cluster's total than this node's own allocation. No
// pod rents a petabyte.
const networkShareThreshold = 100 << 40 // 100 TiB

// torchCUDA caches the CUDA version PyTorch was built against.
var torchCUDA atomic.Pointer[string]

// TorchCUDA returns the cached value, or "" if the probe has not finished.
// Callers treat "" as unknown, never as a fault.
func TorchCUDA() string {
	if p := torchCUDA.Load(); p != nil {
		return *p
	}
	return ""
}

var torchProbeOnce sync.Once

// startTorchProbe imports torch once, in the background.
func startTorchProbe() {
	torchProbeOnce.Do(func() {
		go func() {
			if v := detectTorchCUDA(); v != "" {
				torchCUDA.Store(&v)
			}
		}()
	})
}
