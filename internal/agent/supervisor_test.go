package agent

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/llmfast/gateway/internal/modelspec"
)

// stubSource is a miniature inference engine. It is compiled rather than
// written as a shell script because `nc -l` takes different flags on BSD and
// GNU netcat, and these tests have to pass on a developer's Mac and on the
// Ubuntu box they deploy to.
//
// LLMFAST_STUB_BEHAVIOUR selects the failure mode under test.
const stubSource = `package main

import (
	"flag"
	"fmt"
	"net/http"
	"os"
	"time"
)

func main() {
	port := flag.Int("port", 0, "")
	// The real engines take many more flags than this; ignore whatever else
	// arrives so the stub does not need to track their surface.
	flag.CommandLine.Init("stub", flag.ContinueOnError)
	flag.CommandLine.SetOutput(os.NewFile(0, os.DevNull))
	_ = flag.CommandLine.Parse(engineArgs())

	switch os.Getenv("LLMFAST_STUB_BEHAVIOUR") {
	case "crash":
		fmt.Fprintln(os.Stderr, "fatal: simulated engine failure")
		os.Exit(1)
	case "slow":
		fmt.Println("loading weights...")
		time.Sleep(6 * time.Second)
	}

	http.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, ` + "`" + `{"object":"list","data":[{"id":"stub"}]}` + "`" + `)
	})
	fmt.Printf("engine listening on %d\n", *port)
	if err := http.ListenAndServe(fmt.Sprintf("127.0.0.1:%d", *port), nil); err != nil {
		fmt.Fprintln(os.Stderr, "listen:", err)
		os.Exit(1)
	}
}

// engineArgs keeps only the flags this stub understands.
func engineArgs() []string {
	var out []string
	for i, a := range os.Args {
		if a == "--port" && i+1 < len(os.Args) {
			out = append(out, "-port", os.Args[i+1])
		}
	}
	return out
}
`

// stubBinary compiles the stub once per test run and caches the result, since
// `go build` costs about a second and several tests need it.
var (
	stubOnce sync.Once
	stubPath string
	stubErr  error
)

func buildStub(t *testing.T) string {
	t.Helper()
	stubOnce.Do(func() {
		dir, err := os.MkdirTemp("", "llmfast-stub")
		if err != nil {
			stubErr = err
			return
		}
		if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte(stubSource), 0o644); err != nil {
			stubErr = err
			return
		}
		if err := os.WriteFile(filepath.Join(dir, "go.mod"),
			[]byte("module stub\n\ngo 1.24\n"), 0o644); err != nil {
			stubErr = err
			return
		}
		out := filepath.Join(dir, "stub")
		cmd := exec.Command("go", "build", "-o", out, ".")
		cmd.Dir = dir
		if b, err := cmd.CombinedOutput(); err != nil {
			stubErr = fmt.Errorf("build stub: %v\n%s", err, b)
			return
		}
		stubPath = out
	})
	if stubErr != nil {
		t.Fatalf("could not build the engine stub: %v", stubErr)
	}
	return stubPath
}

// fakeEngine puts the compiled stub on PATH under the given engine name, so the
// supervisor's real LookPath and exec paths are exercised.
func fakeEngine(t *testing.T, name, behaviour string) {
	t.Helper()
	stub := buildStub(t)
	dir := t.TempDir()
	link := filepath.Join(dir, name)
	// A copy rather than a symlink: the supervisor resolves the binary through
	// PATH, and a copy behaves identically on every platform.
	data, err := os.ReadFile(stub)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(link, data, 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", dir+string(os.PathListSeparator)+os.Getenv("PATH"))
	t.Setenv("LLMFAST_STUB_BEHAVIOUR", behaviour)
}

func testSupervisor(t *testing.T) *Supervisor {
	t.Helper()
	s := NewSupervisor(Runtime{Mode: "native"}, t.TempDir(), 19000,
		func(f string, a ...any) { t.Logf(f, a...) })
	// Production timings would make the crash-loop test take over a minute.
	s.RestartBackoff = 50 * time.Millisecond
	s.ProbeInterval = 100 * time.Millisecond
	return s
}

func waitState(t *testing.T, s *Supervisor, name, want string, within time.Duration) {
	t.Helper()
	deadline := time.Now().Add(within)
	var last string
	for time.Now().Before(deadline) {
		if in, ok := s.Get(name); ok {
			last = in.State()
			if last == want {
				return
			}
		}
		time.Sleep(100 * time.Millisecond)
	}
	t.Fatalf("model %q was %q after %v, want %q", name, last, within, want)
}

func TestBuildVLLMCommand(t *testing.T) {
	bin, args, _, err := BuildCommand(Spec{
		HFID: "Qwen/Qwen3-32B", ServedName: "qwen/qwen3-32b", Engine: "vllm",
		Quantization: "fp8", TensorParallel: 2, MaxModelLen: 131072, MaxNumSeqs: 128,
	}, Runtime{Mode: "native"}, 18000)
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	if bin != "vllm" {
		t.Errorf("bin = %q, want vllm", bin)
	}
	joined := strings.Join(args, " ")
	for _, want := range []string{
		"serve Qwen/Qwen3-32B",
		"--served-model-name qwen/qwen3-32b",
		"--port 18000",
		"--tensor-parallel-size 2",
		"--quantization fp8",
		"--max-model-len 131072",
		"--max-num-seqs 128",
		// These two are the difference between competitive and not, so they
		// must always be present rather than left to the operator.
		"--enable-prefix-caching",
		"--enable-chunked-prefill",
	} {
		if !strings.Contains(joined, want) {
			t.Errorf("command missing %q\ngot: %s", want, joined)
		}
	}
}

// TestBF16IsNotPassedAsQuantization pins a real vLLM footgun: bf16 is the
// default dtype, not a quantization scheme, and passing it as one is an error.
func TestBF16IsNotPassedAsQuantization(t *testing.T) {
	_, args, _, err := BuildCommand(Spec{
		HFID: "m/m", ServedName: "m/m", Engine: "vllm", Quantization: "bf16",
	}, Runtime{Mode: "native"}, 18000)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(strings.Join(args, " "), "--quantization") {
		t.Errorf("bf16 must not be passed as --quantization: %v", args)
	}
}

func TestBuildDockerCommandIncludesIPCHost(t *testing.T) {
	bin, args, _, err := BuildCommand(Spec{
		HFID: "Qwen/Qwen3-32B", ServedName: "q", Engine: "vllm", TensorParallel: 2,
	}, Runtime{Mode: "docker", HFCacheDir: "/cache"}, 18000)
	if err != nil {
		t.Fatal(err)
	}
	if bin != "docker" {
		t.Fatalf("bin = %q, want docker", bin)
	}
	joined := strings.Join(args, " ")
	// Without --ipc=host, tensor parallelism crashes on the default 64MB
	// /dev/shm with an error that does not name the cause.
	if !strings.Contains(joined, "--ipc=host") {
		t.Error("docker command is missing --ipc=host")
	}
	if !strings.Contains(joined, "--gpus all") {
		t.Error("docker command is missing --gpus all")
	}
	if !strings.Contains(joined, "/cache:/root/.cache/huggingface") {
		t.Error("HF cache is not mounted; weights would be re-downloaded every restart")
	}
	// The image entrypoint is already "vllm serve", so the args must not repeat it.
	if strings.Contains(joined, "vllm/vllm-openai:latest serve") {
		t.Error("duplicated `serve` after the image name")
	}
}

func TestBuildLlamaCppRequiresGGUF(t *testing.T) {
	if _, _, _, err := BuildCommand(Spec{
		HFID: "Qwen/Qwen3-8B", ServedName: "q", Engine: "llamacpp",
	}, Runtime{Mode: "native"}, 18000); err == nil {
		t.Error("expected an error when no GGUF repo is resolved")
	}

	bin, args, _, err := BuildCommand(Spec{
		HFID: "Qwen/Qwen3-8B", ServedName: "qwen/qwen3-8b", Engine: "llamacpp",
		GGUFRepo: "Qwen/Qwen3-8B-GGUF", Quantization: "Q4_K_M", MaxModelLen: 8192, MaxNumSeqs: 4,
	}, Runtime{Mode: "native"}, 18000)
	if err != nil {
		t.Fatal(err)
	}
	if bin != "llama-server" {
		t.Errorf("bin = %q, want llama-server", bin)
	}
	joined := strings.Join(args, " ")
	for _, want := range []string{"-hf Qwen/Qwen3-8B-GGUF:Q4_K_M", "--port 18000",
		"--alias qwen/qwen3-8b", "-c 8192", "-np 4", "--cont-batching"} {
		if !strings.Contains(joined, want) {
			t.Errorf("command missing %q\ngot: %s", want, joined)
		}
	}
}

func TestUnknownEngineRejected(t *testing.T) {
	if _, _, _, err := BuildCommand(Spec{Engine: "tensorrt"}, Runtime{}, 1); err == nil {
		t.Error("expected an error for an unknown engine")
	}
}

func TestStartReachesReady(t *testing.T) {
	fakeEngine(t, "vllm", "ok")
	s := testSupervisor(t)
	defer s.StopAll()

	in, err := s.Start(Spec{HFID: "org/m", ServedName: "org/m", Engine: "vllm"})
	if err != nil {
		t.Fatalf("start: %v", err)
	}
	if in.Port < 19000 {
		t.Errorf("port = %d, want one allocated from the base", in.Port)
	}
	waitState(t, s, "org/m", StateReady, 25*time.Second)

	views := s.List()
	if len(views) != 1 || views[0].State != StateReady {
		t.Fatalf("List() = %+v", views)
	}
	if views[0].Command == "" {
		t.Error("the launch command should be recorded for the operator to see")
	}
}

func TestDuplicateStartRejected(t *testing.T) {
	fakeEngine(t, "vllm", "ok")
	s := testSupervisor(t)
	defer s.StopAll()

	spec := Spec{HFID: "org/m", ServedName: "org/m", Engine: "vllm"}
	if _, err := s.Start(spec); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Start(spec); err == nil {
		t.Error("expected the second start of the same model to be rejected")
	}
}

func TestPortsDoNotCollide(t *testing.T) {
	fakeEngine(t, "vllm", "ok")
	s := testSupervisor(t)
	defer s.StopAll()

	seen := map[int]bool{}
	for i := 0; i < 3; i++ {
		name := fmt.Sprintf("org/m%d", i)
		in, err := s.Start(Spec{HFID: name, ServedName: name, Engine: "vllm"})
		if err != nil {
			t.Fatalf("start %s: %v", name, err)
		}
		if seen[in.Port] {
			t.Fatalf("port %d allocated twice", in.Port)
		}
		seen[in.Port] = true
	}
}

func TestMissingBinaryReportsClearly(t *testing.T) {
	s := testSupervisor(t)
	// No fake engine installed, so the binary is genuinely absent.
	_, err := s.Start(Spec{HFID: "org/m", ServedName: "org/m", Engine: "vllm"})
	if err == nil {
		t.Fatal("expected an error when the engine binary is missing")
	}
	if !strings.Contains(err.Error(), "not installed") {
		t.Errorf("error %q should explain that the engine is not installed", err)
	}
}

func TestStopIsNotTreatedAsACrash(t *testing.T) {
	fakeEngine(t, "vllm", "ok")
	s := testSupervisor(t)

	if _, err := s.Start(Spec{HFID: "org/m", ServedName: "org/m", Engine: "vllm"}); err != nil {
		t.Fatal(err)
	}
	waitState(t, s, "org/m", StateReady, 25*time.Second)

	if err := s.Stop("org/m"); err != nil {
		t.Fatalf("stop: %v", err)
	}
	in, _ := s.Get("org/m")
	if got := in.State(); got != StateStopped {
		t.Errorf("state = %q, want stopped", got)
	}
	// An operator stop must not trigger the restart logic.
	time.Sleep(500 * time.Millisecond)
	if got := in.Restarts(); got != 0 {
		t.Errorf("restarts = %d after an explicit stop, want 0", got)
	}
}

func TestCrashLoopGivesUpWithDiagnostics(t *testing.T) {
	fakeEngine(t, "vllm", "crash")
	s := testSupervisor(t)
	defer s.StopAll()

	if _, err := s.Start(Spec{HFID: "org/m", ServedName: "org/m", Engine: "vllm"}); err != nil {
		t.Fatal(err)
	}
	// Backoff is 5s per attempt over maxRestarts attempts.
	waitState(t, s, "org/m", StateFailed, 30*time.Second)

	in, _ := s.Get("org/m")
	if got := in.Restarts(); got < s.MaxRestarts {
		t.Errorf("restarts = %d, want %d before giving up", got, s.MaxRestarts)
	}
	// The engine's own output is what identifies the cause, so it must reach
	// the operator without them needing to SSH to the node.
	if e := in.Err(); !strings.Contains(e, "simulated engine failure") {
		t.Errorf("error should include the engine's last output, got: %s", e)
	}
}

func TestRemoveForgetsTheInstance(t *testing.T) {
	fakeEngine(t, "vllm", "ok")
	s := testSupervisor(t)
	defer s.StopAll()

	if _, err := s.Start(Spec{HFID: "org/m", ServedName: "org/m", Engine: "vllm"}); err != nil {
		t.Fatal(err)
	}
	if err := s.Remove("org/m"); err != nil {
		t.Fatalf("remove: %v", err)
	}
	if _, ok := s.Get("org/m"); ok {
		t.Error("instance still present after remove")
	}
}

func TestPersistAndRestore(t *testing.T) {
	fakeEngine(t, "vllm", "ok")
	dir := t.TempDir()

	s1 := NewSupervisor(Runtime{Mode: "native"}, dir, 19100, nil)
	s1.ProbeInterval = 100 * time.Millisecond
	if _, err := s1.Start(Spec{HFID: "org/m", ServedName: "org/m", Engine: "vllm"}); err != nil {
		t.Fatal(err)
	}
	waitState(t, s1, "org/m", StateReady, 25*time.Second)
	s1.StopAll()

	// A fresh supervisor over the same state directory, as after a restart.
	s2 := NewSupervisor(Runtime{Mode: "native"}, dir, 19100, nil)
	s2.ProbeInterval = 100 * time.Millisecond
	defer s2.StopAll()
	s2.Restore()

	if _, ok := s2.Get("org/m"); !ok {
		t.Fatal("model was not restored after restart")
	}
}

// TestStoppedModelsDoNotComeBack: an operator who stopped a model does not want
// it resurrected by a restart.
func TestStoppedModelsDoNotComeBack(t *testing.T) {
	fakeEngine(t, "vllm", "ok")
	dir := t.TempDir()

	s1 := NewSupervisor(Runtime{Mode: "native"}, dir, 19200, nil)
	s1.ProbeInterval = 100 * time.Millisecond
	if _, err := s1.Start(Spec{HFID: "org/m", ServedName: "org/m", Engine: "vllm"}); err != nil {
		t.Fatal(err)
	}
	waitState(t, s1, "org/m", StateReady, 25*time.Second)
	if err := s1.Stop("org/m"); err != nil {
		t.Fatal(err)
	}

	s2 := NewSupervisor(Runtime{Mode: "native"}, dir, 19200, nil)
	s2.ProbeInterval = 100 * time.Millisecond
	defer s2.StopAll()
	s2.Restore()
	if _, ok := s2.Get("org/m"); ok {
		t.Error("an explicitly stopped model was restored on restart")
	}
}

// --- control API -----------------------------------------------------------

func testAPI(t *testing.T) (*httptest.Server, *Supervisor) {
	t.Helper()
	sup := testSupervisor(t)
	node := modelspec.Node{Name: "test", CPUCores: 8, RAMBytes: 32 << 30}
	srv := NewServer(node, sup, Runtime{Mode: "native"}, "secret-token", "test")
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(func() { ts.Close(); sup.StopAll() })
	return ts, sup
}

func TestAPIRequiresToken(t *testing.T) {
	ts, _ := testAPI(t)
	resp, err := http.Get(ts.URL + "/v1/node/info")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusUnauthorized {
		t.Errorf("status = %d, want 401 without a token", resp.StatusCode)
	}
}

func TestAPIHealthIsUnauthenticated(t *testing.T) {
	ts, _ := testAPI(t)
	resp, err := http.Get(ts.URL + "/health")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("status = %d, want 200; systemd must be able to probe this", resp.StatusCode)
	}
}

func TestAPIInfoReportsHardware(t *testing.T) {
	ts, _ := testAPI(t)
	req, _ := http.NewRequest("GET", ts.URL+"/v1/node/info", nil)
	req.Header.Set("Authorization", "Bearer secret-token")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	var info Info
	if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
		t.Fatal(err)
	}
	if info.Node.CPUCores != 8 {
		t.Errorf("cpu cores = %d, want 8", info.Node.CPUCores)
	}
	if info.Node.RAMBytes != 32<<30 {
		t.Errorf("ram = %d, want 32GiB", info.Node.RAMBytes)
	}
}

func TestAPIInstallRejectsMissingEngine(t *testing.T) {
	ts, _ := testAPI(t)
	body := strings.NewReader(`{"hf_id":"org/m","served_name":"org/m","engine":"vllm"}`)
	req, _ := http.NewRequest("POST", ts.URL+"/v1/node/install", body)
	req.Header.Set("Authorization", "Bearer secret-token")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	// No engine is installed in this test's PATH.
	if resp.StatusCode != http.StatusPreconditionFailed {
		t.Errorf("status = %d, want 412 when the engine is absent", resp.StatusCode)
	}
}

// TestAgentShutdownPreservesModels covers the distinction between an operator
// stopping a model and the agent process being shut down. Both terminate the
// engine, but only the former should mean "stay down". Conflating them made
// every agent restart come back with nothing running.
func TestAgentShutdownPreservesModels(t *testing.T) {
	fakeEngine(t, "vllm", "ok")
	dir := t.TempDir()

	s1 := NewSupervisor(Runtime{Mode: "native"}, dir, 19300, nil)
	s1.ProbeInterval = 100 * time.Millisecond
	if _, err := s1.Start(Spec{HFID: "org/m", ServedName: "org/m", Engine: "vllm"}); err != nil {
		t.Fatal(err)
	}
	waitState(t, s1, "org/m", StateReady, 25*time.Second)

	// Agent shutdown, not an operator stop.
	s1.StopAll()

	s2 := NewSupervisor(Runtime{Mode: "native"}, dir, 19300, nil)
	s2.ProbeInterval = 100 * time.Millisecond
	defer s2.StopAll()
	s2.Restore()

	if _, ok := s2.Get("org/m"); !ok {
		t.Fatal("a model running at shutdown was not restored; the agent came back empty")
	}
	waitState(t, s2, "org/m", StateReady, 25*time.Second)
}

// TestStopReturnsPromptly guards against the double-Wait bug, where Stop raced
// the reaper for the process and always burned its full grace period before
// resorting to SIGKILL.
func TestStopReturnsPromptly(t *testing.T) {
	fakeEngine(t, "vllm", "ok")
	s := testSupervisor(t)
	defer s.StopAll()

	if _, err := s.Start(Spec{HFID: "org/m", ServedName: "org/m", Engine: "vllm"}); err != nil {
		t.Fatal(err)
	}
	waitState(t, s, "org/m", StateReady, 25*time.Second)

	start := time.Now()
	if err := s.Stop("org/m"); err != nil {
		t.Fatal(err)
	}
	if elapsed := time.Since(start); elapsed > 5*time.Second {
		t.Errorf("Stop took %v; it should observe the process exiting, not wait out the %v grace period",
			elapsed.Round(time.Millisecond), stopGrace)
	}
}

// TestKVCacheDTypeReachesTheCommand: the fp8 KV cache is the largest lever the
// planner has on concurrency, and it is worthless if the flag never reaches the
// engine. "auto" is vLLM's default and should not be passed.
func TestKVCacheDTypeReachesTheCommand(t *testing.T) {
	_, args, _, err := BuildCommand(Spec{
		HFID: "m/m", ServedName: "m/m", Engine: "vllm", KVCacheDType: "fp8",
	}, Runtime{Mode: "native"}, 18000)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(strings.Join(args, " "), "--kv-cache-dtype fp8") {
		t.Errorf("command is missing --kv-cache-dtype fp8: %v", args)
	}

	for _, v := range []string{"", "auto"} {
		_, args, _, err := BuildCommand(Spec{
			HFID: "m/m", ServedName: "m/m", Engine: "vllm", KVCacheDType: v,
		}, Runtime{Mode: "native"}, 18000)
		if err != nil {
			t.Fatal(err)
		}
		if strings.Contains(strings.Join(args, " "), "--kv-cache-dtype") {
			t.Errorf("%q is the default and should not be passed: %v", v, args)
		}
	}
}

// TestCgroupMemoryLimitParsing covers the values a container actually reports.
// Getting this wrong is not cosmetic: /proc/meminfo shows the host's memory
// inside a container, so a pod limited to 50GB on a 512GB machine reports
// 512GB, and the planner's check that weights can be staged into host memory
// becomes a rubber stamp.
func TestCgroupMemoryLimitParsing(t *testing.T) {
	cases := []struct {
		name string
		body string
		want int64
	}{
		{"cgroup v2 limit", "53687091200\n", 53687091200},
		{"cgroup v2 unlimited", "max\n", 0},
		{"cgroup v1 unlimited sentinel", "9223372036854771712\n", 0},
		{"garbage", "not a number\n", 0},
		{"zero", "0\n", 0},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, "memory.max")
			if err := os.WriteFile(path, []byte(c.body), 0o644); err != nil {
				t.Fatal(err)
			}
			got := parseCgroupMemoryFile(path)
			if got != c.want {
				t.Errorf("parsing %q gave %d, want %d", c.body, got, c.want)
			}
		})
	}
}
