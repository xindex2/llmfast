package agent

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"syscall"
	"time"
)

// Instance lifecycle states.
const (
	StateStarting = "starting" // process launched, weights may still be downloading
	StateReady    = "ready"    // engine answered a health probe
	StateFailed   = "failed"   // exited and will not be retried
	StateStopped  = "stopped"  // stopped on request
)

const (
	logRingSize    = 400
	readyProbeFreq = 3 * time.Second
	// Weight downloads for a large model over a slow link genuinely take this
	// long, and giving up early would leave a half-downloaded cache behind.
	readyTimeout = 2 * time.Hour
	// A crashing engine is usually misconfigured rather than unlucky, so the
	// supervisor gives up rather than restarting forever and burning GPU time
	// on a model that will never start.
	defaultMaxRestarts    = 5
	defaultRestartBackoff = 5 * time.Second
	stopGrace             = 30 * time.Second
)

// Instance is one engine process serving one model.
//
// Spec and Port are fixed at construction. Everything else is mutated by the
// log capture, readiness probe and reaper goroutines, so it is private and
// guarded by mu; callers read it through Snapshot.
type Instance struct {
	Spec Spec
	Port int

	mu        sync.Mutex
	state     string
	errMsg    string
	pid       int
	command   string
	startedAt time.Time
	readyAt   time.Time
	restarts  int
	logs      []string

	cmd    *exec.Cmd
	cancel context.CancelFunc
	// stopping distinguishes an operator stop from a crash, so the supervisor
	// does not fight the operator by restarting what they just shut down.
	stopping bool
	// desired records whether this model should be running. An operator stop
	// clears it; an agent shutdown does not. That difference is what makes
	// "restart the agent" bring models back while "stop this model" keeps it
	// down.
	desired bool
	// exited is closed by reap once the process is gone. Stop waits on this
	// rather than calling cmd.Wait itself: two concurrent Waits on one Cmd is
	// undefined behaviour.
	exited chan struct{}
}

// View is the JSON-safe snapshot sent to the gateway.
type View struct {
	Spec      Spec      `json:"spec"`
	Port      int       `json:"port"`
	State     string    `json:"state"`
	PID       int       `json:"pid"`
	StartedAt time.Time `json:"started_at"`
	ReadyAt   time.Time `json:"ready_at,omitempty"`
	Error     string    `json:"error,omitempty"`
	Restarts  int       `json:"restarts"`
	Command   string    `json:"command"`
}

func (in *Instance) Snapshot() View {
	in.mu.Lock()
	defer in.mu.Unlock()
	return View{
		Spec: in.Spec, Port: in.Port, State: in.state, PID: in.pid,
		StartedAt: in.startedAt, ReadyAt: in.readyAt, Error: in.errMsg,
		Restarts: in.restarts, Command: in.command,
	}
}

func (in *Instance) State() string {
	in.mu.Lock()
	defer in.mu.Unlock()
	return in.state
}

func (in *Instance) Restarts() int {
	in.mu.Lock()
	defer in.mu.Unlock()
	return in.restarts
}

func (in *Instance) Err() string {
	in.mu.Lock()
	defer in.mu.Unlock()
	return in.errMsg
}

func (in *Instance) setState(state, errMsg string) {
	in.mu.Lock()
	in.state, in.errMsg = state, errMsg
	in.mu.Unlock()
}

func (in *Instance) appendLog(line string) {
	in.mu.Lock()
	defer in.mu.Unlock()
	in.logs = append(in.logs, line)
	if len(in.logs) > logRingSize {
		in.logs = in.logs[len(in.logs)-logRingSize:]
	}
}

// Logs returns the last n captured lines.
func (in *Instance) Logs(n int) []string {
	in.mu.Lock()
	defer in.mu.Unlock()
	if n <= 0 || n > len(in.logs) {
		n = len(in.logs)
	}
	out := make([]string, n)
	copy(out, in.logs[len(in.logs)-n:])
	return out
}

type Supervisor struct {
	rt       Runtime
	stateDir string
	portBase int

	mu        sync.RWMutex
	instances map[string]*Instance // keyed by served model name

	// Restart policy, exposed as fields so tests can exercise the crash-loop
	// path without waiting through the production backoff.
	MaxRestarts    int
	RestartBackoff time.Duration
	// ProbeInterval is how often a starting engine is polled for readiness.
	ProbeInterval time.Duration

	log func(string, ...any)
}

func NewSupervisor(rt Runtime, stateDir string, portBase int, logf func(string, ...any)) *Supervisor {
	if portBase <= 0 {
		portBase = 18000
	}
	if logf == nil {
		logf = func(string, ...any) {}
	}
	return &Supervisor{
		rt: rt, stateDir: stateDir, portBase: portBase,
		instances:      map[string]*Instance{},
		MaxRestarts:    defaultMaxRestarts,
		RestartBackoff: defaultRestartBackoff,
		ProbeInterval:  readyProbeFreq,
		log:            logf,
	}
}

// Start launches an engine for a spec. It returns as soon as the process is
// running; readiness is reported asynchronously through the instance state,
// because a first-time weight download can take an hour and the caller must not
// block on it.
func (s *Supervisor) Start(spec Spec) (*Instance, error) {
	if spec.ServedName == "" {
		return nil, fmt.Errorf("served_name is required")
	}
	s.mu.Lock()
	if existing, ok := s.instances[spec.ServedName]; ok {
		if st := existing.State(); st == StateReady || st == StateStarting {
			s.mu.Unlock()
			return nil, fmt.Errorf("model %q is already %s on this node", spec.ServedName, st)
		}
	}
	port, err := s.allocatePort()
	if err != nil {
		s.mu.Unlock()
		return nil, err
	}
	in := &Instance{
		Spec: spec, Port: port, state: StateStarting, startedAt: time.Now(),
		desired: true, exited: make(chan struct{}),
	}
	s.instances[spec.ServedName] = in
	s.mu.Unlock()

	if err := s.launch(in); err != nil {
		in.setState(StateFailed, err.Error())
		return in, err
	}
	s.persist()
	return in, nil
}

func (s *Supervisor) launch(in *Instance) error {
	bin, args, env, err := BuildCommand(in.Spec, s.rt, in.Port)
	if err != nil {
		return err
	}
	if _, err := exec.LookPath(bin); err != nil {
		return fmt.Errorf("%s is not installed on this node (looked for %q on PATH)", in.Spec.Engine, bin)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cmd := exec.CommandContext(ctx, bin, args...)
	cmd.Env = append(os.Environ(), env...)
	// A process group lets us signal the engine and every worker it forks.
	// vLLM spawns one worker per GPU, and killing only the parent orphans them
	// still holding VRAM, which blocks the next model from loading.
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		cancel()
		return err
	}
	cmd.Stderr = cmd.Stdout // engines log to both; one ordered stream is easier to read

	if err := cmd.Start(); err != nil {
		cancel()
		return fmt.Errorf("start %s: %w", bin, err)
	}

	in.mu.Lock()
	in.cmd, in.cancel = cmd, cancel
	in.pid = cmd.Process.Pid
	in.command = bin + " " + shellJoin(args)
	in.stopping = false
	pid := in.pid
	in.mu.Unlock()

	s.log("launched %s pid=%d port=%d", in.Spec.ServedName, pid, in.Port)

	go s.captureLogs(in, stdout)
	go s.waitForReady(in)
	go s.reap(in, cmd)
	return nil
}

func (s *Supervisor) captureLogs(in *Instance, r io.ReadCloser) {
	defer r.Close()
	sc := bufio.NewScanner(r)
	// Engine startup lines (CUDA graph dumps, full configs) can be very long.
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		in.appendLog(sc.Text())
	}
}

// waitForReady probes the engine's own /v1/models until it answers.
func (s *Supervisor) waitForReady(in *Instance) {
	probe := s.ProbeInterval
	if probe <= 0 {
		probe = readyProbeFreq
	}
	deadline := time.Now().Add(readyTimeout)
	client := &http.Client{Timeout: 4 * time.Second}
	url := fmt.Sprintf("http://127.0.0.1:%d/v1/models", in.Port)

	for time.Now().Before(deadline) {
		time.Sleep(probe)
		switch in.State() {
		case StateStopped, StateFailed:
			return
		}
		resp, err := client.Get(url)
		if err != nil {
			continue
		}
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			continue
		}
		in.mu.Lock()
		// Losing a race with a stop must not resurrect the instance's state.
		if in.state == StateStopped || in.state == StateFailed {
			in.mu.Unlock()
			return
		}
		in.state, in.errMsg, in.readyAt = StateReady, "", time.Now()
		since := time.Since(in.startedAt)
		in.mu.Unlock()

		s.log("%s ready on port %d after %s", in.Spec.ServedName, in.Port, since.Round(time.Second))
		s.persist()
		return
	}
	in.setState(StateFailed, "engine did not become ready within "+readyTimeout.String())
}

// reap waits for the process and restarts it if it died unexpectedly.
func (s *Supervisor) reap(in *Instance, cmd *exec.Cmd) {
	err := cmd.Wait()

	in.mu.Lock()
	stopping := in.stopping
	// Signal Stop, and anyone else waiting, that the process is truly gone.
	select {
	case <-in.exited:
	default:
		close(in.exited)
	}
	in.mu.Unlock()

	if stopping {
		in.setState(StateStopped, "")
		return
	}

	tail := in.Logs(logRingSize)
	reason := "exited"
	if err != nil {
		reason = err.Error()
	}
	s.log("%s exited: %s", in.Spec.ServedName, reason)

	in.mu.Lock()
	if in.restarts >= s.MaxRestarts {
		in.state = StateFailed
		// Report the root cause, not the tail. A Python traceback ends with
		// whatever re-raised last -- for vLLM that is invariably
		// "Engine core initialization failed. See root cause above." -- while
		// the line that explains anything is dozens of frames earlier. Showing
		// the last twenty lines therefore showed the operator the one part of
		// the output that carries no information.
		msg := fmt.Sprintf("%s (gave up after %d restarts)", reason, in.restarts)
		if cause := rootCause(tail); cause != "" {
			msg += ".\n" + cause
		} else if len(tail) > 0 {
			msg += ". Last output:\n" + joinLines(lastN(tail, 20))
		}
		in.errMsg = msg
		in.mu.Unlock()
		s.persist()
		return
	}
	in.restarts++
	in.state = StateStarting
	in.errMsg = fmt.Sprintf("restarting after: %s", reason)
	in.exited = make(chan struct{})
	attempt := in.restarts
	in.mu.Unlock()

	// Back off so a model that crashes instantly does not spin the CPU.
	time.Sleep(time.Duration(attempt) * s.RestartBackoff)
	if err := s.launch(in); err != nil {
		in.setState(StateFailed, err.Error())
	}
	s.persist()
}

// Stop terminates a model on operator request. It will not come back on an
// agent restart.
func (s *Supervisor) Stop(name string) error { return s.stop(name, true) }

// stop terminates an instance's process group. clearDesired distinguishes an
// operator stopping a model from the agent shutting down: only the former
// should survive a restart as "stay down".
func (s *Supervisor) stop(name string, clearDesired bool) error {
	s.mu.RLock()
	in, ok := s.instances[name]
	s.mu.RUnlock()
	if !ok {
		return fmt.Errorf("no model %q on this node", name)
	}

	in.mu.Lock()
	in.stopping = true
	if clearDesired {
		in.desired = false
	}
	exited, cmd, cancel := in.exited, in.cmd, in.cancel
	in.mu.Unlock()

	if cmd != nil && cmd.Process != nil {
		// The negative PID targets the whole process group.
		_ = syscall.Kill(-cmd.Process.Pid, syscall.SIGTERM)
		select {
		case <-exited:
		case <-time.After(stopGrace):
			_ = syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
			<-exited
		}
	}
	if cancel != nil {
		cancel()
	}
	in.setState(StateStopped, "")
	s.persist()
	return nil
}

// Remove stops an instance and forgets it entirely.
func (s *Supervisor) Remove(name string) error {
	if err := s.stop(name, true); err != nil {
		return err
	}
	s.mu.Lock()
	delete(s.instances, name)
	s.mu.Unlock()
	s.persist()
	return nil
}

func (s *Supervisor) List() []View {
	s.mu.RLock()
	instances := make([]*Instance, 0, len(s.instances))
	for _, in := range s.instances {
		instances = append(instances, in)
	}
	s.mu.RUnlock()

	out := make([]View, 0, len(instances))
	for _, in := range instances {
		out = append(out, in.Snapshot())
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Spec.ServedName < out[j].Spec.ServedName })
	return out
}

func (s *Supervisor) Get(name string) (*Instance, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	in, ok := s.instances[name]
	return in, ok
}

// StopAll shuts every engine down for agent shutdown. It deliberately does not
// clear the desired state, so restarting the agent brings the same models back.
func (s *Supervisor) StopAll() {
	for _, v := range s.List() {
		_ = s.stop(v.Spec.ServedName, false)
	}
}

// allocatePort finds a free port at or above the base. Callers hold s.mu.
func (s *Supervisor) allocatePort() (int, error) {
	used := map[int]bool{}
	for _, in := range s.instances {
		used[in.Port] = true
	}
	for p := s.portBase; p < s.portBase+200; p++ {
		if used[p] {
			continue
		}
		// Confirm with the kernel: another process on the box may hold it.
		l, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", p))
		if err != nil {
			continue
		}
		l.Close()
		return p, nil
	}
	return 0, fmt.Errorf("no free port in range %d-%d", s.portBase, s.portBase+200)
}

// --- persistence -----------------------------------------------------------
//
// Specs are written to disk so an agent restart brings the same models back
// automatically. Only the desired state is stored; ports and PIDs are
// reallocated on restore because they describe a process that no longer exists.

type persistedState struct {
	Specs []Spec `json:"specs"`
}

func (s *Supervisor) statePath() string { return filepath.Join(s.stateDir, "instances.json") }

func (s *Supervisor) persist() {
	if s.stateDir == "" {
		return
	}
	// Persist the desired state, not the current one: a model that is
	// momentarily restarting, or that was running when the agent was told to
	// shut down, should still come back.
	var st persistedState
	s.mu.RLock()
	for _, in := range s.instances {
		in.mu.Lock()
		want := in.desired
		in.mu.Unlock()
		if want {
			st.Specs = append(st.Specs, in.Spec)
		}
	}
	s.mu.RUnlock()
	sort.Slice(st.Specs, func(i, j int) bool { return st.Specs[i].ServedName < st.Specs[j].ServedName })

	if err := os.MkdirAll(s.stateDir, 0o750); err != nil {
		s.log("persist: %v", err)
		return
	}
	b, err := json.MarshalIndent(st, "", "  ")
	if err != nil {
		return
	}
	tmp := s.statePath() + ".tmp"
	if err := os.WriteFile(tmp, b, 0o640); err != nil {
		s.log("persist: %v", err)
		return
	}
	// Rename is atomic, so a crash mid-write cannot leave a truncated file that
	// would lose every model on the next restart.
	if err := os.Rename(tmp, s.statePath()); err != nil {
		s.log("persist: %v", err)
	}
}

// Restore relaunches the models that were running when the agent last stopped.
func (s *Supervisor) Restore() {
	b, err := os.ReadFile(s.statePath())
	if err != nil {
		return
	}
	var st persistedState
	if err := json.Unmarshal(b, &st); err != nil {
		s.log("restore: state file is corrupt: %v", err)
		return
	}
	for _, spec := range st.Specs {
		if _, err := s.Start(spec); err != nil {
			s.log("restore %s: %v", spec.ServedName, err)
		}
	}
	if n := len(st.Specs); n > 0 {
		s.log("restoring %d model(s) from previous run", n)
	}
}

func shellJoin(args []string) string {
	out := ""
	for i, a := range args {
		if i > 0 {
			out += " "
		}
		out += a
	}
	return out
}

func joinLines(lines []string) string {
	out := ""
	for _, l := range lines {
		out += "  " + l + "\n"
	}
	return out
}

// uselessErrors are the lines a failing engine ends with. They are re-raises
// from an outer frame and say nothing about what actually went wrong, but they
// are what any "show me the last N lines" view displays.
var uselessErrors = []string{
	"Engine core initialization failed",
	"See root cause above",
	"EngineCore failed to start",
	"Engine process failed to start",
}

// interestingErrors are the exception types worth surfacing, most specific
// first: an out-of-memory or an unsupported-feature message tells an operator
// what to change, where a bare RuntimeError often does not.
// buildDefects are failures of the engine binary rather than of the model.
// They are matched first because their line does not look like an exception
// and would otherwise be skipped entirely.
var buildDefects = map[string]string{
	"HTTPS is not supported": "this llama.cpp was built without HTTPS, so it cannot download " +
		"models: install libssl-dev and rebuild with -DLLAMA_OPENSSL=ON",
	"CUDA error": "the engine could not use the GPU",
}

var interestingErrors = []string{
	"torch.cuda.OutOfMemoryError",
	"OutOfMemoryError",
	"NotImplementedError",
	"ValueError",
	"AssertionError",
	"TypeError",
	"KeyError",
	"OSError",
	"ImportError",
	"ModuleNotFoundError",
	"RuntimeError",
}

// rootCause picks the line from an engine's output that explains the failure.
//
// It scans forwards, because in a chained traceback the earliest exception is
// the one that actually happened and everything after it is a re-raise.
func rootCause(lines []string) string {
	for _, raw := range lines {
		for marker, explain := range buildDefects {
			if strings.Contains(raw, marker) {
				return explain
			}
		}
	}
	if q := missingQuantization(lines); q != "" {
		return q
	}
	best, bestRank := "", len(interestingErrors)
	for _, raw := range lines {
		line := strings.TrimSpace(stripPIDPrefix(raw))
		if line == "" {
			continue
		}
		skip := false
		for _, junk := range uselessErrors {
			if strings.Contains(line, junk) {
				skip = true
				break
			}
		}
		if skip {
			continue
		}
		for rank, kind := range interestingErrors {
			if !strings.Contains(line, kind+":") {
				continue
			}
			// A more specific exception replaces a vaguer one, but among
			// equals the earliest wins -- that is the original raise.
			if rank < bestRank {
				best, bestRank = line, rank
			}
			break
		}
	}
	if len(best) > 400 {
		best = best[:400] + "…"
	}
	return best
}

// stripPIDPrefix removes vLLM's "(APIServer pid=123) " decoration so the
// matching above sees the message itself.
func stripPIDPrefix(line string) string {
	if !strings.HasPrefix(line, "(") {
		return line
	}
	if i := strings.Index(line, ") "); i > 0 && i < 40 {
		return line[i+2:]
	}
	return line
}

func lastN(lines []string, n int) []string {
	if len(lines) <= n {
		return lines
	}
	return lines[len(lines)-n:]
}

// missingQuantization turns llama.cpp's "no GGUF files found in repository"
// into something actionable.
//
// The message is misleading on its own: the repository is not empty, it simply
// does not publish the quantization that was asked for. llama.cpp prints the
// ones it does have on the following lines, so the answer is right there and
// worth lifting into the error rather than leaving in the log.
func missingQuantization(lines []string) string {
	repo, listing := "", false
	var have []string
	for _, raw := range lines {
		line := strings.TrimSpace(stripPIDPrefix(raw))
		if i := strings.Index(line, "no GGUF files found in repository"); i >= 0 {
			repo = strings.TrimSpace(line[i+len("no GGUF files found in repository"):])
			listing = false
			have = nil
			continue
		}
		if repo == "" {
			continue
		}
		if strings.Contains(line, "Available GGUF files") {
			listing = true
			continue
		}
		if listing {
			if f := strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(line), "-")); strings.HasSuffix(strings.ToLower(f), ".gguf") {
				have = append(have, f)
				continue
			}
			listing = false
		}
	}
	if repo == "" {
		return ""
	}
	msg := fmt.Sprintf("%s does not publish the quantization that was requested", repo)
	if len(have) > 0 {
		msg += ". It has: " + strings.Join(have, ", ") +
			" -- pick a repository that offers the format you want, or install with one of these"
	}
	return msg
}
