package agent

import (
	"crypto/subtle"
	"encoding/json"
	"net/http"
	"strconv"
	"time"

	"github.com/llmfast/gateway/internal/modelspec"
)

// Server is the agent's control API. It is deliberately small: the gateway can
// ask for hardware, start a model, stop a model and read logs, and nothing
// else. That is the whole point of running an agent rather than handing the
// gateway an SSH key -- the blast radius is this file.
type Server struct {
	Node    modelspec.Node
	Sup     *Supervisor
	Runtime Runtime
	Token   string
	Version string
	started time.Time
}

func NewServer(node modelspec.Node, sup *Supervisor, rt Runtime, token, version string) *Server {
	return &Server{Node: node, Sup: sup, Runtime: rt, Token: token, Version: version, started: time.Now()}
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/node/info", s.auth(s.handleInfo))
	mux.HandleFunc("POST /v1/node/install", s.auth(s.handleInstall))
	mux.HandleFunc("POST /v1/node/stop", s.auth(s.handleStop))
	mux.HandleFunc("POST /v1/node/remove", s.auth(s.handleRemove))
	mux.HandleFunc("GET /v1/node/logs", s.auth(s.handleLogs))
	mux.HandleFunc("GET /v1/node/cache", s.auth(s.handleCacheList))
	mux.HandleFunc("POST /v1/node/cache/delete", s.auth(s.handleCacheDelete))
	// Unauthenticated liveness, so a load balancer or systemd can probe it.
	mux.HandleFunc("GET /health", s.handleHealth)
	return mux
}

func (s *Server) auth(h http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		got := bearer(r.Header.Get("Authorization"))
		if s.Token == "" || subtle.ConstantTimeCompare([]byte(got), []byte(s.Token)) != 1 {
			writeErr(w, http.StatusUnauthorized, "invalid agent token")
			return
		}
		h(w, r)
	}
}

// Info is the agent's full status: what hardware it has, which engines are
// installed, and what it is currently running.
type Info struct {
	Node             modelspec.Node `json:"node"`
	Instances        []View         `json:"instances"`
	EnginesAvailable []string       `json:"engines_available"`
	RuntimeMode      string         `json:"runtime_mode"`
	Version          string         `json:"version"`
	UptimeSeconds    float64        `json:"uptime_seconds"`
}

func (s *Server) handleInfo(w http.ResponseWriter, r *http.Request) {
	// Disk and GPU memory change while the agent runs, so the volatile parts
	// are re-probed rather than served from the boot-time snapshot.
	node := s.Node
	node.DiskFreeBytes = detectDiskFree(dataDirOf(s.Sup))
	if gpus := detectGPUs(); len(gpus) > 0 {
		node.GPUs = gpus
	}

	var engines []string
	for _, e := range []string{"vllm", "llamacpp"} {
		if EngineAvailable(e, s.Runtime) {
			engines = append(engines, e)
		}
	}
	writeJSON(w, http.StatusOK, Info{
		Node: node, Instances: s.Sup.List(), EnginesAvailable: engines,
		RuntimeMode: s.Runtime.Mode, Version: s.Version,
		UptimeSeconds: time.Since(s.started).Seconds(),
	})
}

func (s *Server) handleInstall(w http.ResponseWriter, r *http.Request) {
	var spec Spec
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<20)).Decode(&spec); err != nil {
		writeErr(w, http.StatusBadRequest, "invalid request body: "+err.Error())
		return
	}
	if spec.ServedName == "" {
		spec.ServedName = spec.HFID
	}
	if !EngineAvailable(spec.Engine, s.Runtime) {
		writeErr(w, http.StatusPreconditionFailed,
			"engine "+spec.Engine+" is not installed on this node")
		return
	}
	in, err := s.Sup.Start(spec)
	if err != nil {
		writeErr(w, http.StatusConflict, err.Error())
		return
	}
	// 202: the process is launched but weights may still be downloading. The
	// caller polls /v1/node/info to watch it reach "ready".
	snap := in.Snapshot()
	writeJSON(w, http.StatusAccepted, map[string]any{
		"state": snap.State, "port": snap.Port, "command": snap.Command,
		"served_name": snap.Spec.ServedName,
	})
}

func (s *Server) handleStop(w http.ResponseWriter, r *http.Request) {
	name, ok := s.nameFromBody(w, r)
	if !ok {
		return
	}
	if err := s.Sup.Stop(name); err != nil {
		writeErr(w, http.StatusNotFound, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]bool{"ok": true})
}

func (s *Server) handleRemove(w http.ResponseWriter, r *http.Request) {
	name, ok := s.nameFromBody(w, r)
	if !ok {
		return
	}
	if err := s.Sup.Remove(name); err != nil {
		writeErr(w, http.StatusNotFound, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]bool{"ok": true})
}

func (s *Server) nameFromBody(w http.ResponseWriter, r *http.Request) (string, bool) {
	var body struct {
		ServedName string `json:"served_name"`
	}
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<16)).Decode(&body); err != nil ||
		body.ServedName == "" {
		writeErr(w, http.StatusBadRequest, "served_name is required")
		return "", false
	}
	return body.ServedName, true
}

func (s *Server) handleLogs(w http.ResponseWriter, r *http.Request) {
	name := r.URL.Query().Get("served_name")
	in, ok := s.Sup.Get(name)
	if !ok {
		writeErr(w, http.StatusNotFound, "no model "+name+" on this node")
		return
	}
	n, _ := strconv.Atoi(r.URL.Query().Get("n"))
	if n <= 0 {
		n = 100
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"served_name": name, "state": in.State(), "error": in.Err(), "lines": in.Logs(n),
	})
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	ready := 0
	for _, v := range s.Sup.List() {
		if v.State == StateReady {
			ready++
		}
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"status": "ok", "models_ready": ready, "uptime_seconds": time.Since(s.started).Seconds(),
	})
}

func dataDirOf(s *Supervisor) string {
	if s == nil || s.stateDir == "" {
		return "."
	}
	return s.stateDir
}

func bearer(h string) string {
	const p = "Bearer "
	if len(h) > len(p) && (h[:len(p)] == p || h[:len(p)] == "bearer ") {
		return h[len(p):]
	}
	return ""
}

func readJSON(r *http.Request, v any) error {
	return json.NewDecoder(http.MaxBytesReader(nil, r.Body, 1<<16)).Decode(v)
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeErr(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]any{"error": map[string]string{"message": msg}})
}

// inUseRepos maps the HuggingFace ids this node is currently serving from, so
// weights cannot be deleted out from under a running engine.
func (s *Server) inUseRepos() map[string]bool {
	out := map[string]bool{}
	for _, v := range s.Sup.List() {
		if v.Spec.HFID != "" {
			out[v.Spec.HFID] = true
		}
		if v.Spec.GGUFRepo != "" {
			out[v.Spec.GGUFRepo] = true
		}
	}
	return out
}

func (s *Server) handleCacheList(w http.ResponseWriter, r *http.Request) {
	entries, err := ListCache(s.Runtime.HFCacheDir, s.inUseRepos())
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	var total int64
	for _, e := range entries {
		total += e.Bytes
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"entries": entries, "total_bytes": total, "dir": s.Runtime.HFCacheDir,
	})
}

func (s *Server) handleCacheDelete(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Repo string `json:"repo"`
	}
	if err := readJSON(r, &body); err != nil {
		writeErr(w, http.StatusBadRequest, "invalid request body")
		return
	}
	freed, err := DeleteCache(s.Runtime.HFCacheDir, body.Repo, s.inUseRepos())
	if err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"repo": body.Repo, "freed_bytes": freed})
}
