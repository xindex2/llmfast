// Package gateway is the OpenAI-compatible edge that OpenRouter talks to.
//
// It owns three surfaces: the inference API (/v1/chat/completions and friends),
// the provider model document (/v1/models), and the admin API behind a separate
// listener. The admin listener is separate on purpose -- it should be bound to
// localhost or a private interface and never exposed alongside the public API.
package gateway

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"sync"
	"sync/atomic"
	"time"

	"github.com/llmfast/gateway/internal/config"
	"github.com/llmfast/gateway/internal/modeldoc"
	"github.com/llmfast/gateway/internal/modelspec"
	"github.com/llmfast/gateway/internal/store"
	"github.com/llmfast/gateway/internal/upstream"
)

type Server struct {
	cfg     *config.Config
	store   *store.Store
	pool    *upstream.Pool
	catalog *Catalog
	keys    *keyCache
	limiter *rateLimiter
	log     *slog.Logger

	// Provisioning: talking to node agents and resolving models on HuggingFace.
	nodes *NodeManager
	hf    *modelspec.Client
	// configPath and modelDir are needed to persist a newly installed model and
	// reload the catalog without a restart.
	configPath string
	modelDir   string
	// reloadMu serializes config reloads so two concurrent installs cannot
	// interleave a read-modify-write of the catalog.
	reloadMu sync.Mutex

	// modelsJSON is the rendered provider document. It only changes on config
	// reload, so it is built once and served as bytes -- OpenRouter polls this
	// endpoint and there is no reason to re-marshal it every time.
	modelsJSON atomic.Value // []byte
	startedAt  time.Time
}

func New(cfg *config.Config, st *store.Store, pool *upstream.Pool, log *slog.Logger) *Server {
	s := &Server{
		cfg:       cfg,
		store:     st,
		pool:      pool,
		catalog:   NewCatalog(cfg),
		keys:      newKeyCache(),
		limiter:   newRateLimiter(),
		log:       log,
		hf:        modelspec.NewClient(os.Getenv("HF_TOKEN")),
		startedAt: time.Now(),
	}
	s.nodes = NewNodeManager(cfg.Nodes, pool, log)
	s.rebuildModels()
	return s
}

// SetConfigPath tells the server where its config lives, enabling the install
// flow to persist models and reload them.
func (s *Server) SetConfigPath(path string) {
	s.configPath = path
	s.modelDir = s.cfg.ModelDirPath(path)
}

// Nodes exposes the node manager so the entrypoint can start polling.
func (s *Server) Nodes() *NodeManager { return s.nodes }

// ReloadFromDisk re-reads the config and swaps in the new catalog. It is how a
// freshly installed model becomes servable without a restart.
func (s *Server) ReloadFromDisk() error {
	if s.configPath == "" {
		return fmt.Errorf("config path is unknown; cannot reload")
	}
	s.reloadMu.Lock()
	defer s.reloadMu.Unlock()

	cfg, err := config.Load(s.configPath)
	if err != nil {
		// The previous config stays live: a typo in a model file must not take
		// the whole catalog down.
		return fmt.Errorf("reload: %w", err)
	}
	s.Reload(cfg)
	return nil
}

func (s *Server) rebuildModels() {
	doc := modeldoc.Build(s.cfg)
	b, err := json.Marshal(doc)
	if err != nil {
		s.log.Error("render model document", "err", err)
		return
	}
	s.modelsJSON.Store(b)
}

// Reload swaps in a new config without dropping in-flight requests. The backend
// pool is not rebuilt here: changing backends means changing connection pools
// and admission counters, which needs a restart.
func (s *Server) Reload(cfg *config.Config) {
	s.cfg = cfg
	s.catalog.Replace(cfg)
	s.rebuildModels()
	s.keys.purge()
	s.log.Info("config reloaded", "models", len(cfg.Models))
}

// PublicHandler is the routing table for the internet-facing listener.
func (s *Server) PublicHandler() http.Handler {
	mux := http.NewServeMux()

	mux.HandleFunc("POST /v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		s.handleInference(w, r, "/chat/completions")
	})
	mux.HandleFunc("POST /v1/completions", func(w http.ResponseWriter, r *http.Request) {
		s.handleInference(w, r, "/completions")
	})

	// The provider document is intentionally unauthenticated: OpenRouter's
	// monitor polls it without credentials, and it contains only the pricing
	// and capability information we publish anyway.
	mux.HandleFunc("GET /v1/models", s.handleModels)
	mux.HandleFunc("GET /models", s.handleModels)

	mux.HandleFunc("GET /health", s.handleHealth)
	mux.HandleFunc("GET /", s.handleRoot)

	return s.withRecovery(mux)
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	b, _ := s.modelsJSON.Load().([]byte)
	if b == nil {
		writeError(w, http.StatusInternalServerError, "server_error", "Model document unavailable.")
		return
	}
	w.Header().Set("Content-Type", "application/json")
	// Short cache: OpenRouter polls frequently, but a price change should
	// propagate in seconds rather than minutes.
	w.Header().Set("Cache-Control", "public, max-age=30")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(b)
}

// handleHealth reports readiness. It returns 503 when no backend is reachable,
// which is what a load balancer in front of several gateway instances should
// act on.
func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	type backendHealth struct {
		Name     string `json:"name"`
		Healthy  bool   `json:"healthy"`
		Inflight int    `json:"inflight"`
		MaxConc  int    `json:"max_concurrency"`
		LastErr  string `json:"last_error,omitempty"`
	}
	out := struct {
		Status   string          `json:"status"`
		Uptime   float64         `json:"uptime_seconds"`
		Models   int             `json:"models"`
		Backends []backendHealth `json:"backends"`
	}{Status: "ok", Uptime: time.Since(s.startedAt).Seconds(), Models: len(s.catalog.IDs())}

	anyUp := false
	for _, b := range s.pool.Backends() {
		if b.Healthy() {
			anyUp = true
		}
		out.Backends = append(out.Backends, backendHealth{
			Name: b.Name, Healthy: b.Healthy(), Inflight: b.Inflight(),
			MaxConc: b.MaxConc(), LastErr: b.LastErr(),
		})
	}
	status := http.StatusOK
	if !anyUp {
		out.Status = "degraded"
		status = http.StatusServiceUnavailable
	}
	writeJSON(w, status, out)
}

func (s *Server) handleRoot(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		writeError(w, http.StatusNotFound, "invalid_request_error", "Not found.")
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"provider":   s.cfg.Provider.DisplayName,
		"slug":       s.cfg.Provider.Slug,
		"models_url": "/v1/models",
		"endpoints":  []string{"/v1/chat/completions", "/v1/completions", "/v1/models", "/health"},
	})
}

// withRecovery keeps one bad request from taking down the process. A panic
// after headers are sent cannot be turned into a 500, so it is only logged.
func (s *Server) withRecovery(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if rec := recover(); rec != nil {
				s.log.Error("panic serving request", "path", r.URL.Path, "panic", rec)
			}
		}()
		next.ServeHTTP(w, r)
	})
}

// StartBackground runs the rollup, retention and limiter-sweep jobs.
func (s *Server) StartBackground(ctx context.Context) {
	// Catch up the rollups across the whole retention window at boot. If the
	// process was down for a while, or rows were imported underneath it, the
	// aggregate tables would otherwise stay permanently behind -- the periodic
	// job only looks back two hours. This is one pass over indexed rows and is
	// cheap enough to pay on every start.
	go func() {
		cutoff := time.Now().AddDate(0, 0, -s.cfg.Server.RawRetentionDays)
		if err := s.store.Rollup(ctx, cutoff); err != nil {
			s.log.Error("startup rollup failed", "err", err)
			return
		}
		s.log.Info("rollups caught up", "since", cutoff.Format(time.DateOnly))
	}()

	go func() {
		// Roll up often enough that the dashboard's hourly view is never more
		// than a minute stale, and cheap because each pass only touches the
		// last couple of hours of raw rows.
		rollup := time.NewTicker(time.Minute)
		purge := time.NewTicker(time.Hour)
		sweep := time.NewTicker(5 * time.Minute)
		defer rollup.Stop()
		defer purge.Stop()
		defer sweep.Stop()

		for {
			select {
			case <-ctx.Done():
				return
			case <-rollup.C:
				// Two hours back so the boundary bucket is recomputed once it
				// is complete, rather than left holding a partial hour.
				if err := s.store.Rollup(ctx, time.Now().Add(-2*time.Hour)); err != nil {
					s.log.Error("rollup failed", "err", err)
				}
			case <-purge.C:
				cutoff := time.Now().AddDate(0, 0, -s.cfg.Server.RawRetentionDays)
				// Roll up the full retention window before deleting, so nothing
				// is dropped from the raw log before it has been aggregated.
				if err := s.store.Rollup(ctx, cutoff); err != nil {
					s.log.Error("pre-purge rollup failed", "err", err)
					continue
				}
				n, err := s.store.Purge(ctx, cutoff)
				if err != nil {
					s.log.Error("purge failed", "err", err)
				} else if n > 0 {
					s.log.Info("purged raw requests", "rows", n, "older_than", cutoff)
				}
			case <-sweep.C:
				s.limiter.sweep()
				if d := s.store.Dropped(); d > 0 {
					s.log.Warn("stat records dropped", "count", d)
				}
			}
		}
	}()
}
