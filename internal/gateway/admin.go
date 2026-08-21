package gateway

import (
	"context"
	"crypto/sha256"
	"embed"
	"encoding/hex"
	"io/fs"
	"net/http"
	"strconv"
	"time"

	"github.com/llmfast/gateway/internal/modeldoc"
	"github.com/llmfast/gateway/internal/store"
)

//go:embed ui/*
var uiFS embed.FS

// window describes the time range and resolution behind a dashboard query.
//
// Short ranges read the raw request log so the current, partially-elapsed
// bucket is included and minute resolution is possible. Long ranges read the
// rollup tables, because scanning weeks of raw rows for every dashboard refresh
// would not stay fast as traffic grows.
type window struct {
	from, to time.Time
	bucket   int64
	raw      bool
	table    string
	label    string
}

func parseWindow(s string) window {
	now := time.Now()
	switch s {
	case "1h":
		return window{from: now.Add(-time.Hour), to: now, bucket: 60, raw: true, label: "1h"}
	case "6h":
		return window{from: now.Add(-6 * time.Hour), to: now, bucket: 300, raw: true, label: "6h"}
	case "7d":
		return window{from: now.AddDate(0, 0, -7), to: now, bucket: store.Hour, table: "stats_hourly", label: "7d"}
	case "30d":
		return window{from: now.AddDate(0, 0, -30), to: now, bucket: store.Day, table: "stats_daily", label: "30d"}
	case "all":
		return window{from: time.Unix(0, 0), to: now, bucket: store.Day, table: "stats_daily", label: "all"}
	default: // 24h
		return window{from: now.Add(-24 * time.Hour), to: now, bucket: store.Hour, raw: true, label: "24h"}
	}
}

// AdminHandler is the routing table for the private listener. Bind it to
// localhost or a VPN interface: it exposes keys and full request history.
func (s *Server) AdminHandler() http.Handler {
	mux := http.NewServeMux()

	// Login exchanges the admin token for a cookie so the static UI, which
	// cannot set an Authorization header on a page load, can authenticate.
	mux.HandleFunc("POST /admin/api/login", s.adminLogin)

	mux.HandleFunc("GET /admin/api/overview", s.guard(s.adminOverview))
	mux.HandleFunc("GET /admin/api/series", s.guard(s.adminSeries))
	mux.HandleFunc("GET /admin/api/models", s.guard(s.adminModels))
	mux.HandleFunc("GET /admin/api/backends", s.guard(s.adminBackends))
	mux.HandleFunc("GET /admin/api/requests", s.guard(s.adminRequests))
	mux.HandleFunc("GET /admin/api/modeldoc", s.guard(s.adminModelDoc))

	// Provisioning: inspect a model, install it on a node, and manage what is
	// running there.
	mux.HandleFunc("GET /admin/api/nodes", s.guard(s.adminNodes))
	mux.HandleFunc("POST /admin/api/inspect", s.guard(s.adminInspect))
	mux.HandleFunc("POST /admin/api/install", s.guard(s.adminInstall))
	mux.HandleFunc("POST /admin/api/publish", s.guard(s.adminPublish))
	mux.HandleFunc("POST /admin/api/uninstall", s.guard(s.adminUninstall))
	mux.HandleFunc("GET /admin/api/nodes/{node}/logs", s.guard(s.adminNodeLogs))
	mux.HandleFunc("POST /admin/api/nodes/{node}/stop", s.guard(s.adminNodeStop))

	// Playground: run a real completion against an installed model.
	mux.HandleFunc("GET /admin/api/playground/models", s.guard(s.adminPlaygroundModels))
	mux.HandleFunc("POST /admin/api/playground", s.guard(s.adminPlayground))
	// Load testing: measure the endpoint instead of estimating it.
	mux.HandleFunc("POST /admin/api/benchmark", s.guard(s.adminBenchmark))
	mux.HandleFunc("POST /admin/api/economics", s.guard(s.adminEconomics))

	mux.HandleFunc("GET /admin/api/keys", s.guard(s.adminListKeys))
	mux.HandleFunc("POST /admin/api/keys", s.guard(s.adminCreateKey))
	mux.HandleFunc("DELETE /admin/api/keys/{id}", s.guard(s.adminDeleteKey))
	mux.HandleFunc("POST /admin/api/keys/{id}/toggle", s.guard(s.adminToggleKey))

	sub, err := fs.Sub(uiFS, "ui")
	if err == nil {
		mux.Handle("GET /", staticHandler(sub))
	}
	return s.withRecovery(mux)
}

func (s *Server) guard(h http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if !s.requireAdmin(w, r) {
			return
		}
		h(w, r)
	}
}

func (s *Server) adminLogin(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Token string `json:"token"`
	}
	_ = decodeJSON(r, &body)
	if s.cfg.Server.AdminToken == "" || body.Token != s.cfg.Server.AdminToken {
		writeError(w, http.StatusUnauthorized, "authentication_error", "Invalid admin token.")
		return
	}
	http.SetCookie(w, &http.Cookie{
		Name:     "llmfast_admin",
		Value:    body.Token,
		Path:     "/",
		HttpOnly: true,
		SameSite: http.SameSiteStrictMode,
		MaxAge:   int((12 * time.Hour).Seconds()),
	})
	writeJSON(w, http.StatusOK, map[string]bool{"ok": true})
}

func (s *Server) adminOverview(w http.ResponseWriter, r *http.Request) {
	win := parseWindow(r.URL.Query().Get("range"))
	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()

	totals, err := s.store.Totals(ctx, win.from, win.to)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "server_error", err.Error())
		return
	}
	var byModel []store.ModelStat
	if win.raw {
		byModel, err = s.store.ByModelRaw(ctx, win.from, win.to)
	} else {
		byModel, err = s.store.ByModel(ctx, win.table, win.from, win.to)
	}
	if err != nil {
		writeError(w, http.StatusInternalServerError, "server_error", err.Error())
		return
	}

	type backendView struct {
		Name     string `json:"name"`
		Healthy  bool   `json:"healthy"`
		Inflight int    `json:"inflight"`
		MaxConc  int    `json:"max_concurrency"`
		LastErr  string `json:"last_error,omitempty"`
	}
	backends := []backendView{}
	for _, b := range s.pool.Backends() {
		backends = append(backends, backendView{b.Name, b.Healthy(), b.Inflight(), b.MaxConc(), b.LastErr()})
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"range":          win.label,
		"totals":         totals,
		"by_model":       byModel,
		"backends":       backends,
		"model_count":    len(s.catalog.IDs()),
		"dropped_stats":  s.store.Dropped(),
		"uptime_seconds": time.Since(s.startedAt).Seconds(),
	})
}

func (s *Server) adminSeries(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	win := parseWindow(q.Get("range"))
	model := q.Get("model")

	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()

	var points []store.Point
	var err error
	if win.raw {
		points, err = s.store.SeriesRaw(ctx, win.bucket, model, win.from, win.to)
	} else {
		points, err = s.store.Series(ctx, win.table, model, win.from, win.to)
	}
	if err != nil {
		writeError(w, http.StatusInternalServerError, "server_error", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"range":      win.label,
		"bucket_sec": win.bucket,
		"from":       win.from.Unix(),
		"to":         win.to.Unix(),
		"points":     fillGaps(points, win),
	})
}

// fillGaps inserts empty buckets so a chart shows a flat line through quiet
// periods rather than silently joining two distant points.
func fillGaps(points []store.Point, win window) []store.Point {
	if win.bucket <= 0 {
		return points
	}
	index := make(map[int64]store.Point, len(points))
	for _, p := range points {
		index[p.Bucket] = p
	}
	start := (win.from.Unix() / win.bucket) * win.bucket
	end := (win.to.Unix() / win.bucket) * win.bucket
	// A very wide range at fine resolution would generate an unbounded slice;
	// beyond this the raw points are returned as-is.
	if (end-start)/win.bucket > 5000 {
		return points
	}
	out := make([]store.Point, 0, (end-start)/win.bucket+1)
	for b := start; b <= end; b += win.bucket {
		if p, ok := index[b]; ok {
			out = append(out, p)
		} else {
			out = append(out, store.Point{Bucket: b})
		}
	}
	return out
}

func (s *Server) adminModels(w http.ResponseWriter, r *http.Request) {
	type modelView struct {
		ID            string   `json:"id"`
		Name          string   `json:"name"`
		UpstreamModel string   `json:"upstream_model"`
		Backends      []string `json:"backends"`
		ContextLength int      `json:"context_length"`
		MaxOutput     int      `json:"max_output_tokens"`
		Quantization  string   `json:"quantization"`
		PromptUSD     string   `json:"prompt_usd"`
		CompletionUSD string   `json:"completion_usd"`
		IsFree        bool     `json:"is_free"`
		Ready         bool     `json:"ready"`
		Tools         bool     `json:"tools"`
		Reasoning     bool     `json:"reasoning"`
	}
	out := []modelView{}
	for i := range s.cfg.Models {
		m := &s.cfg.Models[i]
		ready := true
		if m.IsReady != nil {
			ready = *m.IsReady
		}
		out = append(out, modelView{
			ID: m.ID, Name: m.Name, UpstreamModel: m.UpstreamModel, Backends: m.Backends,
			ContextLength: m.ContextLength, MaxOutput: m.MaxOutputTokens,
			Quantization: m.Quantization, PromptUSD: m.Pricing.Prompt,
			CompletionUSD: m.Pricing.Completion, IsFree: m.IsFree, Ready: ready,
			Tools: m.Features.Tools, Reasoning: m.Features.Reasoning,
		})
	}
	writeJSON(w, http.StatusOK, map[string]any{"models": out})
}

func (s *Server) adminBackends(w http.ResponseWriter, r *http.Request) {
	type view struct {
		Name     string `json:"name"`
		BaseURL  string `json:"base_url"`
		Healthy  bool   `json:"healthy"`
		Inflight int    `json:"inflight"`
		MaxConc  int    `json:"max_concurrency"`
		LastErr  string `json:"last_error,omitempty"`
	}
	out := []view{}
	for _, b := range s.pool.Backends() {
		out = append(out, view{b.Name, b.BaseURL, b.Healthy(), b.Inflight(), b.MaxConc(), b.LastErr()})
	}
	writeJSON(w, http.StatusOK, map[string]any{"backends": out})
}

func (s *Server) adminRequests(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	limit, _ := strconv.Atoi(q.Get("limit"))
	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()

	rows, err := s.store.Recent(ctx, limit, q.Get("model"))
	if err != nil {
		writeError(w, http.StatusInternalServerError, "server_error", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"requests": rows})
}

// adminModelDoc previews exactly what OpenRouter will fetch, so the document
// can be checked before it is published.
func (s *Server) adminModelDoc(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, modeldoc.Build(s.cfg))
}

func (s *Server) adminListKeys(w http.ResponseWriter, r *http.Request) {
	keys, err := s.store.ListKeys(r.Context())
	if err != nil {
		writeError(w, http.StatusInternalServerError, "server_error", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"keys": keys})
}

func (s *Server) adminCreateKey(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Name     string `json:"name"`
		RPMLimit int    `json:"rpm_limit"`
	}
	_ = decodeJSON(r, &body)
	if body.Name == "" {
		body.Name = "unnamed"
	}
	key, secret, err := s.store.CreateKey(r.Context(), body.Name, body.RPMLimit)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "server_error", err.Error())
		return
	}
	// The secret is returned exactly once; only its hash is stored.
	writeJSON(w, http.StatusCreated, map[string]any{"key": key, "secret": secret})
}

func (s *Server) adminDeleteKey(w http.ResponseWriter, r *http.Request) {
	id, err := strconv.ParseInt(r.PathValue("id"), 10, 64)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "Invalid key id.")
		return
	}
	if err := s.store.DeleteKey(r.Context(), id); err != nil {
		writeError(w, http.StatusInternalServerError, "server_error", err.Error())
		return
	}
	s.keys.purge()
	writeJSON(w, http.StatusOK, map[string]bool{"ok": true})
}

func (s *Server) adminToggleKey(w http.ResponseWriter, r *http.Request) {
	id, err := strconv.ParseInt(r.PathValue("id"), 10, 64)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "Invalid key id.")
		return
	}
	var body struct {
		Disabled bool `json:"disabled"`
	}
	_ = decodeJSON(r, &body)
	if err := s.store.SetKeyDisabled(r.Context(), id, body.Disabled); err != nil {
		writeError(w, http.StatusInternalServerError, "server_error", err.Error())
		return
	}
	// Drop cached lookups so the change takes effect now rather than after the
	// cache TTL, which matters when disabling a leaked key.
	s.keys.purge()
	writeJSON(w, http.StatusOK, map[string]bool{"ok": true})
}

// staticHandler serves the embedded admin UI with a content-addressed ETag.
//
// http.FileServer alone is not enough here. Files inside an embed.FS have a
// zero modification time, so it cannot send Last-Modified, and a browser with
// no validator of any kind falls back to heuristic caching -- it decides for
// itself how long the asset stays fresh. The visible symptom is an admin UI
// that keeps rendering the previous build after an upgrade, including buttons
// that were added and text that was changed, with nothing in the deploy output
// to suggest anything is wrong.
//
// Hashing the contents gives an exact validator. "no-cache" is not "do not
// store": it means the cached copy may be reused, but only after revalidating,
// so unchanged assets still come back as an empty 304.
func staticHandler(root fs.FS) http.Handler {
	etags := map[string]string{}
	_ = fs.WalkDir(root, ".", func(path string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		b, err := fs.ReadFile(root, path)
		if err != nil {
			return nil
		}
		sum := sha256.Sum256(b)
		etags["/"+path] = `"` + hex.EncodeToString(sum[:8]) + `"`
		return nil
	})

	files := http.FileServer(http.FS(root))
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		p := r.URL.Path
		if p == "/" {
			p = "/index.html"
		}
		if tag, ok := etags[p]; ok {
			w.Header().Set("ETag", tag)
			w.Header().Set("Cache-Control", "no-cache")
			if match := r.Header.Get("If-None-Match"); match == tag {
				w.WriteHeader(http.StatusNotModified)
				return
			}
		}
		files.ServeHTTP(w, r)
	})
}
