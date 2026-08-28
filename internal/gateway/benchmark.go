package gateway

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"
)

// Benchmarking exists because every number the planner produces is an estimate.
// This measures the real thing, through the real request path, on the real
// hardware — which is the only way to know whether an endpoint is competitive
// before publishing it.

// BenchRequest is a load test to run.
type BenchRequest struct {
	Model string `json:"model"`
	// Concurrency levels to sweep. Throughput per request falls as concurrency
	// rises while aggregate throughput climbs, and the point where aggregate
	// stops improving is the number worth configuring as max_concurrency.
	Concurrency      []int `json:"concurrency"`
	RequestsPerLevel int   `json:"requests_per_level"`
	PromptTokens     int   `json:"prompt_tokens"`
	MaxTokens        int   `json:"max_tokens"`
}

// BenchLevel is the outcome at one concurrency level.
type BenchLevel struct {
	Concurrency int   `json:"concurrency"`
	Requests    int   `json:"requests"`
	Errors      int   `json:"errors"`
	TTFTp50     int64 `json:"ttft_p50"`
	TTFTp95     int64 `json:"ttft_p95"`
	TTFTp99     int64 `json:"ttft_p99"`
	// PerRequestTPS is what a single client experiences, and is what OpenRouter
	// records. AggregateTPS is what the GPU produces in total, and is what
	// determines revenue.
	PerRequestTPS float64 `json:"per_request_tps"`
	AggregateTPS  float64 `json:"aggregate_tps"`
	CompletionTok int     `json:"completion_tokens"`
	PromptTok     int     `json:"prompt_tokens"`
	CachedTok     int     `json:"cached_tokens"`
	WallMs        int64   `json:"wall_ms"`
}

type BenchResult struct {
	Model  string       `json:"model"`
	Levels []BenchLevel `json:"levels"`
	// BestConcurrency is where aggregate throughput peaked. Past it the GPU is
	// saturated and extra concurrency only costs latency.
	BestConcurrency int     `json:"best_concurrency"`
	PeakAggregate   float64 `json:"peak_aggregate_tps"`
	Note            string  `json:"note"`
}

// benchWriter stands in for a client connection.
//
// Reusing serveInference with a fake writer means the benchmark exercises the
// identical proxy, admission control and SSE relay that customer traffic does.
// A separate benchmarking path could report healthy numbers for code that is
// not the code actually serving requests.
type benchWriter struct {
	hdr        http.Header
	status     int
	start      time.Time
	firstToken time.Time
	lastToken  time.Time
	usage      *usageInfo
	// partial holds an incomplete SSE frame between writes.
	partial []byte
}

func newBenchWriter() *benchWriter {
	return &benchWriter{hdr: make(http.Header), status: 200, start: time.Now()}
}

func (b *benchWriter) Header() http.Header  { return b.hdr }
func (b *benchWriter) WriteHeader(code int) { b.status = code }

// Flush is required: the SSE relay pushes through an http.ResponseController,
// which refuses to flush a writer that cannot.
func (b *benchWriter) Flush() {}

func (b *benchWriter) Write(p []byte) (int, error) {
	b.partial = append(b.partial, p...)
	for {
		idx := bytes.IndexByte(b.partial, '\n')
		if idx < 0 {
			break
		}
		line := b.partial[:idx+1]
		b.partial = b.partial[idx+1:]

		payload, isData := sseData(line)
		if !isData || bytes.Equal(payload, doneMarker) {
			continue
		}
		if hasUsage(payload) {
			if u := parseUsage(payload); u != nil {
				b.usage = u
			}
			continue
		}
		if carriesToken(payload) {
			now := time.Now()
			if b.firstToken.IsZero() {
				b.firstToken = now
			}
			b.lastToken = now
		}
	}
	return len(p), nil
}

func (s *Server) adminBenchmark(w http.ResponseWriter, r *http.Request) {
	var req BenchRequest
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "invalid request body")
		return
	}
	if req.Model == "" {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "model is required")
		return
	}
	if !s.pool.Available(req.Model) {
		writeError(w, http.StatusPreconditionFailed, "server_error",
			"no healthy backend is serving "+req.Model+" right now")
		return
	}
	if len(req.Concurrency) == 0 {
		req.Concurrency = []int{1, 4, 8, 16}
	}
	if req.RequestsPerLevel <= 0 {
		req.RequestsPerLevel = 8
	}
	if req.PromptTokens <= 0 {
		req.PromptTokens = 512
	}
	if req.MaxTokens <= 0 {
		req.MaxTokens = 128
	}
	// A runaway sweep would occupy the GPU for a long time and, on a live
	// endpoint, starve real traffic.
	const maxTotal = 512
	total := 0
	for _, c := range req.Concurrency {
		total += req.RequestsPerLevel
		if c > 128 {
			writeError(w, http.StatusBadRequest, "invalid_request_error",
				"concurrency above 128 is not accepted; it would saturate the endpoint")
			return
		}
	}
	if total > maxTotal {
		writeError(w, http.StatusBadRequest, "invalid_request_error",
			fmt.Sprintf("that sweep is %d requests; keep it under %d", total, maxTotal))
		return
	}

	// Progress is streamed: a sweep against a real GPU takes minutes, and a
	// silent spinner gives the operator no way to tell it apart from a hang.
	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	sw := newSSEWriter(w)

	emit := func(kind string, v any) {
		b, _ := json.Marshal(map[string]any{"type": kind, "data": v})
		_ = sw.WriteFlush(append(append([]byte("data: "), b...), '\n', '\n'))
	}

	result := BenchResult{Model: req.Model}
	prompt := syntheticPrompt(req.PromptTokens)

	for _, level := range req.Concurrency {
		if r.Context().Err() != nil {
			return // operator navigated away
		}
		emit("progress", map[string]any{"concurrency": level, "state": "running"})
		lv := s.runBenchLevel(r, req, level, prompt)
		result.Levels = append(result.Levels, lv)
		emit("level", lv)
	}

	// Only levels that served every request count. A level that rejects half
	// its load finishes sooner and therefore *reports a higher aggregate*, so
	// including it does not merely add noise -- it actively pulls the
	// recommendation towards the setting that broke.
	for _, lv := range result.Levels {
		if lv.Errors == 0 && lv.AggregateTPS >= result.PeakAggregate {
			result.PeakAggregate, result.BestConcurrency = lv.AggregateTPS, lv.Concurrency
		}
	}
	result.Note = benchNote(result)
	emit("result", result)
	_ = sw.WriteFlush([]byte("data: [DONE]\n\n"))
}

// runBenchLevel fires one batch of concurrent requests and summarises it.
func (s *Server) runBenchLevel(r *http.Request, req BenchRequest, concurrency int, prompt string) BenchLevel {
	body, _ := json.Marshal(map[string]any{
		"model":          req.Model,
		"messages":       []map[string]string{{"role": "user", "content": prompt}},
		"max_tokens":     req.MaxTokens,
		"temperature":    0, // deterministic, so runs are comparable
		"stream":         true,
		"stream_options": map[string]bool{"include_usage": true},
	})

	lv := BenchLevel{Concurrency: concurrency, Requests: req.RequestsPerLevel}
	ttfts := make([]int64, 0, req.RequestsPerLevel)
	perReq := make([]float64, 0, req.RequestsPerLevel)

	var mu sync.Mutex
	var wg sync.WaitGroup
	sem := make(chan struct{}, concurrency)
	wall := time.Now()

	for i := 0; i < req.RequestsPerLevel; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			bw := newBenchWriter()
			inner := r.Clone(r.Context())
			inner.Body = readCloser(bytes.NewReader(body))
			inner.ContentLength = int64(len(body))

			s.serveInference(bw, inner, "/chat/completions", playgroundKeyID, 0,
				bw.start, requestID())

			mu.Lock()
			defer mu.Unlock()
			if bw.status != 200 || bw.firstToken.IsZero() {
				lv.Errors++
				return
			}
			ttfts = append(ttfts, bw.firstToken.Sub(bw.start).Milliseconds())
			out := req.MaxTokens
			if bw.usage != nil {
				out = bw.usage.CompletionTokens
				lv.CompletionTok += bw.usage.CompletionTokens
				lv.PromptTok += bw.usage.PromptTokens
				lv.CachedTok += bw.usage.cached()
			} else {
				lv.CompletionTok += out
			}
			// Per-request throughput, measured the way OpenRouter measures it:
			// output tokens over the whole request, including the wait for the
			// first token.
			if secs := time.Since(bw.start).Seconds(); secs > 0 {
				perReq = append(perReq, float64(out)/secs)
			}
		}()
	}
	wg.Wait()

	lv.WallMs = time.Since(wall).Milliseconds()
	if lv.WallMs > 0 {
		lv.AggregateTPS = float64(lv.CompletionTok) / (float64(lv.WallMs) / 1000)
	}
	lv.TTFTp50, lv.TTFTp95, lv.TTFTp99 = pct(ttfts, .50), pct(ttfts, .95), pct(ttfts, .99)
	lv.PerRequestTPS = medianFloat(perReq)
	return lv
}

// benchNote interprets the sweep.
//
// The peak alone is a poor signal: measurement noise routinely puts the highest
// number at the last level even when throughput plateaued several levels
// earlier. What matters is where the curve flattens, so the knee is the first
// level within a small margin of the peak.
// benchNote turns a sweep into a recommendation for max_concurrency.
//
// That setting is an admission limit, not a latency knob: past it the gateway
// sheds with a 429. So the right value is the highest concurrency the engine
// serves without errors -- lower turns away requests it could have handled,
// higher hands load to an engine that will reject it. Either way the caller
// sees a failure, and failures are what OpenRouter counts against uptime.
func benchNote(r BenchResult) string {
	if len(r.Levels) < 2 {
		return ""
	}
	const plateau = 0.05

	// A level that returned errors is not a candidate, whatever it measured.
	// Rejected requests complete instantly, so such a level reports a *higher*
	// aggregate than a healthy one and would otherwise win.
	var clean, errored []BenchLevel
	for _, lv := range r.Levels {
		if lv.Errors == 0 {
			clean = append(clean, lv)
		} else {
			errored = append(errored, lv)
		}
	}

	var errNote string
	if len(errored) > 0 {
		e := errored[0]
		errNote = fmt.Sprintf(
			" Concurrency %d and above returned errors (%d at %d) and is excluded: a rejected "+
				"request finishes instantly and inflates the aggregate. That is the engine "+
				"refusing load past its slot count -- raise its own limit (llama.cpp --parallel, "+
				"vLLM --max-num-seqs) if you want to serve more at once.",
			e.Concurrency, e.Errors, e.Concurrency)
	}

	if len(clean) == 0 {
		return "Every level returned errors, so none of them is a safe setting." + errNote
	}

	best := clean[len(clean)-1]
	first := clean[0]

	// Still climbing with nothing failing: the ceiling has not been found.
	if len(errored) == 0 && len(clean) >= 2 {
		prev := clean[len(clean)-2]
		if best.AggregateTPS > prev.AggregateTPS*(1+plateau) {
			return fmt.Sprintf(
				"Aggregate throughput was still climbing at %d concurrent (%.0f tok/s, up from "+
					"%.0f), so the backend is not yet saturated. Re-run with higher levels to "+
					"find the ceiling.",
				best.Concurrency, best.AggregateTPS, prev.AggregateTPS)
		}
	}

	// The knee: the lowest concurrency that already reaches peak aggregate.
	// Past it the backend produces no more tokens per second, so admitting
	// more requests divides the same throughput across more clients -- and
	// per-request throughput is what OpenRouter ranks on.
	knee := best
	for _, lv := range clean {
		if lv.AggregateTPS >= r.PeakAggregate*(1-plateau) {
			knee = lv
			break
		}
	}

	note := fmt.Sprintf(
		"Aggregate throughput flattened at about %d concurrent (%.0f tok/s); the peak of %.0f "+
			"tok/s at %d is within noise of it. Set the backend's max_concurrency near %d: "+
			"beyond the knee the backend is saturated and extra concurrency only costs latency.",
		knee.Concurrency, r.PeakAggregate, r.PeakAggregate, r.BestConcurrency, knee.Concurrency)

	// Quantify the trade when it is stark, which it usually is on CPU.
	if best.Concurrency > first.Concurrency && best.PerRequestTPS > 0 && first.PerRequestTPS > 0 {
		gain := best.AggregateTPS / first.AggregateTPS
		loss := first.PerRequestTPS / best.PerRequestTPS
		if gain < 1.25 && loss > 1.5 {
			note += fmt.Sprintf(
				" Going from %d to %d concurrent raised aggregate throughput only %.0f%% "+
					"(%.0f to %.0f tok/s) while cutting per-request throughput %.1fx (%.0f to "+
					"%.0f tok/s). OpenRouter ranks you on per-request, so on this hardware "+
					"concurrency buys almost nothing and costs a great deal.",
				first.Concurrency, best.Concurrency, (gain-1)*100,
				first.AggregateTPS, best.AggregateTPS, loss,
				first.PerRequestTPS, best.PerRequestTPS)
		}
	}
	return note + errNote
}

// syntheticPrompt builds a prompt of roughly n tokens. English averages close
// to four characters per token, which is accurate enough for a load shape.
func syntheticPrompt(n int) string {
	const filler = "The quick brown fox jumps over the lazy dog while the system " +
		"processes tokens and measures latency under sustained load. "
	var b strings.Builder
	target := n * 4
	b.WriteString("Summarise the following text.\n\n")
	for b.Len() < target {
		b.WriteString(filler)
	}
	return b.String()
}

func pct(vals []int64, p float64) int64 {
	if len(vals) == 0 {
		return 0
	}
	sort.Slice(vals, func(i, j int) bool { return vals[i] < vals[j] })
	i := int(float64(len(vals)) * p)
	if i >= len(vals) {
		i = len(vals) - 1
	}
	return vals[i]
}

func medianFloat(v []float64) float64 {
	if len(v) == 0 {
		return 0
	}
	sort.Float64s(v)
	return v[len(v)/2]
}
