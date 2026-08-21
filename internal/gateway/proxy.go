package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"time"

	"github.com/llmfast/gateway/internal/store"
	"github.com/llmfast/gateway/internal/upstream"
)

// maxRequestBody caps the prompt payload. A 1M-token context in JSON is a few
// MB, so 32MB leaves generous headroom while still rejecting garbage early.
const maxRequestBody = 32 << 20

// SSE frame markers. These are compared as raw bytes so the hot loop never
// unmarshals a chunk it does not need.
var (
	dataPrefix   = []byte("data: ")
	doneMarker   = []byte("[DONE]")
	usageKey     = []byte(`"usage":`)
	usageNull    = []byte(`"usage":null`)
	tokContent   = []byte(`"content":"`)
	emptyContent = []byte(`"content":""`)
	tokReasoning = []byte(`"reasoning_content":"`)
	emptyReason  = []byte(`"reasoning_content":""`)
	tokToolCalls = []byte(`"tool_calls":`)
)

type usageInfo struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`

	PromptTokensDetails *struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"prompt_tokens_details"`
	CompletionTokensDetails *struct {
		ReasoningTokens int `json:"reasoning_tokens"`
	} `json:"completion_tokens_details"`
}

func (u *usageInfo) cached() int {
	if u.PromptTokensDetails != nil {
		return u.PromptTokensDetails.CachedTokens
	}
	return 0
}

func (u *usageInfo) reasoning() int {
	if u.CompletionTokensDetails != nil {
		return u.CompletionTokensDetails.ReasoningTokens
	}
	return 0
}

// handleInference serves both /v1/chat/completions and /v1/completions. The two
// differ only in the upstream path, so the transport, admission control,
// streaming and accounting are shared.
func (s *Server) handleInference(w http.ResponseWriter, r *http.Request, upstreamPath string) {
	start := time.Now()
	reqID := requestID()
	w.Header().Set("X-Request-Id", reqID)

	key, ok := s.authenticate(w, r)
	if !ok {
		return
	}
	s.serveInference(w, r, upstreamPath, key.ID, key.RPMLimit, start, reqID)
}

// serveInference is the request path proper, after the caller has been
// authenticated. It is separate from handleInference so the admin playground
// can reach it having authenticated as an operator rather than with an API key,
// and still exercise the identical streaming, admission and accounting code
// rather than a parallel implementation that could drift.
func (s *Server) serveInference(w http.ResponseWriter, r *http.Request, upstreamPath string,
	keyID int64, rpmLimit int, start time.Time, reqID string) {

	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxRequestBody))
	if err != nil {
		// A body that exceeds the cap is a client error (413) and, per
		// OpenRouter's accounting, must not count against our uptime.
		var mbe *http.MaxBytesError
		if errors.As(err, &mbe) {
			writeError(w, http.StatusRequestEntityTooLarge, "invalid_request_error", "Request body too large.")
			return
		}
		writeError(w, http.StatusBadRequest, "invalid_request_error", "Could not read request body.")
		return
	}

	head, err := peekRequest(body)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "Request body is not valid JSON.")
		return
	}

	modelID := head.Model
	if modelID == "" {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "Field `model` is required.")
		return
	}
	entry, known := s.catalog.Get(modelID)
	if !known {
		// 404 counts against uptime at OpenRouter, which is correct: if they
		// ask for a model we advertise and we do not have it, that is our bug.
		writeError(w, http.StatusNotFound, "invalid_request_error", "Unknown model: "+modelID)
		s.logRecord(store.Record{
			TS: start, RequestID: reqID, Model: modelID, APIKeyID: keyID,
			Status: http.StatusNotFound, TTFTMs: -1,
			TotalMs: time.Since(start).Milliseconds(), Error: "unknown model",
		})
		return
	}

	if !s.limiter.allow(keyID, rpmLimit) {
		w.Header().Set("Retry-After", "1")
		writeError(w, http.StatusTooManyRequests, "rate_limit_error", "Per-key rate limit exceeded.")
		s.logRecord(store.Record{
			TS: start, RequestID: reqID, Model: modelID, APIKeyID: keyID,
			Status: http.StatusTooManyRequests, TTFTMs: -1,
			TotalMs: time.Since(start).Milliseconds(), Error: "key rate limit",
		})
		return
	}

	streaming := head.Stream

	outBody, injectedUsage, err := rewriteBody(body, head, entry.UpstreamModel)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "Could not re-encode request.")
		return
	}

	lease, err := s.pool.Acquire(modelID)
	if err != nil {
		status, code, msg := http.StatusServiceUnavailable, "server_error", "No healthy backend for this model."
		if errors.Is(err, upstream.ErrNoCapacity) {
			// Shed rather than queue: see the package comment in internal/upstream.
			status, code, msg = http.StatusTooManyRequests, "rate_limit_error", "At capacity, retry shortly."
			w.Header().Set("Retry-After", "1")
		}
		writeError(w, status, code, msg)
		s.logRecord(store.Record{
			TS: start, RequestID: reqID, Model: modelID, APIKeyID: keyID,
			Status: status, TTFTMs: -1,
			TotalMs: time.Since(start).Milliseconds(), Error: err.Error(),
		})
		return
	}
	defer lease.Release()
	b := lease.B

	// The upstream context is derived from the client's, so a disconnected
	// client immediately frees the GPU instead of generating into the void.
	ctx, cancel := context.WithTimeout(r.Context(), b.Timeout)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, b.BaseURL+upstreamPath, bytes.NewReader(outBody))
	if err != nil {
		writeError(w, http.StatusInternalServerError, "server_error", "Could not build upstream request.")
		return
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	if b.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+b.APIKey)
	}

	resp, err := b.Client().Do(req)
	if err != nil {
		// A cancelled context is the client hanging up, not our failure, so it
		// is recorded with status 499 and excluded from error counts.
		if ctx.Err() != nil && r.Context().Err() != nil {
			s.logRecord(store.Record{
				TS: start, RequestID: reqID, Model: modelID, Backend: b.Name, APIKeyID: keyID,
				Status: 499, Streamed: streaming, TTFTMs: -1,
				TotalMs: time.Since(start).Milliseconds(), Error: "client disconnected",
			})
			return
		}
		writeError(w, http.StatusBadGateway, "server_error", "Upstream request failed.")
		s.logRecord(store.Record{
			TS: start, RequestID: reqID, Model: modelID, Backend: b.Name, APIKeyID: keyID,
			Status: http.StatusBadGateway, Streamed: streaming, TTFTMs: -1,
			TotalMs: time.Since(start).Milliseconds(), Error: err.Error(),
		})
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		s.relayUpstreamError(w, resp, start, reqID, modelID, b.Name, keyID, streaming)
		return
	}

	rec := store.Record{
		TS: start, RequestID: reqID, Model: modelID, Backend: b.Name,
		APIKeyID: keyID, Status: http.StatusOK, Streamed: streaming, TTFTMs: -1,
	}
	if streaming {
		s.relayStream(r.Context(), w, resp, entry, &rec, start, injectedUsage)
	} else {
		s.relayJSON(w, resp, entry, &rec, start)
	}
	s.logRecord(rec)
}

// relayStream forwards SSE frames as they arrive and extracts accounting from
// them in passing.
func (s *Server) relayStream(clientCtx context.Context, w http.ResponseWriter, resp *http.Response, entry *Entry, rec *store.Record, start time.Time, dropUsageFrame bool) {
	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache, no-transform")
	h.Set("Connection", "keep-alive")
	// Defeats proxy buffering (nginx and friends), which would otherwise batch
	// tokens and destroy the inter-token latency we are selling.
	h.Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)

	sw := newSSEWriter(w)
	stopKA := make(chan struct{})
	go func() {
		t := time.NewTicker(s.cfg.Server.KeepAliveInterval)
		defer t.Stop()
		for {
			select {
			case <-stopKA:
				return
			case <-t.C:
				sw.KeepAliveIfIdle(s.cfg.Server.KeepAliveInterval)
			}
		}
	}()
	defer close(stopKA)

	br := getReader(resp.Body)
	defer putReader(br)

	scratchPtr := scratchPool.Get().(*[]byte)
	scratch := (*scratchPtr)[:0]
	// patchBuf is reused across frames so the model-name rewrite does not
	// allocate once per token.
	patchPtr := scratchPool.Get().(*[]byte)
	patchBuf := (*patchPtr)[:0]
	defer func() {
		// Store the grown buffers back so the capacity is reused, but drop any
		// that ballooned on an unusually large frame rather than pinning it.
		const keepUnder = 1 << 20
		if cap(scratch) < keepUnder {
			*scratchPtr = scratch
			scratchPool.Put(scratchPtr)
		}
		if cap(patchBuf) < keepUnder {
			*patchPtr = patchBuf
			scratchPool.Put(patchPtr)
		}
	}()
	var usage *usageInfo
	var firstTok, lastTok time.Time
	chunks := 0

	for {
		var line []byte
		var err error
		line, scratch, err = readSSELine(br, scratch)

		if len(line) > 0 {
			payload, isData := sseData(line)
			switch {
			case isData && bytes.Equal(payload, doneMarker):
				// Pass [DONE] through untouched and stop.
				_ = sw.WriteFlush(line)

			case isData && hasUsage(payload):
				if u := parseUsage(payload); u != nil {
					usage = u
				}
				// Withhold the frame only when the client never asked for it.
				if !dropUsageFrame {
					line, patchBuf = entry.PatchModelName(line, patchBuf)
					_ = sw.WriteFlush(line)
				}

			default:
				if isData && carriesToken(payload) {
					now := time.Now()
					if firstTok.IsZero() {
						firstTok = now
					}
					lastTok = now
					chunks++
				}
				if isData {
					line, patchBuf = entry.PatchModelName(line, patchBuf)
				}
				if werr := sw.WriteFlush(line); werr != nil {
					// Client is gone. Stop reading so the GPU is released.
					rec.Status = 499
					rec.Error = "client disconnected"
					finishStream(rec, entry, usage, chunks, start, firstTok, lastTok)
					return
				}
			}
		}

		if err != nil {
			switch {
			case err == io.EOF:
				// Normal end of stream.
			case clientCtx.Err() != nil:
				// The upstream context is derived from the client's, so a
				// client hanging up surfaces here as a read cancellation
				// rather than a write failure. That is not our error: it is
				// recorded as 499 and excluded from the uptime numerator.
				rec.Status = 499
				rec.Error = "client disconnected"
			default:
				// A genuine failure after headers were sent. It counts against
				// uptime even though the status line already said 200, which is
				// exactly how OpenRouter scores mid-stream errors.
				rec.Error = err.Error()
				rec.Status = http.StatusInternalServerError
			}
			break
		}
	}
	finishStream(rec, entry, usage, chunks, start, firstTok, lastTok)
}

// finishStream fills in timing and cost once the stream has ended.
func finishStream(rec *store.Record, entry *Entry, usage *usageInfo, chunks int, start, firstTok, lastTok time.Time) {
	rec.TotalMs = time.Since(start).Milliseconds()
	if !firstTok.IsZero() {
		rec.TTFTMs = firstTok.Sub(start).Milliseconds()
		rec.GenMs = lastTok.Sub(firstTok).Milliseconds()
	}
	if usage != nil {
		rec.PromptTokens = usage.PromptTokens
		rec.CompletionTokens = usage.CompletionTokens
		rec.CachedTokens = usage.cached()
		rec.ReasoningTokens = usage.reasoning()
	} else {
		// Upstream withheld usage (or the stream broke before the final frame).
		// One content delta is approximately one token, which is close enough
		// to keep throughput charts honest; billing prefers the real number.
		rec.CompletionTokens = chunks
	}
	// Throughput matches OpenRouter's definition -- output tokens over the whole
	// generation time, including fetch latency and TTFT -- so our dashboard and
	// their model page do not disagree.
	if rec.TotalMs > 0 && rec.CompletionTokens > 0 {
		rec.TPS = float64(rec.CompletionTokens) / (float64(rec.TotalMs) / 1000)
	}
	rec.CostUSD = entry.Cost(rec.PromptTokens, rec.CompletionTokens, rec.CachedTokens, rec.ReasoningTokens)
}

// relayJSON handles the non-streaming path.
func (s *Server) relayJSON(w http.ResponseWriter, resp *http.Response, entry *Entry, rec *store.Record, start time.Time) {
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		writeError(w, http.StatusBadGateway, "server_error", "Upstream response truncated.")
		rec.Status = http.StatusBadGateway
		rec.Error = err.Error()
		rec.TotalMs = time.Since(start).Milliseconds()
		return
	}
	// Usage is read before patching: the rewrite only touches the model name,
	// but parsing the untouched body keeps the two concerns independent.
	usage := parseUsage(body)
	out, _ := entry.PatchModelName(body, nil)

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(out)

	rec.TotalMs = time.Since(start).Milliseconds()
	if u := usage; u != nil {
		rec.PromptTokens = u.PromptTokens
		rec.CompletionTokens = u.CompletionTokens
		rec.CachedTokens = u.cached()
		rec.ReasoningTokens = u.reasoning()
	}
	if rec.TotalMs > 0 && rec.CompletionTokens > 0 {
		rec.TPS = float64(rec.CompletionTokens) / (float64(rec.TotalMs) / 1000)
	}
	rec.CostUSD = entry.Cost(rec.PromptTokens, rec.CompletionTokens, rec.CachedTokens, rec.ReasoningTokens)
}

// relayUpstreamError forwards a non-200 from vLLM with its status intact, so a
// client's malformed request stays a 400 and does not masquerade as our 500.
func (s *Server) relayUpstreamError(w http.ResponseWriter, resp *http.Response, start time.Time, reqID, model, backend string, keyID int64, streaming bool) {
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
	ct := resp.Header.Get("Content-Type")
	if ct == "" {
		ct = "application/json"
	}
	w.Header().Set("Content-Type", ct)
	w.WriteHeader(resp.StatusCode)
	_, _ = w.Write(body)

	s.logRecord(store.Record{
		TS: start, RequestID: reqID, Model: model, Backend: backend, APIKeyID: keyID,
		Status: resp.StatusCode, Streamed: streaming, TTFTMs: -1,
		TotalMs: time.Since(start).Milliseconds(),
		Error:   truncate(string(body), 500),
	})
}

// sseData splits a raw line into its payload, reporting whether it is a data
// frame at all. Comments and blank separators return false and pass through.
func sseData(line []byte) (payload []byte, ok bool) {
	trimmed := bytes.TrimRight(line, "\r\n")
	if !bytes.HasPrefix(trimmed, dataPrefix) {
		return nil, false
	}
	return bytes.TrimSpace(trimmed[len(dataPrefix):]), true
}

func hasUsage(payload []byte) bool {
	return bytes.Contains(payload, usageKey) && !bytes.Contains(payload, usageNull)
}

// carriesToken reports whether a delta frame contains generated output, used to
// time the first token. It is a byte scan rather than a parse because it runs on
// every frame; the trade-off is that it recognises content, reasoning traces and
// tool calls, and ignores the opening role-only frame.
func carriesToken(payload []byte) bool {
	if bytes.Contains(payload, tokContent) && !bytes.Contains(payload, emptyContent) {
		return true
	}
	if bytes.Contains(payload, tokReasoning) && !bytes.Contains(payload, emptyReason) {
		return true
	}
	return bytes.Contains(payload, tokToolCalls)
}

func parseUsage(payload []byte) *usageInfo {
	var wrapper struct {
		Usage *usageInfo `json:"usage"`
	}
	if err := json.Unmarshal(payload, &wrapper); err != nil {
		return nil
	}
	return wrapper.Usage
}

// ensureUsage turns on upstream token accounting, returning true when the
// client had not already requested it.
func ensureUsage(fields map[string]json.RawMessage) (injected bool) {
	raw, ok := fields["stream_options"]
	if !ok {
		fields["stream_options"] = json.RawMessage(`{"include_usage":true}`)
		return true
	}
	var opts map[string]any
	if err := json.Unmarshal(raw, &opts); err != nil {
		fields["stream_options"] = json.RawMessage(`{"include_usage":true}`)
		return true
	}
	if v, present := opts["include_usage"].(bool); present && v {
		return false
	}
	opts["include_usage"] = true
	fields["stream_options"] = mustJSON(opts)
	return true
}

func (s *Server) logRecord(rec store.Record) { s.store.Log(rec) }

// requestHead is the handful of fields the proxy needs from a request body.
//
// Decoding into a narrow struct rather than a map is the point: encoding/json
// walks the whole document either way, but it *skips* fields the struct does
// not name instead of copying them. On a request whose `messages` array is
// several megabytes, that is the difference between one allocation and a full
// duplicate of the prompt.
type requestHead struct {
	Model         string `json:"model"`
	Stream        bool   `json:"stream"`
	StreamOptions *struct {
		IncludeUsage *bool `json:"include_usage"`
	} `json:"stream_options"`
}

func peekRequest(body []byte) (requestHead, error) {
	var h requestHead
	err := json.Unmarshal(body, &h)
	return h, err
}

// rewriteBody prepares the upstream request.
//
// The common case is that nothing needs changing: the agent starts every engine
// with --served-model-name set to the public id, and most clients that stream
// already ask for usage. When that holds, the original body is forwarded
// untouched and a large prompt is never re-encoded. Only when a field genuinely
// has to change does it fall back to decoding and re-serialising.
func rewriteBody(body []byte, head requestHead, upstreamModel string) (out []byte, injectedUsage bool, err error) {
	needModel := head.Model != upstreamModel
	needUsage := head.Stream &&
		(head.StreamOptions == nil || head.StreamOptions.IncludeUsage == nil || !*head.StreamOptions.IncludeUsage)

	if !needModel && !needUsage {
		return body, false, nil
	}

	// Something must change, so decode the top level. `messages` and `tools`
	// stay as raw bytes and are copied through verbatim rather than reparsed.
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		return nil, false, err
	}
	if needModel {
		fields["model"] = mustJSON(upstreamModel)
	}
	if needUsage {
		injectedUsage = ensureUsage(fields)
	}
	encoded, err := json.Marshal(fields)
	if err != nil {
		return nil, false, err
	}
	return encoded, injectedUsage, nil
}
