package gateway

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/llmfast/gateway/internal/config"
	"github.com/llmfast/gateway/internal/store"
	"github.com/llmfast/gateway/internal/upstream"
)

// testRig wires a gateway to a fake upstream so the full request path -- auth,
// admission, streaming, accounting -- can be exercised in process.
type testRig struct {
	gw       *httptest.Server
	st       *store.Store
	secret   string
	upstream *httptest.Server
	srv      *Server
	pool     *upstream.Pool
}

func newRig(t *testing.T, maxConc int, handler http.HandlerFunc) *testRig {
	t.Helper()
	up := httptest.NewServer(handler)
	t.Cleanup(up.Close)

	st, err := store.Open(filepath.Join(t.TempDir(), "t.db"))
	if err != nil {
		t.Fatalf("open store: %v", err)
	}
	t.Cleanup(func() { st.Close() })

	_, secret, err := st.CreateKey(context.Background(), "test", 0)
	if err != nil {
		t.Fatalf("create key: %v", err)
	}

	cfg := &config.Config{
		Provider: config.Provider{Slug: "test", DisplayName: "Test"},
		Server: config.Server{
			KeepAliveInterval: 50 * time.Millisecond,
			AdminToken:        "admin-test",
		},
		Backends: []config.Backend{{
			Name: "b1", BaseURL: up.URL + "/v1",
			MaxConcurrency: maxConc, Timeout: 30 * time.Second, Weight: 1,
		}},
		Models: []config.Model{{
			ID: "qwen/qwen3-32b", Name: "Qwen3 32B", UpstreamModel: "Qwen/Qwen3-32B",
			Backends: []string{"b1"}, ContextLength: 131072, MaxOutputTokens: 32768,
			Pricing: config.Pricing{
				Prompt: "0.0000001", Completion: "0.0000003", CachedPrompt: "0.00000002",
			},
		}},
	}
	if err := cfg.Validate(); err != nil {
		t.Fatalf("config: %v", err)
	}

	pool := upstream.NewPool(cfg)
	srv := New(cfg, st, pool, slog.New(slog.NewTextHandler(io.Discard, nil)))
	gw := httptest.NewServer(srv.PublicHandler())
	t.Cleanup(gw.Close)

	return &testRig{gw: gw, st: st, secret: secret, upstream: up, srv: srv, pool: pool}
}

func (r *testRig) post(t *testing.T, body string) *http.Response {
	t.Helper()
	req, _ := http.NewRequest("POST", r.gw.URL+"/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer "+r.secret)
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request: %v", err)
	}
	return resp
}

// waitForRecord polls until the async stat writer has flushed n rows.
func (r *testRig) waitForRecord(t *testing.T, n int) []store.RecentRequest {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		rows, err := r.st.Recent(context.Background(), 50, "")
		if err == nil && len(rows) >= n {
			return rows
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %d stat records", n)
	return nil
}

// streamHandler emits a realistic vLLM stream: a role-only opening frame, a
// prefill pause, content deltas, then usage if it was requested.
func streamHandler(prefill time.Duration, words []string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Model         string `json:"model"`
			StreamOptions *struct {
				IncludeUsage bool `json:"include_usage"`
			} `json:"stream_options"`
		}
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &req)

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		rc := http.NewResponseController(w)
		emit := func(s string) {
			fmt.Fprintf(w, "data: %s\n\n", s)
			_ = rc.Flush()
		}
		emit(fmt.Sprintf(`{"model":%q,"choices":[{"delta":{"role":"assistant","content":""}}]}`, req.Model))
		time.Sleep(prefill)
		for _, word := range words {
			emit(fmt.Sprintf(`{"model":%q,"choices":[{"delta":{"content":%q}}]}`, req.Model, word))
		}
		emit(fmt.Sprintf(`{"model":%q,"choices":[{"delta":{},"finish_reason":"stop"}]}`, req.Model))
		if req.StreamOptions != nil && req.StreamOptions.IncludeUsage {
			emit(fmt.Sprintf(`{"model":%q,"choices":[],"usage":{"prompt_tokens":100,`+
				`"completion_tokens":%d,"prompt_tokens_details":{"cached_tokens":40}}}`, req.Model, len(words)))
		}
		emit("[DONE]")
	}
}

func readFrames(t *testing.T, resp *http.Response) []string {
	t.Helper()
	defer resp.Body.Close()
	var frames []string
	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		line := sc.Text()
		if strings.HasPrefix(line, "data: ") {
			frames = append(frames, strings.TrimPrefix(line, "data: "))
		}
	}
	return frames
}

func TestStreamEndToEnd(t *testing.T) {
	rig := newRig(t, 8, streamHandler(60*time.Millisecond, []string{"Hello", " world", "!"}))

	resp := rig.post(t, `{"model":"qwen/qwen3-32b","messages":[{"role":"user","content":"hi"}],"stream":true}`)
	if resp.StatusCode != 200 {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("content-type = %q, want text/event-stream", ct)
	}
	if resp.Header.Get("X-Accel-Buffering") != "no" {
		t.Error("X-Accel-Buffering: no is missing; a proxy in front would batch tokens")
	}
	frames := readFrames(t, resp)

	if len(frames) == 0 || frames[len(frames)-1] != "[DONE]" {
		t.Fatalf("stream did not terminate with [DONE]: %v", frames)
	}
	// The client did not ask for usage, so the injected frame must be withheld.
	for _, f := range frames {
		if strings.Contains(f, `"usage"`) {
			t.Errorf("usage frame leaked to a client that did not request it: %s", f)
		}
	}
	// Every frame must echo the public model id, not the upstream name.
	for _, f := range frames {
		if strings.Contains(f, "Qwen/Qwen3-32B") {
			t.Errorf("upstream model name leaked to the client: %s", f)
		}
	}

	rows := rig.waitForRecord(t, 1)
	rec := rows[0]
	if rec.Status != 200 {
		t.Errorf("recorded status = %d, want 200", rec.Status)
	}
	if !rec.Streamed {
		t.Error("record not marked as streamed")
	}
	// Usage was extracted from the suppressed frame.
	if rec.PromptTok != 100 || rec.CompTok != 3 {
		t.Errorf("recorded tokens = %d/%d, want 100/3", rec.PromptTok, rec.CompTok)
	}
	// TTFT must reflect the prefill pause, not the role-only opening frame.
	if rec.TTFTMs < 50 {
		t.Errorf("ttft = %dms, want >= 50ms (the opening role frame was counted as a token)", rec.TTFTMs)
	}
	if rec.TTFTMs > 5000 {
		t.Errorf("ttft = %dms, implausibly high", rec.TTFTMs)
	}
	// 600 uncached prompt + 40 cached + 3 completion.
	wantCost := 60*1e-7 + 40*2e-8 + 3*3e-7
	if diff := rec.CostUSD - wantCost; diff > 1e-12 || diff < -1e-12 {
		t.Errorf("cost = %v, want %v", rec.CostUSD, wantCost)
	}
}

func TestStreamPassesUsageWhenRequested(t *testing.T) {
	rig := newRig(t, 8, streamHandler(10*time.Millisecond, []string{"a", "b"}))
	resp := rig.post(t, `{"model":"qwen/qwen3-32b","messages":[],"stream":true,
		"stream_options":{"include_usage":true}}`)
	frames := readFrames(t, resp)

	found := false
	for _, f := range frames {
		if strings.Contains(f, `"usage"`) {
			found = true
		}
	}
	if !found {
		t.Error("client asked for usage but no usage frame was forwarded")
	}
}

func TestKeepAliveDuringSlowPrefill(t *testing.T) {
	// Prefill far exceeds the 50ms keep-alive interval configured in the rig,
	// which is exactly the reasoning-model case OpenRouter would otherwise
	// cancel as a hung stream.
	rig := newRig(t, 8, streamHandler(300*time.Millisecond, []string{"x"}))

	req, _ := http.NewRequest("POST", rig.gw.URL+"/v1/chat/completions",
		strings.NewReader(`{"model":"qwen/qwen3-32b","messages":[],"stream":true}`))
	req.Header.Set("Authorization", "Bearer "+rig.secret)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request: %v", err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(body), ": keepalive") {
		t.Errorf("no keep-alive comment during a %v prefill:\n%s", 300*time.Millisecond, body)
	}
}

func TestCapacityShedsWith429(t *testing.T) {
	release := make(chan struct{})
	// The single slot is held until the test releases it, so the second
	// request necessarily finds the backend full.
	rig := newRig(t, 1, func(w http.ResponseWriter, r *http.Request) {
		<-release
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		fmt.Fprint(w, "data: [DONE]\n\n")
	})

	started := make(chan struct{})
	go func() {
		close(started)
		resp := rig.post(t, `{"model":"qwen/qwen3-32b","messages":[],"stream":true}`)
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
	}()
	<-started
	time.Sleep(150 * time.Millisecond) // let the first request take the slot

	resp := rig.post(t, `{"model":"qwen/qwen3-32b","messages":[],"stream":true}`)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusTooManyRequests {
		t.Errorf("status = %d, want 429 (load must be shed, never queued)", resp.StatusCode)
	}
	if resp.Header.Get("Retry-After") == "" {
		t.Error("Retry-After header missing on a 429")
	}
	close(release)
}

func TestUpstreamErrorStatusPreserved(t *testing.T) {
	// A 400 from vLLM is the client's fault and must reach the client as a 400.
	// Turning it into a 500 would count against our uptime for someone else's bug.
	rig := newRig(t, 8, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprint(w, `{"error":{"message":"bad parameter","type":"invalid_request_error"}}`)
	})
	resp := rig.post(t, `{"model":"qwen/qwen3-32b","messages":[],"stream":true}`)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", resp.StatusCode)
	}
	body, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(body), "bad parameter") {
		t.Errorf("upstream error body was not forwarded: %s", body)
	}

	rows := rig.waitForRecord(t, 1)
	if rows[0].Status != 400 {
		t.Errorf("recorded status = %d, want 400", rows[0].Status)
	}
}

func TestNonStreamingPath(t *testing.T) {
	rig := newRig(t, 8, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"model":"Qwen/Qwen3-32B","choices":[{"message":{"content":"hi"}}],
			"usage":{"prompt_tokens":10,"completion_tokens":2}}`)
	})
	resp := rig.post(t, `{"model":"qwen/qwen3-32b","messages":[]}`)
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	if strings.Contains(string(body), "Qwen/Qwen3-32B") {
		t.Errorf("upstream model name leaked: %s", body)
	}
	rows := rig.waitForRecord(t, 1)
	if rows[0].Streamed {
		t.Error("non-streaming request was recorded as streamed")
	}
	if rows[0].PromptTok != 10 || rows[0].CompTok != 2 {
		t.Errorf("tokens = %d/%d, want 10/2", rows[0].PromptTok, rows[0].CompTok)
	}
	// TTFT is meaningless without a stream and must be recorded as unset.
	if rows[0].TTFTMs != -1 {
		t.Errorf("ttft = %d, want -1 for a non-streaming request", rows[0].TTFTMs)
	}
}

func TestUnknownModelIs404AndLogged(t *testing.T) {
	rig := newRig(t, 8, streamHandler(0, []string{"x"}))
	resp := rig.post(t, `{"model":"nope/nope","messages":[]}`)
	defer resp.Body.Close()
	if resp.StatusCode != 404 {
		t.Errorf("status = %d, want 404", resp.StatusCode)
	}
	rows := rig.waitForRecord(t, 1)
	if rows[0].Status != 404 {
		t.Errorf("recorded status = %d, want 404", rows[0].Status)
	}
}

func TestModelsEndpointIsPublic(t *testing.T) {
	rig := newRig(t, 8, streamHandler(0, nil))
	// OpenRouter's monitor polls this without credentials.
	resp, err := http.Get(rig.gw.URL + "/v1/models")
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("status = %d, want 200 without auth", resp.StatusCode)
	}
	var doc struct {
		Data []map[string]any `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&doc); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(doc.Data) != 1 {
		t.Fatalf("got %d models, want 1", len(doc.Data))
	}
	if doc.Data[0]["schema_version"] != "2.4" {
		t.Errorf("schema_version = %v, want 2.4", doc.Data[0]["schema_version"])
	}
}

func TestOversizedBodyIsUserError(t *testing.T) {
	rig := newRig(t, 8, streamHandler(0, nil))
	// 413 must not count against uptime, so it is important it is not a 500.
	huge := strings.Repeat("a", 33<<20)
	resp := rig.post(t, `{"model":"qwen/qwen3-32b","messages":[{"role":"user","content":"`+huge+`"}]}`)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusRequestEntityTooLarge {
		t.Errorf("status = %d, want 413", resp.StatusCode)
	}
}

// --- playground ------------------------------------------------------------

func (r *testRig) adminPost(t *testing.T, path, body string) *http.Response {
	t.Helper()
	req, _ := http.NewRequest("POST", r.gw.URL+path, strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer admin-test")
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request: %v", err)
	}
	return resp
}

// newAdminRig builds a rig whose handler is the admin surface rather than the
// public API, so the playground endpoints can be exercised.
func newAdminRig(t *testing.T, handler http.HandlerFunc) *testRig {
	t.Helper()
	rig := newRig(t, 8, handler)
	rig.gw.Config.Handler = rig.srv.AdminHandler()
	return rig
}

func TestPlaygroundRequiresAdminAuth(t *testing.T) {
	rig := newAdminRig(t, streamHandler(0, []string{"x"}))
	resp, err := http.Post(rig.gw.URL+"/admin/api/playground", "application/json",
		strings.NewReader(`{"model":"qwen/qwen3-32b","messages":[]}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusUnauthorized {
		t.Errorf("status = %d, want 401 without the admin token", resp.StatusCode)
	}
}

// TestPlaygroundStreamsThroughTheRealPath is the point of the playground: it
// must exercise the same proxy as customer traffic, so a broken stream shows up
// here rather than passing a simplified test console and failing in production.
func TestPlaygroundStreams(t *testing.T) {
	rig := newAdminRig(t, streamHandler(20*time.Millisecond, []string{"a", "b", "c"}))

	resp := rig.adminPost(t, "/admin/api/playground",
		`{"model":"qwen/qwen3-32b","messages":[{"role":"user","content":"hi"}],"stream":true}`)
	if resp.StatusCode != 200 {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("content-type = %q, want text/event-stream", ct)
	}
	frames := readFrames(t, resp)
	if len(frames) == 0 || frames[len(frames)-1] != "[DONE]" {
		t.Fatalf("stream did not terminate with [DONE]: %v", frames)
	}
	// The upstream model name must be rewritten here too.
	for _, f := range frames {
		if strings.Contains(f, "Qwen/Qwen3-32B") {
			t.Errorf("upstream model name leaked into the playground: %s", f)
		}
	}

	// Playground traffic is real load on a real backend, so it is recorded
	// rather than hidden, and tagged so it can be told from customer traffic.
	rows := rig.waitForRecord(t, 1)
	if rows[0].Status != 200 {
		t.Errorf("recorded status = %d, want 200", rows[0].Status)
	}
	if rows[0].TTFTMs < 0 {
		t.Error("playground request should have a measured TTFT")
	}
}

func TestPlaygroundUsesDistinctKeyID(t *testing.T) {
	rig := newAdminRig(t, streamHandler(0, []string{"a"}))
	resp := rig.adminPost(t, "/admin/api/playground",
		`{"model":"qwen/qwen3-32b","messages":[],"stream":true}`)
	io.Copy(io.Discard, resp.Body)
	resp.Body.Close()
	rig.waitForRecord(t, 1)

	var keyID int64
	err := rig.st.DB().QueryRow(`SELECT api_key_id FROM requests ORDER BY id DESC LIMIT 1`).Scan(&keyID)
	if err != nil {
		t.Fatalf("query: %v", err)
	}
	if keyID != playgroundKeyID {
		t.Errorf("api_key_id = %d, want %d so playground traffic is identifiable",
			keyID, playgroundKeyID)
	}
}

func TestPlaygroundModelsReportsAvailability(t *testing.T) {
	rig := newAdminRig(t, streamHandler(0, nil))
	req, _ := http.NewRequest("GET", rig.gw.URL+"/admin/api/playground/models", nil)
	req.Header.Set("Authorization", "Bearer admin-test")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	var out struct {
		Models []PlaygroundModel `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	if len(out.Models) != 1 {
		t.Fatalf("got %d models, want 1", len(out.Models))
	}
	m := out.Models[0]
	if m.ID != "qwen/qwen3-32b" {
		t.Errorf("id = %q", m.ID)
	}
	// The static backend in the rig is healthy, so the model is usable.
	if !m.Available {
		t.Errorf("model should be available; reason given: %q", m.Reason)
	}
	if m.ContextLength != 131072 {
		t.Errorf("context = %d, want 131072", m.ContextLength)
	}
}

// TestPlaygroundUnavailableModelExplainsWhy: offering a dead option with no
// explanation is worse than not offering it.
func TestPlaygroundUnavailableModelExplainsWhy(t *testing.T) {
	rig := newAdminRig(t, streamHandler(0, nil))
	// Mark the only backend unhealthy.
	for _, b := range rig.pool.Backends() {
		b.SetHealthyForTest(false)
	}
	req, _ := http.NewRequest("GET", rig.gw.URL+"/admin/api/playground/models", nil)
	req.Header.Set("Authorization", "Bearer admin-test")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	var out struct {
		Models []PlaygroundModel `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	if out.Models[0].Available {
		t.Fatal("model reported available with no healthy backend")
	}
	if out.Models[0].Reason == "" {
		t.Error("an unavailable model must explain why")
	}
}
