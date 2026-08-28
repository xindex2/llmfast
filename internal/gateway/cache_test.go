package gateway

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"testing/fstest"
)

// TestShellIsNeverCached: index.html is what pulls in every other asset, so a
// stale copy of it pins the whole UI to an old build. An ETag is not enough
// for a copy that was cached before ETags were sent -- the browser keeps its
// own heuristic freshness and never asks again. The shell is two kilobytes;
// it is not worth caching at all.
func TestShellIsNeverCached(t *testing.T) {
	h := staticHandler(fstest.MapFS{
		"index.html": {Data: []byte("<h1>hi</h1>")},
		"app.js":     {Data: []byte("console.log(1)")},
	})
	for _, path := range []string{"/", "/index.html"} {
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, path, nil))
		if cc := rec.Header().Get("Cache-Control"); !strings.Contains(cc, "no-store") {
			t.Errorf("%s Cache-Control = %q, want no-store", path, cc)
		}
	}
	// Assets keep the cheaper revalidation, since they are the large ones.
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/app.js", nil))
	if rec.Header().Get("ETag") == "" {
		t.Error("app.js lost its ETag")
	}
}

// TestLocalCheckpointGetsASensibleID: a checkpoint loaded from disk arrives
// with a filesystem path where a repository id belongs. Publishing that as the
// model id would leak the machine's directory layout to every caller and read
// as nonsense in an API response.
func TestLocalCheckpointGetsASensibleID(t *testing.T) {
	cases := map[string]struct{ id, name string }{
		"/Users/pro/Desktop/llmfffff2/models/qwen3-0.6b": {"local/qwen3-0.6b", "qwen3-0.6b"},
		"/srv/checkpoints/my-finetune/":                  {"local/my-finetune", "my-finetune"},
	}
	for path, want := range cases {
		if got := SuggestModelID(path); got != want.id {
			t.Errorf("SuggestModelID(%q) = %q, want %q", path, got, want.id)
		}
		if got := SuggestDisplayName(path); got != want.name {
			t.Errorf("SuggestDisplayName(%q) = %q, want %q", path, got, want.name)
		}
	}
	// Repository ids are untouched.
	if got := SuggestModelID("Qwen/Qwen3-8B"); got != "qwen/qwen3-8b" {
		t.Errorf("repo id changed: %q", got)
	}
}

// TestBenchNoteIgnoresLevelsThatShedLoad reproduces a real measurement from a
// CPU node whose engine had four slots. At 8 and 16 concurrent it rejected
// half the requests -- and *reported a higher aggregate* than the levels that
// served everything, because a rejected request finishes instantly and
// shortens the wall clock. The note recommended max_concurrency 8, which is
// precisely the setting that sheds load, and shed load is what OpenRouter
// counts against uptime.
func TestBenchNoteIgnoresLevelsThatShedLoad(t *testing.T) {
	r := BenchResult{Levels: []BenchLevel{
		{Concurrency: 1, AggregateTPS: 52, Errors: 0},
		{Concurrency: 4, AggregateTPS: 54, Errors: 0},
		{Concurrency: 8, AggregateTPS: 67, Errors: 4},
		{Concurrency: 16, AggregateTPS: 67, Errors: 4},
	}}
	for _, lv := range r.Levels {
		if lv.Errors == 0 && lv.AggregateTPS > r.PeakAggregate {
			r.PeakAggregate, r.BestConcurrency = lv.AggregateTPS, lv.Concurrency
		}
	}
	if r.BestConcurrency != 4 || r.PeakAggregate != 54 {
		t.Fatalf("peak = %.0f at %d; only error-free levels count",
			r.PeakAggregate, r.BestConcurrency)
	}

	note := benchNote(r)
	if strings.Contains(note, "near 8") || strings.Contains(note, "near 16") {
		t.Errorf("recommended a level that shed load: %s", note)
	}
	if !strings.Contains(note, "returned errors") {
		t.Errorf("note = %q, want the errors called out", note)
	}
	if !strings.Contains(note, "--parallel") {
		t.Errorf("note = %q, want the engine slot limit named", note)
	}
}

// TestBenchNoteFlagsABadConcurrencyTrade: on a CPU node, going from 1 to 4
// concurrent bought 4% more aggregate throughput and cost 3.3x per-request.
// OpenRouter ranks on per-request, so that is worth saying out loud rather
// than leaving in two columns of a table.
func TestBenchNoteFlagsABadConcurrencyTrade(t *testing.T) {
	r := BenchResult{Levels: []BenchLevel{
		{Concurrency: 1, AggregateTPS: 52, PerRequestTPS: 55.5, Errors: 0},
		{Concurrency: 4, AggregateTPS: 54, PerRequestTPS: 16.7, Errors: 0},
	}}
	for _, lv := range r.Levels {
		if lv.Errors == 0 && lv.AggregateTPS >= r.PeakAggregate {
			r.PeakAggregate, r.BestConcurrency = lv.AggregateTPS, lv.Concurrency
		}
	}
	note := benchNote(r)
	if !strings.Contains(note, "buys almost nothing") {
		t.Errorf("note = %q, want the per-request cost called out", note)
	}
	if !strings.Contains(note, "3.3x") {
		t.Errorf("note = %q, want the ratio quantified", note)
	}
}

// TestBenchNoteWhenEverythingErrors: no level is a safe setting, and saying
// nothing would leave the operator to read a plausible-looking table.
func TestBenchNoteWhenEverythingErrors(t *testing.T) {
	r := BenchResult{Levels: []BenchLevel{
		{Concurrency: 1, AggregateTPS: 20, Errors: 1},
		{Concurrency: 4, AggregateTPS: 40, Errors: 4},
	}}
	if note := benchNote(r); !strings.Contains(note, "Every level returned errors") {
		t.Errorf("note = %q", note)
	}
}
