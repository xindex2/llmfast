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
