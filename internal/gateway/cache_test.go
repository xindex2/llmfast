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
