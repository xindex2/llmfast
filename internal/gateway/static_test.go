package gateway

import (
	"net/http"
	"net/http/httptest"
	"testing"
	"testing/fstest"
)

// TestStaticAssetsRevalidate pins the behaviour that made an upgraded admin UI
// invisible: without a validator, browsers cached the embedded assets on a
// heuristic of their own choosing and kept serving the previous build.
func TestStaticAssetsRevalidate(t *testing.T) {
	h := staticHandler(fstest.MapFS{
		"index.html": {Data: []byte("<h1>hi</h1>")},
		"app.js":     {Data: []byte("console.log(1)")},
	})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/app.js", nil))
	tag := rec.Header().Get("ETag")
	if tag == "" {
		t.Fatal("no ETag, so a browser has nothing to revalidate against")
	}
	if got := rec.Header().Get("Cache-Control"); got != "no-cache" {
		t.Errorf("Cache-Control = %q, want no-cache", got)
	}

	// An unchanged asset must still be cheap: 304, no body.
	req := httptest.NewRequest(http.MethodGet, "/app.js", nil)
	req.Header.Set("If-None-Match", tag)
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusNotModified {
		t.Errorf("matching ETag returned %d, want 304", rec.Code)
	}
	if rec.Body.Len() != 0 {
		t.Errorf("304 carried a %d-byte body", rec.Body.Len())
	}

	// A changed asset must not be served from cache.
	h2 := staticHandler(fstest.MapFS{"app.js": {Data: []byte("console.log(2)")}})
	rec = httptest.NewRecorder()
	req = httptest.NewRequest(http.MethodGet, "/app.js", nil)
	req.Header.Set("If-None-Match", tag)
	h2.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Errorf("changed asset returned %d, want 200", rec.Code)
	}
	if rec.Body.String() != "console.log(2)" {
		t.Errorf("served %q, want the new content", rec.Body.String())
	}

	// "/" is index.html and needs the same treatment.
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/", nil))
	if rec.Header().Get("ETag") == "" {
		t.Error("/ has no ETag, so the shell page can go stale too")
	}
}
