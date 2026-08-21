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

// TestLoginLimiterBoundsGuessing: the sign-in endpoint is reachable from the
// internet and a password, unlike an API key, is guessable. Each attempt also
// costs a deliberate ~100ms of KDF, so unbounded attempts are a denial of
// service as well as a credential risk.
func TestLoginLimiterBoundsGuessing(t *testing.T) {
	l := newLoginLimiter()
	for i := 0; i < loginMaxTries; i++ {
		if !l.allow("203.0.113.7") {
			t.Fatalf("attempt %d was blocked; the limit is %d", i+1, loginMaxTries)
		}
	}
	if l.allow("203.0.113.7") {
		t.Error("attempt past the limit was allowed")
	}

	// One attacker must not be able to lock everyone else out.
	if !l.allow("198.51.100.4") {
		t.Error("a different address was blocked by another's failures")
	}

	// A successful sign-in clears the count, so a person who mistypes their
	// password a few times is not then locked out by their own success.
	l.reset("203.0.113.7")
	if !l.allow("203.0.113.7") {
		t.Error("the counter was not cleared after a successful sign-in")
	}
}

// TestClientIPPrefersForwardedAddress: every request arrives through the
// Cloudflare tunnel, so RemoteAddr is always loopback. Rate limiting on that
// would put every visitor in one bucket and let one attacker lock out the
// operator.
func TestClientIPPrefersForwardedAddress(t *testing.T) {
	cases := []struct{ name, cf, xff, remote, want string }{
		{"cloudflare header wins", "203.0.113.7", "10.0.0.1", "127.0.0.1:5000", "203.0.113.7"},
		{"first hop of x-forwarded-for", "", "203.0.113.9, 10.0.0.1", "127.0.0.1:5000", "203.0.113.9"},
		{"falls back to the socket", "", "", "192.0.2.5:5000", "192.0.2.5"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			r := httptest.NewRequest(http.MethodPost, "/admin/api/login", nil)
			r.RemoteAddr = c.remote
			if c.cf != "" {
				r.Header.Set("CF-Connecting-IP", c.cf)
			}
			if c.xff != "" {
				r.Header.Set("X-Forwarded-For", c.xff)
			}
			if got := clientIP(r); got != c.want {
				t.Errorf("clientIP = %q, want %q", got, c.want)
			}
		})
	}
}

// TestSessionCookieIsSecureBehindTheTunnel: cloudflared speaks plain HTTP to
// the origin, so r.TLS is nil even when the browser is on HTTPS. Reading only
// r.TLS would leave the cookie without Secure on every real deployment;
// setting it unconditionally would stop the cookie being stored at all over a
// plain-HTTP SSH forward.
func TestSessionCookieIsSecureBehindTheTunnel(t *testing.T) {
	plain := httptest.NewRequest(http.MethodGet, "/", nil)
	if isHTTPS(plain) {
		t.Error("a plain HTTP request was treated as secure")
	}
	fwd := httptest.NewRequest(http.MethodGet, "/", nil)
	fwd.Header.Set("X-Forwarded-Proto", "https")
	if !isHTTPS(fwd) {
		t.Error("a request forwarded from HTTPS was treated as insecure")
	}
}
