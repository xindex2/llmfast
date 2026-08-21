package gateway

import (
	"context"
	"crypto/subtle"
	"net/http"
	"sync"
	"time"

	"github.com/llmfast/gateway/internal/store"
)

// keyCacheTTL bounds how long a revoked key keeps working. Hitting SQLite on
// every inference request would put a disk read in front of every stream, so
// lookups are cached; 30s is the compromise between that and revocation lag.
const keyCacheTTL = 30 * time.Second

type cachedKey struct {
	key store.APIKey
	at  time.Time
	ok  bool
}

type keyCache struct {
	mu sync.RWMutex
	m  map[string]cachedKey
}

func newKeyCache() *keyCache { return &keyCache{m: make(map[string]cachedKey)} }

func (c *keyCache) get(secret string) (cachedKey, bool) {
	c.mu.RLock()
	e, ok := c.m[secret]
	c.mu.RUnlock()
	if !ok || time.Since(e.at) > keyCacheTTL {
		return cachedKey{}, false
	}
	return e, true
}

func (c *keyCache) put(secret string, k store.APIKey, ok bool) {
	c.mu.Lock()
	// Negative entries are cached too, so a flood of bad credentials cannot be
	// turned into a database read amplification attack.
	if len(c.m) > 10000 {
		c.m = make(map[string]cachedKey)
	}
	c.m[secret] = cachedKey{key: k, at: time.Now(), ok: ok}
	c.mu.Unlock()
}

func (c *keyCache) purge() {
	c.mu.Lock()
	c.m = make(map[string]cachedKey)
	c.mu.Unlock()
}

// authenticate resolves the bearer token. It returns the key and true on
// success; on failure it has already written the error response.
//
// Rejections are logged, not just returned. OpenRouter counts 401 against our
// uptime, so a rotated or mistyped key would otherwise tank our score on their
// side while this dashboard showed a clean sheet. The cost is that unauthorised
// scanners also land in the log; bind the public listener behind a firewall if
// that becomes noisy.
func (s *Server) authenticate(w http.ResponseWriter, r *http.Request) (store.APIKey, bool) {
	reject := func(reason string) {
		writeError(w, http.StatusUnauthorized, "authentication_error", "Invalid API key.")
		s.store.Log(store.Record{
			TS: time.Now(), RequestID: w.Header().Get("X-Request-Id"),
			Model: "-", Status: http.StatusUnauthorized, TTFTMs: -1, Error: reason,
		})
	}

	secret := store.ExtractBearer(r.Header.Get("Authorization"))
	if secret == "" {
		writeError(w, http.StatusUnauthorized, "authentication_error",
			"Missing Authorization header. Pass your key as: Authorization: Bearer "+store.KeyPrefix+"...")
		s.store.Log(store.Record{
			TS: time.Now(), RequestID: w.Header().Get("X-Request-Id"),
			Model: "-", Status: http.StatusUnauthorized, TTFTMs: -1, Error: "missing authorization header",
		})
		return store.APIKey{}, false
	}
	if e, hit := s.keys.get(secret); hit {
		if !e.ok {
			reject("unknown api key")
			return store.APIKey{}, false
		}
		return e.key, true
	}
	ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
	defer cancel()
	k, err := s.store.LookupKey(ctx, secret)
	s.keys.put(secret, k, err == nil)
	if err != nil {
		reject("unknown or disabled api key")
		return store.APIKey{}, false
	}
	return k, true
}

// rateLimiter is a fixed-window counter per API key. A fixed window can allow a
// 2x burst across a boundary, which is acceptable here: this is a fairness
// guard on our own keys, while the real capacity limit is the backend
// admission control in the upstream pool.
type rateLimiter struct {
	mu      sync.Mutex
	windows map[int64]*rlWindow
}

type rlWindow struct {
	start time.Time
	count int
}

func newRateLimiter() *rateLimiter { return &rateLimiter{windows: make(map[int64]*rlWindow)} }

// allow reports whether the key may make another request. limit <= 0 disables
// the check.
func (rl *rateLimiter) allow(keyID int64, limit int) bool {
	if limit <= 0 {
		return true
	}
	now := time.Now()
	rl.mu.Lock()
	defer rl.mu.Unlock()
	w, ok := rl.windows[keyID]
	if !ok || now.Sub(w.start) >= time.Minute {
		rl.windows[keyID] = &rlWindow{start: now, count: 1}
		return true
	}
	if w.count >= limit {
		return false
	}
	w.count++
	return true
}

// sweep drops windows for keys that have gone quiet, so the map does not grow
// without bound across key rotations.
func (rl *rateLimiter) sweep() {
	now := time.Now()
	rl.mu.Lock()
	for id, w := range rl.windows {
		if now.Sub(w.start) > 5*time.Minute {
			delete(rl.windows, id)
		}
	}
	rl.mu.Unlock()
}

// requireAdmin gates the admin surface.
//
// Two credentials are accepted, for two different callers. A person signs in
// with an email and password and carries a session cookie; a script sends the
// configured bearer token. Sessions can be listed, expired and revoked one at
// a time, which a single shared token cannot be, so the token is not offered
// as a way to sign into the UI once any account exists.
func (s *Server) requireAdmin(w http.ResponseWriter, r *http.Request) bool {
	if u, ok := s.sessionUser(r); ok {
		_ = u
		return true
	}

	want := s.cfg.Server.AdminToken
	got := store.ExtractBearer(r.Header.Get("Authorization"))
	if want != "" && got != "" &&
		subtle.ConstantTimeCompare([]byte(got), []byte(want)) == 1 {
		return true
	}

	// Say which credentials would actually work here, rather than a flat
	// "unauthorized" that leaves an operator guessing.
	if want == "" && !s.hasAdminUsers() {
		writeError(w, http.StatusForbidden, "config_error",
			"No admin account exists and no admin token is configured. "+
				"Create an account with: llmfast -config <file> -add-admin <email>")
		return false
	}
	writeError(w, http.StatusUnauthorized, "authentication_error", "Not signed in.")
	return false
}

// sessionUser resolves the session cookie, if there is a valid one.
func (s *Server) sessionUser(r *http.Request) (store.AdminUser, bool) {
	c, err := r.Cookie(sessionCookie)
	if err != nil || c.Value == "" {
		return store.AdminUser{}, false
	}
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()
	u, err := s.store.LookupSession(ctx, c.Value)
	if err != nil {
		return store.AdminUser{}, false
	}
	return u, true
}

// hasAdminUsers reports whether any account exists, so the login page can offer
// the right thing and requireAdmin can explain itself.
func (s *Server) hasAdminUsers() bool {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	n, err := s.store.CountAdminUsers(ctx)
	return err == nil && n > 0
}
