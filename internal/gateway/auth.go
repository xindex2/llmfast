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

// requireAdmin gates the admin surface with a constant-time token comparison.
func (s *Server) requireAdmin(w http.ResponseWriter, r *http.Request) bool {
	want := s.cfg.Server.AdminToken
	if want == "" {
		writeError(w, http.StatusForbidden, "config_error",
			"Admin token is not configured. Set server.admin_token or LLMFAST_ADMIN_TOKEN.")
		return false
	}
	got := store.ExtractBearer(r.Header.Get("Authorization"))
	if got == "" {
		// The UI is a plain page load, so it cannot set a header; it passes the
		// token as a cookie set at login instead.
		if c, err := r.Cookie("llmfast_admin"); err == nil {
			got = c.Value
		}
	}
	if subtle.ConstantTimeCompare([]byte(got), []byte(want)) != 1 {
		writeError(w, http.StatusUnauthorized, "authentication_error", "Invalid admin token.")
		return false
	}
	return true
}
