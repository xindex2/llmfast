package gateway

import (
	"context"
	"errors"
	"net"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/llmfast/gateway/internal/store"
)

const sessionCookie = "llmfast_session"

// loginLimiter throttles password guessing.
//
// A password is guessable in a way an API key is not, and this endpoint is
// reachable from the internet, so without a limit an attacker gets unlimited
// attempts at whatever the operator chose. The KDF makes each attempt cost
// about a tenth of a second of CPU, which is also a denial-of-service lever if
// attempts are unbounded.
type loginLimiter struct {
	mu       sync.Mutex
	attempts map[string][]time.Time
}

const (
	loginWindow   = 15 * time.Minute
	loginMaxTries = 10
)

func newLoginLimiter() *loginLimiter {
	return &loginLimiter{attempts: map[string][]time.Time{}}
}

// allow records an attempt and reports whether it may proceed.
func (l *loginLimiter) allow(ip string) bool {
	now := time.Now()
	l.mu.Lock()
	defer l.mu.Unlock()

	kept := l.attempts[ip][:0]
	for _, t := range l.attempts[ip] {
		if now.Sub(t) < loginWindow {
			kept = append(kept, t)
		}
	}
	// Bound the map: without this an attacker rotating source addresses grows
	// it without limit.
	if len(l.attempts) > 10_000 {
		for k, v := range l.attempts {
			if len(v) == 0 || now.Sub(v[len(v)-1]) > loginWindow {
				delete(l.attempts, k)
			}
		}
	}
	if len(kept) >= loginMaxTries {
		l.attempts[ip] = kept
		return false
	}
	l.attempts[ip] = append(kept, now)
	return true
}

func (l *loginLimiter) reset(ip string) {
	l.mu.Lock()
	delete(l.attempts, ip)
	l.mu.Unlock()
}

// clientIP prefers the address Cloudflare reports, since every request arrives
// through the tunnel and would otherwise share one loopback address -- which
// would let one attacker's failures lock out everybody.
func clientIP(r *http.Request) string {
	if ip := r.Header.Get("CF-Connecting-IP"); ip != "" {
		return ip
	}
	if fwd := r.Header.Get("X-Forwarded-For"); fwd != "" {
		if i := strings.IndexByte(fwd, ','); i > 0 {
			return strings.TrimSpace(fwd[:i])
		}
		return strings.TrimSpace(fwd)
	}
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return r.RemoteAddr
	}
	return host
}

// adminAuthState tells the login page what to render: a sign-in form, or a
// first-run form that creates the initial account.
func (s *Server) adminAuthState(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()
	n, err := s.store.CountAdminUsers(ctx)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "server_error", err.Error())
		return
	}
	res := map[string]any{"needs_setup": n == 0, "signed_in": false}
	if u, ok := s.sessionUser(r); ok {
		res["signed_in"] = true
		res["email"] = u.Email
	}
	writeJSON(w, http.StatusOK, res)
}

// adminSetup creates the first account. It is only available while no account
// exists: afterwards, an open registration endpoint on a public hostname would
// hand the dashboard to whoever found it first.
func (s *Server) adminSetup(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Email    string `json:"email"`
		Password string `json:"password"`
	}
	_ = decodeJSON(r, &body)

	ctx, cancel := context.WithTimeout(r.Context(), 15*time.Second)
	defer cancel()
	n, err := s.store.CountAdminUsers(ctx)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "server_error", err.Error())
		return
	}
	if n > 0 {
		writeError(w, http.StatusForbidden, "invalid_request_error",
			"An admin account already exists. Sign in, or add more accounts from the dashboard.")
		return
	}
	u, err := s.store.CreateAdminUser(ctx, body.Email, body.Password)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	s.log.Info("first admin account created", "email", u.Email)
	s.startSession(w, r, u)
}

func (s *Server) adminLogin(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Email    string `json:"email"`
		Password string `json:"password"`
	}
	_ = decodeJSON(r, &body)

	ip := clientIP(r)
	if !s.logins.allow(ip) {
		writeError(w, http.StatusTooManyRequests, "rate_limit_error",
			"Too many sign-in attempts. Wait 15 minutes and try again.")
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 15*time.Second)
	defer cancel()
	u, err := s.store.VerifyAdminLogin(ctx, body.Email, body.Password)
	if err != nil {
		if errors.Is(err, store.ErrBadCredential) {
			// One message for both a wrong address and a wrong password, so
			// this cannot be used to discover which addresses have accounts.
			s.log.Warn("failed admin sign-in", "email", store.NormalizeEmail(body.Email), "ip", ip)
			writeError(w, http.StatusUnauthorized, "authentication_error",
				"Incorrect email or password.")
			return
		}
		writeError(w, http.StatusInternalServerError, "server_error", err.Error())
		return
	}
	s.logins.reset(ip)
	s.log.Info("admin signed in", "email", u.Email, "ip", ip)
	s.startSession(w, r, u)
}

// startSession mints a session and sets it as a cookie.
func (s *Server) startSession(w http.ResponseWriter, r *http.Request, u store.AdminUser) {
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()
	token, err := s.store.CreateSession(ctx, u.ID)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "server_error", err.Error())
		return
	}
	http.SetCookie(w, &http.Cookie{
		Name:     sessionCookie,
		Value:    token,
		Path:     "/",
		HttpOnly: true, // never readable from JavaScript, so an XSS cannot lift it
		Secure:   isHTTPS(r),
		SameSite: http.SameSiteLaxMode,
		MaxAge:   int(store.SessionTTL.Seconds()),
	})
	writeJSON(w, http.StatusOK, map[string]any{"ok": true, "email": u.Email})
}

// isHTTPS reports whether the browser's connection is encrypted. Behind the
// Cloudflare tunnel the origin hop is plain HTTP, so r.TLS is nil even though
// the user is on HTTPS; the forwarded header is what says so. Marking the
// cookie Secure on a plain-HTTP localhost session would stop it being stored
// at all, which is why this is conditional rather than always on.
func isHTTPS(r *http.Request) bool {
	if r.TLS != nil {
		return true
	}
	return strings.EqualFold(r.Header.Get("X-Forwarded-Proto"), "https")
}

func (s *Server) adminLogout(w http.ResponseWriter, r *http.Request) {
	if c, err := r.Cookie(sessionCookie); err == nil && c.Value != "" {
		ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
		defer cancel()
		_ = s.store.DeleteSession(ctx, c.Value)
	}
	http.SetCookie(w, &http.Cookie{
		Name: sessionCookie, Value: "", Path: "/",
		HttpOnly: true, Secure: isHTTPS(r), SameSite: http.SameSiteLaxMode, MaxAge: -1,
	})
	writeJSON(w, http.StatusOK, map[string]bool{"ok": true})
}

// --- account management, for signed-in operators ---------------------------

func (s *Server) adminListUsers(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()
	users, err := s.store.ListAdminUsers(ctx)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "server_error", err.Error())
		return
	}
	me := ""
	if u, ok := s.sessionUser(r); ok {
		me = u.Email
	}
	writeJSON(w, http.StatusOK, map[string]any{"users": users, "me": me})
}

func (s *Server) adminCreateUser(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Email    string `json:"email"`
		Password string `json:"password"`
	}
	_ = decodeJSON(r, &body)
	ctx, cancel := context.WithTimeout(r.Context(), 15*time.Second)
	defer cancel()
	u, err := s.store.CreateAdminUser(ctx, body.Email, body.Password)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, u)
}

// adminChangePassword requires the current password even though the caller is
// already signed in, so that a session left open on an unattended machine
// cannot be used to take the account over.
func (s *Server) adminChangePassword(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Current string `json:"current_password"`
		New     string `json:"new_password"`
	}
	_ = decodeJSON(r, &body)

	u, ok := s.sessionUser(r)
	if !ok {
		writeError(w, http.StatusForbidden, "invalid_request_error",
			"Changing a password requires signing in; a bearer token cannot do it.")
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()
	if _, err := s.store.VerifyAdminLogin(ctx, u.Email, body.Current); err != nil {
		writeError(w, http.StatusUnauthorized, "authentication_error",
			"Your current password is not correct.")
		return
	}
	if err := s.store.SetAdminPassword(ctx, u.Email, body.New); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	// Every session was just invalidated, including this one, so issue a fresh
	// one rather than signing the operator out of the page they are on.
	s.startSession(w, r, u)
}

func (s *Server) adminDeleteUser(w http.ResponseWriter, r *http.Request) {
	id, err := strconv.ParseInt(r.PathValue("id"), 10, 64)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "invalid user id")
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()
	if err := s.store.DeleteAdminUser(ctx, id); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]bool{"ok": true})
}
