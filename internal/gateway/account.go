package gateway

import (
	"context"
	"embed"
	"errors"
	"io/fs"
	"net/http"
	"strconv"
	"time"

	"github.com/llmfast/gateway/internal/store"
)

// The marketing site and the customer dashboard. Both are served from the
// public listener: the site at /, the dashboard at /app.
//
//go:embed webui/*
var webFS embed.FS

const userCookie = "llmfast_user"

// PublicRoutes adds everything a customer touches: the marketing pages, the
// dashboard, sign-up and sign-in, their own keys and their own usage.
func (s *Server) publicRoutes(mux *http.ServeMux) {
	mux.HandleFunc("POST /api/register", s.userRegister)
	mux.HandleFunc("POST /api/login", s.userLogin)
	mux.HandleFunc("POST /api/logout", s.userLogout)
	mux.HandleFunc("GET /api/me", s.userMe)

	mux.HandleFunc("GET /api/keys", s.userGuard(s.userListKeys))
	mux.HandleFunc("POST /api/keys", s.userGuard(s.userCreateKey))
	mux.HandleFunc("DELETE /api/keys/{id}", s.userGuard(s.userDeleteKey))
	mux.HandleFunc("GET /api/usage", s.userGuard(s.userUsage))
	mux.HandleFunc("GET /api/requests", s.userGuard(s.userRequests))
	mux.HandleFunc("POST /api/password", s.userGuard(s.userChangePassword))

	if sub, err := fs.Sub(webFS, "webui"); err == nil {
		mux.Handle("GET /app", http.RedirectHandler("/app/", http.StatusMovedPermanently))
		mux.Handle("GET /app/", http.StripPrefix("/app/", staticHandler(sub)))
	}
}

// userGuard requires a signed-in customer account. Admin accounts are accepted
// too, so an operator can see the dashboard their customers see.
func (s *Server) userGuard(h func(http.ResponseWriter, *http.Request, store.AdminUser)) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		u, ok := s.currentUser(r)
		if !ok {
			writeError(w, http.StatusUnauthorized, "authentication_error", "Not signed in.")
			return
		}
		h(w, r, u)
	}
}

func (s *Server) currentUser(r *http.Request) (store.AdminUser, bool) {
	c, err := r.Cookie(userCookie)
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

func (s *Server) startUserSession(w http.ResponseWriter, r *http.Request, u store.AdminUser) {
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()
	token, err := s.store.CreateSession(ctx, u.ID)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "server_error", err.Error())
		return
	}
	http.SetCookie(w, &http.Cookie{
		Name: userCookie, Value: token, Path: "/",
		HttpOnly: true, Secure: isHTTPS(r), SameSite: http.SameSiteLaxMode,
		MaxAge: int(store.SessionTTL.Seconds()),
	})
	writeJSON(w, http.StatusOK, map[string]any{"ok": true, "email": u.Email, "role": u.Role})
}

func (s *Server) userRegister(w http.ResponseWriter, r *http.Request) {
	var body struct{ Email, Password string }
	_ = decodeJSON(r, &body)

	// Sign-up is a password endpoint reachable from the internet, so it is
	// throttled on the same limiter as sign-in: without it, an attacker can
	// both enumerate addresses and burn CPU on the KDF for free.
	ip := clientIP(r)
	if !s.logins.allow(ip) {
		writeError(w, http.StatusTooManyRequests, "rate_limit_error",
			"Too many attempts. Wait 15 minutes and try again.")
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 15*time.Second)
	defer cancel()
	u, err := s.store.CreateUser(ctx, body.Email, body.Password)
	if err != nil {
		if errors.Is(err, store.ErrUserExists) {
			// Deliberately the same wording a caller would see for a wrong
			// password, so sign-up cannot be used to discover who has an
			// account here.
			writeError(w, http.StatusBadRequest, "invalid_request_error",
				"That email cannot be registered. If you already have an account, sign in.")
			return
		}
		writeError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	s.log.Info("customer registered", "email", u.Email)
	s.startUserSession(w, r, u)
}

func (s *Server) userLogin(w http.ResponseWriter, r *http.Request) {
	var body struct{ Email, Password string }
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
		writeError(w, http.StatusUnauthorized, "authentication_error", "Incorrect email or password.")
		return
	}
	s.logins.reset(ip)
	s.startUserSession(w, r, u)
}

func (s *Server) userLogout(w http.ResponseWriter, r *http.Request) {
	if c, err := r.Cookie(userCookie); err == nil && c.Value != "" {
		ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
		defer cancel()
		_ = s.store.DeleteSession(ctx, c.Value)
	}
	http.SetCookie(w, &http.Cookie{Name: userCookie, Value: "", Path: "/",
		HttpOnly: true, Secure: isHTTPS(r), SameSite: http.SameSiteLaxMode, MaxAge: -1})
	writeJSON(w, http.StatusOK, map[string]bool{"ok": true})
}

func (s *Server) userMe(w http.ResponseWriter, r *http.Request) {
	u, ok := s.currentUser(r)
	if !ok {
		writeJSON(w, http.StatusOK, map[string]any{"signed_in": false})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"signed_in": true, "email": u.Email, "role": u.Role,
		"is_admin": u.IsAdmin(), "created_at": u.CreatedAt,
	})
}

func (s *Server) userListKeys(w http.ResponseWriter, r *http.Request, u store.AdminUser) {
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()
	keys, err := s.store.ListKeysFor(ctx, u.ID)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "server_error", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"keys": keys})
}

func (s *Server) userCreateKey(w http.ResponseWriter, r *http.Request, u store.AdminUser) {
	var body struct {
		Name string `json:"name"`
	}
	_ = decodeJSON(r, &body)
	if body.Name == "" {
		body.Name = "default"
	}
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()

	// A cap per account, so one sign-up cannot fill the table.
	if existing, err := s.store.ListKeysFor(ctx, u.ID); err == nil && len(existing) >= 10 {
		writeError(w, http.StatusBadRequest, "invalid_request_error",
			"You already have 10 keys. Delete one before creating another.")
		return
	}
	key, secret, err := s.store.CreateKeyFor(ctx, u.ID, body.Name, 0)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "server_error", err.Error())
		return
	}
	// The secret is returned exactly once; only its hash is stored.
	writeJSON(w, http.StatusOK, map[string]any{"key": key, "secret": secret})
}

func (s *Server) userDeleteKey(w http.ResponseWriter, r *http.Request, u store.AdminUser) {
	id, err := strconv.ParseInt(r.PathValue("id"), 10, 64)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "invalid key id")
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()

	// Check ownership before deleting. Without this, any signed-in customer
	// could revoke any other customer's key by guessing its id.
	owner, err := s.store.KeyOwner(ctx, id)
	if err != nil || owner != u.ID {
		writeError(w, http.StatusNotFound, "invalid_request_error", "No such key.")
		return
	}
	if err := s.store.DeleteKey(ctx, id); err != nil {
		writeError(w, http.StatusInternalServerError, "server_error", err.Error())
		return
	}
	s.keys.purge()
	writeJSON(w, http.StatusOK, map[string]bool{"ok": true})
}

func (s *Server) userUsage(w http.ResponseWriter, r *http.Request, u store.AdminUser) {
	win := parseWindow(r.URL.Query().Get("range"))
	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()

	total, err := s.store.UsageFor(ctx, u.ID, win.from, win.to)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "server_error", err.Error())
		return
	}
	byModel, err := s.store.UsageByModelFor(ctx, u.ID, win.from, win.to)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "server_error", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"range": r.URL.Query().Get("range"), "total": total, "by_model": byModel,
	})
}

func (s *Server) userRequests(w http.ResponseWriter, r *http.Request, u store.AdminUser) {
	limit, _ := strconv.Atoi(r.URL.Query().Get("limit"))
	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()
	rows, err := s.store.RecentFor(ctx, u.ID, limit)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "server_error", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"requests": rows})
}

func (s *Server) userChangePassword(w http.ResponseWriter, r *http.Request, u store.AdminUser) {
	var body struct {
		Current string `json:"current_password"`
		New     string `json:"new_password"`
	}
	_ = decodeJSON(r, &body)
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
	// Every session was just dropped, including this one.
	s.startUserSession(w, r, u)
}
