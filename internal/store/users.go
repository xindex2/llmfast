package store

import (
	"context"
	"crypto/pbkdf2"
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"database/sql"
	"encoding/base64"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"time"
)

var (
	ErrUserNotFound  = errors.New("admin user not found")
	ErrBadCredential = errors.New("incorrect email or password")
	ErrWeakPassword  = errors.New("password must be at least 10 characters")
	ErrUserExists    = errors.New("an account with that email already exists")
)

// SessionTTL is how long a browser stays signed in. Long enough not to be
// irritating, short enough that a forgotten open tab on a shared machine stops
// working within a day.
const SessionTTL = 12 * time.Hour

// pbkdf2Iterations follows OWASP's 2023 guidance for PBKDF2-HMAC-SHA256.
//
// Unlike API keys -- which are high-entropy random strings and so are stored as
// a plain SHA-256 -- a password is chosen by a person and is guessable. It
// needs a deliberately slow KDF so that an attacker holding the database file
// cannot test candidates in bulk.
const pbkdf2Iterations = 600_000

type AdminUser struct {
	ID        int64  `json:"id"`
	Email     string `json:"email"`
	CreatedAt int64  `json:"created_at"`
	LastLogin int64  `json:"last_login"`
}

// NormalizeEmail lowercases and trims, so that an address entered with
// different capitalisation on a later visit still signs in.
func NormalizeEmail(email string) string {
	return strings.ToLower(strings.TrimSpace(email))
}

// hashPassword returns a self-describing hash: the algorithm, its cost and the
// salt travel with the digest, so the iteration count can be raised later
// without invalidating existing passwords.
func hashPassword(password string) (string, error) {
	salt := make([]byte, 16)
	if _, err := rand.Read(salt); err != nil {
		return "", err
	}
	dk, err := pbkdf2.Key(sha256.New, password, salt, pbkdf2Iterations, 32)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("pbkdf2-sha256$%d$%s$%s", pbkdf2Iterations,
		base64.RawStdEncoding.EncodeToString(salt),
		base64.RawStdEncoding.EncodeToString(dk)), nil
}

// verifyPassword compares in constant time so that no timing signal
// distinguishes a wrong password from a nearly-right one.
func verifyPassword(encoded, password string) bool {
	parts := strings.Split(encoded, "$")
	if len(parts) != 4 || parts[0] != "pbkdf2-sha256" {
		return false
	}
	iter, err := strconv.Atoi(parts[1])
	if err != nil || iter <= 0 {
		return false
	}
	salt, err := base64.RawStdEncoding.DecodeString(parts[2])
	if err != nil {
		return false
	}
	want, err := base64.RawStdEncoding.DecodeString(parts[3])
	if err != nil {
		return false
	}
	got, err := pbkdf2.Key(sha256.New, password, salt, iter, len(want))
	if err != nil {
		return false
	}
	return subtle.ConstantTimeCompare(got, want) == 1
}

// CountAdminUsers reports how many accounts exist. Zero means the admin UI has
// no way to sign anyone in yet and must fall back to the configured token.
func (s *Store) CountAdminUsers(ctx context.Context) (int, error) {
	var n int
	err := s.r.QueryRowContext(ctx, `SELECT COUNT(*) FROM admin_users`).Scan(&n)
	return n, err
}

func (s *Store) CreateAdminUser(ctx context.Context, email, password string) (AdminUser, error) {
	email = NormalizeEmail(email)
	if email == "" || !strings.Contains(email, "@") {
		return AdminUser{}, errors.New("a valid email address is required")
	}
	if len([]rune(password)) < 10 {
		return AdminUser{}, ErrWeakPassword
	}
	hash, err := hashPassword(password)
	if err != nil {
		return AdminUser{}, err
	}
	now := time.Now().Unix()
	res, err := s.w.ExecContext(ctx,
		`INSERT INTO admin_users (email, password_hash, created_at) VALUES (?,?,?)`,
		email, hash, now)
	if err != nil {
		if strings.Contains(err.Error(), "UNIQUE") {
			return AdminUser{}, ErrUserExists
		}
		return AdminUser{}, err
	}
	id, _ := res.LastInsertId()
	return AdminUser{ID: id, Email: email, CreatedAt: now}, nil
}

// SetAdminPassword changes a password and invalidates every existing session
// for that account, so a password change actually evicts whoever else was
// signed in -- which is the entire point of changing it.
func (s *Store) SetAdminPassword(ctx context.Context, email, password string) error {
	email = NormalizeEmail(email)
	if len([]rune(password)) < 10 {
		return ErrWeakPassword
	}
	hash, err := hashPassword(password)
	if err != nil {
		return err
	}
	res, err := s.w.ExecContext(ctx,
		`UPDATE admin_users SET password_hash = ? WHERE email = ?`, hash, email)
	if err != nil {
		return err
	}
	if n, _ := res.RowsAffected(); n == 0 {
		return ErrUserNotFound
	}
	_, err = s.w.ExecContext(ctx,
		`DELETE FROM admin_sessions WHERE user_id = (SELECT id FROM admin_users WHERE email = ?)`, email)
	return err
}

// VerifyAdminLogin checks a password and returns the account it belongs to.
//
// The KDF runs even when the email is unknown. Skipping it would make a
// non-existent account answer in microseconds and a real one in a hundred
// milliseconds, which is a reliable oracle for which addresses are registered.
func (s *Store) VerifyAdminLogin(ctx context.Context, email, password string) (AdminUser, error) {
	email = NormalizeEmail(email)
	var u AdminUser
	var hash string
	err := s.r.QueryRowContext(ctx,
		`SELECT id, email, password_hash, created_at, last_login FROM admin_users WHERE email = ?`,
		email).Scan(&u.ID, &u.Email, &hash, &u.CreatedAt, &u.LastLogin)
	if errors.Is(err, sql.ErrNoRows) {
		verifyPassword("pbkdf2-sha256$"+strconv.Itoa(pbkdf2Iterations)+
			"$AAAAAAAAAAAAAAAAAAAAAA$AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", password)
		return AdminUser{}, ErrBadCredential
	}
	if err != nil {
		return AdminUser{}, err
	}
	if !verifyPassword(hash, password) {
		return AdminUser{}, ErrBadCredential
	}
	now := time.Now().Unix()
	_, _ = s.w.ExecContext(ctx, `UPDATE admin_users SET last_login = ? WHERE id = ?`, now, u.ID)
	u.LastLogin = now
	return u, nil
}

func (s *Store) ListAdminUsers(ctx context.Context) ([]AdminUser, error) {
	rows, err := s.r.QueryContext(ctx,
		`SELECT id, email, created_at, last_login FROM admin_users ORDER BY created_at`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []AdminUser
	for rows.Next() {
		var u AdminUser
		if err := rows.Scan(&u.ID, &u.Email, &u.CreatedAt, &u.LastLogin); err != nil {
			return nil, err
		}
		out = append(out, u)
	}
	return out, rows.Err()
}

// DeleteAdminUser refuses to remove the last account, which would lock everyone
// out of the dashboard with no way back in short of editing the database.
func (s *Store) DeleteAdminUser(ctx context.Context, id int64) error {
	n, err := s.CountAdminUsers(ctx)
	if err != nil {
		return err
	}
	if n <= 1 {
		return errors.New("this is the only admin account; create another before removing it")
	}
	res, err := s.w.ExecContext(ctx, `DELETE FROM admin_users WHERE id = ?`, id)
	if err != nil {
		return err
	}
	if k, _ := res.RowsAffected(); k == 0 {
		return ErrUserNotFound
	}
	_, err = s.w.ExecContext(ctx, `DELETE FROM admin_sessions WHERE user_id = ?`, id)
	return err
}

// CreateSession mints a session and returns the cookie value exactly once.
// Only its hash is stored, so a stolen database yields no usable sessions.
func (s *Store) CreateSession(ctx context.Context, userID int64) (string, error) {
	buf := make([]byte, 32)
	if _, err := rand.Read(buf); err != nil {
		return "", err
	}
	token := base64.RawURLEncoding.EncodeToString(buf)
	now := time.Now()
	_, err := s.w.ExecContext(ctx,
		`INSERT INTO admin_sessions (token_hash, user_id, created_at, expires_at) VALUES (?,?,?,?)`,
		HashKey(token), userID, now.Unix(), now.Add(SessionTTL).Unix())
	if err != nil {
		return "", err
	}
	return token, nil
}

// LookupSession resolves a cookie to its account, or reports ErrUserNotFound
// for anything unknown or expired.
func (s *Store) LookupSession(ctx context.Context, token string) (AdminUser, error) {
	if token == "" {
		return AdminUser{}, ErrUserNotFound
	}
	var u AdminUser
	err := s.r.QueryRowContext(ctx, `
		SELECT u.id, u.email, u.created_at, u.last_login
		  FROM admin_sessions s JOIN admin_users u ON u.id = s.user_id
		 WHERE s.token_hash = ? AND s.expires_at > ?`,
		HashKey(token), time.Now().Unix()).Scan(&u.ID, &u.Email, &u.CreatedAt, &u.LastLogin)
	if errors.Is(err, sql.ErrNoRows) {
		return AdminUser{}, ErrUserNotFound
	}
	return u, err
}

func (s *Store) DeleteSession(ctx context.Context, token string) error {
	_, err := s.w.ExecContext(ctx, `DELETE FROM admin_sessions WHERE token_hash = ?`, HashKey(token))
	return err
}

// PurgeExpiredSessions keeps the table from growing without bound.
func (s *Store) PurgeExpiredSessions(ctx context.Context) error {
	_, err := s.w.ExecContext(ctx, `DELETE FROM admin_sessions WHERE expires_at <= ?`, time.Now().Unix())
	return err
}
