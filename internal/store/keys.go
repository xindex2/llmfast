package store

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"errors"
	"strings"
	"time"
)

// KeyPrefix is the visible, non-secret leader on every issued key. It lets us
// reject obviously-wrong credentials without touching the database, and makes
// leaked keys greppable in logs and repos.
const KeyPrefix = "sk-llmfast-"

var ErrKeyNotFound = errors.New("api key not found")

type APIKey struct {
	ID        int64  `json:"id"`
	Name      string `json:"name"`
	Prefix    string `json:"prefix"`
	CreatedAt int64  `json:"created_at"`
	Disabled  bool   `json:"disabled"`
	RPMLimit  int    `json:"rpm_limit"`
}

// HashKey is the one place the secret-to-hash mapping is defined. Plain SHA-256
// is deliberate: these are high-entropy random tokens, not user passwords, so a
// slow KDF would only add latency to every request without adding security.
func HashKey(secret string) string {
	sum := sha256.Sum256([]byte(secret))
	return hex.EncodeToString(sum[:])
}

// CreateKey mints a key and returns the secret exactly once. Only the hash is
// persisted, so a lost key must be replaced rather than recovered.
func (s *Store) CreateKey(ctx context.Context, name string, rpmLimit int) (APIKey, string, error) {
	buf := make([]byte, 24)
	if _, err := rand.Read(buf); err != nil {
		return APIKey{}, "", err
	}
	secret := KeyPrefix + base64.RawURLEncoding.EncodeToString(buf)
	prefix := secret[:len(KeyPrefix)+6]
	now := time.Now().Unix()

	res, err := s.w.ExecContext(ctx,
		`INSERT INTO api_keys (name, key_hash, key_prefix, created_at, rpm_limit) VALUES (?,?,?,?,?)`,
		name, HashKey(secret), prefix, now, rpmLimit)
	if err != nil {
		return APIKey{}, "", err
	}
	id, _ := res.LastInsertId()
	return APIKey{ID: id, Name: name, Prefix: prefix, CreatedAt: now, RPMLimit: rpmLimit}, secret, nil
}

// LookupKey resolves a presented secret. It returns ErrKeyNotFound for both
// unknown and disabled keys so callers cannot distinguish the two.
func (s *Store) LookupKey(ctx context.Context, secret string) (APIKey, error) {
	var k APIKey
	var disabled int
	err := s.r.QueryRowContext(ctx,
		`SELECT id, name, key_prefix, created_at, disabled, rpm_limit FROM api_keys WHERE key_hash = ?`,
		HashKey(secret)).Scan(&k.ID, &k.Name, &k.Prefix, &k.CreatedAt, &disabled, &k.RPMLimit)
	if err != nil {
		return APIKey{}, ErrKeyNotFound
	}
	if disabled == 1 {
		return APIKey{}, ErrKeyNotFound
	}
	return k, nil
}

func (s *Store) ListKeys(ctx context.Context) ([]APIKey, error) {
	rows, err := s.r.QueryContext(ctx,
		`SELECT id, name, key_prefix, created_at, disabled, rpm_limit FROM api_keys ORDER BY id DESC`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []APIKey{}
	for rows.Next() {
		var k APIKey
		var disabled int
		if err := rows.Scan(&k.ID, &k.Name, &k.Prefix, &k.CreatedAt, &disabled, &k.RPMLimit); err != nil {
			return nil, err
		}
		k.Disabled = disabled == 1
		out = append(out, k)
	}
	return out, rows.Err()
}

func (s *Store) SetKeyDisabled(ctx context.Context, id int64, disabled bool) error {
	_, err := s.w.ExecContext(ctx, `UPDATE api_keys SET disabled = ? WHERE id = ?`, b2i(disabled), id)
	return err
}

func (s *Store) DeleteKey(ctx context.Context, id int64) error {
	_, err := s.w.ExecContext(ctx, `DELETE FROM api_keys WHERE id = ?`, id)
	return err
}

// CountKeys reports how many keys exist, used at boot to decide whether to mint
// a bootstrap key.
func (s *Store) CountKeys(ctx context.Context) (int, error) {
	var n int
	err := s.r.QueryRowContext(ctx, `SELECT COUNT(*) FROM api_keys`).Scan(&n)
	return n, err
}

// ExtractBearer pulls the credential out of an Authorization header.
func ExtractBearer(header string) string {
	const p = "Bearer "
	if len(header) > len(p) && strings.EqualFold(header[:len(p)], p) {
		return strings.TrimSpace(header[len(p):])
	}
	return ""
}
