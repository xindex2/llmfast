// Package store owns the SQLite database: API keys, the raw request log and
// the hourly/daily rollups behind the dashboard.
//
// Writes never happen on the request path. Handlers hand a Record to a buffered
// channel and return; a single writer goroutine drains it in batched
// transactions. If the channel is full we drop the record and count the drop --
// losing a stats row is always better than adding latency to a live stream.
package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"time"

	_ "modernc.org/sqlite"
)

type Store struct {
	// SQLite allows one writer and many concurrent readers under WAL, so the
	// pools are split rather than shared to keep readers off the write lock.
	w *sql.DB
	r *sql.DB

	ch      chan Record
	dropped atomic.Int64
	done    chan struct{}
}

// Record is one completed inference request.
type Record struct {
	TS               time.Time
	RequestID        string
	Model            string
	Backend          string
	APIKeyID         int64
	Status           int
	Streamed         bool
	PromptTokens     int
	CompletionTokens int
	ReasoningTokens  int
	CachedTokens     int
	// TTFTMs is time to first token; -1 for non-streaming requests where the
	// value is not meaningful.
	TTFTMs int64
	// TotalMs is the full wall clock, GenMs is first-token-to-last-token.
	TotalMs int64
	GenMs   int64
	TPS     float64
	CostUSD float64
	Error   string
}

const schema = `
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS api_keys (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  name        TEXT NOT NULL,
  key_hash    TEXT NOT NULL UNIQUE,
  key_prefix  TEXT NOT NULL,
  created_at  INTEGER NOT NULL,
  disabled    INTEGER NOT NULL DEFAULT 0,
  rpm_limit   INTEGER NOT NULL DEFAULT 0,
  -- Which account owns this key. 0 is the service's own key, issued before
  -- customer accounts existed and visible only to admins.
  user_id     INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_keys_user ON api_keys(user_id);

-- Dashboard accounts. Separate from api_keys: those are machine credentials
-- and are stored as a plain hash of a random secret, whereas these are chosen
-- by a person and need a slow KDF.
CREATE TABLE IF NOT EXISTS admin_users (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  email         TEXT NOT NULL UNIQUE,
  password_hash TEXT NOT NULL,
  created_at    INTEGER NOT NULL,
  last_login    INTEGER NOT NULL DEFAULT 0,
  -- "admin" runs the service; "user" is a customer who buys tokens. The table
  -- is shared because the credential handling is identical and duplicating it
  -- would mean two places to get password storage wrong.
  role          TEXT NOT NULL DEFAULT 'admin',
  disabled      INTEGER NOT NULL DEFAULT 0
);

-- Only the hash of a session token is kept, so the database is not a source of
-- live sessions if it leaks.
CREATE TABLE IF NOT EXISTS admin_sessions (
  token_hash TEXT PRIMARY KEY,
  user_id    INTEGER NOT NULL,
  created_at INTEGER NOT NULL,
  expires_at INTEGER NOT NULL,
  FOREIGN KEY(user_id) REFERENCES admin_users(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_sessions_expiry ON admin_sessions(expires_at);

CREATE TABLE IF NOT EXISTS requests (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  ts                INTEGER NOT NULL,
  request_id        TEXT NOT NULL,
  model             TEXT NOT NULL,
  backend           TEXT NOT NULL DEFAULT '',
  api_key_id        INTEGER NOT NULL DEFAULT 0,
  status            INTEGER NOT NULL,
  streamed          INTEGER NOT NULL DEFAULT 0,
  prompt_tokens     INTEGER NOT NULL DEFAULT 0,
  completion_tokens INTEGER NOT NULL DEFAULT 0,
  reasoning_tokens  INTEGER NOT NULL DEFAULT 0,
  cached_tokens     INTEGER NOT NULL DEFAULT 0,
  ttft_ms           INTEGER NOT NULL DEFAULT -1,
  total_ms          INTEGER NOT NULL DEFAULT 0,
  gen_ms            INTEGER NOT NULL DEFAULT 0,
  tps               REAL    NOT NULL DEFAULT 0,
  cost_usd          REAL    NOT NULL DEFAULT 0,
  error             TEXT    NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_requests_ts    ON requests(ts);
CREATE INDEX IF NOT EXISTS idx_requests_model ON requests(model, ts);

-- Rollups are keyed by (bucket, model) so a model breakdown needs no scan of
-- the raw log. bucket is the unix second at the start of the period.
CREATE TABLE IF NOT EXISTS stats_hourly (
  bucket            INTEGER NOT NULL,
  model             TEXT NOT NULL,
  requests          INTEGER NOT NULL DEFAULT 0,
  errors            INTEGER NOT NULL DEFAULT 0,
  rate_limited      INTEGER NOT NULL DEFAULT 0,
  user_errors       INTEGER NOT NULL DEFAULT 0,
  prompt_tokens     INTEGER NOT NULL DEFAULT 0,
  completion_tokens INTEGER NOT NULL DEFAULT 0,
  cached_tokens     INTEGER NOT NULL DEFAULT 0,
  ttft_p50          INTEGER NOT NULL DEFAULT 0,
  ttft_p95          INTEGER NOT NULL DEFAULT 0,
  ttft_p99          INTEGER NOT NULL DEFAULT 0,
  tps_avg           REAL    NOT NULL DEFAULT 0,
  cost_usd          REAL    NOT NULL DEFAULT 0,
  PRIMARY KEY (bucket, model)
);

CREATE TABLE IF NOT EXISTS stats_daily (
  bucket            INTEGER NOT NULL,
  model             TEXT NOT NULL,
  requests          INTEGER NOT NULL DEFAULT 0,
  errors            INTEGER NOT NULL DEFAULT 0,
  rate_limited      INTEGER NOT NULL DEFAULT 0,
  user_errors       INTEGER NOT NULL DEFAULT 0,
  prompt_tokens     INTEGER NOT NULL DEFAULT 0,
  completion_tokens INTEGER NOT NULL DEFAULT 0,
  cached_tokens     INTEGER NOT NULL DEFAULT 0,
  ttft_p50          INTEGER NOT NULL DEFAULT 0,
  ttft_p95          INTEGER NOT NULL DEFAULT 0,
  ttft_p99          INTEGER NOT NULL DEFAULT 0,
  tps_avg           REAL    NOT NULL DEFAULT 0,
  cost_usd          REAL    NOT NULL DEFAULT 0,
  PRIMARY KEY (bucket, model)
);
`

func Open(path string) (*Store, error) {
	// Create the parent directory rather than failing. SQLite reports a missing
	// directory as error 14, which the driver surfaces as "out of memory" -- a
	// genuinely misleading message to hit on a first deploy.
	if dir := filepath.Dir(path); dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0o750); err != nil {
			return nil, fmt.Errorf("create database directory %s: %w", dir, err)
		}
	}
	// _txlock=immediate avoids upgrade deadlocks when a write txn starts as a
	// reader; busy_timeout covers the rollup job overlapping the writer.
	dsn := path + "?_pragma=busy_timeout(5000)&_pragma=journal_mode(WAL)&_pragma=synchronous(NORMAL)&_txlock=immediate"
	w, err := sql.Open("sqlite", dsn)
	if err != nil {
		return nil, err
	}
	w.SetMaxOpenConns(1)
	if _, err := w.Exec(schema); err != nil {
		w.Close()
		return nil, fmt.Errorf("open database at %s: %w (check the path is writable)", path, err)
	}
	// CREATE TABLE IF NOT EXISTS does nothing to a table that already exists,
	// so columns added later need an explicit ALTER. Each is attempted once and
	// its "duplicate column" error ignored, which is the whole migration story
	// this schema needs.
	for _, alter := range []string{
		`ALTER TABLE admin_users ADD COLUMN role TEXT NOT NULL DEFAULT 'admin'`,
		`ALTER TABLE admin_users ADD COLUMN disabled INTEGER NOT NULL DEFAULT 0`,
		`ALTER TABLE api_keys ADD COLUMN user_id INTEGER NOT NULL DEFAULT 0`,
	} {
		if _, err := w.Exec(alter); err != nil && !strings.Contains(err.Error(), "duplicate column") {
			w.Close()
			return nil, fmt.Errorf("migrate database: %w", err)
		}
	}
	r, err := sql.Open("sqlite", dsn)
	if err != nil {
		w.Close()
		return nil, err
	}
	r.SetMaxOpenConns(8)

	s := &Store{w: w, r: r, ch: make(chan Record, 8192), done: make(chan struct{})}
	go s.writeLoop()
	return s, nil
}

func (s *Store) Close() error {
	close(s.ch)
	<-s.done
	return errors.Join(s.w.Close(), s.r.Close())
}

// Log queues a record. It never blocks.
func (s *Store) Log(rec Record) {
	select {
	case s.ch <- rec:
	default:
		s.dropped.Add(1)
	}
}

func (s *Store) Dropped() int64 { return s.dropped.Load() }

// DB exposes the read pool for tests and ad-hoc queries. Production code should
// use the typed methods on Store rather than reaching for raw SQL.
func (s *Store) DB() *sql.DB { return s.r }

// writeLoop batches queued records into transactions. It flushes when the batch
// fills or after a short idle tick, so a trickle of traffic is not stranded in
// memory.
func (s *Store) writeLoop() {
	defer close(s.done)
	const maxBatch = 256
	batch := make([]Record, 0, maxBatch)
	tick := time.NewTicker(time.Second)
	defer tick.Stop()

	flush := func() {
		if len(batch) == 0 {
			return
		}
		if err := s.insertBatch(batch); err != nil {
			// Stats are best effort; a failed flush must not take down the gateway.
			fmt.Printf("store: insert batch failed: %v\n", err)
		}
		batch = batch[:0]
	}
	for {
		select {
		case rec, ok := <-s.ch:
			if !ok {
				flush()
				return
			}
			batch = append(batch, rec)
			if len(batch) >= maxBatch {
				flush()
			}
		case <-tick.C:
			flush()
		}
	}
}

func (s *Store) insertBatch(batch []Record) error {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	tx, err := s.w.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()
	stmt, err := tx.PrepareContext(ctx, `INSERT INTO requests
		(ts, request_id, model, backend, api_key_id, status, streamed,
		 prompt_tokens, completion_tokens, reasoning_tokens, cached_tokens,
		 ttft_ms, total_ms, gen_ms, tps, cost_usd, error)
		VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`)
	if err != nil {
		return err
	}
	defer stmt.Close()
	for _, r := range batch {
		if _, err := stmt.ExecContext(ctx,
			r.TS.UnixMilli(), r.RequestID, r.Model, r.Backend, r.APIKeyID, r.Status, b2i(r.Streamed),
			r.PromptTokens, r.CompletionTokens, r.ReasoningTokens, r.CachedTokens,
			r.TTFTMs, r.TotalMs, r.GenMs, r.TPS, r.CostUSD, r.Error); err != nil {
			return err
		}
	}
	return tx.Commit()
}

func b2i(b bool) int {
	if b {
		return 1
	}
	return 0
}
