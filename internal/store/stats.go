package store

import (
	"context"
	"database/sql"
	"fmt"
	"strings"
	"time"
)

// Bucket sizes for rollups and for the dashboard's time series.
const (
	Hour = 3600
	Day  = 86400
)

// errorClass mirrors OpenRouter's uptime accounting. Their formula is
// successful / total, with user-caused failures excluded from the denominator
// entirely, so we bucket statuses the same way rather than lumping all non-2xx
// together.
//
//	uptime-affecting: 401, 402, 404, 5xx, mid-stream failures
//	user errors:      400, 413, 403 (excluded from the denominator)
//	tracked apart:    429
const (
	sqlIsRateLimited = `status = 429`
	sqlIsUserError   = `status IN (400, 403, 413)`
	sqlIsError       = `(status >= 500 OR status IN (401, 402, 404) OR (error <> '' AND status < 400))`
)

// Rollup recomputes hourly and daily aggregates covering `since`. It is
// idempotent: buckets are replaced wholesale, so re-running it over an
// overlapping window cannot double count.
func (s *Store) Rollup(ctx context.Context, since time.Time) error {
	if err := s.rollupInto(ctx, "stats_hourly", Hour, since); err != nil {
		return fmt.Errorf("hourly: %w", err)
	}
	if err := s.rollupInto(ctx, "stats_daily", Day, since); err != nil {
		return fmt.Errorf("daily: %w", err)
	}
	return nil
}

func (s *Store) rollupInto(ctx context.Context, table string, size int64, since time.Time) error {
	// Percentiles come from a ranked window rather than an extension function,
	// so this stays portable to a plain SQLite build with no loadable modules.
	q := fmt.Sprintf(`
INSERT OR REPLACE INTO %s
  (bucket, model, requests, errors, rate_limited, user_errors,
   prompt_tokens, completion_tokens, cached_tokens,
   ttft_p50, ttft_p95, ttft_p99, tps_avg, cost_usd)
WITH src AS (
  SELECT (ts/1000/%d)*%d AS bucket, model, ttft_ms, tps, status, error,
         prompt_tokens, completion_tokens, cached_tokens, cost_usd
  FROM requests WHERE ts >= ?
),
agg AS (
  SELECT bucket, model,
    COUNT(*)                                        AS requests,
    SUM(CASE WHEN %s THEN 1 ELSE 0 END)             AS errors,
    SUM(CASE WHEN %s THEN 1 ELSE 0 END)             AS rate_limited,
    SUM(CASE WHEN %s THEN 1 ELSE 0 END)             AS user_errors,
    SUM(prompt_tokens)                              AS prompt_tokens,
    SUM(completion_tokens)                          AS completion_tokens,
    SUM(cached_tokens)                              AS cached_tokens,
    COALESCE(AVG(CASE WHEN tps > 0 THEN tps END),0) AS tps_avg,
    SUM(cost_usd)                                   AS cost_usd
  FROM src GROUP BY bucket, model
),
ranked AS (
  SELECT bucket, model, ttft_ms,
    ROW_NUMBER() OVER (PARTITION BY bucket, model ORDER BY ttft_ms) AS rn,
    COUNT(*)     OVER (PARTITION BY bucket, model)                  AS n
  FROM src WHERE ttft_ms >= 0
),
pct AS (
  SELECT bucket, model,
    COALESCE(MAX(CASE WHEN rn = MAX(1, CAST(n*0.50 AS INTEGER)) THEN ttft_ms END), 0) AS p50,
    COALESCE(MAX(CASE WHEN rn = MAX(1, CAST(n*0.95 AS INTEGER)) THEN ttft_ms END), 0) AS p95,
    COALESCE(MAX(CASE WHEN rn = MAX(1, CAST(n*0.99 AS INTEGER)) THEN ttft_ms END), 0) AS p99
  FROM ranked GROUP BY bucket, model
)
SELECT a.bucket, a.model, a.requests, a.errors, a.rate_limited, a.user_errors,
       a.prompt_tokens, a.completion_tokens, a.cached_tokens,
       COALESCE(p.p50,0), COALESCE(p.p95,0), COALESCE(p.p99,0),
       a.tps_avg, a.cost_usd
FROM agg a LEFT JOIN pct p ON p.bucket = a.bucket AND p.model = a.model`,
		table, size, size, sqlIsError, sqlIsRateLimited, sqlIsUserError)

	_, err := s.w.ExecContext(ctx, q, since.UnixMilli())
	return err
}

// Purge drops raw request rows past the retention window. Rollups are kept.
func (s *Store) Purge(ctx context.Context, olderThan time.Time) (int64, error) {
	res, err := s.w.ExecContext(ctx, `DELETE FROM requests WHERE ts < ?`, olderThan.UnixMilli())
	if err != nil {
		return 0, err
	}
	return res.RowsAffected()
}

// Point is one bucket of the dashboard time series.
type Point struct {
	Bucket           int64   `json:"bucket"`
	Requests         int64   `json:"requests"`
	Errors           int64   `json:"errors"`
	RateLimited      int64   `json:"rate_limited"`
	UserErrors       int64   `json:"user_errors"`
	PromptTokens     int64   `json:"prompt_tokens"`
	CompletionTokens int64   `json:"completion_tokens"`
	CachedTokens     int64   `json:"cached_tokens"`
	TTFTp50          int64   `json:"ttft_p50"`
	TTFTp95          int64   `json:"ttft_p95"`
	TTFTp99          int64   `json:"ttft_p99"`
	TPSAvg           float64 `json:"tps_avg"`
	CostUSD          float64 `json:"cost_usd"`
}

// Series returns buckets in [from, to). Percentiles are averaged across models
// when several are folded into one bucket; for a single-model view they are
// exact. Pass model == "" for all models.
func (s *Store) Series(ctx context.Context, table, model string, from, to time.Time) ([]Point, error) {
	var where strings.Builder
	args := []any{from.Unix(), to.Unix()}
	where.WriteString(`bucket >= ? AND bucket < ?`)
	if model != "" {
		where.WriteString(` AND model = ?`)
		args = append(args, model)
	}
	// Percentiles are weighted by request count so a bucket with one slow
	// request on a quiet model does not dominate a busy one.
	q := fmt.Sprintf(`
SELECT bucket,
  SUM(requests), SUM(errors), SUM(rate_limited), SUM(user_errors),
  SUM(prompt_tokens), SUM(completion_tokens), SUM(cached_tokens),
  CAST(COALESCE(SUM(ttft_p50*requests)/NULLIF(SUM(requests),0),0) AS INTEGER),
  CAST(COALESCE(SUM(ttft_p95*requests)/NULLIF(SUM(requests),0),0) AS INTEGER),
  CAST(COALESCE(SUM(ttft_p99*requests)/NULLIF(SUM(requests),0),0) AS INTEGER),
  COALESCE(SUM(tps_avg*requests)/NULLIF(SUM(requests),0),0),
  SUM(cost_usd)
FROM %s WHERE %s GROUP BY bucket ORDER BY bucket`, table, where.String())

	rows, err := s.r.QueryContext(ctx, q, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Point
	for rows.Next() {
		var p Point
		if err := rows.Scan(&p.Bucket, &p.Requests, &p.Errors, &p.RateLimited, &p.UserErrors,
			&p.PromptTokens, &p.CompletionTokens, &p.CachedTokens,
			&p.TTFTp50, &p.TTFTp95, &p.TTFTp99, &p.TPSAvg, &p.CostUSD); err != nil {
			return nil, err
		}
		out = append(out, p)
	}
	return out, rows.Err()
}

// ModelStat is the per-model breakdown table.
type ModelStat struct {
	Model string `json:"model"`
	Point
}

func (s *Store) ByModel(ctx context.Context, table string, from, to time.Time) ([]ModelStat, error) {
	q := fmt.Sprintf(`
SELECT model,
  SUM(requests), SUM(errors), SUM(rate_limited), SUM(user_errors),
  SUM(prompt_tokens), SUM(completion_tokens), SUM(cached_tokens),
  CAST(COALESCE(SUM(ttft_p50*requests)/NULLIF(SUM(requests),0),0) AS INTEGER),
  CAST(COALESCE(SUM(ttft_p95*requests)/NULLIF(SUM(requests),0),0) AS INTEGER),
  CAST(COALESCE(SUM(ttft_p99*requests)/NULLIF(SUM(requests),0),0) AS INTEGER),
  COALESCE(SUM(tps_avg*requests)/NULLIF(SUM(requests),0),0),
  SUM(cost_usd)
FROM %s WHERE bucket >= ? AND bucket < ?
GROUP BY model ORDER BY SUM(requests) DESC`, table)

	rows, err := s.r.QueryContext(ctx, q, from.Unix(), to.Unix())
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []ModelStat
	for rows.Next() {
		var m ModelStat
		if err := rows.Scan(&m.Model, &m.Requests, &m.Errors, &m.RateLimited, &m.UserErrors,
			&m.PromptTokens, &m.CompletionTokens, &m.CachedTokens,
			&m.TTFTp50, &m.TTFTp95, &m.TTFTp99, &m.TPSAvg, &m.CostUSD); err != nil {
			return nil, err
		}
		out = append(out, m)
	}
	return out, rows.Err()
}

// RecentRequest is a row of the live request log.
type RecentRequest struct {
	TS        int64   `json:"ts"`
	RequestID string  `json:"request_id"`
	Model     string  `json:"model"`
	Backend   string  `json:"backend"`
	Status    int     `json:"status"`
	Streamed  bool    `json:"streamed"`
	PromptTok int     `json:"prompt_tokens"`
	CompTok   int     `json:"completion_tokens"`
	TTFTMs    int64   `json:"ttft_ms"`
	TotalMs   int64   `json:"total_ms"`
	TPS       float64 `json:"tps"`
	CostUSD   float64 `json:"cost_usd"`
	Error     string  `json:"error"`
}

func (s *Store) Recent(ctx context.Context, limit int, model string) ([]RecentRequest, error) {
	if limit <= 0 || limit > 500 {
		limit = 100
	}
	q := `SELECT ts, request_id, model, backend, status, streamed,
	             prompt_tokens, completion_tokens, ttft_ms, total_ms, tps, cost_usd, error
	      FROM requests`
	args := []any{}
	if model != "" {
		q += ` WHERE model = ?`
		args = append(args, model)
	}
	q += ` ORDER BY id DESC LIMIT ?`
	args = append(args, limit)

	rows, err := s.r.QueryContext(ctx, q, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []RecentRequest
	for rows.Next() {
		var r RecentRequest
		var streamed int
		if err := rows.Scan(&r.TS, &r.RequestID, &r.Model, &r.Backend, &r.Status, &streamed,
			&r.PromptTok, &r.CompTok, &r.TTFTMs, &r.TotalMs, &r.TPS, &r.CostUSD, &r.Error); err != nil {
			return nil, err
		}
		r.Streamed = streamed == 1
		out = append(out, r)
	}
	return out, rows.Err()
}

// Totals is the headline summary for a window, read from the raw log so the
// numbers include the current partial hour that rollups have not covered yet.
type Totals struct {
	Requests    int64   `json:"requests"`
	Errors      int64   `json:"errors"`
	RateLimited int64   `json:"rate_limited"`
	UserErrors  int64   `json:"user_errors"`
	PromptTok   int64   `json:"prompt_tokens"`
	CompTok     int64   `json:"completion_tokens"`
	CachedTok   int64   `json:"cached_tokens"`
	TTFTp50     int64   `json:"ttft_p50"`
	TTFTp95     int64   `json:"ttft_p95"`
	TTFTp99     int64   `json:"ttft_p99"`
	TPSAvg      float64 `json:"tps_avg"`
	CostUSD     float64 `json:"cost_usd"`
	// Uptime is OpenRouter's formula: successes over requests that are not
	// user errors. -1 means not enough data (they require 100+ requests).
	Uptime float64 `json:"uptime"`
}

func (s *Store) Totals(ctx context.Context, from, to time.Time) (Totals, error) {
	var t Totals
	q := fmt.Sprintf(`SELECT
	  COUNT(*),
	  SUM(CASE WHEN %s THEN 1 ELSE 0 END),
	  SUM(CASE WHEN %s THEN 1 ELSE 0 END),
	  SUM(CASE WHEN %s THEN 1 ELSE 0 END),
	  COALESCE(SUM(prompt_tokens),0), COALESCE(SUM(completion_tokens),0),
	  COALESCE(SUM(cached_tokens),0),
	  COALESCE(AVG(CASE WHEN tps > 0 THEN tps END),0),
	  COALESCE(SUM(cost_usd),0)
	FROM requests WHERE ts >= ? AND ts < ?`, sqlIsError, sqlIsRateLimited, sqlIsUserError)

	var errs, rl, ue sql.NullInt64
	err := s.r.QueryRowContext(ctx, q, from.UnixMilli(), to.UnixMilli()).Scan(
		&t.Requests, &errs, &rl, &ue, &t.PromptTok, &t.CompTok, &t.CachedTok, &t.TPSAvg, &t.CostUSD)
	if err != nil {
		return t, err
	}
	t.Errors, t.RateLimited, t.UserErrors = errs.Int64, rl.Int64, ue.Int64

	p, err := s.ttftPercentiles(ctx, from, to)
	if err != nil {
		return t, err
	}
	t.TTFTp50, t.TTFTp95, t.TTFTp99 = p[0], p[1], p[2]

	// Below OpenRouter's 100-request floor the ratio is noise, so report it as
	// unknown rather than showing a scary number off three failed calls.
	denom := t.Requests - t.UserErrors
	switch {
	case t.Requests < 100:
		t.Uptime = -1
	case denom <= 0:
		t.Uptime = 1
	default:
		t.Uptime = float64(denom-t.Errors) / float64(denom)
	}
	return t, nil
}

func (s *Store) ttftPercentiles(ctx context.Context, from, to time.Time) ([3]int64, error) {
	var out [3]int64
	rows, err := s.r.QueryContext(ctx, `
WITH ranked AS (
  SELECT ttft_ms,
    ROW_NUMBER() OVER (ORDER BY ttft_ms) AS rn,
    COUNT(*)     OVER ()                 AS n
  FROM requests WHERE ts >= ? AND ts < ? AND ttft_ms >= 0
)
SELECT
  COALESCE(MAX(CASE WHEN rn = MAX(1, CAST(n*0.50 AS INTEGER)) THEN ttft_ms END),0),
  COALESCE(MAX(CASE WHEN rn = MAX(1, CAST(n*0.95 AS INTEGER)) THEN ttft_ms END),0),
  COALESCE(MAX(CASE WHEN rn = MAX(1, CAST(n*0.99 AS INTEGER)) THEN ttft_ms END),0)
FROM ranked`, from.UnixMilli(), to.UnixMilli())
	if err != nil {
		return out, err
	}
	defer rows.Close()
	if rows.Next() {
		if err := rows.Scan(&out[0], &out[1], &out[2]); err != nil {
			return out, err
		}
	}
	return out, rows.Err()
}

// SeriesRaw buckets the raw request log at an arbitrary interval. Rollups only
// exist at hour and day granularity, so short dashboard ranges read straight
// from the log: it keeps minute-level resolution available and means the
// current, not-yet-rolled-up period is always included.
func (s *Store) SeriesRaw(ctx context.Context, bucketSec int64, model string, from, to time.Time) ([]Point, error) {
	if bucketSec <= 0 {
		bucketSec = Hour
	}
	args := []any{from.UnixMilli(), to.UnixMilli()}
	filter := `ts >= ? AND ts < ?`
	if model != "" {
		filter += ` AND model = ?`
		args = append(args, model)
	}
	q := fmt.Sprintf(`
WITH src AS (
  SELECT (ts/1000/%d)*%d AS bucket, ttft_ms, tps, status, error,
         prompt_tokens, completion_tokens, cached_tokens, cost_usd
  FROM requests WHERE %s
),
agg AS (
  SELECT bucket,
    COUNT(*)                                        AS requests,
    SUM(CASE WHEN %s THEN 1 ELSE 0 END)             AS errors,
    SUM(CASE WHEN %s THEN 1 ELSE 0 END)             AS rate_limited,
    SUM(CASE WHEN %s THEN 1 ELSE 0 END)             AS user_errors,
    SUM(prompt_tokens)                              AS prompt_tokens,
    SUM(completion_tokens)                          AS completion_tokens,
    SUM(cached_tokens)                              AS cached_tokens,
    COALESCE(AVG(CASE WHEN tps > 0 THEN tps END),0) AS tps_avg,
    SUM(cost_usd)                                   AS cost_usd
  FROM src GROUP BY bucket
),
ranked AS (
  SELECT bucket, ttft_ms,
    ROW_NUMBER() OVER (PARTITION BY bucket ORDER BY ttft_ms) AS rn,
    COUNT(*)     OVER (PARTITION BY bucket)                  AS n
  FROM src WHERE ttft_ms >= 0
),
pct AS (
  SELECT bucket,
    COALESCE(MAX(CASE WHEN rn = MAX(1, CAST(n*0.50 AS INTEGER)) THEN ttft_ms END),0) AS p50,
    COALESCE(MAX(CASE WHEN rn = MAX(1, CAST(n*0.95 AS INTEGER)) THEN ttft_ms END),0) AS p95,
    COALESCE(MAX(CASE WHEN rn = MAX(1, CAST(n*0.99 AS INTEGER)) THEN ttft_ms END),0) AS p99
  FROM ranked GROUP BY bucket
)
SELECT a.bucket, a.requests, a.errors, a.rate_limited, a.user_errors,
       a.prompt_tokens, a.completion_tokens, a.cached_tokens,
       COALESCE(p.p50,0), COALESCE(p.p95,0), COALESCE(p.p99,0), a.tps_avg, a.cost_usd
FROM agg a LEFT JOIN pct p ON p.bucket = a.bucket ORDER BY a.bucket`,
		bucketSec, bucketSec, filter, sqlIsError, sqlIsRateLimited, sqlIsUserError)

	rows, err := s.r.QueryContext(ctx, q, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []Point{}
	for rows.Next() {
		var p Point
		if err := rows.Scan(&p.Bucket, &p.Requests, &p.Errors, &p.RateLimited, &p.UserErrors,
			&p.PromptTokens, &p.CompletionTokens, &p.CachedTokens,
			&p.TTFTp50, &p.TTFTp95, &p.TTFTp99, &p.TPSAvg, &p.CostUSD); err != nil {
			return nil, err
		}
		out = append(out, p)
	}
	return out, rows.Err()
}

// ByModelRaw is the per-model breakdown read from the raw log, used for the
// same short ranges that SeriesRaw serves.
func (s *Store) ByModelRaw(ctx context.Context, from, to time.Time) ([]ModelStat, error) {
	q := fmt.Sprintf(`
SELECT model,
  COUNT(*),
  SUM(CASE WHEN %s THEN 1 ELSE 0 END),
  SUM(CASE WHEN %s THEN 1 ELSE 0 END),
  SUM(CASE WHEN %s THEN 1 ELSE 0 END),
  COALESCE(SUM(prompt_tokens),0), COALESCE(SUM(completion_tokens),0),
  COALESCE(SUM(cached_tokens),0),
  CAST(COALESCE(AVG(CASE WHEN ttft_ms >= 0 THEN ttft_ms END),0) AS INTEGER), 0, 0,
  COALESCE(AVG(CASE WHEN tps > 0 THEN tps END),0),
  COALESCE(SUM(cost_usd),0)
FROM requests WHERE ts >= ? AND ts < ?
GROUP BY model ORDER BY COUNT(*) DESC`, sqlIsError, sqlIsRateLimited, sqlIsUserError)

	rows, err := s.r.QueryContext(ctx, q, from.UnixMilli(), to.UnixMilli())
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []ModelStat{}
	for rows.Next() {
		var m ModelStat
		if err := rows.Scan(&m.Model, &m.Requests, &m.Errors, &m.RateLimited, &m.UserErrors,
			&m.PromptTokens, &m.CompletionTokens, &m.CachedTokens,
			&m.TTFTp50, &m.TTFTp95, &m.TTFTp99, &m.TPSAvg, &m.CostUSD); err != nil {
			return nil, err
		}
		out = append(out, m)
	}
	return out, rows.Err()
}
