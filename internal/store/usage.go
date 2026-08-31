package store

import (
	"context"
	"database/sql"
	"fmt"
	"time"
)

// Usage is one account's consumption over a period.
type Usage struct {
	Requests  int64   `json:"requests"`
	Errors    int64   `json:"errors"`
	PromptTok int64   `json:"prompt_tokens"`
	CompTok   int64   `json:"completion_tokens"`
	CachedTok int64   `json:"cached_tokens"`
	CostUSD   float64 `json:"cost_usd"`
	TPSAvg    float64 `json:"tps_avg"`
}

// UsageByModel is what an account spent on one model.
type UsageByModel struct {
	Model     string  `json:"model"`
	Requests  int64   `json:"requests"`
	PromptTok int64   `json:"prompt_tokens"`
	CompTok   int64   `json:"completion_tokens"`
	CachedTok int64   `json:"cached_tokens"`
	CostUSD   float64 `json:"cost_usd"`
	TPSAvg    float64 `json:"tps_avg"`
	LastUsed  int64   `json:"last_used"`
}

// UsageFor totals one account's requests over a period.
//
// Requests record the API key they arrived on, not the account, so the join is
// through api_keys. Doing it in SQL rather than filtering in a handler means an
// account can only ever see rows its own keys produced.
func (s *Store) UsageFor(ctx context.Context, userID int64, from, to time.Time) (Usage, error) {
	var u Usage
	// sqlIsError is written against an unaliased "status" column, so the
	// condition is spelled out here rather than reused with an alias.
	q := fmt.Sprintf(`SELECT
	  COUNT(*),
	  SUM(CASE WHEN r.status >= 400 THEN 1 ELSE 0 END),
	  COALESCE(SUM(r.prompt_tokens),0), COALESCE(SUM(r.completion_tokens),0),
	  COALESCE(SUM(r.cached_tokens),0),
	  COALESCE(AVG(CASE WHEN r.tps > 0 THEN r.tps END),0),
	  COALESCE(SUM(r.cost_usd),0)
	FROM requests r JOIN api_keys k ON k.id = r.api_key_id
	WHERE k.user_id = ? AND r.ts >= ? AND r.ts < ?`)

	var errs sql.NullInt64
	err := s.r.QueryRowContext(ctx, q, userID, from.UnixMilli(), to.UnixMilli()).Scan(
		&u.Requests, &errs, &u.PromptTok, &u.CompTok, &u.CachedTok, &u.TPSAvg, &u.CostUSD)
	u.Errors = errs.Int64
	return u, err
}

// UsageByModelFor breaks one account's spend down by model, most used first.
func (s *Store) UsageByModelFor(ctx context.Context, userID int64, from, to time.Time) ([]UsageByModel, error) {
	rows, err := s.r.QueryContext(ctx, `
	  SELECT r.model, COUNT(*),
	         COALESCE(SUM(r.prompt_tokens),0), COALESCE(SUM(r.completion_tokens),0),
	         COALESCE(SUM(r.cached_tokens),0),
	         COALESCE(AVG(CASE WHEN r.tps > 0 THEN r.tps END),0),
	         COALESCE(SUM(r.cost_usd),0),
	         MAX(r.ts)
	    FROM requests r JOIN api_keys k ON k.id = r.api_key_id
	   WHERE k.user_id = ? AND r.ts >= ? AND r.ts < ?
	   GROUP BY r.model
	   ORDER BY COUNT(*) DESC`, userID, from.UnixMilli(), to.UnixMilli())
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []UsageByModel{}
	for rows.Next() {
		var m UsageByModel
		if err := rows.Scan(&m.Model, &m.Requests, &m.PromptTok, &m.CompTok,
			&m.CachedTok, &m.TPSAvg, &m.CostUSD, &m.LastUsed); err != nil {
			return nil, err
		}
		out = append(out, m)
	}
	return out, rows.Err()
}

// RecentFor is an account's own request log.
func (s *Store) RecentFor(ctx context.Context, userID int64, limit int) ([]RecentRequest, error) {
	if limit <= 0 || limit > 500 {
		limit = 100
	}
	rows, err := s.r.QueryContext(ctx, `
	  SELECT r.ts, r.request_id, r.model, r.status, r.streamed,
	         r.prompt_tokens, r.completion_tokens,
	         r.ttft_ms, r.total_ms, r.tps, r.cost_usd, r.error
	    FROM requests r JOIN api_keys k ON k.id = r.api_key_id
	   WHERE k.user_id = ?
	   ORDER BY r.ts DESC LIMIT ?`, userID, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []RecentRequest{}
	for rows.Next() {
		var q RecentRequest
		var streamed int
		var ts int64
		if err := rows.Scan(&ts, &q.RequestID, &q.Model, &q.Status, &streamed,
			&q.PromptTok, &q.CompTok,
			&q.TTFTMs, &q.TotalMs, &q.TPS, &q.CostUSD, &q.Error); err != nil {
			return nil, err
		}
		q.TS = ts
		q.Streamed = streamed == 1
		out = append(out, q)
	}
	return out, rows.Err()
}
