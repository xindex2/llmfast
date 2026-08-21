package store

import (
	"context"
	"path/filepath"
	"testing"
	"time"
)

func openTest(t *testing.T) *Store {
	t.Helper()
	s, err := Open(filepath.Join(t.TempDir(), "test.db"))
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	t.Cleanup(func() { s.Close() })
	return s
}

// flush forces the async write loop to drain. The loop batches on a one second
// tick, so tests wait for the row count to settle rather than sleeping blindly.
func flush(t *testing.T, s *Store, want int) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		var n int
		if err := s.r.QueryRow(`SELECT COUNT(*) FROM requests`).Scan(&n); err == nil && n >= want {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %d rows to flush", want)
}

func TestLogAndTotals(t *testing.T) {
	s := openTest(t)
	base := time.Now().Add(-30 * time.Minute)

	// Ten successes with known TTFTs so percentiles are predictable.
	for i := 0; i < 10; i++ {
		s.Log(Record{
			TS: base, Model: "m1", Status: 200, PromptTokens: 100, CompletionTokens: 50,
			CachedTokens: 40, TTFTMs: int64((i + 1) * 100), TotalMs: 1000, TPS: 50, CostUSD: 0.001,
		})
	}
	flush(t, s, 10)

	got, err := s.Totals(context.Background(), base.Add(-time.Minute), time.Now())
	if err != nil {
		t.Fatalf("totals: %v", err)
	}
	if got.Requests != 10 {
		t.Errorf("requests = %d, want 10", got.Requests)
	}
	if got.PromptTok != 1000 || got.CompTok != 500 || got.CachedTok != 400 {
		t.Errorf("tokens = %d/%d/%d, want 1000/500/400", got.PromptTok, got.CompTok, got.CachedTok)
	}
	// With values 100..1000, the row at index floor(10*0.5)=5 is 500.
	if got.TTFTp50 != 500 {
		t.Errorf("ttft p50 = %d, want 500", got.TTFTp50)
	}
	if got.TTFTp95 != 900 {
		t.Errorf("ttft p95 = %d, want 900", got.TTFTp95)
	}
	// Below the 100-request floor uptime is reported as unknown.
	if got.Uptime != -1 {
		t.Errorf("uptime = %v, want -1 (insufficient data)", got.Uptime)
	}
}

// TestErrorClassification pins the status buckets to OpenRouter's uptime rules:
// user errors leave the denominator, 429 is tracked apart, and only real
// failures count against us.
func TestErrorClassification(t *testing.T) {
	s := openTest(t)
	now := time.Now().Add(-time.Minute)

	cases := []struct {
		status                      int
		wantErr, wantUser, wantRate bool
	}{
		{200, false, false, false},
		{400, false, true, false}, // user error
		{403, false, true, false}, // geo restriction, tracked apart from uptime
		{413, false, true, false}, // oversized payload
		{429, false, false, true}, // rate limit
		{401, true, false, false}, // auth
		{402, true, false, false}, // payment
		{404, true, false, false}, // model missing
		{500, true, false, false}, // server error
		{503, true, false, false},
		{499, false, false, false}, // client disconnect: nobody's error
	}
	for _, c := range cases {
		s.Log(Record{TS: now, Model: "m", Status: c.status, TTFTMs: -1})
	}
	flush(t, s, len(cases))

	got, err := s.Totals(context.Background(), now.Add(-time.Minute), time.Now())
	if err != nil {
		t.Fatalf("totals: %v", err)
	}
	var wantErrs, wantUsers, wantRates int64
	for _, c := range cases {
		if c.wantErr {
			wantErrs++
		}
		if c.wantUser {
			wantUsers++
		}
		if c.wantRate {
			wantRates++
		}
	}
	if got.Errors != wantErrs {
		t.Errorf("errors = %d, want %d", got.Errors, wantErrs)
	}
	if got.UserErrors != wantUsers {
		t.Errorf("user errors = %d, want %d", got.UserErrors, wantUsers)
	}
	if got.RateLimited != wantRates {
		t.Errorf("rate limited = %d, want %d", got.RateLimited, wantRates)
	}
}

// TestMidStreamErrorCounts covers the case that a 200 status line does not mean
// success: a stream that breaks after headers must still count against uptime.
func TestMidStreamErrorCounts(t *testing.T) {
	s := openTest(t)
	now := time.Now().Add(-time.Minute)
	s.Log(Record{TS: now, Model: "m", Status: 200, TTFTMs: 100, Error: "unexpected EOF"})
	s.Log(Record{TS: now, Model: "m", Status: 200, TTFTMs: 100})
	flush(t, s, 2)

	got, _ := s.Totals(context.Background(), now.Add(-time.Minute), time.Now())
	if got.Errors != 1 {
		t.Errorf("errors = %d, want 1 (the mid-stream failure)", got.Errors)
	}
}

func TestUptimeFormula(t *testing.T) {
	s := openTest(t)
	now := time.Now().Add(-time.Minute)
	// 100 successes, 10 real errors, 20 user errors.
	// Uptime = (130 - 20 - 10) / (130 - 20) = 100/110.
	for i := 0; i < 100; i++ {
		s.Log(Record{TS: now, Model: "m", Status: 200, TTFTMs: 10})
	}
	for i := 0; i < 10; i++ {
		s.Log(Record{TS: now, Model: "m", Status: 500, TTFTMs: -1})
	}
	for i := 0; i < 20; i++ {
		s.Log(Record{TS: now, Model: "m", Status: 400, TTFTMs: -1})
	}
	flush(t, s, 130)

	got, _ := s.Totals(context.Background(), now.Add(-time.Minute), time.Now())
	want := 100.0 / 110.0
	if diff := got.Uptime - want; diff > 1e-9 || diff < -1e-9 {
		t.Errorf("uptime = %v, want %v", got.Uptime, want)
	}
}

func TestRollupHourly(t *testing.T) {
	s := openTest(t)
	// Two distinct hours, so bucketing is actually exercised.
	h1 := time.Now().Add(-3 * time.Hour).Truncate(time.Hour).Add(10 * time.Minute)
	h2 := h1.Add(time.Hour)

	for i := 0; i < 4; i++ {
		s.Log(Record{TS: h1, Model: "m1", Status: 200, PromptTokens: 10, CompletionTokens: 5,
			TTFTMs: int64(100 * (i + 1)), TPS: 40, CostUSD: 0.01})
	}
	for i := 0; i < 2; i++ {
		s.Log(Record{TS: h2, Model: "m1", Status: 500, TTFTMs: -1})
	}
	s.Log(Record{TS: h2, Model: "m2", Status: 200, PromptTokens: 7, CompletionTokens: 3, TTFTMs: 250, TPS: 20})
	flush(t, s, 7)

	ctx := context.Background()
	if err := s.Rollup(ctx, h1.Add(-time.Hour)); err != nil {
		t.Fatalf("rollup: %v", err)
	}

	pts, err := s.Series(ctx, "stats_hourly", "", h1.Add(-time.Hour), time.Now())
	if err != nil {
		t.Fatalf("series: %v", err)
	}
	if len(pts) != 2 {
		t.Fatalf("got %d buckets, want 2", len(pts))
	}
	if pts[0].Requests != 4 || pts[0].PromptTokens != 40 || pts[0].CompletionTokens != 20 {
		t.Errorf("bucket 1 = %+v, want 4 requests / 40 prompt / 20 completion", pts[0])
	}
	// TTFTs 100,200,300,400: index floor(4*0.5)=2 is 200.
	if pts[0].TTFTp50 != 200 {
		t.Errorf("bucket 1 p50 = %d, want 200", pts[0].TTFTp50)
	}
	if pts[1].Requests != 3 || pts[1].Errors != 2 {
		t.Errorf("bucket 2 = %+v, want 3 requests / 2 errors", pts[1])
	}

	// Re-running over an overlapping range must not double count.
	if err := s.Rollup(ctx, h1.Add(-time.Hour)); err != nil {
		t.Fatalf("second rollup: %v", err)
	}
	pts2, _ := s.Series(ctx, "stats_hourly", "", h1.Add(-time.Hour), time.Now())
	if len(pts2) != 2 || pts2[0].Requests != 4 {
		t.Errorf("rollup is not idempotent: %+v", pts2)
	}

	// Per-model filtering.
	only, _ := s.Series(ctx, "stats_hourly", "m2", h1.Add(-time.Hour), time.Now())
	if len(only) != 1 || only[0].Requests != 1 {
		t.Errorf("model filter = %+v, want a single bucket with 1 request", only)
	}
}

func TestSeriesRawBucketing(t *testing.T) {
	s := openTest(t)
	base := time.Now().Add(-10 * time.Minute).Truncate(time.Minute)
	// Three requests in one minute, one in the next.
	for i := 0; i < 3; i++ {
		s.Log(Record{TS: base, Model: "m", Status: 200, CompletionTokens: 5, TTFTMs: 100})
	}
	s.Log(Record{TS: base.Add(time.Minute), Model: "m", Status: 200, CompletionTokens: 5, TTFTMs: 100})
	flush(t, s, 4)

	pts, err := s.SeriesRaw(context.Background(), 60, "", base.Add(-time.Minute), time.Now())
	if err != nil {
		t.Fatalf("series raw: %v", err)
	}
	var total int64
	var nonEmpty int
	for _, p := range pts {
		total += p.Requests
		if p.Requests > 0 {
			nonEmpty++
		}
	}
	if total != 4 {
		t.Errorf("total requests = %d, want 4", total)
	}
	if nonEmpty != 2 {
		t.Errorf("non-empty buckets = %d, want 2", nonEmpty)
	}
}

func TestPurgeKeepsRollups(t *testing.T) {
	s := openTest(t)
	old := time.Now().AddDate(0, 0, -40)
	s.Log(Record{TS: old, Model: "m", Status: 200, CompletionTokens: 5, TTFTMs: 100})
	s.Log(Record{TS: time.Now(), Model: "m", Status: 200, CompletionTokens: 5, TTFTMs: 100})
	flush(t, s, 2)

	ctx := context.Background()
	cutoff := time.Now().AddDate(0, 0, -30)
	if err := s.Rollup(ctx, old.Add(-time.Hour)); err != nil {
		t.Fatalf("rollup: %v", err)
	}
	n, err := s.Purge(ctx, cutoff)
	if err != nil {
		t.Fatalf("purge: %v", err)
	}
	if n != 1 {
		t.Errorf("purged %d rows, want 1", n)
	}
	// The aggregate for the purged day must survive.
	pts, _ := s.Series(ctx, "stats_daily", "", old.Add(-24*time.Hour), time.Now())
	if len(pts) == 0 {
		t.Error("rollups were lost when raw rows were purged")
	}
}

func TestAPIKeyLifecycle(t *testing.T) {
	s := openTest(t)
	ctx := context.Background()

	key, secret, err := s.CreateKey(ctx, "openrouter", 600)
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	if len(secret) < len(KeyPrefix)+16 {
		t.Errorf("secret looks too short: %q", secret)
	}
	got, err := s.LookupKey(ctx, secret)
	if err != nil {
		t.Fatalf("lookup: %v", err)
	}
	if got.ID != key.ID || got.RPMLimit != 600 {
		t.Errorf("lookup returned %+v, want id %d rpm 600", got, key.ID)
	}
	if _, err := s.LookupKey(ctx, "sk-llmfast-nonsense"); err != ErrKeyNotFound {
		t.Errorf("unknown key error = %v, want ErrKeyNotFound", err)
	}
	// A disabled key must be indistinguishable from an unknown one.
	if err := s.SetKeyDisabled(ctx, key.ID, true); err != nil {
		t.Fatalf("disable: %v", err)
	}
	if _, err := s.LookupKey(ctx, secret); err != ErrKeyNotFound {
		t.Errorf("disabled key error = %v, want ErrKeyNotFound", err)
	}
	if err := s.DeleteKey(ctx, key.ID); err != nil {
		t.Fatalf("delete: %v", err)
	}
	keys, _ := s.ListKeys(ctx)
	if len(keys) != 0 {
		t.Errorf("listed %d keys after delete, want 0", len(keys))
	}
}

func TestExtractBearer(t *testing.T) {
	cases := map[string]string{
		"Bearer abc":  "abc",
		"bearer abc":  "abc",
		"BEARER  abc": "abc",
		"abc":         "",
		"":            "",
		"Bearer":      "",
	}
	for header, want := range cases {
		if got := ExtractBearer(header); got != want {
			t.Errorf("ExtractBearer(%q) = %q, want %q", header, got, want)
		}
	}
}
