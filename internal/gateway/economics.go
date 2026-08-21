package gateway

import (
	"context"
	"fmt"
	"net/http"
	"time"
)

func contextWithTimeout(d time.Duration) (context.Context, context.CancelFunc) {
	return context.WithTimeout(context.Background(), d)
}

// hoursPerMonth is the billing month used throughout: 730 hours is the average
// of a 365-day year, and is what GPU providers quote against.
const hoursPerMonth = 730.0

// EconomicsRequest asks "given this hardware cost and this measured
// throughput, does this make money?"
type EconomicsRequest struct {
	Model string `json:"model"`
	// GPUCostPerMonth is what the hardware costs. Include storage and any
	// stopped-instance charges: they are real money.
	GPUCostPerMonth float64 `json:"gpu_cost_per_month"`
	// AggregateTPS is output tokens per second the endpoint sustains, ideally
	// taken from a benchmark run rather than guessed.
	AggregateTPS float64 `json:"aggregate_tps"`
	// Utilisation is the share of the month the GPU is actually generating.
	// This is the number most business plans get wrong: a new provider on
	// OpenRouter competes for traffic and will not be busy.
	Utilisation float64 `json:"utilisation"`
	// InputRatio is prompt tokens per completion token. Chat workloads are
	// input-heavy, and input is priced far lower, so this materially changes
	// the answer.
	InputRatio float64 `json:"input_ratio"`
	// CacheHitRate is the share of prompt tokens served from the prefix cache
	// and billed at the cheaper rate.
	CacheHitRate float64 `json:"cache_hit_rate"`
}

type EconomicsResult struct {
	Model             string  `json:"model"`
	PromptUSDPerM     float64 `json:"prompt_usd_per_m"`
	CompletionUSDPerM float64 `json:"completion_usd_per_m"`
	CachedUSDPerM     float64 `json:"cached_usd_per_m"`

	OutputTokensPerMonth float64 `json:"output_tokens_per_month"`
	InputTokensPerMonth  float64 `json:"input_tokens_per_month"`

	RevenueOutput float64 `json:"revenue_output"`
	RevenueInput  float64 `json:"revenue_input"`
	RevenueTotal  float64 `json:"revenue_total"`
	Cost          float64 `json:"cost"`
	Margin        float64 `json:"margin"`
	MarginPercent float64 `json:"margin_percent"`

	RevenuePerDay float64 `json:"revenue_per_day"`
	MarginPerDay  float64 `json:"margin_per_day"`

	// BreakEvenUtilisation is the share of the month the GPU must be generating
	// for revenue to cover the hardware. Below it you lose money however you
	// price, because you cannot sell tokens you did not generate.
	BreakEvenUtilisation float64 `json:"break_even_utilisation"`
	// BreakEvenTPS is the aggregate throughput needed at the stated
	// utilisation to cover cost.
	BreakEvenTPS float64 `json:"break_even_tps"`

	Verdict string   `json:"verdict"`
	Notes   []string `json:"notes"`
}

func (s *Server) adminEconomics(w http.ResponseWriter, r *http.Request) {
	var req EconomicsRequest
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "invalid request body")
		return
	}
	entry, ok := s.catalog.Get(req.Model)
	if !ok {
		writeError(w, http.StatusNotFound, "invalid_request_error", "unknown model "+req.Model)
		return
	}
	if req.AggregateTPS <= 0 {
		writeError(w, http.StatusBadRequest, "invalid_request_error",
			"aggregate_tps is required; run a benchmark first rather than guessing it")
		return
	}
	if req.Utilisation <= 0 || req.Utilisation > 1 {
		req.Utilisation = 0.25
	}
	if req.InputRatio <= 0 {
		req.InputRatio = 5
	}
	if req.CacheHitRate < 0 || req.CacheHitRate > 1 {
		req.CacheHitRate = 0
	}

	res := computeEconomics(req, entry)
	res.Model = req.Model
	writeJSON(w, http.StatusOK, res)
}

// computeEconomics is separated from the handler so it can be tested directly.
func computeEconomics(req EconomicsRequest, e *Entry) EconomicsResult {
	const secondsPerMonth = hoursPerMonth * 3600

	res := EconomicsResult{
		PromptUSDPerM:     e.PromptUSD * 1e6,
		CompletionUSDPerM: e.CompletionUSD * 1e6,
		CachedUSDPerM:     e.CachedPromptUSD * 1e6,
		Cost:              req.GPUCostPerMonth,
	}

	res.OutputTokensPerMonth = req.AggregateTPS * secondsPerMonth * req.Utilisation
	res.InputTokensPerMonth = res.OutputTokensPerMonth * req.InputRatio

	cached := res.InputTokensPerMonth * req.CacheHitRate
	full := res.InputTokensPerMonth - cached

	res.RevenueOutput = res.OutputTokensPerMonth * e.CompletionUSD
	res.RevenueInput = full*e.PromptUSD + cached*e.CachedPromptUSD
	res.RevenueTotal = res.RevenueOutput + res.RevenueInput
	res.Margin = res.RevenueTotal - res.Cost
	if res.RevenueTotal > 0 {
		res.MarginPercent = res.Margin / res.RevenueTotal * 100
	}
	res.RevenuePerDay = res.RevenueTotal / 30
	res.MarginPerDay = res.Margin / 30

	// Revenue is linear in utilisation, so break-even is a simple ratio.
	if res.RevenueTotal > 0 {
		res.BreakEvenUtilisation = req.Utilisation * res.Cost / res.RevenueTotal
	}
	if req.Utilisation > 0 && res.RevenueTotal > 0 {
		revenuePerTPS := res.RevenueTotal / req.AggregateTPS
		res.BreakEvenTPS = res.Cost / revenuePerTPS
	}

	switch {
	case res.BreakEvenUtilisation <= 0:
		res.Verdict = "This model is priced at zero, so it cannot cover any hardware cost."
	case res.BreakEvenUtilisation > 1:
		res.Verdict = fmt.Sprintf(
			"Loses money even at 100%% utilisation. You would need %.0f tok/s aggregate to break "+
				"even, against the %.0f measured.", res.BreakEvenTPS, req.AggregateTPS)
	case res.BreakEvenUtilisation > 0.5:
		res.Verdict = fmt.Sprintf(
			"Break-even needs %.0f%% utilisation. That is a lot of traffic for a new provider; "+
				"treat this as marginal.", res.BreakEvenUtilisation*100)
	default:
		res.Verdict = fmt.Sprintf(
			"Break-even at %.0f%% utilisation, so there is real headroom above it.",
			res.BreakEvenUtilisation*100)
	}

	res.Notes = append(res.Notes,
		"Utilisation is the share of the month the GPU is actually generating tokens. A new "+
			"provider competing for routed traffic will not be busy: assume low and be pleased "+
			"to be wrong.")
	if req.CacheHitRate > 0 {
		res.Notes = append(res.Notes, fmt.Sprintf(
			"A %.0f%% prefix-cache hit rate lowers revenue here because cached tokens bill "+
				"cheaper — but it also raises the throughput you can serve from the same GPU, "+
				"which this figure does not credit you for.", req.CacheHitRate*100))
	}
	res.Notes = append(res.Notes,
		"Revenue assumes you are paid your published prices for every token. Verify what "+
			"OpenRouter actually remits before relying on it.")
	return res
}

// EconomicsFromStats seeds the calculator from what the gateway has actually
// served, so the operator starts from measured reality rather than a guess.
func (s *Server) economicsSeed(window time.Duration) (aggregateTPS, inputRatio, cacheHit float64) {
	ctx, cancel := contextWithTimeout(5 * time.Second)
	defer cancel()
	t, err := s.store.Totals(ctx, time.Now().Add(-window), time.Now())
	if err != nil || t.Requests == 0 {
		return 0, 0, 0
	}
	if t.CompTok > 0 {
		inputRatio = float64(t.PromptTok) / float64(t.CompTok)
	}
	if t.PromptTok > 0 {
		cacheHit = float64(t.CachedTok) / float64(t.PromptTok)
	}
	return t.TPSAvg, inputRatio, cacheHit
}
