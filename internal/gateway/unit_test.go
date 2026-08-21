package gateway

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestEnsureUsage(t *testing.T) {
	cases := []struct {
		name         string
		in           string
		wantInjected bool
	}{
		{"absent", `{}`, true},
		{"explicitly false", `{"stream_options":{"include_usage":false}}`, true},
		{"already true", `{"stream_options":{"include_usage":true}}`, false},
		{"other options present", `{"stream_options":{"something":1}}`, true},
		{"malformed options", `{"stream_options":"nonsense"}`, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			var fields map[string]json.RawMessage
			if err := json.Unmarshal([]byte(c.in), &fields); err != nil {
				t.Fatalf("setup: %v", err)
			}
			got := ensureUsage(fields)
			if got != c.wantInjected {
				t.Errorf("injected = %v, want %v", got, c.wantInjected)
			}
			// Regardless of the input, upstream must end up being asked for usage.
			var opts struct {
				IncludeUsage bool `json:"include_usage"`
			}
			if err := json.Unmarshal(fields["stream_options"], &opts); err != nil {
				t.Fatalf("stream_options is not an object: %v", err)
			}
			if !opts.IncludeUsage {
				t.Error("include_usage was not enabled upstream")
			}
		})
	}
}

// TestEnsureUsagePreservesOtherOptions guards against clobbering fields we do
// not own when merging into an existing stream_options object.
func TestEnsureUsagePreservesOtherOptions(t *testing.T) {
	var fields map[string]json.RawMessage
	_ = json.Unmarshal([]byte(`{"stream_options":{"continuous_usage_stats":true}}`), &fields)
	ensureUsage(fields)

	var opts map[string]any
	_ = json.Unmarshal(fields["stream_options"], &opts)
	if opts["continuous_usage_stats"] != true {
		t.Errorf("existing option was dropped: %v", opts)
	}
	if opts["include_usage"] != true {
		t.Errorf("include_usage not set: %v", opts)
	}
}

func TestCarriesToken(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want bool
	}{
		{"opening role frame", `{"choices":[{"delta":{"role":"assistant","content":""}}]}`, false},
		{"content delta", `{"choices":[{"delta":{"content":"Hello"}}]}`, true},
		{"null content", `{"choices":[{"delta":{"content":null}}]}`, false},
		{"reasoning delta", `{"choices":[{"delta":{"reasoning_content":"thinking"}}]}`, true},
		{"empty reasoning", `{"choices":[{"delta":{"reasoning_content":""}}]}`, false},
		{"tool call", `{"choices":[{"delta":{"tool_calls":[{"index":0}]}}]}`, true},
		{"finish frame", `{"choices":[{"delta":{},"finish_reason":"stop"}]}`, false},
		{"usage frame", `{"choices":[],"usage":{"prompt_tokens":5}}`, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := carriesToken([]byte(c.in)); got != c.want {
				t.Errorf("carriesToken = %v, want %v", got, c.want)
			}
		})
	}
}

func TestSSEData(t *testing.T) {
	cases := []struct {
		in      string
		want    string
		wantHit bool
	}{
		{"data: {\"a\":1}\n", `{"a":1}`, true},
		{"data: [DONE]\n", "[DONE]", true},
		{"data: {\"a\":1}\r\n", `{"a":1}`, true},
		{": keepalive\n", "", false},
		{"\n", "", false},
		{"event: message\n", "", false},
	}
	for _, c := range cases {
		got, hit := sseData([]byte(c.in))
		if hit != c.wantHit || (hit && string(got) != c.want) {
			t.Errorf("sseData(%q) = %q,%v; want %q,%v", c.in, got, hit, c.want, c.wantHit)
		}
	}
}

func TestHasUsage(t *testing.T) {
	if hasUsage([]byte(`{"usage":null,"choices":[]}`)) {
		t.Error("a null usage field must not be treated as usage")
	}
	if !hasUsage([]byte(`{"usage":{"prompt_tokens":1},"choices":[]}`)) {
		t.Error("a populated usage field was not detected")
	}
}

func TestParseUsageDetails(t *testing.T) {
	u := parseUsage([]byte(`{"usage":{"prompt_tokens":100,"completion_tokens":50,
		"prompt_tokens_details":{"cached_tokens":80},
		"completion_tokens_details":{"reasoning_tokens":20}}}`))
	if u == nil {
		t.Fatal("usage did not parse")
	}
	if u.PromptTokens != 100 || u.CompletionTokens != 50 {
		t.Errorf("tokens = %d/%d, want 100/50", u.PromptTokens, u.CompletionTokens)
	}
	if u.cached() != 80 {
		t.Errorf("cached = %d, want 80", u.cached())
	}
	if u.reasoning() != 20 {
		t.Errorf("reasoning = %d, want 20", u.reasoning())
	}
	// Absent detail objects must read as zero, not panic.
	plain := parseUsage([]byte(`{"usage":{"prompt_tokens":1}}`))
	if plain.cached() != 0 || plain.reasoning() != 0 {
		t.Error("missing detail objects should read as zero")
	}
}

func TestCost(t *testing.T) {
	e := &Entry{
		PromptUSD:       0.0000001,  // $0.10/M
		CompletionUSD:   0.0000003,  // $0.30/M
		CachedPromptUSD: 0.00000002, // $0.02/M
	}
	// 1000 prompt of which 400 cached, 500 completion:
	//   600*1e-7 + 400*2e-8 + 500*3e-7 = 6e-5 + 8e-6 + 1.5e-4 = 2.18e-4
	got := e.Cost(1000, 500, 400, 0)
	if diff := got - 0.000218; diff > 1e-12 || diff < -1e-12 {
		t.Errorf("cost = %v, want 0.000218", got)
	}

	// Cached tokens are billed at the cache rate instead of, not on top of,
	// the full prompt rate.
	if e.Cost(1000, 0, 1000, 0) >= e.Cost(1000, 0, 0, 0) {
		t.Error("a fully cached prompt should cost less than an uncached one")
	}

	free := &Entry{IsFree: true, PromptUSD: 1, CompletionUSD: 1}
	if free.Cost(1000, 1000, 0, 0) != 0 {
		t.Error("a free model must never accrue cost")
	}

	disc := &Entry{CompletionUSD: 0.000001, DiscountToUser: 0.5}
	if got := disc.Cost(0, 1000, 0, 0); got != 0.0005 {
		t.Errorf("discounted cost = %v, want 0.0005", got)
	}
}

func TestCostNeverNegative(t *testing.T) {
	// A reasoning rate cheaper than the completion rate must not produce a
	// negative bill through the delta adjustment.
	e := &Entry{CompletionUSD: 0.000001, ReasoningUSD: 0.0000001}
	if got := e.Cost(0, 10, 0, 10); got < 0 {
		t.Errorf("cost = %v, want >= 0", got)
	}
}

func TestPatchModelName(t *testing.T) {
	// Names match: no rewrite, and the input slice is returned untouched.
	same := &Entry{ID: "a/b", UpstreamModel: "a/b"}
	in := []byte(`data: {"model":"a/b"}`)
	out, _ := same.PatchModelName(in, nil)
	if string(out) != string(in) {
		t.Errorf("unexpected rewrite: %s", out)
	}

	e := &Entry{
		ID: "qwen/qwen3-32b", RewriteModel: true,
		modelNeedles: [][]byte{[]byte(`"model":"Qwen/Qwen3-32B"`), []byte(`"model": "Qwen/Qwen3-32B"`)},
		modelRepls:   [][]byte{[]byte(`"model":"qwen/qwen3-32b"`), []byte(`"model": "qwen/qwen3-32b"`)},
	}
	got, buf := e.PatchModelName([]byte(`data: {"id":"x","model":"Qwen/Qwen3-32B","object":"chunk"}`), nil)
	want := `data: {"id":"x","model":"qwen/qwen3-32b","object":"chunk"}`
	if string(got) != want {
		t.Errorf("compact form: got %s, want %s", got, want)
	}
	// Indented encoders emit a space after the colon.
	got2, _ := e.PatchModelName([]byte(`{"model": "Qwen/Qwen3-32B"}`), buf)
	if string(got2) != `{"model": "qwen/qwen3-32b"}` {
		t.Errorf("spaced form: got %s", got2)
	}
	// The upstream name appearing in generated content must not be rewritten,
	// because the needle includes the "model" key.
	got3, _ := e.PatchModelName([]byte(`{"content":"I am Qwen/Qwen3-32B"}`), nil)
	if string(got3) != `{"content":"I am Qwen/Qwen3-32B"}` {
		t.Errorf("content was wrongly rewritten: %s", got3)
	}
}

func TestRateLimiter(t *testing.T) {
	rl := newRateLimiter()
	for i := 0; i < 3; i++ {
		if !rl.allow(1, 3) {
			t.Fatalf("request %d denied while under the limit", i+1)
		}
	}
	if rl.allow(1, 3) {
		t.Error("request over the limit was allowed")
	}
	// A different key has its own window.
	if !rl.allow(2, 3) {
		t.Error("a separate key was blocked by another key's limit")
	}
	// Zero means unlimited.
	for i := 0; i < 100; i++ {
		if !rl.allow(3, 0) {
			t.Fatal("a zero limit should not restrict anything")
		}
	}
}

// --- economics --------------------------------------------------------------

func qwenEntry() *Entry {
	// Qwen3.8-27B at the market rate observed on OpenRouter: $0.45 in,
	// $3.20 out, $0.05 cached, per million tokens.
	return &Entry{
		ID:              "qwen/qwen3.8-27b",
		PromptUSD:       0.45 / 1e6,
		CompletionUSD:   3.20 / 1e6,
		CachedPromptUSD: 0.05 / 1e6,
	}
}

func TestEconomicsBasics(t *testing.T) {
	res := computeEconomics(EconomicsRequest{
		GPUCostPerMonth: 336, AggregateTPS: 400, Utilisation: 0.25, InputRatio: 5,
	}, qwenEntry())

	// 400 tok/s x 730h x 3600s x 25% = 1.051e9 output tokens.
	wantOut := 400 * hoursPerMonth * 3600 * 0.25
	if diff := res.OutputTokensPerMonth - wantOut; diff > 1 || diff < -1 {
		t.Errorf("output tokens = %.0f, want %.0f", res.OutputTokensPerMonth, wantOut)
	}
	if res.InputTokensPerMonth != res.OutputTokensPerMonth*5 {
		t.Errorf("input tokens should be 5x output, got %.0f vs %.0f",
			res.InputTokensPerMonth, res.OutputTokensPerMonth)
	}
	if res.RevenueTotal <= res.Cost {
		t.Errorf("revenue %.0f should exceed cost %.0f at these rates", res.RevenueTotal, res.Cost)
	}
	if res.Margin != res.RevenueTotal-res.Cost {
		t.Error("margin must be revenue minus cost")
	}
	if diff := res.RevenuePerDay*30 - res.RevenueTotal; diff > 0.01 || diff < -0.01 {
		t.Error("daily revenue must be the monthly figure over 30")
	}
}

// TestBreakEvenUtilisation is the number that actually matters to a new
// provider: revenue scales linearly with how busy the GPU is, and a new
// provider is not busy.
func TestBreakEvenUtilisation(t *testing.T) {
	req := EconomicsRequest{GPUCostPerMonth: 336, AggregateTPS: 400, Utilisation: 0.25, InputRatio: 5}
	res := computeEconomics(req, qwenEntry())

	// Re-running at exactly the break-even utilisation should yield zero margin.
	check := req
	check.Utilisation = res.BreakEvenUtilisation
	at := computeEconomics(check, qwenEntry())
	if diff := at.Margin; diff > 1 || diff < -1 {
		t.Errorf("at break-even utilisation %.4f the margin was %.2f, want about 0",
			res.BreakEvenUtilisation, at.Margin)
	}
}

func TestEconomicsFlagsHopelessConfigurations(t *testing.T) {
	// A CPU box: 5 tok/s against a $60/mo server, on cheap tokens.
	cheap := &Entry{PromptUSD: 0.065 / 1e6, CompletionUSD: 0.14 / 1e6}
	res := computeEconomics(EconomicsRequest{
		GPUCostPerMonth: 60, AggregateTPS: 5, Utilisation: 1.0, InputRatio: 5,
	}, cheap)
	if res.Margin >= 0 {
		t.Errorf("5 tok/s on $0.14/M output should lose money even at 100%%, margin was %.2f", res.Margin)
	}
	if res.BreakEvenUtilisation <= 1 {
		t.Errorf("break-even utilisation = %.2f; it should exceed 1 for an impossible case",
			res.BreakEvenUtilisation)
	}
	if !strings.Contains(res.Verdict, "even at 100%") {
		t.Errorf("verdict should say it cannot break even: %q", res.Verdict)
	}
}

// TestCacheHitLowersBilledRevenue: cached prompt tokens bill at the cheaper
// rate, so a high hit rate reduces revenue on the same traffic. It also raises
// the traffic the GPU can serve, which this calculation deliberately does not
// credit — better to understate.
func TestCacheHitLowersBilledRevenue(t *testing.T) {
	base := EconomicsRequest{GPUCostPerMonth: 336, AggregateTPS: 400, Utilisation: 0.25, InputRatio: 5}
	none := computeEconomics(base, qwenEntry())

	cached := base
	cached.CacheHitRate = 0.5
	with := computeEconomics(cached, qwenEntry())

	if with.RevenueInput >= none.RevenueInput {
		t.Errorf("cached tokens bill cheaper, so input revenue should fall: %.2f vs %.2f",
			with.RevenueInput, none.RevenueInput)
	}
	if with.RevenueOutput != none.RevenueOutput {
		t.Error("a prompt cache hit must not change output revenue")
	}
}

func TestEconomicsDefaultsAreConservative(t *testing.T) {
	// Unset utilisation must not default to a flattering 100%.
	res := computeEconomics(EconomicsRequest{
		GPUCostPerMonth: 336, AggregateTPS: 400,
	}, qwenEntry())
	full := computeEconomics(EconomicsRequest{
		GPUCostPerMonth: 336, AggregateTPS: 400, Utilisation: 1, InputRatio: 5,
	}, qwenEntry())
	if res.RevenueTotal >= full.RevenueTotal {
		t.Error("an unspecified utilisation should not be treated as fully busy")
	}
}

// --- benchmark interpretation ----------------------------------------------

// TestBenchNoteFindsTheKnee: the highest measured number is a poor signal,
// because noise routinely puts it at the last level even when throughput
// plateaued earlier. What the operator needs is where the curve flattens.
func TestBenchNoteFindsTheKnee(t *testing.T) {
	// Plateaus at 8; the tiny rise at 32 is noise.
	r := BenchResult{Levels: []BenchLevel{
		{Concurrency: 1, AggregateTPS: 54.5},
		{Concurrency: 4, AggregateTPS: 199.9},
		{Concurrency: 8, AggregateTPS: 393.8},
		{Concurrency: 16, AggregateTPS: 390.2},
		{Concurrency: 32, AggregateTPS: 394.7},
	}}
	for _, lv := range r.Levels {
		if lv.AggregateTPS > r.PeakAggregate {
			r.PeakAggregate, r.BestConcurrency = lv.AggregateTPS, lv.Concurrency
		}
	}
	note := benchNote(r)
	if strings.Contains(note, "still climbing") {
		t.Errorf("a 0.2%% rise is noise, not headroom: %q", note)
	}
	if !strings.Contains(note, "near 8") {
		t.Errorf("the knee is at 8 concurrent, note said: %q", note)
	}
}

func TestBenchNoteDetectsRealHeadroom(t *testing.T) {
	// Genuinely still climbing: each level meaningfully above the last.
	r := BenchResult{Levels: []BenchLevel{
		{Concurrency: 1, AggregateTPS: 50},
		{Concurrency: 4, AggregateTPS: 190},
		{Concurrency: 8, AggregateTPS: 370},
		{Concurrency: 16, AggregateTPS: 700},
	}}
	for _, lv := range r.Levels {
		if lv.AggregateTPS > r.PeakAggregate {
			r.PeakAggregate, r.BestConcurrency = lv.AggregateTPS, lv.Concurrency
		}
	}
	if note := benchNote(r); !strings.Contains(note, "still climbing") {
		t.Errorf("throughput nearly doubled at the last level; that is headroom: %q", note)
	}
}

func TestSyntheticPromptLength(t *testing.T) {
	// Roughly four characters per token is close enough for a load shape, but
	// the prompt must not be wildly short or the prefill cost is unrepresentative.
	for _, n := range []int{128, 512, 4096} {
		got := len(syntheticPrompt(n))
		if got < n*3 || got > n*6 {
			t.Errorf("syntheticPrompt(%d) produced %d chars, want roughly %d", n, got, n*4)
		}
	}
}
