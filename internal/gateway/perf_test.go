package gateway

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

// chatBody builds a realistic request whose prompt dominates the payload, which
// is what makes the difference between copying and not copying it matter.
func chatBody(promptKB int, streaming, askUsage bool) []byte {
	content := strings.Repeat("The quick brown fox jumps over the lazy dog. ", promptKB*1024/44)
	req := map[string]any{
		"model": "qwen/qwen3.8-27b",
		"messages": []map[string]string{
			{"role": "system", "content": "You are a helpful assistant."},
			{"role": "user", "content": content},
		},
		"max_tokens":  512,
		"temperature": 0.7,
	}
	if streaming {
		req["stream"] = true
		if askUsage {
			req["stream_options"] = map[string]bool{"include_usage": true}
		}
	}
	b, _ := json.Marshal(req)
	return b
}

// oldRewrite is the previous implementation, kept so the benchmark measures a
// real difference rather than an assertion about one.
func oldRewrite(body []byte, upstreamModel string) ([]byte, error) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		return nil, err
	}
	if boolField(fields, "stream") {
		ensureUsage(fields)
	}
	fields["model"] = mustJSON(upstreamModel)
	return json.Marshal(fields)
}

func BenchmarkRewriteBody(b *testing.B) {
	for _, kb := range []int{1, 64, 1024} {
		// The case the agent produces: served-model-name matches the public id
		// and the client already asked for usage, so nothing needs changing.
		body := chatBody(kb, true, true)
		head, err := peekRequest(body)
		if err != nil {
			b.Fatal(err)
		}
		b.Run(fmt.Sprintf("passthrough/%dKB", kb), func(b *testing.B) {
			b.SetBytes(int64(len(body)))
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				if _, _, err := rewriteBody(body, head, "qwen/qwen3.8-27b"); err != nil {
					b.Fatal(err)
				}
			}
		})
		b.Run(fmt.Sprintf("old/%dKB", kb), func(b *testing.B) {
			b.SetBytes(int64(len(body)))
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				if _, err := oldRewrite(body, "qwen/qwen3.8-27b"); err != nil {
					b.Fatal(err)
				}
			}
		})

		// The case that genuinely needs a rewrite, so the fallback path is
		// measured too and cannot quietly regress.
		nb := chatBody(kb, true, false)
		nh, _ := peekRequest(nb)
		b.Run(fmt.Sprintf("rewrite/%dKB", kb), func(b *testing.B) {
			b.SetBytes(int64(len(nb)))
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				if _, _, err := rewriteBody(nb, nh, "Qwen/Qwen3.8-27B"); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkPeekRequest(b *testing.B) {
	for _, kb := range []int{1, 64, 1024} {
		body := chatBody(kb, true, true)
		b.Run(fmt.Sprintf("%dKB", kb), func(b *testing.B) {
			b.SetBytes(int64(len(body)))
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				if _, err := peekRequest(body); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkRequestPrep measures what the handler actually does: read the fields
// it needs, then prepare the upstream body. This is the comparison that matters,
// since peekRequest runs on every request whether or not a rewrite follows.
func BenchmarkRequestPrep(b *testing.B) {
	for _, kb := range []int{64, 1024} {
		body := chatBody(kb, true, true)

		b.Run(fmt.Sprintf("new/%dKB", kb), func(b *testing.B) {
			b.SetBytes(int64(len(body)))
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				head, err := peekRequest(body)
				if err != nil {
					b.Fatal(err)
				}
				if _, _, err := rewriteBody(body, head, "qwen/qwen3.8-27b"); err != nil {
					b.Fatal(err)
				}
			}
		})
		b.Run(fmt.Sprintf("old/%dKB", kb), func(b *testing.B) {
			b.SetBytes(int64(len(body)))
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				if _, err := oldRewrite(body, "qwen/qwen3.8-27b"); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// TestRewriteBodyPassesThroughUnchanged pins the optimisation's correctness.
// Forwarding the original bytes is only safe when nothing actually needs to
// change; if that check is ever wrong, requests reach the engine with the wrong
// model name or without usage accounting, and billing goes quietly wrong.
func TestRewriteBodyPassesThroughUnchanged(t *testing.T) {
	body := chatBody(4, true, true)
	head, err := peekRequest(body)
	if err != nil {
		t.Fatal(err)
	}
	out, injected, err := rewriteBody(body, head, "qwen/qwen3.8-27b")
	if err != nil {
		t.Fatal(err)
	}
	if &out[0] != &body[0] {
		t.Error("nothing needed changing, so the original bytes should be forwarded")
	}
	if injected {
		t.Error("the client already asked for usage; nothing was injected")
	}
}

func TestRewriteBodyRewritesWhenNeeded(t *testing.T) {
	cases := []struct {
		name           string
		body           []byte
		upstream       string
		wantModel      string
		wantInjected   bool
		wantUsageAsked bool
	}{
		{
			name: "model name differs",
			body: chatBody(2, true, true), upstream: "Qwen/Qwen3.8-27B",
			wantModel: "Qwen/Qwen3.8-27B", wantInjected: false, wantUsageAsked: true,
		},
		{
			name: "usage must be injected",
			body: chatBody(2, true, false), upstream: "qwen/qwen3.8-27b",
			wantModel: "qwen/qwen3.8-27b", wantInjected: true, wantUsageAsked: true,
		},
		{
			name: "non-streaming needs no usage injection",
			body: chatBody(2, false, false), upstream: "Qwen/Qwen3.8-27B",
			wantModel: "Qwen/Qwen3.8-27B", wantInjected: false, wantUsageAsked: false,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			head, err := peekRequest(c.body)
			if err != nil {
				t.Fatal(err)
			}
			out, injected, err := rewriteBody(c.body, head, c.upstream)
			if err != nil {
				t.Fatal(err)
			}
			var got struct {
				Model         string `json:"model"`
				StreamOptions *struct {
					IncludeUsage bool `json:"include_usage"`
				} `json:"stream_options"`
				Messages  []map[string]string `json:"messages"`
				MaxTokens int                 `json:"max_tokens"`
			}
			if err := json.Unmarshal(out, &got); err != nil {
				t.Fatalf("output is not valid JSON: %v", err)
			}
			if got.Model != c.wantModel {
				t.Errorf("model = %q, want %q", got.Model, c.wantModel)
			}
			if injected != c.wantInjected {
				t.Errorf("injected = %v, want %v", injected, c.wantInjected)
			}
			if c.wantUsageAsked && (got.StreamOptions == nil || !got.StreamOptions.IncludeUsage) {
				t.Error("upstream must be asked for usage on a streaming request")
			}
			// Everything we do not touch has to survive intact.
			if len(got.Messages) != 2 {
				t.Errorf("messages = %d, want 2 preserved", len(got.Messages))
			}
			if got.MaxTokens != 512 {
				t.Errorf("max_tokens = %d, want 512 preserved", got.MaxTokens)
			}
		})
	}
}

// TestPeekRequestDoesNotCopyThePrompt is the property the whole optimisation
// rests on: decoding into a narrow struct must skip the message array rather
// than allocate a copy of it.
func TestPeekRequestDoesNotCopyThePrompt(t *testing.T) {
	small := chatBody(1, true, true)
	large := chatBody(512, true, true)

	perOp := func(body []byte) float64 {
		r := testing.Benchmark(func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				if _, err := peekRequest(body); err != nil {
					b.Fatal(err)
				}
			}
		})
		return float64(r.AllocedBytesPerOp())
	}
	smallAlloc, largeAlloc := perOp(small), perOp(large)
	// The large body is 512x the small one. If the prompt were being copied,
	// allocation would scale with it.
	if largeAlloc > smallAlloc*4 {
		t.Errorf("peekRequest allocated %.0f B on a 512KB body against %.0f B on 1KB; "+
			"the prompt is being copied", largeAlloc, smallAlloc)
	}
}
