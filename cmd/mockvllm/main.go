// Command mockvllm is a stand-in for a real vLLM server.
//
// It speaks enough of the OpenAI API for the gateway to be exercised
// end-to-end -- streaming, usage accounting, prefix-cache reporting and error
// paths -- on a laptop with no GPU. Latency is simulated so the dashboard's
// TTFT and throughput charts show realistic shapes during development.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math/rand"
	"net/http"
	"os"
	"strings"
	"time"
)

var (
	addr = flag.String("addr", ":8000", "listen address")
	// port mirrors the flag real engines take, so this binary can stand in for
	// one on PATH and exercise the node agent's launch path without a GPU.
	port      = flag.Int("port", 0, "listen port (overrides -addr; matches the flag vllm and llama-server take)")
	host      = flag.String("host", "", "listen host (accepted for compatibility with real engines)")
	ttftMin   = flag.Duration("ttft-min", 80*time.Millisecond, "minimum simulated time to first token")
	ttftMax   = flag.Duration("ttft-max", 400*time.Millisecond, "maximum simulated time to first token")
	tokDelay  = flag.Duration("token-delay", 12*time.Millisecond, "simulated delay between tokens")
	failRate  = flag.Float64("fail-rate", 0, "fraction of requests to fail with a 500, for testing error handling")
	modelName = flag.String("model", "Qwen/Qwen3-32B", "model name to report")
)

const sample = `Streaming works end to end. This mock emits one token per chunk with a ` +
	`configurable delay so time-to-first-token and throughput can be measured against a ` +
	`realistic shape before any GPU is involved. Swap it for a real vLLM server by pointing ` +
	`the backend base_url at that host instead.`

func main() {
	// Real engines are given many flags this mock does not model. Ignoring
	// unknown ones lets it stand in for them unmodified.
	flag.CommandLine.Init(os.Args[0], flag.ContinueOnError)
	flag.CommandLine.SetOutput(io.Discard)
	if err := flag.CommandLine.Parse(knownFlags(os.Args[1:])); err != nil {
		log.Printf("ignoring unparsed flags: %v", err)
	}
	if *port != 0 {
		*addr = fmt.Sprintf("%s:%d", *host, *port)
	}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/models", handleModels)
	mux.HandleFunc("POST /v1/chat/completions", handleChat)
	mux.HandleFunc("POST /v1/completions", handleChat)

	log.Printf("mock vLLM listening on %s as %q", *addr, *modelName)
	// No WriteTimeout: streams are long-lived by design.
	srv := &http.Server{Addr: *addr, Handler: mux, ReadHeaderTimeout: 10 * time.Second}
	log.Fatal(srv.ListenAndServe())
}

func handleModels(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, 200, map[string]any{
		"object": "list",
		"data": []map[string]any{
			{"id": *modelName, "object": "model", "owned_by": "vllm", "created": time.Now().Unix()},
		},
	})
}

func handleChat(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Model         string `json:"model"`
		Stream        bool   `json:"stream"`
		MaxTokens     int    `json:"max_tokens"`
		StreamOptions *struct {
			IncludeUsage bool `json:"include_usage"`
		} `json:"stream_options"`
		Messages []struct {
			Role    string `json:"role"`
			Content any    `json:"content"`
		} `json:"messages"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, 400, errBody("invalid request body"))
		return
	}
	if *failRate > 0 && rand.Float64() < *failRate {
		writeJSON(w, 500, errBody("simulated upstream failure"))
		return
	}

	words := strings.Fields(sample)
	if req.MaxTokens > 0 && req.MaxTokens < len(words) {
		words = words[:req.MaxTokens]
	}
	// Prompt length is estimated rather than tokenized; the gateway only needs
	// a plausible number to exercise its accounting.
	promptTokens := 12
	for _, m := range req.Messages {
		if s, ok := m.Content.(string); ok {
			promptTokens += len(strings.Fields(s))
		}
	}
	// Report a share of the prompt as cache hits so the cached-prompt SKU and
	// the hit-rate card have something to show.
	cached := int(float64(promptTokens) * 0.4)

	usage := map[string]any{
		"prompt_tokens":         promptTokens,
		"completion_tokens":     len(words),
		"total_tokens":          promptTokens + len(words),
		"prompt_tokens_details": map[string]any{"cached_tokens": cached},
	}

	prefill := *ttftMin + time.Duration(rand.Int63n(int64(*ttftMax-*ttftMin)+1))
	id := fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
	created := time.Now().Unix()

	if !req.Stream {
		time.Sleep(prefill + time.Duration(len(words))*(*tokDelay))
		writeJSON(w, 200, map[string]any{
			"id": id, "object": "chat.completion", "created": created, "model": req.Model,
			"choices": []map[string]any{{
				"index":         0,
				"message":       map[string]any{"role": "assistant", "content": strings.Join(words, " ")},
				"finish_reason": "stop",
			}},
			"usage": usage,
		})
		return
	}

	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	w.WriteHeader(200)
	rc := http.NewResponseController(w)

	send := func(v any) bool {
		b, _ := json.Marshal(v)
		if _, err := fmt.Fprintf(w, "data: %s\n\n", b); err != nil {
			return false
		}
		return rc.Flush() == nil
	}
	chunk := func(delta map[string]any, finish any) map[string]any {
		return map[string]any{
			"id": id, "object": "chat.completion.chunk", "created": created, "model": req.Model,
			"choices": []map[string]any{{"index": 0, "delta": delta, "finish_reason": finish}},
		}
	}

	// vLLM opens with a role-only frame carrying no content. The gateway must
	// not count this as the first token, so the mock reproduces it.
	if !send(chunk(map[string]any{"role": "assistant", "content": ""}, nil)) {
		return
	}
	time.Sleep(prefill)

	for i, word := range words {
		text := word
		if i < len(words)-1 {
			text += " "
		}
		if !send(chunk(map[string]any{"content": text}, nil)) {
			return // client hung up
		}
		time.Sleep(*tokDelay)
	}
	if !send(chunk(map[string]any{}, "stop")) {
		return
	}

	// Only sent when asked for, matching vLLM: this is what lets the gateway
	// verify its own usage-injection and frame-suppression logic.
	if req.StreamOptions != nil && req.StreamOptions.IncludeUsage {
		if !send(map[string]any{
			"id": id, "object": "chat.completion.chunk", "created": created,
			"model": req.Model, "choices": []any{}, "usage": usage,
		}) {
			return
		}
	}
	fmt.Fprint(w, "data: [DONE]\n\n")
	_ = rc.Flush()
}

// knownFlags keeps only the arguments this mock understands, so it can be
// invoked with a real engine's full command line.
func knownFlags(args []string) []string {
	known := map[string]bool{
		"-addr": true, "-port": true, "-host": true, "-model": true,
		"-ttft-min": true, "-ttft-max": true, "-token-delay": true, "-fail-rate": true,
	}
	var out []string
	for i := 0; i < len(args); i++ {
		a := args[i]
		// Accept both -flag and --flag spellings.
		norm := strings.TrimPrefix(a, "-")
		norm = "-" + strings.TrimPrefix(norm, "-")
		base, inline, hasInline := strings.Cut(norm, "=")
		if !known[base] {
			continue
		}
		if hasInline {
			out = append(out, base+"="+inline)
			continue
		}
		if i+1 < len(args) && !strings.HasPrefix(args[i+1], "-") {
			out = append(out, base, args[i+1])
			i++
		} else {
			out = append(out, base)
		}
	}
	return out
}

func errBody(msg string) map[string]any {
	return map[string]any{"error": map[string]any{"message": msg, "type": "server_error"}}
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
