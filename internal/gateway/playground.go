package gateway

import (
	"net/http"
	"time"
)

// playgroundKeyID marks requests made from the admin playground.
//
// They are logged like any other request, because they are real load on a real
// backend and hiding them would make the dashboard lie about GPU utilisation.
// The distinct key id lets them be told apart from customer traffic.
const playgroundKeyID = -1

// adminPlayground runs a chat completion from the admin UI.
//
// It reuses serveInference rather than reimplementing the proxy, so what the
// playground exercises is exactly what OpenRouter would hit: the same admission
// control, the same SSE relay, the same token accounting. A separate simplified
// path would be able to work while the real one was broken, which is precisely
// the failure a test console exists to catch.
func (s *Server) adminPlayground(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	reqID := requestID()
	w.Header().Set("X-Request-Id", reqID)
	// rpmLimit 0 disables per-key rate limiting: an operator testing their own
	// endpoint should not be throttled by a customer-facing quota.
	s.serveInference(w, r, "/chat/completions", playgroundKeyID, 0, start, reqID)
}

// PlaygroundModel is one entry in the playground's model picker.
type PlaygroundModel struct {
	ID            string `json:"id"`
	Name          string `json:"name"`
	Available     bool   `json:"available"`
	Ready         bool   `json:"ready"`
	ContextLength int    `json:"context_length"`
	MaxOutput     int    `json:"max_output_tokens"`
	Tools         bool   `json:"tools"`
	Reasoning     bool   `json:"reasoning"`
	PromptUSD     string `json:"prompt_usd"`
	CompletionUSD string `json:"completion_usd"`
	// Reason explains why a model cannot be used, so the picker does not simply
	// present a dead option with no explanation.
	Reason string `json:"reason,omitempty"`
}

func (s *Server) adminPlaygroundModels(w http.ResponseWriter, r *http.Request) {
	out := make([]PlaygroundModel, 0, len(s.cfg.Models))
	for i := range s.cfg.Models {
		m := &s.cfg.Models[i]
		ready := true
		if m.IsReady != nil {
			ready = *m.IsReady
		}
		pm := PlaygroundModel{
			ID: m.ID, Name: m.Name, Ready: ready,
			ContextLength: m.ContextLength, MaxOutput: m.MaxOutputTokens,
			Tools: m.Features.Tools, Reasoning: m.Features.Reasoning,
			PromptUSD: m.Pricing.Prompt, CompletionUSD: m.Pricing.Completion,
			Available: s.pool.Available(m.ID),
		}
		if !pm.Available {
			// A staged model with no engine yet is the common case here, and it
			// is worth distinguishing from a backend that has gone down.
			pm.Reason = "no healthy backend is serving this model right now"
		}
		out = append(out, pm)
	}
	writeJSON(w, http.StatusOK, map[string]any{"models": out})
}
