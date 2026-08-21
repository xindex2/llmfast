package gateway

import (
	"bytes"
	"strconv"
	"sync"

	"github.com/llmfast/gateway/internal/config"
)

// Entry is a model resolved into the exact form the request path needs. Prices
// arrive from config as decimal strings for precision; they are parsed once
// here so per-request billing is float arithmetic and no allocation.
type Entry struct {
	ID            string
	UpstreamModel string
	ContextLength int
	MaxOutput     int
	IsFree        bool

	PromptUSD       float64
	CompletionUSD   float64
	CachedPromptUSD float64
	ReasoningUSD    float64
	// DiscountToUser is applied to what the user pays, matching OpenRouter's
	// user price = base x (1 - discount) formula, so our recorded revenue
	// reflects what we actually collect.
	DiscountToUser float64

	// RewriteModel is set when the upstream model name differs from the public
	// id. vLLM echoes back the name it was started with, but clients key off
	// the id they asked for, so those frames need patching on the way out.
	// Starting vLLM with --served-model-name equal to the public id makes the
	// names match and skips this work entirely.
	RewriteModel bool
	modelNeedles [][]byte
	modelRepls   [][]byte
}

// PatchModelName rewrites the echoed model name in one response frame, using
// dst as scratch to avoid allocating per chunk. It returns the original slice
// untouched when no rewrite is needed, which is the common case.
func (e *Entry) PatchModelName(frame, dst []byte) ([]byte, []byte) {
	if !e.RewriteModel {
		return frame, dst
	}
	for i, needle := range e.modelNeedles {
		idx := bytes.Index(frame, needle)
		if idx < 0 {
			continue
		}
		repl := e.modelRepls[i]
		dst = append(dst[:0], frame[:idx]...)
		dst = append(dst, repl...)
		dst = append(dst, frame[idx+len(needle):]...)
		return dst, dst
	}
	return frame, dst
}

// Cost prices a completed request. Cached prompt tokens are billed at the cache
// rate and subtracted from the full-price prompt count rather than billed twice.
func (e *Entry) Cost(promptTok, completionTok, cachedTok, reasoningTok int) float64 {
	if e.IsFree {
		return 0
	}
	billablePrompt := promptTok - cachedTok
	if billablePrompt < 0 {
		billablePrompt = 0
	}
	total := float64(billablePrompt)*e.PromptUSD +
		float64(cachedTok)*e.CachedPromptUSD +
		float64(completionTok)*e.CompletionUSD
	// Reasoning tokens are already counted inside completion_tokens by vLLM, so
	// only the delta between the reasoning rate and the completion rate applies.
	if e.ReasoningUSD > 0 && reasoningTok > 0 {
		total += float64(reasoningTok) * (e.ReasoningUSD - e.CompletionUSD)
	}
	if e.DiscountToUser != 0 {
		total *= 1 - e.DiscountToUser
	}
	if total < 0 {
		return 0
	}
	return total
}

// Catalog is an immutable snapshot of the model table, swapped wholesale on
// config reload so readers never see a half-updated map.
type Catalog struct {
	mu      sync.RWMutex
	entries map[string]*Entry
}

func NewCatalog(cfg *config.Config) *Catalog {
	c := &Catalog{}
	c.Replace(cfg)
	return c
}

func (c *Catalog) Replace(cfg *config.Config) {
	m := make(map[string]*Entry, len(cfg.Models))
	for i := range cfg.Models {
		src := &cfg.Models[i]
		e := &Entry{
			ID:              src.ID,
			UpstreamModel:   src.UpstreamModel,
			ContextLength:   src.ContextLength,
			MaxOutput:       src.MaxOutputTokens,
			IsFree:          src.IsFree,
			PromptUSD:       parsePrice(src.Pricing.Prompt),
			CompletionUSD:   parsePrice(src.Pricing.Completion),
			CachedPromptUSD: parsePrice(src.Pricing.CachedPrompt),
			ReasoningUSD:    parsePrice(src.Pricing.InternalReasoning),
			DiscountToUser:  src.DiscountToUser,
		}
		if src.UpstreamModel != src.ID {
			e.RewriteModel = true
			// Both spacings are covered because compact and indented JSON
			// encoders differ, and we do not control the upstream's choice.
			for _, sep := range []string{`"model":"`, `"model": "`} {
				e.modelNeedles = append(e.modelNeedles, []byte(sep+src.UpstreamModel+`"`))
				e.modelRepls = append(e.modelRepls, []byte(sep+src.ID+`"`))
			}
		}
		m[src.ID] = e
	}
	c.mu.Lock()
	c.entries = m
	c.mu.Unlock()
}

func (c *Catalog) Get(id string) (*Entry, bool) {
	c.mu.RLock()
	e, ok := c.entries[id]
	c.mu.RUnlock()
	return e, ok
}

func (c *Catalog) IDs() []string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	out := make([]string, 0, len(c.entries))
	for id := range c.entries {
		out = append(out, id)
	}
	return out
}

// parsePrice tolerates an unset price, which means the SKU is not billed.
func parsePrice(s string) float64 {
	if s == "" {
		return 0
	}
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0
	}
	return v
}
