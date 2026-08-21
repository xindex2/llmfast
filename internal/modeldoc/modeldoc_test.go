package modeldoc

import (
	"encoding/json"
	"testing"

	"github.com/llmfast/gateway/internal/config"
)

func base() *config.Model {
	return &config.Model{
		ID: "qwen/qwen3-32b", Name: "Qwen3 32B", UpstreamModel: "Qwen/Qwen3-32B",
		Created: "2025-04-29", Quantization: "fp8", ContextLength: 131072, MaxOutputTokens: 32768,
		Pricing: config.Pricing{Prompt: "0.0000001", Completion: "0.0000003"},
	}
}

func render(t *testing.T, m *config.Model) map[string]any {
	t.Helper()
	b, err := json.Marshal(BuildOne(m))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var out map[string]any
	if err := json.Unmarshal(b, &out); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	return out
}

func TestRequiredShape(t *testing.T) {
	doc := render(t, base())
	if doc["schema_version"] != "2.4" {
		t.Errorf("schema_version = %v", doc["schema_version"])
	}
	// The spec requires at least one of each.
	if len(doc["input_modalities"].([]any)) == 0 {
		t.Error("no input modality emitted")
	}
	if len(doc["output_modalities"].([]any)) == 0 {
		t.Error("no output modality emitted")
	}
	// created is a unix timestamp, not the date string.
	if doc["created"].(float64) != 1745884800 {
		t.Errorf("created = %v, want a unix timestamp", doc["created"])
	}
}

// TestQuantizationIsExplicitNull covers the spec's instruction to send null
// rather than omit the field when precision is undeclared.
func TestQuantizationIsExplicitNull(t *testing.T) {
	m := base()
	m.Quantization = ""
	doc := render(t, m)
	v, present := doc["quantization"]
	if !present {
		t.Fatal("quantization key is missing; the spec wants an explicit null")
	}
	if v != nil {
		t.Errorf("quantization = %v, want null", v)
	}
}

// TestNoZeroStuffing pins the rule that unbilled SKUs are omitted rather than
// declared at zero.
func TestNoZeroStuffing(t *testing.T) {
	doc := render(t, base()) // cached_prompt and cache_write are unset

	in := doc["input_modalities"].([]any)[0].(map[string]any)
	prices := in["pricing"].([]any)
	if len(prices) != 1 {
		t.Fatalf("got %d input prices, want only the prompt SKU: %v", len(prices), prices)
	}
	if prices[0].(map[string]any)["type"] != "prompt" {
		t.Errorf("unexpected input SKU: %v", prices[0])
	}

	out := doc["output_modalities"].([]any)[0].(map[string]any)
	outPrices := out["pricing"].([]any)
	if len(outPrices) != 1 {
		t.Fatalf("got %d output prices, want only completion: %v", len(outPrices), outPrices)
	}
}

func TestPricesLandOnTheOwningModality(t *testing.T) {
	m := base()
	m.Pricing.CachedPrompt = "0.00000002"
	m.Pricing.CacheWrite = "0.0000001"
	m.Pricing.InternalReasoning = "0.0000005"
	doc := render(t, m)

	inTypes := map[string]bool{}
	for _, p := range doc["input_modalities"].([]any)[0].(map[string]any)["pricing"].([]any) {
		inTypes[p.(map[string]any)["type"].(string)] = true
	}
	for _, want := range []string{"prompt", "cached_prompt", "cache_write"} {
		if !inTypes[want] {
			t.Errorf("input modality is missing the %q SKU", want)
		}
	}
	// Cache pricing is input-side only; it must never appear on an output.
	outTypes := map[string]bool{}
	for _, p := range doc["output_modalities"].([]any)[0].(map[string]any)["pricing"].([]any) {
		outTypes[p.(map[string]any)["type"].(string)] = true
	}
	for _, forbidden := range []string{"cached_prompt", "cache_write", "prompt"} {
		if outTypes[forbidden] {
			t.Errorf("%q must not appear on an output modality", forbidden)
		}
	}
	if !outTypes["completion"] || !outTypes["internal_reasoning"] {
		t.Errorf("output modality is missing expected SKUs: %v", outTypes)
	}
	// The root carries no per-token pricing at all.
	if p, ok := doc["pricing"]; ok && p != nil {
		t.Errorf("root pricing should be absent for a plain text model, got %v", p)
	}
}

func TestCapacityScoping(t *testing.T) {
	m := base()
	m.Capacity = config.Capacity{
		PromptTPM: 1000, CachedPromptTPM: 2000, CompletionTPM: 500,
		RequestsPerMin: 100, Concurrency: 8,
	}
	doc := render(t, m)

	root := doc["capacity"].([]any)
	if len(root) != 1 {
		t.Fatalf("root capacity = %v, want only the request entry", root)
	}
	if root[0].(map[string]any)["type"] != "request" {
		t.Errorf("root capacity must be request-scoped, got %v", root[0])
	}

	out := doc["output_modalities"].([]any)[0].(map[string]any)["capacity"].([]any)
	for _, c := range out {
		e := c.(map[string]any)
		// concurrency has no window; the spec rejects a `per` on it.
		if e["type"] == "concurrency" {
			if _, hasPer := e["per"]; hasPer {
				t.Errorf("concurrency entry must not carry a per window: %v", e)
			}
		}
	}
}

func TestFeatureFlagsGateParameters(t *testing.T) {
	params := func(m *config.Model) map[string]any {
		doc := render(t, m)
		return doc["output_modalities"].([]any)[0].(map[string]any)["supported_parameters"].(map[string]any)
	}
	// Absent means unsupported, per the descriptor grammar.
	if p := params(base()); p["tools"] != nil || p["reasoning"] != nil {
		t.Errorf("unset features should be absent, got tools=%v reasoning=%v", p["tools"], p["reasoning"])
	}

	m := base()
	m.Features = config.Features{Tools: true, Reasoning: true, StructuredOutputs: true}
	p := params(m)
	for _, want := range []string{"tools", "tool_choice", "reasoning", "structured_outputs"} {
		if p[want] == nil {
			t.Errorf("%q missing when the feature is enabled", want)
		}
	}
	// max_tokens is bounded by the model's own output limit.
	mt := p["max_tokens"].(map[string]any)
	if mt["max"].(float64) != 32768 {
		t.Errorf("max_tokens max = %v, want 32768", mt["max"])
	}
}

func TestFreeModelOmitsPricing(t *testing.T) {
	m := base()
	m.IsFree = true
	doc := render(t, m)
	in := doc["input_modalities"].([]any)[0].(map[string]any)
	if p, ok := in["pricing"]; ok && p != nil {
		t.Errorf("a free model should not declare pricing, got %v", p)
	}
	if doc["is_free"] != true {
		t.Error("is_free was not set")
	}
}

func TestVisionAddsImageModality(t *testing.T) {
	m := base()
	m.Vision = &config.Vision{MaxImageBytes: 20971520}
	m.Pricing.ImagePrompt = "0.0048"
	doc := render(t, m)

	mods := doc["input_modalities"].([]any)
	if len(mods) != 2 {
		t.Fatalf("got %d input modalities, want text plus image", len(mods))
	}
	img := mods[1].(map[string]any)
	if img["type"] != "image" {
		t.Fatalf("second modality = %v, want image", img["type"])
	}
	price := img["pricing"].([]any)[0].(map[string]any)
	if price["unit"] != "image" {
		t.Errorf("image price unit = %v, want image", price["unit"])
	}
}

func TestBuildAllModels(t *testing.T) {
	cfg := &config.Config{Models: []config.Model{*base(), *base()}}
	cfg.Models[1].ID = "other/model"
	if got := len(Build(cfg).Data); got != 2 {
		t.Errorf("rendered %d documents, want 2", got)
	}
}
