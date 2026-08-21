// Package modeldoc renders the OpenRouter provider model document (schema 2.4).
//
// The shape is deliberately close to the spec rather than to our config: every
// price and capacity entry hangs off the modality that owns it, and only
// request-scoped entries stay at the document root. Anything we do not bill is
// omitted entirely -- the spec explicitly rejects zero-stuffing.
package modeldoc

import (
	"time"

	"github.com/llmfast/gateway/internal/config"
)

const SchemaVersion = "2.4"

type Response struct {
	Data []Document `json:"data"`
}

type Document struct {
	SchemaVersion string `json:"schema_version"`

	ID            string  `json:"id"`
	Name          string  `json:"name"`
	HuggingFaceID string  `json:"hugging_face_id,omitempty"`
	Created       int64   `json:"created,omitempty"`
	Quantization  *string `json:"quantization"`
	Tokenizer     string  `json:"tokenizer,omitempty"`
	Description   string  `json:"description,omitempty"`

	InputModalities  []InputModality  `json:"input_modalities"`
	OutputModalities []OutputModality `json:"output_modalities"`

	Pricing  []Price    `json:"pricing,omitempty"`
	Capacity []Capacity `json:"capacity,omitempty"`

	PassthroughParameters map[string]Descriptor `json:"passthrough_parameters,omitempty"`

	DeprecationDate  string      `json:"deprecation_date,omitempty"`
	IsReady          *bool       `json:"is_ready,omitempty"`
	IsFree           bool        `json:"is_free"`
	DiscountToUser   float64     `json:"discount_to_user,omitempty"`
	OpenRouter       *ORRef      `json:"openrouter,omitempty"`
	Datacenters      []DC        `json:"datacenters,omitempty"`
	DeploymentRegion string      `json:"deployment_region,omitempty"`
	Compliance       *Compliance `json:"compliance,omitempty"`
}

type ORRef struct {
	Slug string `json:"slug"`
}

type DC struct {
	CountryCode string `json:"country_code"`
	Region      string `json:"region,omitempty"`
}

type Compliance struct {
	ZDR   bool `json:"zdr"`
	HIPAA bool `json:"hipaa"`
}

type InputModality struct {
	Type            string                `json:"type"`
	SupportedInputs map[string]any        `json:"supported_inputs,omitempty"`
	Pricing         []Price               `json:"pricing,omitempty"`
	Capacity        []Capacity            `json:"capacity,omitempty"`
	Passthrough     map[string]Descriptor `json:"passthrough_parameters,omitempty"`
}

type OutputModality struct {
	Type                string                `json:"type"`
	MaxLength           *Measure              `json:"max_length,omitempty"`
	Streaming           bool                  `json:"streaming"`
	SupportedParameters map[string]Descriptor `json:"supported_parameters"`
	Pricing             []Price               `json:"pricing,omitempty"`
	Capacity            []Capacity            `json:"capacity,omitempty"`
	Passthrough         map[string]Descriptor `json:"passthrough_parameters,omitempty"`
}

// Measure is the spec's single-value limit object.
type Measure struct {
	Value int64  `json:"value"`
	Unit  string `json:"unit,omitempty"`
}

// Price is one billable SKU. TTLSeconds and the UTC window are qualifier
// fields: two entries of the same type with different qualifiers are distinct
// SKUs, which is why they live here rather than in the type string.
type Price struct {
	Type       string     `json:"type"`
	Unit       string     `json:"unit"`
	CostUSD    string     `json:"cost_usd"`
	TTLSeconds int        `json:"ttl_seconds,omitempty"`
	Implicit   bool       `json:"implicit,omitempty"`
	UTCStart   *int       `json:"utc_start,omitempty"`
	UTCEnd     *int       `json:"utc_end,omitempty"`
	Overrides  []Override `json:"overrides,omitempty"`
}

type Override struct {
	When    map[string]any `json:"when"`
	CostUSD string         `json:"cost_usd"`
}

// Capacity is a declared throughput limit. Per is omitted for concurrency,
// which has no window.
type Capacity struct {
	Type  string `json:"type"`
	Unit  string `json:"unit"`
	Per   string `json:"per,omitempty"`
	Value int    `json:"value"`
}

// Descriptor is the spec's capability grammar: range, integer, boolean, enum,
// array, object or unknown.
type Descriptor struct {
	Type     string   `json:"type"`
	Min      *float64 `json:"min,omitempty"`
	Max      *float64 `json:"max,omitempty"`
	Unit     string   `json:"unit,omitempty"`
	Values   []any    `json:"values,omitempty"`
	MaxItems *int     `json:"max_items,omitempty"`
	Default  any      `json:"default,omitempty"`
}

func f(v float64) *float64 { return &v }
func i(v int) *int         { return &v }

func rangeDesc(min, max float64) Descriptor {
	return Descriptor{Type: "range", Min: f(min), Max: f(max)}
}

func intDesc(min, max int, unit string) Descriptor {
	return Descriptor{Type: "integer", Min: f(float64(min)), Max: f(float64(max)), Unit: unit}
}

func boolDesc() Descriptor { return Descriptor{Type: "boolean"} }

// Build renders every configured model. Models flagged not-ready are still
// emitted: is_ready false is how OpenRouter is told to keep them hidden.
func Build(cfg *config.Config) Response {
	out := Response{Data: make([]Document, 0, len(cfg.Models))}
	for i := range cfg.Models {
		out.Data = append(out.Data, BuildOne(&cfg.Models[i]))
	}
	return out
}

func BuildOne(m *config.Model) Document {
	doc := Document{
		SchemaVersion:   SchemaVersion,
		ID:              m.ID,
		Name:            m.Name,
		HuggingFaceID:   m.HuggingFaceID,
		Created:         parseCreated(m.Created),
		Tokenizer:       m.Tokenizer,
		Description:     m.Description,
		IsReady:         m.IsReady,
		IsFree:          m.IsFree,
		DiscountToUser:  m.DiscountToUser,
		DeprecationDate: m.DeprecationDate,
	}
	// The spec wants an explicit null when precision is undeclared, so this
	// field is a pointer rather than omitempty.
	if m.Quantization != "" {
		q := m.Quantization
		doc.Quantization = &q
	}
	if m.OpenRouterSlug != "" {
		doc.OpenRouter = &ORRef{Slug: m.OpenRouterSlug}
	}
	for _, dc := range m.Datacenters {
		doc.Datacenters = append(doc.Datacenters, DC{CountryCode: dc.CountryCode, Region: dc.Region})
	}
	doc.Compliance = &Compliance{ZDR: m.Compliance.ZDR, HIPAA: m.Compliance.HIPAA}

	doc.InputModalities = buildInputs(m)
	doc.OutputModalities = []OutputModality{buildTextOutput(m)}

	if m.Capacity.RequestsPerMin > 0 {
		doc.Capacity = append(doc.Capacity, Capacity{
			Type: "request", Unit: "request", Per: "minute", Value: m.Capacity.RequestsPerMin,
		})
	}
	return doc
}

func buildInputs(m *config.Model) []InputModality {
	text := InputModality{
		Type: "text",
		SupportedInputs: map[string]any{
			"max_context_length": Measure{Value: int64(m.ContextLength), Unit: "token"},
		},
	}
	// Free endpoints ignore pricing entirely, so emitting it would be noise.
	if !m.IsFree {
		text.Pricing = appendPrice(text.Pricing, "prompt", "token", m.Pricing.Prompt)
		text.Pricing = appendPrice(text.Pricing, "cached_prompt", "token", m.Pricing.CachedPrompt)
		text.Pricing = appendPrice(text.Pricing, "cache_write", "token", m.Pricing.CacheWrite)
	}
	if m.Capacity.PromptTPM > 0 {
		text.Capacity = append(text.Capacity, Capacity{Type: "prompt", Unit: "token", Per: "minute", Value: m.Capacity.PromptTPM})
	}
	if m.Capacity.CachedPromptTPM > 0 {
		text.Capacity = append(text.Capacity, Capacity{Type: "cached_prompt", Unit: "token", Per: "minute", Value: m.Capacity.CachedPromptTPM})
	}
	inputs := []InputModality{text}

	if m.Vision != nil {
		formats := m.Vision.Formats
		if len(formats) == 0 {
			formats = []string{"image/png", "image/jpeg", "image/webp"}
		}
		si := map[string]any{
			"sources": Descriptor{Type: "enum", Values: []any{"url", "base64"}},
			"formats": Descriptor{Type: "enum", Values: toAny(formats)},
		}
		if m.Vision.MaxImageBytes > 0 {
			si["max_content_size_bytes"] = Measure{Value: int64(m.Vision.MaxImageBytes), Unit: "byte"}
		}
		img := InputModality{Type: "image", SupportedInputs: si}
		if !m.IsFree {
			img.Pricing = appendPrice(img.Pricing, "prompt", "image", m.Pricing.ImagePrompt)
		}
		inputs = append(inputs, img)
	}
	return inputs
}

func buildTextOutput(m *config.Model) OutputModality {
	params := map[string]Descriptor{
		"temperature":        rangeDesc(0, 2),
		"top_p":              rangeDesc(0, 1),
		"top_k":              intDesc(0, 1000, ""),
		"min_p":              rangeDesc(0, 1),
		"frequency_penalty":  rangeDesc(-2, 2),
		"presence_penalty":   rangeDesc(-2, 2),
		"repetition_penalty": rangeDesc(0.1, 2),
		"max_tokens":         intDesc(1, m.MaxOutputTokens, "token"),
		"stop":               {Type: "array", MaxItems: i(4)},
	}
	if m.Features.Tools {
		params["tools"] = boolDesc()
		params["tool_choice"] = Descriptor{Type: "enum", Values: []any{"none", "auto", "required"}}
	}
	if m.Features.StructuredOutputs {
		params["structured_outputs"] = boolDesc()
	}
	if m.Features.ResponseFormat {
		params["response_format"] = Descriptor{Type: "object"}
	}
	if m.Features.Reasoning {
		params["reasoning"] = boolDesc()
	}
	if m.Features.Seed {
		params["seed"] = intDesc(0, 2147483647, "")
	}
	if m.Features.Logprobs {
		params["logprobs"] = boolDesc()
		params["top_logprobs"] = intDesc(0, 20, "")
	}

	out := OutputModality{
		Type:                "text",
		MaxLength:           &Measure{Value: int64(m.MaxOutputTokens), Unit: "token"},
		Streaming:           true,
		SupportedParameters: params,
	}
	if !m.IsFree {
		out.Pricing = appendPrice(out.Pricing, "completion", "token", m.Pricing.Completion)
		out.Pricing = appendPrice(out.Pricing, "internal_reasoning", "token", m.Pricing.InternalReasoning)
	}
	if m.Capacity.CompletionTPM > 0 {
		out.Capacity = append(out.Capacity, Capacity{Type: "completion", Unit: "token", Per: "minute", Value: m.Capacity.CompletionTPM})
	}
	if m.Capacity.Concurrency > 0 {
		out.Capacity = append(out.Capacity, Capacity{Type: "concurrency", Unit: "request", Value: m.Capacity.Concurrency})
	}
	return out
}

// appendPrice skips unset SKUs. An explicit "0" is preserved: the spec allows a
// genuinely free line, it only forbids inventing zero prices for SKUs we do not
// bill at all.
func appendPrice(dst []Price, typ, unit, cost string) []Price {
	if cost == "" {
		return dst
	}
	return append(dst, Price{Type: typ, Unit: unit, CostUSD: cost})
}

func toAny(ss []string) []any {
	out := make([]any, len(ss))
	for i, s := range ss {
		out[i] = s
	}
	return out
}

func parseCreated(s string) int64 {
	if s == "" {
		return 0
	}
	for _, layout := range []string{time.RFC3339, "2006-01-02"} {
		if t, err := time.Parse(layout, s); err == nil {
			return t.Unix()
		}
	}
	return 0
}
