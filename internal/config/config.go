// Package config loads the provider, backend and model catalog from YAML.
//
// The model entries here are the single source of truth for three things:
// what we send upstream to vLLM, what OpenRouter sees at /v1/models, and how
// requests are billed and rate limited. Keeping them in one file means a new
// model is a config change and a SIGHUP, not a deploy.
package config

import (
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

type Config struct {
	Provider Provider  `yaml:"provider"`
	Server   Server    `yaml:"server"`
	Backends []Backend `yaml:"backends"`
	Nodes    []Node    `yaml:"nodes"`
	Models   []Model   `yaml:"models"`
}

// Node is an inference host running llmfast-agent. The gateway asks it what
// hardware it has and tells it which models to serve; the engines it starts
// become routable backends automatically.
type Node struct {
	Name string `yaml:"name"`
	URL  string `yaml:"url"`
	// Token authenticates to the agent. A leading $ reads it from that
	// environment variable instead, so it need not sit in the file.
	Token string `yaml:"token"`
	// MaxConcurrency is the admission limit applied to each engine this node
	// runs. It should not exceed the engine's own --max-num-seqs.
	MaxConcurrency int `yaml:"max_concurrency"`
	Weight         int `yaml:"weight"`
}

type Provider struct {
	Slug        string `yaml:"slug"`
	DisplayName string `yaml:"display_name"`
	PublicURL   string `yaml:"public_url"`
}

type Server struct {
	Listen      string `yaml:"listen"`
	AdminListen string `yaml:"admin_listen"`
	// AdminToken gates the admin API and UI. A leading $ reads it from that
	// environment variable, so the secret need not sit in the file. Left empty,
	// it falls back to LLMFAST_ADMIN_TOKEN.
	AdminToken string `yaml:"admin_token"`
	DBPath     string `yaml:"db_path"`
	// ReadTimeout bounds how long a client may take to send its request body.
	// Deliberately not applied to responses: streams run for minutes.
	ReadTimeout time.Duration `yaml:"read_timeout"`
	// KeepAliveInterval is how often we emit an SSE comment while waiting on a
	// slow upstream. OpenRouter cancels silent streams, so this must stay well
	// under their fetch timeout.
	KeepAliveInterval time.Duration `yaml:"keepalive_interval"`
	// RawRetentionDays bounds the per-request log. Rollups are kept forever.
	RawRetentionDays int `yaml:"raw_retention_days"`
	// ModelDir holds one YAML file per model added through the admin UI.
	// Keeping them out of the main file means installing a model never has to
	// rewrite config.yaml and discard its comments.
	ModelDir string `yaml:"model_dir"`
}

type Backend struct {
	Name    string `yaml:"name"`
	BaseURL string `yaml:"base_url"`
	APIKey  string `yaml:"api_key"`
	// MaxConcurrency is the number of in-flight requests we allow before
	// returning 429. We never queue: OpenRouter measures queueing as our
	// throughput, so shedding load beats absorbing it.
	MaxConcurrency int           `yaml:"max_concurrency"`
	Timeout        time.Duration `yaml:"timeout"`
	// Weight biases least-loaded selection toward bigger nodes.
	Weight int `yaml:"weight"`
}

type Model struct {
	ID            string   `yaml:"id"`
	Name          string   `yaml:"name"`
	UpstreamModel string   `yaml:"upstream_model"`
	Backends      []string `yaml:"backends"`

	HuggingFaceID string `yaml:"hugging_face_id,omitempty"`
	Created       string `yaml:"created,omitempty"`
	Quantization  string `yaml:"quantization,omitempty"`
	Tokenizer     string `yaml:"tokenizer,omitempty"`
	Description   string `yaml:"description,omitempty"`

	ContextLength   int `yaml:"context_length"`
	MaxOutputTokens int `yaml:"max_output_tokens,omitempty"`

	Pricing  Pricing  `yaml:"pricing,omitempty"`
	Capacity Capacity `yaml:"capacity,omitempty"`
	Features Features `yaml:"features,omitempty"`
	Vision   *Vision  `yaml:"vision,omitempty"`

	Datacenters []Datacenter `yaml:"datacenters,omitempty"`
	Compliance  Compliance   `yaml:"compliance,omitempty"`

	IsReady         *bool   `yaml:"is_ready"`
	IsFree          bool    `yaml:"is_free,omitempty"`
	DiscountToUser  float64 `yaml:"discount_to_user,omitempty"`
	DeprecationDate string  `yaml:"deprecation_date,omitempty"`
	OpenRouterSlug  string  `yaml:"openrouter_slug,omitempty"`
}

// Pricing values are USD per single token, as decimal strings. OpenRouter
// requires strings so nobody round-trips these through a float64.
type Pricing struct {
	Prompt            string `yaml:"prompt,omitempty"`
	Completion        string `yaml:"completion,omitempty"`
	CachedPrompt      string `yaml:"cached_prompt,omitempty"`
	CacheWrite        string `yaml:"cache_write,omitempty"`
	InternalReasoning string `yaml:"internal_reasoning,omitempty"`
	ImagePrompt       string `yaml:"image_prompt,omitempty"`
}

type Capacity struct {
	PromptTPM       int `yaml:"prompt_tpm,omitempty"`
	CachedPromptTPM int `yaml:"cached_prompt_tpm,omitempty"`
	CompletionTPM   int `yaml:"completion_tpm,omitempty"`
	RequestsPerMin  int `yaml:"requests_per_minute,omitempty"`
	Concurrency     int `yaml:"concurrency,omitempty"`
}

type Features struct {
	Tools             bool `yaml:"tools,omitempty"`
	StructuredOutputs bool `yaml:"structured_outputs,omitempty"`
	Reasoning         bool `yaml:"reasoning,omitempty"`
	ResponseFormat    bool `yaml:"response_format,omitempty"`
	Logprobs          bool `yaml:"logprobs,omitempty"`
	Seed              bool `yaml:"seed,omitempty"`
}

type Vision struct {
	MaxImageBytes int      `yaml:"max_image_bytes"`
	Formats       []string `yaml:"formats"`
}

type Datacenter struct {
	CountryCode string `yaml:"country_code"`
	Region      string `yaml:"region"`
}

type Compliance struct {
	ZDR   bool `yaml:"zdr,omitempty"`
	HIPAA bool `yaml:"hipaa,omitempty"`
}

// Load reads, defaults and validates the config file.
func Load(path string) (*Config, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var c Config
	dec := yaml.NewDecoder(strings.NewReader(string(raw)))
	dec.KnownFields(true)
	if err := dec.Decode(&c); err != nil {
		return nil, fmt.Errorf("parse %s: %w", path, err)
	}
	c.applyDefaults()
	if err := c.loadModelDir(filepath.Dir(path)); err != nil {
		return nil, err
	}
	c.applyDefaults() // models loaded from the directory need defaults too
	return &c, c.Validate()
}

// loadModelDir merges in one-model-per-file YAML from the model directory.
//
// Models added through the admin UI land here rather than in config.yaml, so
// the main file keeps its structure and comments and an installed model can be
// removed by deleting a single file.
func (c *Config) loadModelDir(base string) error {
	dir := c.Server.ModelDir
	if dir == "" {
		return nil
	}
	if !filepath.IsAbs(dir) {
		dir = filepath.Join(base, dir)
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil // nothing installed yet
		}
		return fmt.Errorf("read model dir %s: %w", dir, err)
	}
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if !strings.HasSuffix(name, ".yaml") && !strings.HasSuffix(name, ".yml") {
			continue
		}
		raw, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			return fmt.Errorf("read %s: %w", name, err)
		}
		var m Model
		dec := yaml.NewDecoder(strings.NewReader(string(raw)))
		dec.KnownFields(true)
		if err := dec.Decode(&m); err != nil {
			return fmt.Errorf("parse %s: %w", name, err)
		}
		c.Models = append(c.Models, m)
	}
	return nil
}

// ModelDirPath resolves the model directory against the config file's location.
func (c *Config) ModelDirPath(configPath string) string {
	dir := c.Server.ModelDir
	if dir == "" {
		return ""
	}
	if filepath.IsAbs(dir) {
		return dir
	}
	return filepath.Join(filepath.Dir(configPath), dir)
}

func (c *Config) applyDefaults() {
	if c.Server.Listen == "" {
		c.Server.Listen = ":8080"
	}
	if c.Server.AdminListen == "" {
		c.Server.AdminListen = ":8081"
	}
	if c.Server.DBPath == "" {
		c.Server.DBPath = "llmfast.db"
	}
	if c.Server.ReadTimeout == 0 {
		c.Server.ReadTimeout = 60 * time.Second
	}
	if c.Server.KeepAliveInterval == 0 {
		c.Server.KeepAliveInterval = 10 * time.Second
	}
	if c.Server.RawRetentionDays == 0 {
		c.Server.RawRetentionDays = 30
	}
	c.Server.AdminToken = resolveSecret(c.Server.AdminToken)
	if c.Server.AdminToken == "" {
		c.Server.AdminToken = os.Getenv("LLMFAST_ADMIN_TOKEN")
	}
	for i := range c.Nodes {
		n := &c.Nodes[i]
		if n.MaxConcurrency <= 0 {
			n.MaxConcurrency = 64
		}
		if n.Weight <= 0 {
			n.Weight = 1
		}
		n.Token = resolveSecret(n.Token)
	}
	for i := range c.Backends {
		b := &c.Backends[i]
		if b.MaxConcurrency <= 0 {
			b.MaxConcurrency = 64
		}
		if b.Timeout == 0 {
			b.Timeout = 10 * time.Minute
		}
		if b.Weight <= 0 {
			b.Weight = 1
		}
		b.APIKey = resolveSecret(b.APIKey)
	}
	for i := range c.Models {
		m := &c.Models[i]
		if m.UpstreamModel == "" {
			m.UpstreamModel = m.ID
		}
		if m.Name == "" {
			m.Name = m.ID
		}
		if m.MaxOutputTokens == 0 {
			m.MaxOutputTokens = m.ContextLength
		}
	}
}

// resolveSecret expands a "$NAME" value from the environment, so credentials
// can stay out of the config file.
//
// Every field that accepts a secret goes through this. Applying the convention
// to only some of them is worse than not having it: a field that silently keeps
// the literal string "$LLMFAST_ADMIN_TOKEN" as its value looks configured, and
// the only symptom is a login that rejects the right password.
func resolveSecret(v string) string {
	if !strings.HasPrefix(v, "$") {
		return v
	}
	name := strings.TrimPrefix(v, "$")
	// Tolerate ${NAME} as well; people write it both ways.
	name = strings.TrimPrefix(strings.TrimSuffix(name, "}"), "{")
	return os.Getenv(name)
}

func (c *Config) Validate() error {
	if c.Provider.Slug == "" {
		return fmt.Errorf("provider.slug is required")
	}
	if len(c.Backends) == 0 && len(c.Nodes) == 0 {
		return fmt.Errorf("configure at least one backend or node")
	}
	nodeNames := map[string]bool{}
	for _, n := range c.Nodes {
		if n.Name == "" {
			return fmt.Errorf("node name is required")
		}
		if nodeNames[n.Name] {
			return fmt.Errorf("duplicate node %q", n.Name)
		}
		u, err := url.Parse(n.URL)
		if err != nil || u.Scheme == "" || u.Host == "" {
			return fmt.Errorf("node %q: url must be absolute, got %q", n.Name, n.URL)
		}
		nodeNames[n.Name] = true
	}

	known := map[string]bool{}
	for _, b := range c.Backends {
		if b.Name == "" {
			return fmt.Errorf("backend name is required")
		}
		if known[b.Name] {
			return fmt.Errorf("duplicate backend %q", b.Name)
		}
		u, err := url.Parse(b.BaseURL)
		if err != nil || u.Scheme == "" || u.Host == "" {
			return fmt.Errorf("backend %q: base_url must be an absolute URL, got %q", b.Name, b.BaseURL)
		}
		known[b.Name] = true
	}
	seen := map[string]bool{}
	for _, m := range c.Models {
		if m.ID == "" {
			return fmt.Errorf("model id is required")
		}
		if seen[m.ID] {
			return fmt.Errorf("duplicate model %q", m.ID)
		}
		seen[m.ID] = true
		if m.ContextLength <= 0 {
			return fmt.Errorf("model %q: context_length must be > 0", m.ID)
		}
		if len(m.Backends) == 0 {
			return fmt.Errorf("model %q: at least one backend is required", m.ID)
		}
		for _, name := range m.Backends {
			// A model routes to statically configured backends, to agent-managed
			// nodes, or to both.
			if !known[name] && !nodeNames[name] {
				return fmt.Errorf("model %q references unknown backend or node %q", m.ID, name)
			}
		}
		if m.DiscountToUser >= 1 {
			return fmt.Errorf("model %q: discount_to_user must be < 1", m.ID)
		}
	}
	return nil
}

// Model returns the catalog entry for an id.
func (c *Config) Model(id string) (*Model, bool) {
	for i := range c.Models {
		if c.Models[i].ID == id {
			return &c.Models[i], true
		}
	}
	return nil, false
}
