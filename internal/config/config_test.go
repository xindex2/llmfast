package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func write(t *testing.T, body string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "c.yaml")
	if err := os.WriteFile(p, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	return p
}

const minimal = `
provider:
  slug: llmfast
backends:
  - name: b1
    base_url: http://localhost:8000/v1
models:
  - id: a/b
    backends: [b1]
    context_length: 1000
`

func TestLoadDefaults(t *testing.T) {
	cfg, err := Load(write(t, minimal))
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if cfg.Server.Listen != ":8080" || cfg.Server.AdminListen != ":8081" {
		t.Errorf("listen defaults not applied: %+v", cfg.Server)
	}
	if cfg.Backends[0].MaxConcurrency != 64 || cfg.Backends[0].Weight != 1 {
		t.Errorf("backend defaults not applied: %+v", cfg.Backends[0])
	}
	m := cfg.Models[0]
	// upstream_model defaults to the public id, which is the right default when
	// vLLM is started with --served-model-name matching.
	if m.UpstreamModel != "a/b" {
		t.Errorf("upstream_model = %q, want the model id", m.UpstreamModel)
	}
	if m.Name != "a/b" {
		t.Errorf("name = %q, want the model id as fallback", m.Name)
	}
	if m.MaxOutputTokens != 1000 {
		t.Errorf("max_output_tokens = %d, want the context length as fallback", m.MaxOutputTokens)
	}
}

func TestValidationRejects(t *testing.T) {
	cases := map[string]string{
		"missing slug": `
backends: [{name: b1, base_url: "http://x/v1"}]
models: [{id: a/b, backends: [b1], context_length: 10}]`,

		"unknown backend reference": `
provider: {slug: s}
backends: [{name: b1, base_url: "http://x/v1"}]
models: [{id: a/b, backends: [nope], context_length: 10}]`,

		"duplicate model id": `
provider: {slug: s}
backends: [{name: b1, base_url: "http://x/v1"}]
models:
  - {id: a/b, backends: [b1], context_length: 10}
  - {id: a/b, backends: [b1], context_length: 10}`,

		"relative backend url": `
provider: {slug: s}
backends: [{name: b1, base_url: "/v1"}]
models: [{id: a/b, backends: [b1], context_length: 10}]`,

		"zero context length": `
provider: {slug: s}
backends: [{name: b1, base_url: "http://x/v1"}]
models: [{id: a/b, backends: [b1], context_length: 0}]`,

		"discount of 1 would make it free": `
provider: {slug: s}
backends: [{name: b1, base_url: "http://x/v1"}]
models: [{id: a/b, backends: [b1], context_length: 10, discount_to_user: 1.0}]`,

		"no backends at all": `
provider: {slug: s}
models: [{id: a/b, backends: [b1], context_length: 10}]`,
	}
	for name, body := range cases {
		t.Run(name, func(t *testing.T) {
			if _, err := Load(write(t, body)); err == nil {
				t.Error("expected a validation error, got none")
			}
		})
	}
}

// TestUnknownFieldRejected catches typos in config keys at boot rather than
// silently ignoring them, which would otherwise mean shipping a model with, say,
// pricing that never took effect.
func TestUnknownFieldRejected(t *testing.T) {
	body := minimal + "\n  " // keep valid, then add a bogus top-level key
	body += "\nnonsense_key: 1\n"
	if _, err := Load(write(t, body)); err == nil {
		t.Error("an unknown config key was silently accepted")
	}
}

func TestAPIKeyFromEnv(t *testing.T) {
	t.Setenv("TEST_BACKEND_KEY", "secret-value")
	body := `
provider: {slug: s}
backends: [{name: b1, base_url: "http://x/v1", api_key: "$TEST_BACKEND_KEY"}]
models: [{id: a/b, backends: [b1], context_length: 10}]`
	cfg, err := Load(write(t, body))
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if cfg.Backends[0].APIKey != "secret-value" {
		t.Errorf("api_key = %q, want the resolved env value", cfg.Backends[0].APIKey)
	}
}

func TestModelLookup(t *testing.T) {
	cfg, _ := Load(write(t, minimal))
	if _, ok := cfg.Model("a/b"); !ok {
		t.Error("known model not found")
	}
	if _, ok := cfg.Model("missing"); ok {
		t.Error("unknown model was found")
	}
}

func TestModelFileName(t *testing.T) {
	cases := map[string]string{
		"qwen/qwen3-32b":         "qwen_qwen3-32b.yaml",
		"Qwen/Qwen3-32B":         "qwen_qwen3-32b.yaml",
		"deepseek/deepseek-v3.1": "deepseek_deepseek-v3.1.yaml",
		"../../etc/passwd":       ".._.._etc_passwd.yaml",
		"a b/c":                  "a_b_c.yaml",
	}
	for id, want := range cases {
		if got := ModelFileName(id); got != want {
			t.Errorf("ModelFileName(%q) = %q, want %q", id, got, want)
		}
	}
}

// TestModelFileNameContainsNoSeparator is the security-relevant property: a
// model id is attacker-influenced in the sense that it comes from a form field,
// and it must never be able to escape the model directory.
func TestModelFileNameContainsNoSeparator(t *testing.T) {
	for _, id := range []string{"../../etc/passwd", "a/../../b", "/absolute/path", "x\\y"} {
		got := ModelFileName(id)
		if strings.ContainsAny(got, `/\`) {
			t.Errorf("ModelFileName(%q) = %q, which contains a path separator", id, got)
		}
	}
}

func TestWriteAndLoadModelDir(t *testing.T) {
	dir := t.TempDir()
	modelDir := filepath.Join(dir, "models.d")

	m := Model{
		ID: "qwen/qwen3-8b", Name: "Qwen3 8B", UpstreamModel: "qwen/qwen3-8b",
		Backends: []string{"gpu-a"}, ContextLength: 32768, MaxOutputTokens: 8192,
		Pricing:  Pricing{Prompt: "0.00000002", Completion: "0.00000006"},
		Features: Features{Tools: true},
	}
	path, err := WriteModel(modelDir, m)
	if err != nil {
		t.Fatalf("write: %v", err)
	}
	if filepath.Base(path) != "qwen_qwen3-8b.yaml" {
		t.Errorf("wrote %q, unexpected filename", path)
	}

	cfgPath := filepath.Join(dir, "config.yaml")
	if err := os.WriteFile(cfgPath, []byte(`
provider: {slug: s}
server: {model_dir: models.d}
nodes: [{name: gpu-a, url: "http://x:9900"}]
models: []
`), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg, err := Load(cfgPath)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	got, ok := cfg.Model("qwen/qwen3-8b")
	if !ok {
		t.Fatal("model from the model dir was not loaded")
	}
	if got.ContextLength != 32768 || !got.Features.Tools {
		t.Errorf("round trip lost data: %+v", got)
	}

	// Removing the file removes it from the catalog.
	if err := RemoveModel(modelDir, "qwen/qwen3-8b"); err != nil {
		t.Fatalf("remove: %v", err)
	}
	cfg2, err := Load(cfgPath)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := cfg2.Model("qwen/qwen3-8b"); ok {
		t.Error("model still present after its file was removed")
	}
}

// TestModelDirDoesNotBreakOnMissingDirectory: a fresh install has no models.d
// yet, and that must not be a startup error.
func TestModelDirDoesNotBreakOnMissingDirectory(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config.yaml")
	os.WriteFile(cfgPath, []byte(`
provider: {slug: s}
server: {model_dir: models.d}
backends: [{name: b1, base_url: "http://x/v1"}]
models: [{id: a/b, backends: [b1], context_length: 10}]
`), 0o600)
	if _, err := Load(cfgPath); err != nil {
		t.Errorf("a missing model dir should not be an error: %v", err)
	}
}

func TestModelReferencingNodeIsValid(t *testing.T) {
	body := `
provider: {slug: s}
nodes: [{name: gpu-a, url: "http://10.0.0.1:9900"}]
models: [{id: a/b, backends: [gpu-a], context_length: 100}]`
	if _, err := Load(write(t, body)); err != nil {
		t.Errorf("a model routing to a node should validate: %v", err)
	}
}
