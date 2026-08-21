package config

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"gopkg.in/yaml.v3"
)

// WriteModel persists one model as its own file in the model directory.
//
// Writing a separate file rather than editing config.yaml means installing a
// model never reformats the operator's hand-written configuration or discards
// its comments, and uninstalling is a single file deletion.
func WriteModel(dir string, m Model) (string, error) {
	if dir == "" {
		return "", fmt.Errorf("server.model_dir is not configured, so models cannot be added from the admin UI")
	}
	if m.ID == "" {
		return "", fmt.Errorf("model id is required")
	}
	if err := os.MkdirAll(dir, 0o750); err != nil {
		return "", fmt.Errorf("create model dir: %w", err)
	}
	path := filepath.Join(dir, ModelFileName(m.ID))

	var buf strings.Builder
	buf.WriteString("# Managed by the LLMFast admin UI.\n")
	buf.WriteString("# Edit freely; it is plain config and is reloaded on SIGHUP.\n")
	buf.WriteString("# Delete this file to remove the model from the catalog.\n\n")

	enc := yaml.NewEncoder(&buf)
	enc.SetIndent(2)
	if err := enc.Encode(m); err != nil {
		return "", fmt.Errorf("encode model: %w", err)
	}
	if err := enc.Close(); err != nil {
		return "", err
	}

	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, []byte(buf.String()), 0o640); err != nil {
		return "", fmt.Errorf("write %s: %w", path, err)
	}
	// Rename is atomic, so a reload racing this write sees either the old file
	// or the complete new one, never a truncated document that would fail to
	// parse and take the whole catalog down.
	if err := os.Rename(tmp, path); err != nil {
		return "", fmt.Errorf("install %s: %w", path, err)
	}
	return path, nil
}

// RemoveModel deletes a model's file from the model directory.
func RemoveModel(dir, id string) error {
	if dir == "" {
		return fmt.Errorf("server.model_dir is not configured")
	}
	path := filepath.Join(dir, ModelFileName(id))
	if err := os.Remove(path); err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("model %q was not added through the admin UI; "+
				"if it is in config.yaml, remove it there", id)
		}
		return err
	}
	return nil
}

// ModelFileName maps a model id to a safe filename. Model ids contain a slash
// and may contain other characters that are awkward in a path, so everything
// outside a conservative set is replaced.
func ModelFileName(id string) string {
	var b strings.Builder
	for _, r := range strings.ToLower(id) {
		switch {
		case r >= 'a' && r <= 'z', r >= '0' && r <= '9', r == '-', r == '.':
			b.WriteRune(r)
		default:
			b.WriteRune('_')
		}
	}
	return b.String() + ".yaml"
}
