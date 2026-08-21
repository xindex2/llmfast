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
	// Write back to whichever file already declares this model, if any. The
	// canonical name is only a default: a file the operator named themselves,
	// or one written under an older naming scheme, still holds the model.
	// Writing to the canonical path regardless would leave two files declaring
	// the same id, and the next reload fails with "duplicate model" -- from a
	// state the admin UI offers no way out of.
	path := filepath.Join(dir, ModelFileName(m.ID))
	if existing, err := FindModelFile(dir, m.ID); err == nil && existing != "" {
		path = existing
	}

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
	path, err := FindModelFile(dir, id)
	if err != nil {
		return err
	}
	if path == "" {
		return fmt.Errorf("model %q is not declared in the model directory; "+
			"if it is in config.yaml, remove it there and restart", id)
	}
	return os.Remove(path)
}

// FindModelFile returns the path of the file in dir that declares this model
// id, or "" if no file does.
//
// Files are searched by content rather than by name because the name is only a
// convention. Anything in the directory that does not parse as a model is
// skipped rather than failing the search: a stray file should not make an
// installed model unremovable.
func FindModelFile(dir, id string) (string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return "", nil
		}
		return "", fmt.Errorf("read model dir %s: %w", dir, err)
	}
	// The canonical name first, so the common case does not read the directory.
	canonical := filepath.Join(dir, ModelFileName(id))
	names := []string{filepath.Base(canonical)}
	for _, e := range entries {
		if n := e.Name(); n != names[0] {
			names = append(names, n)
		}
	}
	for _, name := range names {
		if !strings.HasSuffix(name, ".yaml") && !strings.HasSuffix(name, ".yml") {
			continue
		}
		raw, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			continue
		}
		var m Model
		if err := yaml.Unmarshal(raw, &m); err != nil {
			continue
		}
		if m.ID == id {
			return filepath.Join(dir, name), nil
		}
	}
	return "", nil
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
