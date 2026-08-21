package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestWriteModelReplacesAFileNamedAnythingElse covers a state the admin UI had
// no way out of. Publishing a model whose file was not named by ModelFileName
// wrote a second file declaring the same id; the next reload then failed with
// "duplicate model", and since every later write repeated the mistake, the
// catalog could not be repaired from the UI at all.
func TestWriteModelReplacesAFileNamedAnythingElse(t *testing.T) {
	dir := t.TempDir()
	odd := filepath.Join(dir, "my-favourite-model.yaml")
	if err := os.WriteFile(odd, []byte("id: qwen/qwen3.8-27b\nname: Old\n"), 0o640); err != nil {
		t.Fatal(err)
	}

	ready := true
	path, err := WriteModel(dir, Model{ID: "qwen/qwen3.8-27b", Name: "New", IsReady: &ready})
	if err != nil {
		t.Fatalf("write: %v", err)
	}
	if path != odd {
		t.Errorf("wrote to %s, want the existing %s", path, odd)
	}

	entries, _ := os.ReadDir(dir)
	if len(entries) != 1 {
		var names []string
		for _, e := range entries {
			names = append(names, e.Name())
		}
		t.Errorf("model directory holds %v, want exactly one file", names)
	}

	// And it must be removable, which it was not: RemoveModel looked only at
	// the canonical name and reported success having deleted nothing.
	if err := RemoveModel(dir, "qwen/qwen3.8-27b"); err != nil {
		t.Fatalf("remove: %v", err)
	}
	if entries, _ := os.ReadDir(dir); len(entries) != 0 {
		t.Error("model file still present after RemoveModel")
	}
}

// TestRemoveModelSaysWhenItIsNotOursToRemove: a model declared in config.yaml
// has no file here, and silently succeeding would leave the operator watching
// a model that never disappears.
func TestRemoveModelSaysWhenItIsNotOursToRemove(t *testing.T) {
	dir := t.TempDir()
	err := RemoveModel(dir, "qwen/from-config-yaml")
	if err == nil {
		t.Fatal("expected an error for a model with no file in the model dir")
	}
	if !strings.Contains(err.Error(), "config.yaml") {
		t.Errorf("error should point at config.yaml, got: %v", err)
	}
}

// TestFindModelFileSkipsUnparseableFiles: a stray file in the directory must
// not make an installed model unremovable.
func TestFindModelFileSkipsUnparseableFiles(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "notes.yaml"), []byte("\tthis: [is not yaml"), 0o640); err != nil {
		t.Fatal(err)
	}
	if _, err := WriteModel(dir, Model{ID: "qwen/real"}); err != nil {
		t.Fatal(err)
	}
	got, err := FindModelFile(dir, "qwen/real")
	if err != nil {
		t.Fatalf("find: %v", err)
	}
	if got != filepath.Join(dir, ModelFileName("qwen/real")) {
		t.Errorf("found %q, want the canonical file", got)
	}
}
