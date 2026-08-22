package agent

import (
	"os"
	"path/filepath"
	"testing"
)

func TestRepoNameRoundTrip(t *testing.T) {
	for _, repo := range []string{
		"Qwen/Qwen3-32B",
		"Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
		"cyankiwi/Qwen3.8-27B-AWQ-INT4",
	} {
		dir := cacheDirForRepo(repo)
		if got := repoFromCacheDir(dir); got != repo {
			t.Errorf("%q -> %q -> %q", repo, dir, got)
		}
	}
	// Anything that is not a model cache directory is ignored rather than
	// guessed at: the cache also holds datasets, locks and .no_exist markers.
	for _, junk := range []string{"datasets--foo--bar", ".locks", "version.txt"} {
		if got := repoFromCacheDir(junk); got != "" {
			t.Errorf("repoFromCacheDir(%q) = %q, want empty", junk, got)
		}
	}
}

// TestCacheListingAndDeletion covers the behaviour that made a volume fill up
// with no explanation: uninstalling a model leaves its weights on disk on
// purpose, so that reinstalling does not mean re-downloading tens of
// gigabytes. That is the right default, and it needs to be visible and
// reversible.
func TestCacheListingAndDeletion(t *testing.T) {
	cache := t.TempDir()
	hub := filepath.Join(cache, "hub")
	for name, size := range map[string]int{
		"models--Qwen--Qwen3-32B":        4096,
		"models--Qwen--Qwen3-8B":         1024,
		"datasets--someone--not-a-model": 512,
	} {
		d := filepath.Join(hub, name, "blobs")
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(d, "weights"), make([]byte, size), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	inUse := map[string]bool{"Qwen/Qwen3-32B": true}
	entries, err := ListCache(cache, inUse)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 2 {
		t.Fatalf("listed %d entries, want the 2 models (datasets are not models): %+v", len(entries), entries)
	}
	// Largest first, so the thing worth deleting is at the top.
	if entries[0].Repo != "Qwen/Qwen3-32B" {
		t.Errorf("entries not sorted by size: %+v", entries)
	}
	if !entries[0].InUse {
		t.Error("a model being served must be marked in use")
	}

	// Weights a running engine is serving from must not be removable.
	if _, err := DeleteCache(cache, "Qwen/Qwen3-32B", inUse); err == nil {
		t.Error("deleted weights out from under a running engine")
	}

	freed, err := DeleteCache(cache, "Qwen/Qwen3-8B", inUse)
	if err != nil {
		t.Fatalf("delete: %v", err)
	}
	if freed < 1024 {
		t.Errorf("freed %d bytes, want at least the 1024 written", freed)
	}
	if _, err := os.Stat(filepath.Join(hub, "models--Qwen--Qwen3-8B")); !os.IsNotExist(err) {
		t.Error("the cache directory is still there")
	}

	// A traversal attempt must not escape the cache.
	if _, err := DeleteCache(cache, "../../etc", inUse); err == nil {
		t.Error("a repo name containing .. was accepted")
	}
}
