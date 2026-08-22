package agent

import (
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// CacheEntry is one model's downloaded weights on this node.
type CacheEntry struct {
	// Repo is the HuggingFace id, reconstructed from the cache directory name.
	Repo  string `json:"repo"`
	Bytes int64  `json:"bytes"`
	// InUse marks weights a running engine is currently serving from, which
	// must not be deleted underneath it.
	InUse bool `json:"in_use"`
	dir   string
}

// hubDir is where huggingface_hub keeps downloaded repositories.
func hubDir(hfCache string) string {
	if hfCache == "" {
		return ""
	}
	return filepath.Join(hfCache, "hub")
}

// repoFromCacheDir turns "models--Qwen--Qwen3-32B" back into "Qwen/Qwen3-32B".
//
// A repository name can itself contain "--", which the cache encodes the same
// way as the owner separator, so only the first pair is treated as the split.
func repoFromCacheDir(name string) string {
	if !strings.HasPrefix(name, "models--") {
		return ""
	}
	rest := strings.TrimPrefix(name, "models--")
	owner, model, ok := strings.Cut(rest, "--")
	if !ok {
		return rest
	}
	return owner + "/" + strings.ReplaceAll(model, "--", "/")
}

// cacheDirForRepo is the inverse, used to delete a specific repository.
func cacheDirForRepo(repo string) string {
	return "models--" + strings.ReplaceAll(repo, "/", "--")
}

// ListCache reports what the weight cache is holding and how large each entry
// is.
//
// Uninstalling a model deliberately leaves its weights in place so that
// reinstalling it does not mean downloading tens of gigabytes again. That is
// the right default and a bad one to be unable to see: a pod volume fills up
// with the remains of models that are no longer in the catalog, and nothing
// says where the space went.
func ListCache(hfCache string, inUse map[string]bool) ([]CacheEntry, error) {
	dir := hubDir(hfCache)
	if dir == "" {
		return nil, nil
	}
	items, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var out []CacheEntry
	for _, it := range items {
		if !it.IsDir() {
			continue
		}
		repo := repoFromCacheDir(it.Name())
		if repo == "" {
			continue
		}
		full := filepath.Join(dir, it.Name())
		out = append(out, CacheEntry{
			Repo: repo, Bytes: dirSize(full), InUse: inUse[repo], dir: full,
		})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Bytes > out[j].Bytes })
	return out, nil
}

// dirSize sums the apparent size of a tree.
//
// The blobs are content-addressed files that the snapshot directories link to,
// so counting link targets once is what gives a figure matching the volume.
func dirSize(root string) int64 {
	var total int64
	seen := map[uint64]bool{}
	_ = filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		info, err := d.Info()
		if err != nil {
			return nil
		}
		if ino, nlink, ok := inode(info); ok && nlink > 1 {
			if seen[ino] {
				return nil
			}
			seen[ino] = true
		}
		total += info.Size()
		return nil
	})
	return total
}

// DeleteCache removes one repository's weights.
func DeleteCache(hfCache, repo string, inUse map[string]bool) (int64, error) {
	if repo == "" {
		return 0, fmt.Errorf("repo is required")
	}
	if inUse[repo] {
		return 0, fmt.Errorf("%s is being served right now; stop it before deleting its weights", repo)
	}
	if !validRepo(repo) {
		return 0, fmt.Errorf("invalid repo name %q", repo)
	}
	dir := hubDir(hfCache)
	if dir == "" {
		return 0, fmt.Errorf("no HuggingFace cache directory is configured")
	}
	target := filepath.Join(dir, cacheDirForRepo(repo))
	// Belt and braces: the name is already validated, and encoding replaces
	// every separator, but a deletion is not the place to rely on one check.
	if filepath.Dir(target) != filepath.Clean(dir) {
		return 0, fmt.Errorf("invalid repo name %q", repo)
	}
	if _, err := os.Stat(target); err != nil {
		if os.IsNotExist(err) {
			return 0, fmt.Errorf("%s is not in the weight cache", repo)
		}
		return 0, err
	}
	freed := dirSize(target)
	if err := os.RemoveAll(target); err != nil {
		return 0, err
	}
	return freed, nil
}

// validRepo accepts the characters HuggingFace allows in an owner/name pair
// and nothing else.
//
// Rejecting rather than sanitising matters here: cacheDirForRepo replaces every
// slash, so "../../etc" encodes to a harmless filename that simply does not
// exist -- the deletion would report success having removed nothing. Refusing
// the input says what actually happened.
func validRepo(repo string) bool {
	owner, name, ok := strings.Cut(repo, "/")
	if !ok || owner == "" || name == "" || strings.Contains(name, "/") {
		return false
	}
	for _, part := range []string{owner, name} {
		if part == "." || part == ".." || strings.HasPrefix(part, ".") {
			return false
		}
		for _, r := range part {
			switch {
			case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9':
			case r == '-', r == '_', r == '.':
			default:
				return false
			}
		}
	}
	return true
}
