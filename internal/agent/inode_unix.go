//go:build unix

package agent

import (
	"os"
	"syscall"
)

// inode exposes the identity and link count of a file, so a tree of hard links
// is not counted once per link. The HuggingFace cache links every snapshot
// entry to a shared blob, so ignoring this roughly doubles every figure.
func inode(info os.FileInfo) (ino uint64, nlink uint64, ok bool) {
	st, ok := info.Sys().(*syscall.Stat_t)
	if !ok {
		return 0, 0, false
	}
	return st.Ino, uint64(st.Nlink), true
}
