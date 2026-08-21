//go:build unix

package main

import (
	"os"
	"syscall"
	"unsafe"
)

// disableEcho turns off terminal echo so a typed password is not shown, and
// returns a function that restores the previous setting.
//
// This is done with a raw ioctl rather than golang.org/x/term to avoid adding a
// dependency for one call. If stdin is not a terminal -- a pipe, or a here-doc
// -- the ioctl fails and echo was never on in the first place, so the caller
// simply reads the line as normal.
func disableEcho(fd int) (func(), error) {
	var termios syscall.Termios
	if err := ioctl(fd, ioctlReadTermios, &termios); err != nil {
		return nil, err
	}
	previous := termios
	termios.Lflag &^= syscall.ECHO
	termios.Lflag |= syscall.ICANON | syscall.ISIG
	if err := ioctl(fd, ioctlWriteTermios, &termios); err != nil {
		return nil, err
	}
	return func() { _ = ioctl(fd, ioctlWriteTermios, &previous) }, nil
}

func ioctl(fd int, req uintptr, t *syscall.Termios) error {
	_, _, errno := syscall.Syscall6(syscall.SYS_IOCTL, uintptr(fd), req,
		uintptr(unsafe.Pointer(t)), 0, 0, 0)
	if errno != 0 {
		return errno
	}
	return nil
}

var _ = os.Stdin
