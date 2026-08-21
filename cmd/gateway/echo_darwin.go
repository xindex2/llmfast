//go:build darwin

package main

const (
	ioctlReadTermios  = 0x40487413 // TIOCGETA
	ioctlWriteTermios = 0x80487414 // TIOCSETA
)
