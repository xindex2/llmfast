//go:build linux

package main

const (
	ioctlReadTermios  = 0x5401 // TCGETS
	ioctlWriteTermios = 0x5402 // TCSETS
)
