package gateway

import (
	"bufio"
	"io"
	"net/http"
	"sync"
	"sync/atomic"
	"time"
)

// readerPool recycles the buffered readers used to relay upstream streams.
//
// Each streaming request previously allocated a fresh 64KB buffer. At a few
// hundred requests per second that is tens of megabytes per second of garbage,
// and GC pressure shows up as latency jitter on exactly the streams we are
// trying to keep smooth.
var readerPool = sync.Pool{
	New: func() any { return bufio.NewReaderSize(nil, 64<<10) },
}

// getReader borrows a buffered reader wrapping r.
func getReader(r io.Reader) *bufio.Reader {
	br := readerPool.Get().(*bufio.Reader)
	br.Reset(r)
	return br
}

// putReader returns a reader to the pool. Resetting to nil first drops the
// reference to the response body so a pooled reader cannot pin a connection.
func putReader(br *bufio.Reader) {
	br.Reset(nil)
	readerPool.Put(br)
}

// scratchPool recycles the line-accumulation buffers used for SSE frames longer
// than the reader's buffer, such as large tool-call arguments.
var scratchPool = sync.Pool{
	New: func() any { b := make([]byte, 0, 8<<10); return &b },
}

// sseWriter serializes writes to the client. Two goroutines touch the response:
// the relay loop forwarding upstream tokens, and the keep-alive ticker. An
// http.ResponseWriter is not safe for concurrent use, so every write goes
// through this mutex.
type sseWriter struct {
	mu   sync.Mutex
	w    http.ResponseWriter
	rc   *http.ResponseController
	last atomic.Int64 // unix nanos of the last successful write
	err  error
}

func newSSEWriter(w http.ResponseWriter) *sseWriter {
	s := &sseWriter{w: w, rc: http.NewResponseController(w)}
	s.last.Store(time.Now().UnixNano())
	return s
}

// WriteFlush writes a frame and pushes it to the socket immediately. Buffering
// here would show up directly in the client's measured inter-token latency, so
// every frame is flushed rather than batched.
func (s *sseWriter) WriteFlush(p []byte) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.err != nil {
		return s.err
	}
	if _, err := s.w.Write(p); err != nil {
		s.err = err
		return err
	}
	if err := s.rc.Flush(); err != nil {
		s.err = err
		return err
	}
	s.last.Store(time.Now().UnixNano())
	return nil
}

// KeepAliveIfIdle emits an SSE comment when the stream has been silent. Comments
// are ignored by every compliant SSE client, but they prove to OpenRouter that
// we are still working -- otherwise a long prefill or a reasoning model's
// thinking phase looks like a hung connection and gets cancelled.
func (s *sseWriter) KeepAliveIfIdle(idle time.Duration) {
	if time.Since(time.Unix(0, s.last.Load())) < idle {
		return
	}
	_ = s.WriteFlush([]byte(": keepalive\n\n"))
}

func (s *sseWriter) Err() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.err
}

// readSSELine returns one line including its trailing newline.
//
// The fast path returns the bufio.Reader's internal slice with no copy, valid
// only until the next read -- which is fine because the caller writes it
// downstream before reading again. Lines longer than the buffer (large tool
// call arguments) fall back to accumulating into scratch.
func readSSELine(br *bufio.Reader, scratch []byte) (line []byte, newScratch []byte, err error) {
	line, err = br.ReadSlice('\n')
	if err != bufio.ErrBufferFull {
		return line, scratch, err
	}
	scratch = append(scratch[:0], line...)
	for err == bufio.ErrBufferFull {
		line, err = br.ReadSlice('\n')
		scratch = append(scratch, line...)
	}
	return scratch, scratch, err
}
