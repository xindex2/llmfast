// Package upstream owns the vLLM backends: connection pooling, health, and
// admission control.
//
// The guiding constraint is that OpenRouter measures our throughput as output
// tokens divided by total generation time, including any time we spend
// queueing. Queueing a request therefore looks identical to being slow. So we
// never queue: if every replica for a model is saturated we return 429
// immediately and let OpenRouter route elsewhere, which costs us one request
// but protects the metric that decides how much traffic we get.
package upstream

import (
	"context"
	"encoding/json"
	"errors"
	"net"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"github.com/llmfast/gateway/internal/config"
)

var ErrNoCapacity = errors.New("no backend capacity")
var ErrNoBackend = errors.New("no healthy backend for model")

type Backend struct {
	Name    string
	BaseURL string
	APIKey  string
	Timeout time.Duration

	maxConc int32
	weight  int32

	inflight atomic.Int32
	healthy  atomic.Bool
	// lastErr records why a health probe failed, surfaced in the admin UI.
	lastErr atomic.Value // string

	client *http.Client
}

func (b *Backend) Inflight() int        { return int(b.inflight.Load()) }
func (b *Backend) MaxConc() int         { return int(b.maxConc) }
func (b *Backend) Healthy() bool        { return b.healthy.Load() }
func (b *Backend) Client() *http.Client { return b.client }

// SetHealthyForTest overrides health without waiting for a probe cycle. It
// exists so tests can exercise the degraded paths deterministically.
func (b *Backend) SetHealthyForTest(v bool) { b.healthy.Store(v) }

func (b *Backend) LastErr() string {
	if v, ok := b.lastErr.Load().(string); ok {
		return v
	}
	return ""
}

// acquire takes an admission slot without blocking. The compare-and-swap loop
// keeps the counter exact under concurrency; a plain Add would let a burst
// briefly exceed the cap before we could decrement back.
func (b *Backend) acquire() bool {
	for {
		cur := b.inflight.Load()
		if cur >= b.maxConc {
			return false
		}
		if b.inflight.CompareAndSwap(cur, cur+1) {
			return true
		}
	}
}

func (b *Backend) release() { b.inflight.Add(-1) }

// Lease is a held admission slot. Release exactly once, when the response body
// is fully consumed -- releasing at handler entry would let us oversubscribe
// the replica for the whole duration of a stream.
type Lease struct {
	B    *Backend
	once sync.Once
}

func (l *Lease) Release() {
	if l == nil {
		return
	}
	l.once.Do(func() { l.B.release() })
}

type Pool struct {
	mu       sync.RWMutex
	backends map[string]*Backend
	// byModel maps a model id to the replicas that can serve it.
	byModel map[string][]*Backend

	// Dynamic backends are engines started by node agents. They come and go as
	// models are installed and stopped, so they are tracked separately from the
	// static ones declared in config and merged at lookup time.
	dynamic        map[string]*Backend // keyed by "node/servedName"
	dynamicByModel map[string][]*Backend
	// dynamicByNode lets a sync replace exactly one node's entries without
	// disturbing the others.
	dynamicByNode map[string][]string

	stop chan struct{}
	wg   sync.WaitGroup
}

// DynamicBackend describes one engine process an agent is running.
type DynamicBackend struct {
	Node           string
	ModelID        string
	BaseURL        string
	MaxConcurrency int
	Weight         int
}

func NewPool(cfg *config.Config) *Pool {
	p := &Pool{
		backends:       make(map[string]*Backend, len(cfg.Backends)),
		byModel:        make(map[string][]*Backend, len(cfg.Models)),
		dynamic:        make(map[string]*Backend),
		dynamicByModel: make(map[string][]*Backend),
		dynamicByNode:  make(map[string][]string),
		stop:           make(chan struct{}),
	}
	for i := range cfg.Backends {
		bc := &cfg.Backends[i]
		b := &Backend{
			Name:    bc.Name,
			BaseURL: trimSlash(bc.BaseURL),
			APIKey:  bc.APIKey,
			Timeout: bc.Timeout,
			maxConc: int32(bc.MaxConcurrency),
			weight:  int32(bc.Weight),
			client:  newClient(bc),
		}
		// Assume healthy at boot so a cold start can serve immediately; the
		// first probe corrects this within seconds if the replica is down.
		b.healthy.Store(true)
		p.backends[b.Name] = b
	}
	for i := range cfg.Models {
		m := &cfg.Models[i]
		for _, name := range m.Backends {
			if b, ok := p.backends[name]; ok {
				p.byModel[m.ID] = append(p.byModel[m.ID], b)
			}
		}
	}
	return p
}

// newClient builds a transport tuned for long-lived streaming to a small set of
// known hosts on a fast network.
func newClient(bc *config.Backend) *http.Client {
	t := &http.Transport{
		DialContext: (&net.Dialer{
			Timeout: 5 * time.Second,
			// Detect a dead replica quickly rather than holding a stream open
			// against a host that has gone away.
			KeepAlive: 30 * time.Second,
		}).DialContext,
		// A large idle pool means steady-state requests almost never pay for a
		// TCP+TLS handshake, which is the single biggest avoidable slice of TTFT.
		MaxIdleConns:        512,
		MaxIdleConnsPerHost: 256,
		MaxConnsPerHost:     0,
		IdleConnTimeout:     90 * time.Second,
		TLSHandshakeTimeout: 5 * time.Second,
		// Go would otherwise add Accept-Encoding: gzip and transparently
		// decompress. On an SSE stream that introduces buffering between the
		// token arriving and us being able to forward it.
		DisableCompression: true,
		// Nagle is already off in Go, but skipping the 100-continue round trip
		// matters for large prompt bodies.
		ExpectContinueTimeout: 0,
		ForceAttemptHTTP2:     true,
		WriteBufferSize:       64 << 10,
		ReadBufferSize:        64 << 10,
	}
	// No client-level Timeout: it would abort long generations mid-stream.
	// Deadlines are applied per request via context instead.
	return &http.Client{Transport: t}
}

// SyncNode replaces the dynamic backends for one node.
//
// Existing Backend objects are reused when the endpoint is unchanged, so their
// warm connection pools and in-flight counters survive a sync. Rebuilding them
// on every poll would discard exactly the pooled connections that keep TTFT
// low.
func (p *Pool) SyncNode(node string, entries []DynamicBackend) {
	p.mu.Lock()
	defer p.mu.Unlock()

	keep := make(map[string]bool, len(entries))
	for _, e := range entries {
		key := node + "/" + e.ModelID
		keep[key] = true

		if b, ok := p.dynamic[key]; ok && b.BaseURL == trimSlash(e.BaseURL) {
			// Same endpoint: only the admission limit may have changed.
			b.maxConc = int32(maxOf(e.MaxConcurrency, 1))
			continue
		}
		weight := e.Weight
		if weight <= 0 {
			weight = 1
		}
		b := &Backend{
			Name:    key,
			BaseURL: trimSlash(e.BaseURL),
			Timeout: 10 * time.Minute,
			maxConc: int32(maxOf(e.MaxConcurrency, 1)),
			weight:  int32(weight),
			client:  newClient(&config.Backend{}),
		}
		// The agent only reports a model once its engine answered a health
		// probe, so it starts healthy rather than waiting a probe cycle.
		b.healthy.Store(true)
		p.dynamic[key] = b
	}

	// Drop entries this node no longer serves.
	for _, key := range p.dynamicByNode[node] {
		if !keep[key] {
			delete(p.dynamic, key)
		}
	}
	keys := make([]string, 0, len(keep))
	for k := range keep {
		keys = append(keys, k)
	}
	p.dynamicByNode[node] = keys

	// Rebuild the model index across every node.
	byModel := make(map[string][]*Backend, len(p.dynamic))
	for _, e := range entries {
		if b, ok := p.dynamic[node+"/"+e.ModelID]; ok {
			byModel[e.ModelID] = append(byModel[e.ModelID], b)
		}
	}
	for otherNode, otherKeys := range p.dynamicByNode {
		if otherNode == node {
			continue
		}
		for _, key := range otherKeys {
			b, ok := p.dynamic[key]
			if !ok {
				continue
			}
			model := key[len(otherNode)+1:]
			byModel[model] = append(byModel[model], b)
		}
	}
	p.dynamicByModel = byModel
}

// DropNode removes every dynamic backend for a node, used when its agent has
// been unreachable long enough that we should stop routing to it.
func (p *Pool) DropNode(node string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	for _, key := range p.dynamicByNode[node] {
		delete(p.dynamic, key)
	}
	delete(p.dynamicByNode, node)

	byModel := make(map[string][]*Backend)
	for n, keys := range p.dynamicByNode {
		for _, key := range keys {
			if b, ok := p.dynamic[key]; ok {
				byModel[key[len(n)+1:]] = append(byModel[key[len(n)+1:]], b)
			}
		}
	}
	p.dynamicByModel = byModel
}

// Available reports whether a model has at least one healthy replica, static or
// agent-managed. The playground uses it to offer only models that can actually
// answer, rather than everything in the catalog.
func (p *Pool) Available(model string) bool {
	p.mu.RLock()
	defer p.mu.RUnlock()
	for _, set := range [][]*Backend{p.byModel[model], p.dynamicByModel[model]} {
		for _, b := range set {
			if b.healthy.Load() {
				return true
			}
		}
	}
	return false
}

// DynamicModels lists the model ids currently served by agent-managed engines.
func (p *Pool) DynamicModels() []string {
	p.mu.RLock()
	defer p.mu.RUnlock()
	out := make([]string, 0, len(p.dynamicByModel))
	for m := range p.dynamicByModel {
		out = append(out, m)
	}
	return out
}

// Acquire picks the least-loaded healthy replica for a model and reserves a
// slot on it. Load is measured as inflight/weight so a bigger node takes
// proportionally more traffic.
func (p *Pool) Acquire(model string) (*Lease, error) {
	p.mu.RLock()
	static := p.byModel[model]
	dyn := p.dynamicByModel[model]
	p.mu.RUnlock()

	// Statically configured replicas and agent-managed ones are equal citizens
	// for routing; only their lifecycle differs.
	var candidates []*Backend
	switch {
	case len(dyn) == 0:
		candidates = static
	case len(static) == 0:
		candidates = dyn
	default:
		candidates = make([]*Backend, 0, len(static)+len(dyn))
		candidates = append(candidates, static...)
		candidates = append(candidates, dyn...)
	}
	if len(candidates) == 0 {
		return nil, ErrNoBackend
	}

	var best *Backend
	bestLoad := float64(1 << 30)
	anyHealthy := false
	for _, b := range candidates {
		if !b.healthy.Load() {
			continue
		}
		anyHealthy = true
		load := float64(b.inflight.Load()) / float64(b.weight)
		if load < bestLoad {
			best, bestLoad = b, load
		}
	}
	if !anyHealthy {
		return nil, ErrNoBackend
	}
	// Try the best candidate first, then fall through the rest: the winner may
	// have filled up between the scan and the acquire.
	if best != nil && best.acquire() {
		return &Lease{B: best}, nil
	}
	for _, b := range candidates {
		if b.healthy.Load() && b.acquire() {
			return &Lease{B: b}, nil
		}
	}
	return nil, ErrNoCapacity
}

// Backends returns every backend, static and dynamic, for health reporting.
func (p *Pool) Backends() []*Backend {
	p.mu.RLock()
	defer p.mu.RUnlock()
	out := make([]*Backend, 0, len(p.backends)+len(p.dynamic))
	for _, b := range p.backends {
		out = append(out, b)
	}
	for _, b := range p.dynamic {
		out = append(out, b)
	}
	return out
}

func maxOf(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func (p *Pool) Get(name string) (*Backend, bool) {
	p.mu.RLock()
	defer p.mu.RUnlock()
	b, ok := p.backends[name]
	return b, ok
}

// StartHealthChecks probes every replica's /models endpoint on an interval.
// vLLM serves it cheaply and it proves the process is actually up rather than
// just accepting TCP.
func (p *Pool) StartHealthChecks(interval time.Duration) {
	p.wg.Add(1)
	go func() {
		defer p.wg.Done()
		t := time.NewTicker(interval)
		defer t.Stop()
		p.probeAll()
		for {
			select {
			case <-p.stop:
				return
			case <-t.C:
				p.probeAll()
			}
		}
	}()
}

func (p *Pool) Stop() {
	close(p.stop)
	p.wg.Wait()
}

// probeAll checks the statically configured backends. Dynamic ones are not
// probed here: their agent already reports readiness, and probing them from two
// places would produce contradictory health.
func (p *Pool) probeAll() {
	p.mu.RLock()
	statics := make([]*Backend, 0, len(p.backends))
	for _, b := range p.backends {
		statics = append(statics, b)
	}
	p.mu.RUnlock()

	var wg sync.WaitGroup
	for _, b := range statics {
		wg.Add(1)
		go func(b *Backend) {
			defer wg.Done()
			b.probe()
		}(b)
	}
	wg.Wait()
}

func (b *Backend) probe() {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, b.BaseURL+"/models", nil)
	if err != nil {
		b.markDown(err.Error())
		return
	}
	if b.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+b.APIKey)
	}
	resp, err := b.client.Do(req)
	if err != nil {
		b.markDown(err.Error())
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b.markDown("health probe status " + resp.Status)
		return
	}
	// Drain so the connection returns to the idle pool instead of being closed.
	var discard struct{}
	_ = json.NewDecoder(resp.Body).Decode(&discard)
	b.healthy.Store(true)
	b.lastErr.Store("")
}

func (b *Backend) markDown(reason string) {
	b.healthy.Store(false)
	b.lastErr.Store(reason)
}

func trimSlash(s string) string {
	for len(s) > 0 && s[len(s)-1] == '/' {
		s = s[:len(s)-1]
	}
	return s
}
