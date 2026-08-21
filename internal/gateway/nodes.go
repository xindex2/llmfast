package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"sync"
	"time"

	"github.com/llmfast/gateway/internal/agent"
	"github.com/llmfast/gateway/internal/config"
	"github.com/llmfast/gateway/internal/upstream"
)

// nodeUnreachableGrace is how long a node may fail to answer before its engines
// are removed from routing. A brief network blip should not cost us every model
// on that box, but a genuinely dead node must stop receiving traffic.
const nodeUnreachableGrace = 90 * time.Second

// NodeManager talks to the llmfast-agent processes on the inference hosts.
//
// It polls each node for hardware and running engines, and mirrors those
// engines into the routing pool as dynamic backends. The gateway therefore
// never needs shell access to a GPU box: everything it can do to a node is
// bounded by the agent's small control API.
type NodeManager struct {
	pool   *upstream.Pool
	client *http.Client
	log    logger

	mu     sync.RWMutex
	nodes  map[string]config.Node
	status map[string]*NodeStatus
}

type logger interface {
	Info(msg string, args ...any)
	Warn(msg string, args ...any)
	Error(msg string, args ...any)
}

// NodeStatus is the gateway's view of one node.
type NodeStatus struct {
	Name      string      `json:"name"`
	URL       string      `json:"url"`
	Reachable bool        `json:"reachable"`
	LastSeen  time.Time   `json:"last_seen,omitempty"`
	LastError string      `json:"last_error,omitempty"`
	Info      *agent.Info `json:"info,omitempty"`
}

func NewNodeManager(nodes []config.Node, pool *upstream.Pool, log logger) *NodeManager {
	m := &NodeManager{
		pool: pool,
		log:  log,
		// No global timeout: install requests are quick to return (the agent
		// answers 202 immediately) but polling a busy node can be slow.
		client: &http.Client{Timeout: 20 * time.Second},
		nodes:  make(map[string]config.Node, len(nodes)),
		status: make(map[string]*NodeStatus, len(nodes)),
	}
	for _, n := range nodes {
		m.nodes[n.Name] = n
		m.status[n.Name] = &NodeStatus{Name: n.Name, URL: n.URL}
	}
	return m
}

// Start begins polling. The first poll happens immediately so the admin UI is
// populated as soon as the gateway is up.
func (m *NodeManager) Start(ctx context.Context, interval time.Duration) {
	if len(m.nodes) == 0 {
		return
	}
	go func() {
		m.pollAll(ctx)
		t := time.NewTicker(interval)
		defer t.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-t.C:
				m.pollAll(ctx)
			}
		}
	}()
}

func (m *NodeManager) pollAll(ctx context.Context) {
	m.mu.RLock()
	nodes := make([]config.Node, 0, len(m.nodes))
	for _, n := range m.nodes {
		nodes = append(nodes, n)
	}
	m.mu.RUnlock()

	var wg sync.WaitGroup
	for _, n := range nodes {
		wg.Add(1)
		go func(n config.Node) {
			defer wg.Done()
			m.poll(ctx, n)
		}(n)
	}
	wg.Wait()
}

func (m *NodeManager) poll(ctx context.Context, n config.Node) {
	info, err := m.Info(ctx, n.Name)

	m.mu.Lock()
	st := m.status[n.Name]
	if err != nil {
		st.Reachable = false
		st.LastError = err.Error()
		unreachableFor := time.Since(st.LastSeen)
		hadSeen := !st.LastSeen.IsZero()
		m.mu.Unlock()

		// Only drop a node's engines from routing once it has been down long
		// enough that this is not a transient blip.
		if !hadSeen || unreachableFor > nodeUnreachableGrace {
			m.pool.DropNode(n.Name)
		}
		return
	}
	st.Reachable = true
	st.LastError = ""
	st.LastSeen = time.Now()
	st.Info = info
	m.mu.Unlock()

	// Mirror the node's ready engines into the routing pool.
	host, err := hostOf(n.URL)
	if err != nil {
		m.log.Error("node has an unusable url", "node", n.Name, "err", err)
		return
	}
	var entries []upstream.DynamicBackend
	for _, in := range info.Instances {
		if in.State != agent.StateReady {
			continue
		}
		entries = append(entries, upstream.DynamicBackend{
			Node:    n.Name,
			ModelID: in.Spec.ServedName,
			BaseURL: fmt.Sprintf("http://%s:%d/v1", host, in.Port),
			// The engine's own --max-num-seqs is the real ceiling; the node's
			// configured limit is how much of it we are willing to use.
			MaxConcurrency: minPositive(n.MaxConcurrency, in.Spec.MaxNumSeqs),
			Weight:         n.Weight,
		})
	}
	m.pool.SyncNode(n.Name, entries)
}

// Info fetches a node's current status directly.
func (m *NodeManager) Info(ctx context.Context, name string) (*agent.Info, error) {
	var info agent.Info
	if err := m.call(ctx, name, http.MethodGet, "/v1/node/info", nil, &info); err != nil {
		return nil, err
	}
	return &info, nil
}

// Install asks a node to start serving a model.
func (m *NodeManager) Install(ctx context.Context, name string, spec agent.Spec) (map[string]any, error) {
	var out map[string]any
	if err := m.call(ctx, name, http.MethodPost, "/v1/node/install", spec, &out); err != nil {
		return nil, err
	}
	return out, nil
}

func (m *NodeManager) StopModel(ctx context.Context, name, servedName string) error {
	body := map[string]string{"served_name": servedName}
	return m.call(ctx, name, http.MethodPost, "/v1/node/stop", body, nil)
}

func (m *NodeManager) RemoveModel(ctx context.Context, name, servedName string) error {
	body := map[string]string{"served_name": servedName}
	return m.call(ctx, name, http.MethodPost, "/v1/node/remove", body, nil)
}

// Logs fetches an engine's recent output, which is where a failed install
// explains itself.
func (m *NodeManager) Logs(ctx context.Context, name, servedName string, n int) (map[string]any, error) {
	var out map[string]any
	path := fmt.Sprintf("/v1/node/logs?served_name=%s&n=%d", url.QueryEscape(servedName), n)
	if err := m.call(ctx, name, http.MethodGet, path, nil, &out); err != nil {
		return nil, err
	}
	return out, nil
}

func (m *NodeManager) call(ctx context.Context, name, method, path string, in, out any) error {
	m.mu.RLock()
	n, ok := m.nodes[name]
	m.mu.RUnlock()
	if !ok {
		return fmt.Errorf("unknown node %q", name)
	}

	var body io.Reader
	if in != nil {
		b, err := json.Marshal(in)
		if err != nil {
			return err
		}
		body = bytes.NewReader(b)
	}
	req, err := http.NewRequestWithContext(ctx, method, n.URL+path, body)
	if err != nil {
		return err
	}
	if n.Token != "" {
		req.Header.Set("Authorization", "Bearer "+n.Token)
	}
	if in != nil {
		req.Header.Set("Content-Type", "application/json")
	}

	resp, err := m.client.Do(req)
	if err != nil {
		return fmt.Errorf("node %s unreachable: %w", name, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		// Surface the agent's own message: it is far more useful than a status
		// code, especially for "engine not installed" or "already running".
		var e struct {
			Error struct {
				Message string `json:"message"`
			} `json:"error"`
		}
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 8<<10))
		if json.Unmarshal(raw, &e) == nil && e.Error.Message != "" {
			return fmt.Errorf("node %s: %s", name, e.Error.Message)
		}
		return fmt.Errorf("node %s returned %s", name, resp.Status)
	}
	if out == nil {
		io.Copy(io.Discard, resp.Body)
		return nil
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

// Statuses returns a snapshot of every node, for the admin UI.
func (m *NodeManager) Statuses() []NodeStatus {
	m.mu.RLock()
	defer m.mu.RUnlock()
	out := make([]NodeStatus, 0, len(m.status))
	for _, st := range m.status {
		out = append(out, *st)
	}
	return out
}

func (m *NodeManager) Has(name string) bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	_, ok := m.nodes[name]
	return ok
}

func (m *NodeManager) Count() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return len(m.nodes)
}

func hostOf(raw string) (string, error) {
	u, err := url.Parse(raw)
	if err != nil {
		return "", err
	}
	if u.Hostname() == "" {
		return "", fmt.Errorf("no host in %q", raw)
	}
	return u.Hostname(), nil
}

// minPositive returns the smaller of two limits, ignoring unset (non-positive)
// values so an undeclared limit never wins.
func minPositive(a, b int) int {
	switch {
	case a <= 0:
		return b
	case b <= 0:
		return a
	case a < b:
		return a
	}
	return b
}
