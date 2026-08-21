# LLMFast

An OpenRouter-compatible inference provider stack: a Go gateway that fronts
vLLM replicas, publishes the provider model document OpenRouter reads, and
records the latency and throughput metrics that decide how much traffic you get.

```
OpenRouter ──▶ gateway (Go, single binary) ──▶ engines on your nodes
                  │                              (vLLM on GPU, llama.cpp on CPU)
                  ├── /v1/chat/completions   streaming proxy, admission control
                  ├── /v1/models             OpenRouter provider doc, schema 2.4
                  ├── /health                per-backend health
                  └── :8081 admin UI         stats, one-click installs, keys
                          │
                          │ control only, never in the latency path
                          ▼
                   llmfast-agent on each node
                     detects hardware, launches and supervises engines
```

## Install

```bash
git clone https://github.com/xindex2/llmfast.git
cd llmfast
make build          # gateway, agent and planner for this machine
make build-linux    # static binaries for your server (pure Go, no cgo)
```

Requires Go 1.24+. Nothing else — SQLite is pure Go and the admin UI is
embedded in the binary, so there is no Node build step and no CGO toolchain.

To update an existing install:

```bash
git pull
make build-linux
scp dist/llmfast-linux-amd64 root@GATEWAY:/opt/llmfast/llmfast
ssh root@GATEWAY 'systemctl restart llmfast'
```

Models and pricing live in `config/` and `models.d/`, not in the binary, so an
upgrade never touches your catalog. A `systemctl reload` (SIGHUP) picks up
config changes without dropping in-flight streams; a full restart is only
needed when the binary or the `backends`/`nodes` list changes.

## Quick start

No GPU needed — a mock vLLM backend ships with it.

```bash
make dev
```

Will a model fit on hardware you are considering? Ask before you rent it:

```bash
make plan MODEL=Qwen/Qwen3-8B      # compares across a standard set of machines
go run ./cmd/llmplan Qwen/Qwen3-32B --gpu "H100:80,H100:80"
```

Then in another shell:

```bash
# The bootstrap API key is printed once on first start.
KEY=sk-llmfast-...

curl -N localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model":"qwen/qwen3-32b","messages":[{"role":"user","content":"hi"}],"stream":true}'
```

Admin UI: <http://localhost:8081> — token `dev-admin-token` (from
`config/dev.yaml`).

## Adding models

**Admin UI → Add Model.** Paste a HuggingFace id. The gateway reads the model's
config, computes its weight and KV-cache footprint, and tells you which of your
nodes can actually serve it — with the tensor-parallel size, quantization and
context length worked out, and a suggested price. One button installs it: the
node agent launches the engine, the catalog entry is written, and the model
becomes routable as soon as the engine reports ready.

Installed models are staged hidden (`is_ready: false`) until you publish them,
because advertising a model whose weights are still downloading earns 404s, and
404s count against your uptime at OpenRouter.

Everything is also plain config. A model is one YAML file under
`server.model_dir`; delete it to remove the model. See [docs/nodes.md](docs/nodes.md).

## What it does

**Streaming proxy.** Forwards SSE frames with per-frame flushing and no
intermediate buffering. Measures true time-to-first-token by ignoring the
opening role-only frame that carries no content. Emits SSE keep-alive comments
during long prefills so OpenRouter does not cancel a reasoning model mid-think.

**Load shedding, never queueing.** OpenRouter measures throughput as output
tokens over total generation time, so a queued request is indistinguishable from
a slow one. Each backend has a hard admission limit; past it the gateway returns
an immediate 429 with `Retry-After`. Losing a request is cheaper than losing the
metric that routes traffic to you.

**Token accounting.** Asks upstream for usage even when the client did not, then
withholds that frame from a client that never requested it. Cached prompt tokens
are billed at the cache rate instead of, not on top of, the full prompt rate.

**Provider document.** Renders OpenRouter's schema 2.4 format from your config —
nested modalities, per-modality pricing and capacity, request-scoped entries at
the root only, and no zero-stuffed SKUs. Preview it in Admin → Model Doc before
you submit anything.

**Measured, not estimated.** A benchmark that sweeps concurrency through the
real request path and reports TTFT, per-request and aggregate throughput, and
where the GPU saturates — then feeds those numbers into a break-even calculator
using your configured prices. Renting a GPU is a decision you should make on
measurement.

**A request path that does not copy your prompt.** The proxy decodes only the
fields it needs and forwards the original bytes untouched when nothing has to
change. On a 1MB prompt that is 360 bytes of allocation instead of 2.4MB, and
about half the CPU time.

**Playground.** A test console in the admin UI that runs completions through the
real `/v1/chat/completions` path — the same admission control, SSE relay and
token accounting a customer gets — and reports TTFT, throughput, cache hits and
cost per request. A simplified test console could pass while the real path was
broken, which is exactly the failure a console exists to catch.

**Statistics.** Per-request TTFT, throughput, token counts, cache hit rate and
error class, rolled up hourly and daily. Uptime is computed with OpenRouter's
own formula — successes over requests excluding user errors — so your dashboard
and their provider page should agree.

**Capacity planning that refuses to flatter you.** The planner sizes weights and
KV cache from the model's real config, and reports whether a plan is not just
*possible* but *competitive*. A configuration that runs at 5 tok/s is labelled
as unfit for live traffic, because OpenRouter deprioritizes any endpoint more
than 1.5σ below the peer median regardless of price.

## Layout

```
cmd/gateway/        the service
cmd/agent/          llmfast-agent, runs on each inference node
cmd/llmplan/        "will this model fit on this hardware?" CLI
cmd/mockvllm/       fake engine for local development
internal/config/    YAML catalog: backends, nodes, models, pricing
internal/upstream/  connection pooling, health, admission control
internal/gateway/   HTTP surface, streaming proxy, admin API, embedded UI
internal/agent/     hardware detection, engine launch and supervision
internal/modelspec/ HuggingFace metadata, memory sizing, serving plans
internal/store/     SQLite: keys, request log, rollups
internal/modeldoc/  OpenRouter schema 2.4 renderer
config/             config.yaml (production shape), dev.yaml (mock backend)
site/               static public site for llmfa.st (home, terms, privacy)
deploy/             systemd units, nginx config
docs/               application answers, policies, deployment, node guides
```

## Adding a model

Append to `models:` in `config/config.yaml`, then `systemctl reload llmfast`
(SIGHUP) — models and pricing reload without dropping in-flight streams.

```yaml
- id: qwen/qwen3-32b              # what OpenRouter calls
  upstream_model: Qwen/Qwen3-32B  # what vLLM was started with
  backends: [gpu-a]
  context_length: 131072
  pricing:
    prompt: "0.00000009"          # USD per token; $0.09 / M
    completion: "0.00000028"
    cached_prompt: "0.00000002"
```

Set `is_ready: false` to stage a model that OpenRouter should keep hidden.

Starting vLLM with `--served-model-name` equal to the public id lets the gateway
skip rewriting the model name in every response frame.

## Deploying for real

Full walkthrough in **[docs/launch.md](docs/launch.md)**. The short version:

### 1. GPU node (RunPod)

Deploy a pod, attach a volume of at least 60GB, and run the agent on it:

```bash
export LLMFAST_AGENT_TOKEN=$(openssl rand -hex 32)   # give this to the gateway
export HF_TOKEN=hf_...                               # only for gated repos

./llmfast-agent -hardware -state-dir /workspace/state   # check it sees the GPU

./llmfast-agent \
  -listen 0.0.0.0:9900 \
  -name gpu-a \
  -state-dir /workspace/state \
  -hf-cache /workspace/hf \
  -mode docker
```

Put `-hf-cache` on the volume. Without it every restart re-downloads the weights,
and that is downtime OpenRouter scores you on.

### 2. Gateway

Runs on a small VM **in the same region as the GPU** — every millisecond between
them lands inside your TTFT.

```bash
sudo useradd --system --no-create-home llmfast
sudo mkdir -p /opt/llmfast /etc/llmfast /var/lib/llmfast
sudo chown llmfast:llmfast /var/lib/llmfast

sudo install -m 0755 dist/llmfast-linux-amd64 /opt/llmfast/llmfast
sudo install -m 0640 config/config.yaml /etc/llmfast/config.yaml
sudo sh -c 'echo "LLMFAST_ADMIN_TOKEN=$(openssl rand -hex 32)" > /etc/llmfast/env'
sudo sh -c 'echo "LLMFAST_AGENT_TOKEN=<from step 1>" >> /etc/llmfast/env'
sudo chmod 0600 /etc/llmfast/env

sudo install -m 0644 deploy/llmfast.service /etc/systemd/system/
sudo systemctl enable --now llmfast
```

### 3. Domain and TLS

Point DNS at the gateway:

| Record | Name | Value |
|---|---|---|
| A | `api.llmfa.st` | gateway public IP |
| A | `llmfa.st` | static host for [site/](site/) |

Then issue a certificate. Caddy is the least error-prone option because it also
gets the streaming settings right by default:

```bash
sudo apt install caddy
sudo tee /etc/caddy/Caddyfile <<'EOF'
api.llmfa.st {
    reverse_proxy 127.0.0.1:8080 {
        flush_interval -1     # stream immediately, never buffer
        transport http {
            read_timeout 600s
        }
    }
}
EOF
sudo systemctl reload caddy
```

`flush_interval -1` is the important line. With nginx use
`proxy_buffering off` — see [deploy.md](docs/deploy.md). A buffering proxy does
not break anything visibly; it just makes your measured throughput worse than
the hardware you are paying for.

Verify from outside your network that frames arrive spread over time:

```bash
curl -N https://api.llmfa.st/v1/chat/completions \
  -H "Authorization: Bearer sk-llmfast-..." \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen/qwen3.8-27b","messages":[{"role":"user","content":"hi"}],"stream":true}'
```

### 4. Firewall

| Port | Who reaches it |
|---|---|
| 443 | public — ideally restricted to OpenRouter's egress ranges |
| 8080 | localhost only; the proxy forwards to it |
| 8081 | **localhost only** — admin UI exposes keys and request history |
| 9900 | the gateway only — the agent's control API |
| 18000+ | the gateway only — engines have no authentication of their own |

Reach the admin UI over a tunnel: `ssh -L 8081:127.0.0.1:8081 gateway-host`

## Documentation

| | |
|---|---|
| [docs/launch.md](docs/launch.md) | **Start here** — zero to serving, with the benchmark and break-even steps |
| [docs/openrouter-application.md](docs/openrouter-application.md) | Prepared answers for the provider application form |
| [docs/nodes.md](docs/nodes.md) | Node agents, one-click installs, capacity planning |
| [docs/hosting.md](docs/hosting.md) | Where to rent or buy GPUs, and what not to buy yet |
| [site/](site/) | The public website: home page, Terms of Use, Privacy Policy |
| [docs/deploy.md](docs/deploy.md) | Production deployment, TLS, the SSE buffering trap |
| [docs/models.md](docs/models.md) | vLLM installation, tuning, hardware sizing |
| [site/terms.html](site/terms.html) | Terms of Use — draft, needs your details and a lawyer |
| [site/privacy.html](site/privacy.html) | Privacy Policy — same |

The same operator guides are embedded in the admin UI under **Guides**, so
they are available on a fresh server before anyone has cloned the repo.

## Development

```bash
make test        # full suite
make test-race   # under the race detector
make lint        # vet + gofmt check
make build-linux # cross-compile gateway, agent and planner for the server
make hardware    # what the agent detects on this machine
make plan MODEL=Qwen/Qwen3-8B
```

## License

MIT — see [LICENSE](LICENSE).
