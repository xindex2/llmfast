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

## What runs where

Three things exist. Only one of them needs a server you pay for.

```
  llmfa.st                      api.llmfa.st
  the website                   the inference API
       │                              │
       ▼                              ▼
  ┌──────────────┐          ┌────────────────────────────────┐
  │   Netlify    │          │  ONE machine with a GPU        │
  │  or Pages    │          │                                │
  │   (free)     │          │   llmfast        the gateway   │
  │              │          │   llmfast-agent  supervises    │
  │  static HTML │          │   vLLM           the model     │
  └──────────────┘          └────────────────────────────────┘
     site/                          RunPod pod
```

**The website does not need a VPS.** `site/` is three static HTML files, so
Netlify or Cloudflare Pages hosts it free.

**The gateway does not need its own VPS either — put it on the GPU pod.** It is
a single Go binary that used under 1% CPU at 880 requests/second. Running it
beside the engine also removes every millisecond of network between them, and
that time lands inside your TTFT, which is the number OpenRouter scores you on.

So the minimum is: **a free static host + one RunPod pod.** No VPS.

### When you do want a separate VPS

Add a small always-on box (~$5/month) when either becomes true:

- **You have more than one GPU node.** The gateway routes across them, so it
  should not live inside any one of them.
- **You destroy and recreate pods often.** A pod's address changes when it is
  recreated; a VPS keeps `api.llmfa.st` pointing somewhere stable.

Put it in the same region as the GPU. A €5 VM next to your GPU beats a €50 one
in another country for this workload.

---

## Deploying — RunPod + Cloudflare, step by step

This is the exact path for: website already on Netlify, GPU pod on RunPod, DNS
on Cloudflare.

### The one thing to understand first

**You do not point `api.llmfa.st` at an IP address.**

RunPod gives you something like `69.30.85.5:22105`. That is a *proxy address with
a port*, and DNS cannot express a port — an A record is only an IP, and HTTPS
has to arrive on :443. So there is no IP to paste into Cloudflare.

Instead the pod opens an outbound connection to Cloudflare and Cloudflare sends
traffic back down it. That is a **Cloudflare Tunnel**. It needs no open port, no
public IP and no certificate, and it survives the pod's address changing.

```
  browser ──▶ Cloudflare edge ──▶ tunnel ──▶ 127.0.0.1:8080 on your pod
              api.llmfa.st                    the gateway
```

> **Two machines are involved, and mixing them up is the easiest mistake to
> make.** Every command below is labelled. 💻 means your own computer, 🖥️ means
> the pod's terminal.

### 1. 💻 Send the code to the pod — on your computer

Open a terminal **on your own machine**, go to the llmfast folder, and run one
command:

```bash
# 💻 ON YOUR COMPUTER
cd ~/Desktop/llmfast          # wherever you cloned it
./scripts/deploy-to-pod.sh 69.30.85.5 22105
```

Those two values are the ones RunPod shows under **Direct TCP ports** — for
`SSH → 69.30.85.5:22105 → :22`, the IP is `69.30.85.5` and the port is `22105`.

The script builds the Linux binaries, checks the pod is reachable, copies
everything up, and runs the setup remotely. It refuses to run from the wrong
folder or from the pod itself, so if you paste it in the wrong place it will
tell you rather than half-work.

**If it says SSH failed**, RunPod does not have your public key. Add it at
[runpod.io/console/user/settings](https://www.runpod.io/console/user/settings)
and restart the pod — RunPod only injects keys at start. Or skip SSH entirely
and use the deploy key route below.

<details>
<summary>🖥️ No SSH? Pull from the pod instead with a deploy key</summary>

This repository is private, so `git clone` from the pod gets a 404 — GitHub
answers unauthenticated requests that way rather than 403. Give the pod
read-only access:

```bash
# 🖥️ on the pod, in the web terminal
ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519 -C "runpod"
cat ~/.ssh/id_ed25519.pub
```

Copy that line, then open
`github.com/xindex2/llmfast/settings/keys` → **Add deploy key**, paste it,
**leave "Allow write access" unchecked**, save. Back on the pod:

```bash
# 🖥️ on the pod
git clone git@github.com:xindex2/llmfast.git /workspace/llmfast
bash /workspace/llmfast/scripts/setup-pod.sh
```

</details>

<details>
<summary>Or make the repository public</summary>

Then the pod can just fetch it:

```bash
# 🖥️ on the pod
curl -fsSL https://raw.githubusercontent.com/xindex2/llmfast/main/scripts/setup-pod.sh | bash
```

Worth a thought first. Public means anyone can read and reuse the gateway, the
planner and the agent. It is **not** a security question — no secrets are in the
repo, and tokens are generated on the pod into `/workspace/llmfast.env`, which is
gitignored. It is only about whether you want the work visible.

</details>

### 2. 🖥️ Check the setup found your GPU

The setup script prints the hardware it detected. It should say something like:

```
  node: gpu-a
  cpu:  AMD EPYC 7402P 24-Core Processor (9 cores)
  ram:  50.0 GiB
  disk: 118.4 GiB free (NVMe)
  gpu 0: NVIDIA A40, 48.0 GiB
  total VRAM: 48.0 GiB across 1 GPU(s)
```

**If the GPU line is missing, stop here.** Nothing after this will work, and the
cause is the driver rather than anything in this project. Run `nvidia-smi` on the
pod to confirm.

### 3. 🖥️ Start the agent

```bash
# 🖥️ ON THE POD
source /workspace/llmfast.env
/workspace/llmfast/dist/llmfast-agent \
  -listen 127.0.0.1:9900 -name gpu-a \
  -state-dir /workspace/state -hf-cache /workspace/hf -mode native
```

Leave it running. `-hf-cache /workspace/hf` puts model weights on the volume, so
a restart does not re-download 20GB.

### 4. 🖥️ Start the gateway

New terminal tab:

```bash
# 🖥️ ON THE POD
source /workspace/llmfast.env
cd /workspace && /workspace/llmfast/dist/llmfast -config /workspace/config.yaml
```

It prints an API key the first time it runs. Save that — it is shown once.

### 5. 🖥️ Point api.llmfa.st at it

New terminal tab:

```bash
# 🖥️ ON THE POD
cloudflared tunnel login          # opens a link; pick llmfa.st in the browser
cloudflared tunnel create llmfast
cloudflared tunnel route dns llmfast api.llmfa.st
cloudflared tunnel --url http://127.0.0.1:8080 run llmfast
```

`route dns` **creates the Cloudflare DNS record for you**. You do not add
anything by hand. Look in your Cloudflare dashboard afterwards and you will see
a proxied CNAME for `api` pointing at `<something>.cfargotunnel.com`. That is
correct — leave it proxied, a tunnel only works that way.

`llmfa.st` itself stays exactly as it is on Netlify. You are only adding the
`api` subdomain.

### 6. 💻 Check that it actually streams

This is the step worth not skipping:

```bash
# 💻 ON YOUR COMPUTER
./scripts/check-streaming.sh https://api.llmfa.st sk-llmfast-... qwen/qwen3.8-27b
```

```
  frames:            51
  first frame at:    0.03s
  last frame at:     1.52s
  spread:            1.49s

  STREAMING. Frames arrived spread over time, which is what you want.
```

If it says BUFFERED, Cloudflare is accumulating the response before forwarding
it. Nothing looks broken when that happens — the answers are correct, they just
all arrive at the end, and your throughput on OpenRouter ends up worse than the
GPU you are paying for. Two known causes: compression or Rocket Loader enabled
on the API hostname, or a Free/Pro plan cutting a request off after 100 seconds
of silence. If you cannot clear it, move the gateway to a small VPS and
terminate TLS there instead — see [deploy/nginx.conf](deploy/nginx.conf).

### 7. 💻 Open the admin UI

The admin listener stays on localhost deliberately: it exposes API keys and your
full request history. Reach it through a tunnel from your own machine:

```bash
# 💻 ON YOUR COMPUTER
ssh -N -L 8081:127.0.0.1:8081 root@69.30.85.5 -p 22105
```

Then open <http://localhost:8081>. Your token:

```bash
# 🖥️ ON THE POD
source /workspace/llmfast.env && echo $LLMFAST_ADMIN_TOKEN
```

### 8. 💻 Install a model and measure it

In the admin UI:

1. **Add Model** — paste `Qwen/Qwen3.8-27B`, press Inspect, read the plan, press
   Install on `gpu-a`. First install downloads ~20GB, so give it time. It is
   staged hidden until you publish it.
2. **Benchmark** — once it reports ready, sweep `1, 4, 8, 16`. Note the
   concurrency where aggregate throughput stops climbing, and put that number in
   `max_concurrency` in `/workspace/config.yaml`.
3. **Playground** — send it a few real prompts.
4. Publish it, then follow
   [docs/openrouter-application.md](docs/openrouter-application.md).

### Keeping it running

Pods restart. Use `tmux` so the three processes survive your terminal closing:

```bash
# 🖥️ ON THE POD
tmux new -s llmfast
# start the agent, Ctrl-b c for a new window, start the gateway, again for the tunnel
# Ctrl-b d to detach; tmux attach -t llmfast to come back
```

To send a new build later, run the same command again on your computer:

```bash
# 💻 on your computer
./scripts/deploy-to-pod.sh 69.30.85.5 22105
```

Then restart the three processes on the pod. If you used a deploy key instead:
`cd /workspace/llmfast && git pull && make build`.

Your config and installed models live in `/workspace`, outside the repo, so a
`git pull` never touches them.

### When to add a VPS

Not yet. Add a small always-on box (~$5/month, same region as the GPU) once you
have a second GPU node, or if you recreate pods often enough that restarting
three processes by hand gets old.

---

## Adding models

**Admin UI → Add Model.** Paste a HuggingFace id. The gateway reads the model's
config, works out its weight and KV-cache footprint, and tells you which of your
nodes can actually serve it — with the quantization, tensor-parallel size,
KV-cache format and context length worked out, plus a suggested price. One
button installs it: the node agent launches the engine, the catalog entry is
written, and the model becomes routable as soon as the engine reports ready.

New models are staged hidden (`is_ready: false`) until you publish them, because
advertising a model whose weights are still downloading earns 404s — and 404s
count against your uptime at OpenRouter.

It is all still plain config underneath. Each model is one YAML file in
`server.model_dir`; delete the file to remove the model, or write one by hand:

```yaml
- id: qwen/qwen3.8-27b            # what OpenRouter calls
  upstream_model: qwen/qwen3.8-27b  # what the engine was started with
  backends: [gpu-a]
  context_length: 16384
  pricing:
    prompt: "0.00000045"          # USD per token; $0.45 / M
    completion: "0.0000032"
    cached_prompt: "0.00000005"
  compliance:
    zdr: true                     # must match what your privacy policy says
```

`systemctl reload` (SIGHUP) picks up changes without dropping in-flight streams.

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
