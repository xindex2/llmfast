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

<details open>
<summary>🖥️ No SSH? Pull from the pod instead with a deploy key</summary>

This repository is private, so `git clone` from the pod gets
`Permission denied (publickey)` until the pod has a key GitHub recognises.
**Do these in order — the clone is last.**

**1.** Make a key on the pod:

```bash
# 🖥️ ON THE POD
ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519 -C "runpod"
```

**2.** Print it and copy the whole line, `ssh-ed25519` through `runpod`:

```bash
# 🖥️ ON THE POD
cat ~/.ssh/id_ed25519.pub
```

**3.** In a browser, open
[github.com/xindex2/llmfast/settings/keys](https://github.com/xindex2/llmfast/settings/keys)
→ **Add deploy key**. Title it `runpod`, paste the line into the Key box, and
**leave "Allow write access" unchecked** — the pod only ever needs to read.
Click **Add key**.

**4.** Now the clone works:

```bash
# 🖥️ ON THE POD
git clone git@github.com:xindex2/llmfast.git /workspace/llmfast
bash /workspace/llmfast/scripts/setup-pod.sh
```

If step 4 still says `Permission denied (publickey)`, the key did not land.
Check what the pod is offering against what GitHub has:

```bash
# 🖥️ ON THE POD
ssh -T git@github.com
```

A successful key prints `Hi xindex2/llmfast! You've successfully authenticated`.
Anything else means the paste in step 3 was incomplete — it must be one single
line with no wrapping.

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

### 3. 🖥️ Start everything

```bash
# 🖥️ ON THE POD
bash /workspace/llmfast/scripts/llmfast.sh start
```

That checks everything first, then launches the agent, the gateway and the
tunnel as detached processes. **You can close the terminal and the browser —
they keep running.**

If something is wrong it says so before starting anything:

```
==> Checking
  ✓ dist/llmfast
  ✓ dist/llmfast-agent
  ✓ llmfast.env
  ✓ config.yaml
  ✗ port 9900 is held by an older llmfast process — run: bash scripts/llmfast.sh stop
```

The first run prints an API key. Save it; it is shown once and only its hash is
stored.

Four commands are all you need afterwards:

| | |
|---|---|
| `llmfast.sh status` | what is running, and whether each part answers |
| `llmfast.sh logs` | follow all three logs (Ctrl-C to stop watching) |
| `llmfast.sh restart` | after a `git pull && make build` |
| `llmfast.sh stop` | shut it all down, strays included |

`status` is the one to reach for. It checks the agent, the gateway, the admin UI
and your public hostname, and lists the installed models:

```
==> Processes
  ✓ agent    pid 10468
  ✓ gateway  pid 10486
  ✓ tunnel   pid 10502

==> Health
  ✓ agent    127.0.0.1:9900
  ✓ gateway  127.0.0.1:8080
  ✓ admin    127.0.0.1:8090
  ✓ public   api.llmfa.st

==> Models
  qwen/qwen3.8-27b-fp8         ready=true  ctx=32,768
```

If a process died, `status` prints the tail of its log instead of just saying
it is gone. Everything is written to `/workspace/logs/`, so a crash at 3am is
still readable the next morning.

**A pod restart stops everything.** Run `llmfast.sh start` again and you are
back — your models, config, database and downloaded weights all live in
`/workspace`, which survives.

### 4. 🖥️ Point api.llmfa.st at it

```bash
# 🖥️ ON THE POD
cloudflared tunnel login          # opens a link; pick llmfa.st in the browser
cloudflared tunnel create llmfast
bash /workspace/llmfast/scripts/setup-tunnel.sh llmfast api.llmfa.st admin.llmfa.st
```

That writes `/root/.cloudflared/config.yml` with both hostnames and creates the
DNS records for you. You add nothing by hand — look in Cloudflare afterwards and
you will see proxied CNAMEs pointing at `<id>.cfargotunnel.com`. Leave them
proxied; a tunnel only works that way.

Then restart everything so the tunnel picks up the new config:

```bash
# 🖥️ ON THE POD
bash /workspace/llmfast/scripts/llmfast.sh restart
```

If you see **error 1033** when you visit the domain, the tunnel is not running.
That is the whole meaning of 1033: Cloudflare has the DNS record but nothing is
connected behind it.

Cloudflare will offer to "migrate" the tunnel to dashboard management. You do
not need it, and it is irreversible — the config file above is easier to reason
about and can live in version control.

### 5. 🔒 Put Cloudflare Access in front of admin.llmfa.st

**Do this before you open that hostname.** The dashboard exposes your API keys,
your full request history, and the ability to install and stop models. The
gateway does check a bearer token, but a token is one secret with no second
factor, no expiry and no audit trail. Access adds a real login in front of it,
and it is free.

1. Cloudflare dashboard → **Zero Trust** → **Access** → **Applications**
2. **Add an application** → **Self-hosted**
3. Subdomain `admin`, domain `llmfa.st`
4. Add a policy: action **Allow**, include **Emails** → your address
5. Save

You will get a one-time code by email the first time you visit. Until that
policy exists, treat `admin.llmfa.st` as public.

> **Not** `api.llmfa.st`. That one has to stay open — OpenRouter's monitor polls
> `/v1/models` without credentials, and their traffic cannot pass through a login
> page. The API is protected by its own bearer keys instead.

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

If you set up Access in step 5b, just open <https://admin.llmfa.st> and sign in
with the email code.

<details>
<summary>Or reach it over SSH instead, without exposing a hostname at all</summary>

```bash
# 💻 ON YOUR COMPUTER
ssh -N -L 8090:127.0.0.1:8090 root@69.30.85.5 -p 22105
```

Then <http://localhost:8090>. RunPod pods have no root password — access is by
SSH key only, and the key has to be in your RunPod account settings *before* the
pod starts. If SSH is not set up, use the Access route above; it is easier and
gives you an audit trail.

</details>

Either way, your token:

```bash
# 🖥️ ON THE POD
source /workspace/llmfast.env && echo $LLMFAST_ADMIN_TOKEN
```

### 8. 💻 Install a model and measure it

In the admin UI:

1. **Add Model** — paste `Qwen/Qwen3.8-27B`, press Inspect, read the plan, press
   Install on `gpu-a`. First install downloads ~20GB, so give it time. It is
   staged hidden until you publish it.

   **Raise the context to 32768** before installing. The planner defaults to a
   safe 16384 because KV cache is what actually limits you — on a 48GB A40 with
   a 27B model you have about 25 GiB left for KV after the weights, and every
   token of every in-flight request costs 128 KB of it. That is ~204k tokens
   total to share out: 16384 gives you 12 concurrent requests, 32768 gives you
   6. Six is the right trade here, because the apps routing to this class of
   model are coding agents that send long prompts and would be truncated at 16k.

   Do not chase the 1M context other providers advertise. The model's native
   context is 262,144; 1M is RoPE scaling, and one such request alone would need
   122 GiB of KV cache.
2. **Benchmark** — once it reports ready, sweep `1, 4, 8, 16`. Note the
   concurrency where aggregate throughput stops climbing, and put that number in
   `max_concurrency` in `/workspace/config.yaml`.
3. **Playground** — send it a few real prompts.
4. **Earnings** — enter **26** as the input:output ratio, not the default. Real
   traffic on these models runs about 26:1 (long prompts, short completions),
   which is what makes the economics work: you bill for the prompt tokens too,
   so a $335/mo A40 breaks even at roughly 8.6 output tok/s sustained rather
   than the ~24 you would need at 5:1.
5. Publish it, then follow
   [docs/openrouter-application.md](docs/openrouter-application.md).

### Running it day to day

Nothing needs to stay open on your laptop. The three processes run detached on
the pod, writing to `/workspace/logs/`:

```bash
# 🖥️ ON THE POD
bash /workspace/llmfast/scripts/llmfast.sh status    # is everything up?
bash /workspace/llmfast/scripts/llmfast.sh logs      # watch it live
bash /workspace/llmfast/scripts/llmfast.sh restart   # after an upgrade
```

Close the browser, shut the laptop, go to bed. The pod carries on.

**After a pod restart**, nothing is running, so start it again:

```bash
# 🖥️ ON THE POD
bash /workspace/llmfast/scripts/llmfast.sh start
```

Your models, config, database and downloaded weights all live in `/workspace`,
which survives, so nothing is reinstalled and nothing is re-downloaded.

Worth checking `status` daily at first. Uptime is what decides how much traffic
OpenRouter sends you, and a process that died quietly at 3am costs you a day of
routing before you notice.

To upgrade:

```bash
# 🖥️ ON THE POD
cd /workspace/llmfast && git pull && make build
bash scripts/llmfast.sh restart
```

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
