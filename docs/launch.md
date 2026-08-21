# Launch guide: zero to serving on OpenRouter

Written for the setup you have chosen: one rented A40, one model, one gateway.
Every command here has been run against the real code.

Budget about three hours. Roughly $1.50 of that is GPU time before you commit to
anything.

---

## Step 0 — Decide the model first

Do not rent hardware and then look for something to serve. Pick the model, then
buy what serves it.

The opening you are aiming at is a **recently released model with few
providers**. Qwen3.8-27B at the time of writing had 7 providers and output
priced at $3.20/M, against DeepSeek-V4-Flash's 27 providers at $0.14/M. The same
GPU earns twenty times more on the first than the second.

Check before committing:

```bash
llmplan Qwen/Qwen3.8-27B --compare          # what hardware it needs
llmplan Qwen/Qwen3.8-27B --gpu "NVIDIA A40:48" --ram 50 --disk 122
```

Then open the model's page on OpenRouter and read two columns:

- **Provider count.** Under ten is an opening. Over twenty is a price war you
  will lose.
- **Throughput.** If the median is far above what the planner predicts for your
  card, everyone serving it is on hardware you cannot match. Pick another model.

---

## Step 1 — Rent the GPU (~$1.50)

RunPod bills per millisecond, so you can prove the whole stack works for the
price of a coffee before committing to a month.

1. Deploy an **A40 pod**, 48GB VRAM. Confirm the total, not the GPU line: the
   headline $0.44/hr becomes **$0.46/hr** once container and volume disk are
   added, which is **$335.80/month**.
2. Attach a **volume** of at least 60GB and mount it at `/workspace`. Weights
   live here. Without it every restart re-downloads the model, and that is
   downtime you are scored on.
3. Note the pod's IP and the port you expose.

Do not enable a public IP for the engine ports. The gateway is the only thing
that should reach them.

---

> **On RunPod specifically**, `scripts/setup-pod.sh` does steps 2 and 3 for you:
> it installs Go, vLLM and cloudflared, builds the binaries, generates tokens
> and writes a config. See the README for the exact sequence.

## Step 2 — Install the agent on the GPU box

```bash
# On the pod
apt-get update && apt-get install -y curl
mkdir -p /workspace/llmfast /var/lib/llmfast-agent

# Copy the binary you cross-compiled: make build-linux
scp dist/llmfast-agent-linux-amd64 root@POD_IP:/workspace/llmfast/llmfast-agent
chmod +x /workspace/llmfast/llmfast-agent

# The token authenticates the gateway. It is read from the environment, never a
# flag: a credential in a flag is visible in `ps` to every user on the box.
export LLMFAST_AGENT_TOKEN=$(openssl rand -hex 32)
export HF_TOKEN=hf_your_token        # only needed for gated repos
echo "$LLMFAST_AGENT_TOKEN"          # you will need this on the gateway

# Check it sees the hardware before going further.
/workspace/llmfast/llmfast-agent -hardware -state-dir /workspace/llmfast/state
```

You should see the A40 and its 48GB. If the GPU is missing here it is missing
everywhere — fix the driver before continuing.

```bash
/workspace/llmfast/llmfast-agent \
  -listen 127.0.0.1:9900 \
  -name gpu-a \
  -state-dir /workspace/llmfast/state \
  -hf-cache /workspace/hf \
  -mode docker
```

`-hf-cache` on the volume is what makes restarts fast instead of a 20-minute
re-download.

---

## Step 3 — Run the gateway

**On the same pod.** The gateway is a single Go binary that used under 1% CPU at
880 req/s; it does not need its own machine, and running it beside the engine
removes every millisecond of network between them. That time lands inside your
TTFT, which is the number you are competing on.

Add a separate VM (~$5/month, same region) only once you have more than one GPU
node, or if you recreate pods often enough that a changing address is a problem.

```yaml
# /workspace/config.yaml
provider:
  slug: llmfast
  display_name: LLMFast
  public_url: https://api.llmfa.st

server:
  listen: "127.0.0.1:8080"     # the tunnel or proxy is what faces the internet
  admin_listen: "127.0.0.1:8081"     # never expose this
  admin_token: "$LLMFAST_ADMIN_TOKEN"
  db_path: "/var/lib/llmfast/llmfast.db"
  model_dir: "models.d"

nodes:
  - name: gpu-a
    url: http://127.0.0.1:9900   # same box, so localhost
    token: $LLMFAST_AGENT_TOKEN
    max_concurrency: 13              # set from the benchmark in step 5
```

```bash
export LLMFAST_ADMIN_TOKEN=$(openssl rand -hex 32)
./llmfast -config config/config.yaml
```

Reach the admin UI over an SSH tunnel — it exposes API keys and full request
history, and has no business being on the internet:

```bash
ssh -L 8081:127.0.0.1:8081 gateway-host
```

**Nodes** should show `gpu-a` as reachable with its hardware listed. If it does
not, the token or the firewall is wrong.

---

## Step 4 — Install the model

**Admin UI → Add Model.** Paste the HuggingFace id, press Inspect, read the plan,
press Install on `gpu-a`.

The model is staged **hidden** (`is_ready: false`). Leave it that way. Weights
are still downloading; advertising it now earns 404s, and 404s count against
your uptime.

Watch the engine's own log output in the install panel until it reports ready.
First install of a 27B takes 10–30 minutes depending on the link.

---

## Step 5 — Measure it (this is the step people skip)

**Admin UI → Benchmark.** This is where you find out whether the GPU was worth
renting. It sweeps concurrency through the identical request path customers use.

Start with:

| Setting | Value |
|---|---|
| Concurrency levels | `1, 4, 8, 16, 32` |
| Requests per level | 8 |
| Prompt tokens | 512 |
| Output tokens | 128 |

Read three things from the result:

- **TTFT p50 and p95.** Compare against the Latency column on the model's
  OpenRouter page. The field there ranged from 0.42s to 3.08s.
- **Per request throughput.** This is what OpenRouter records against you.
  It falls as concurrency rises.
- **Aggregate throughput.** This is what the GPU produces in total, and it is
  what determines revenue. It rises with concurrency until the GPU saturates.

The report names the **knee** — the concurrency where aggregate stops improving.
Set `max_concurrency` in `config.yaml` to that number. Past the knee you are
only adding latency, and latency is scored.

Then run a long-prompt pass (`prompt_tokens: 8192`) to check prefill, and if the
model takes images, a real image request: the vision encoder needs memory the
text arithmetic does not model.

---

## Step 6 — Check the economics before publishing

The Benchmark tab feeds straight into the earnings calculator with the number
you just measured. Enter your real GPU cost and be pessimistic about
utilisation.

At **$335.80/month**, Qwen3.8-27B market prices, 100 tok/s aggregate, 5:1
input:output, 40% cache hits:

| Busy | Revenue | Margin |
|---|---|---|
| 5% | $61/mo · $2.0/day | **−$275/mo** |
| 10% | $122/mo · $4.1/day | **−$214/mo** |
| 25% | $306/mo · $10.2/day | **−$30/mo** |
| 50% | $611/mo · $20.4/day | **+$275/mo** |
| 100% | $1,222/mo · $40.7/day | **+$886/mo** |

**Break-even is 27.5% utilisation.**

That single number is the business. Below it you lose money no matter how you
price, because you cannot sell tokens you did not generate. Everything about
being a provider — uptime, latency, model choice — is really about getting
routed enough traffic to clear 27.5%.

A new provider does not start busy. Plan for the first month to lose money.

---

## Step 7 — Publish and apply

1. **Publish the model** in the admin UI (`is_ready: true`).
2. Check **Model Doc** — that is byte-for-byte what OpenRouter fetches.
3. Verify from outside your network:

```bash
curl -s https://api.llmfa.st/v1/models | jq '.data[0].schema_version'   # "2.4"

./scripts/check-streaming.sh https://api.llmfa.st sk-llmfast-... qwen/qwen3.8-27b
```

The script reports the spread between the first and last frame. If everything
arrives at once, something is buffering — most often Cloudflare's proxy on the
API subdomain, or a reverse proxy missing `proxy_buffering off`. This matters
more than it sounds: a buffering proxy does not fail visibly, it just makes your
measured throughput worse than the hardware you are paying for.

4. Create a key named `openrouter` in the admin UI.
5. Submit the form using [openrouter-application.md](openrouter-application.md).

---

## Step 8 — The first two weeks

Watch **Overview** daily. The numbers that matter:

- **Uptime.** Below 95% you are degraded; below 80% you are fallback-only. It is
  computed with OpenRouter's own formula so your dashboard and theirs agree.
- **TTFT p95.** If it drifts up, you are over `max_concurrency`. Lower it. A 429
  costs you one request; slow responses cost you the routing slot.
- **Cache hit rate.** Low hit rates mean prefix caching is not working. Check
  `--enable-prefix-caching` is on.
- **429 count.** Some is healthy — it means admission control is doing its job.
  A lot means you need another GPU, which is a good problem.

Do not add a second model until the first is earning. One model served reliably
beats three served badly, because uptime is per-endpoint and a bad one drags
nothing but itself down — while your attention is finite.

---

## What to do when it is not working

**No traffic at all.** Check `/v1/models` is reachable unauthenticated, and that
`is_ready` is true. Then check your price against the field: routing is
price-weighted by default.

**Traffic then it stopped.** Look at uptime. One bad hour can drop you below 95%
and take days of clean running to recover.

**Slow according to OpenRouter, fast in your benchmark.** Something between you
and them is buffering, or you are queueing. This gateway never queues, so look
at the reverse proxy.

**Losing money.** Compare measured utilisation against the 27.5% break-even. If
you are far below it after a month, the answer is a different model, not a
bigger GPU.
