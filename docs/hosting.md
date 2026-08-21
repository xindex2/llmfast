# Where to buy the server

Prices below were checked in August 2026 and move constantly. Treat them as
orders of magnitude and confirm on the provider's own page before committing.

## The short answer

**Do not buy a monthly server yet.** Rent an RTX 4090 by the hour for about
$0.35, spend an evening putting your stack in front of it, and you will have
learned more than a month's commitment would teach you — for about $5.

Only sign up for monthly once you have real OpenRouter traffic and know which
model earns.

---

## Step 1 — Rent hourly to validate (spend ~$5)

| Provider | RTX 4090 | Notes |
|---|---|---|
| [Vast.ai](https://vast.ai/pricing/gpu/RTX-4090) | from ~$0.13/hr | Marketplace of individual hosts. Cheapest by far, quality varies host to host. Ideal for testing, risky for production uptime. |
| [RunPod](https://www.runpod.io/) | ~$0.34/hr community, more for Secure Cloud | Good middle ground. Secure Cloud is datacenter-grade; Community is peer hosts. |
| [CloudRift](https://www.cloudrift.ai/pricing) | competitive | Per-minute billing. |

What to do with those hours:

```bash
# On the rented box
docker run --gpus all --ipc=host -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest --model Qwen/Qwen3-8B \
  --served-model-name qwen/qwen3-8b --quantization fp8 \
  --enable-prefix-caching --enable-chunked-prefill

# Then install llmfast-agent on it and add it as a node. See docs/nodes.md.
```

Measure your real TTFT and throughput under load before you believe any
estimate, including this project's planner.

**A note on marketplace GPUs.** OpenRouter tracks your uptime and drops you to
degraded routing below 95%. A cheap host that reboots without warning will cost
you more in lost routing than it saves in rent. Use Vast.ai to learn, not to
serve.

---

## Step 2 — Monthly, once you have traffic

### Cheapest real entry point

| Provider | Hardware | Rough price | Good for |
|---|---|---|---|
| [Hetzner](https://www.hetzner.com/dedicated-rootserver/matrix-gpu/) | GEX44: RTX 4000 SFF Ada, 20GB | ~€184/mo | 8B models at FP8. Excellent value, German datacenters, well-run. 20GB is the constraint. |
| [Hostkey](https://hostkey.com/dedicated-servers/rent-nvidia-servers/) | various NVIDIA dedicated | varies | EU/US, dedicated rather than shared. |
| [OVHcloud](https://www.ovhcloud.com/en/public-cloud/gpu/) | L4, L40S, A100, H100 | varies | Strong network, predictable monthly billing, EU presence. L40S is the value pick for inference. |

### The sweet spot for this business

**One L40S (48GB)** is the best single-card choice for an OpenRouter provider:

- Qwen3-32B at FP8 fits with real concurrency (the planner says ~10 concurrent)
- Hardware FP8, so you get the memory saving without an emulated slow path
- Far cheaper than an H100 while serving the models that actually get traffic

**One RTX 4090 (24GB)** is the budget option: fine for 8B–14B models, too tight
for 32B (the planner will tell you it fits at 4-bit but with only ~2 concurrent
requests, which will not hold a routing slot).

Check what fits before you pay:

```bash
llmplan Qwen/Qwen3-32B --compare
llmplan Qwen/Qwen3-8B --gpu "RTX 4090:24"
```

### What not to buy yet

- **H100 / H200** (~$2.00–2.50/hr, ~$1,500–1,800/mo). Only once you have enough
  routed traffic to keep it busy. An idle H100 loses money faster than a busy
  4090 makes it.
- **8×H200 nodes** for DeepSeek-V3 or Kimi-K2. That is a five-figure monthly
  commitment against providers with far better economics of scale.
- **Any CPU-only server for inference.** See below.

---

## About the Xeon E5-2683v4 server

The one you were looking at — 16 cores, 32GB DDR4, 8TB SATA HDD, no GPU — cannot
serve OpenRouter traffic. Measured against real model configs:

| Model | Throughput on that box | Peer median |
|---|---|---|
| Qwen3-8B (4-bit) | ~5.5 tok/s | 50–100+ tok/s |
| Qwen3-32B (4-bit) | ~1.4 tok/s | 50–100+ tok/s |

CPU decoding reads every active weight once per token, so throughput is memory
bandwidth divided by weight size. Quad-channel DDR4-2400 gives roughly 40–60
GB/s, and no amount of tuning changes that arithmetic. OpenRouter deprioritizes
any endpoint more than 1.5 standard deviations below the peer median, so it
would be routed last regardless of price.

Two further problems: 32GB of RAM cannot hold Qwen3-32B's weights at any useful
precision, and an 8TB spinning disk at ~150 MB/s turns every model load into a
multi-minute outage. Weights want NVMe.

**It is still useful for two things:**

1. **Running the gateway.** It is heavily over-specced for that — the gateway
   used under 1% CPU at 880 req/s in benchmarking — but it works.
2. **Validating the whole pipeline** with `llama-server` and a 1.7B model, so
   streaming, stats, billing and your `/v1/models` document are all proven
   before you rent a GPU.

If you buy it, buy it as the gateway host, and only if it is cheap.

---

## Co-location matters more than specs

Put the gateway in the **same datacenter** as the GPUs. If the gateway is at
provider A and the GPUs at provider B in another city, every request pays that
round trip inside your TTFT — and TTFT is the number you are competing on.

A €5/month VM next to your GPU beats a €50/month dedicated server in another
country for this workload. If in doubt, run the gateway on the GPU box itself;
it is light enough.

## Ask about egress

Streaming text is tiny — roughly 200 bytes per second per active stream, so a
1 Gbit link is nowhere near the constraint. But some providers meter egress
aggressively. Confirm it is unmetered or generously included before signing.
