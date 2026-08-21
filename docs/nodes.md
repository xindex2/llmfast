# Node agents and one-click model installs

## What the agent is

`llmfast-agent` runs on every inference host. It reports the machine's hardware
to the gateway and starts, supervises and stops inference engines on request.

```
       gateway (control)                 node agent (execution)
  ┌──────────────────────┐          ┌────────────────────────────┐
  │ Admin UI             │  HTTPS   │ /v1/node/info    hardware  │
  │  "install Qwen3-8B" ─┼─────────▶│ /v1/node/install launch    │
  │                      │  token   │ /v1/node/stop              │
  │ routes traffic ──────┼──────┐   │ /v1/node/logs              │
  └──────────────────────┘      │   └─────────────┬──────────────┘
                                │                 │ supervises
                                │        ┌────────▼────────┐
                                └───────▶│ vLLM :18000     │
                                  direct │ vLLM :18001     │
                                         └─────────────────┘
```

The gateway never gets SSH access to a GPU box. Everything it can do to a node
is bounded by the four endpoints above. Inference traffic goes **directly** from
the gateway to the engine port, not through the agent, so the agent is never in
the latency path.

## Why an agent rather than SSH

An SSH key on the gateway can run any command on a machine holding your GPUs and
your HuggingFace token. If the gateway is compromised — and it is the part
exposed to the internet — that key is the whole fleet. The agent's API can start
a model, stop a model, and read logs. That is the entire blast radius.

## Install the agent

```bash
# On the inference host
sudo useradd --system --no-create-home --shell /usr/sbin/nologin llmfast
sudo mkdir -p /opt/llmfast /etc/llmfast /var/lib/llmfast-agent
sudo chown llmfast:llmfast /var/lib/llmfast-agent

sudo install -m 0755 dist/llmfast-agent-linux-amd64 /opt/llmfast/llmfast-agent

# The token authenticates the gateway. Generate one and give the gateway the
# same value. It is read from the environment, never a flag: a credential
# passed as a flag is visible in `ps` to every user on the box.
sudo sh -c 'cat > /etc/llmfast/agent.env' <<EOF
LLMFAST_AGENT_TOKEN=$(openssl rand -hex 32)
HF_TOKEN=hf_your_token_here
EOF
sudo chmod 0600 /etc/llmfast/agent.env

sudo install -m 0644 deploy/llmfast-agent.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now llmfast-agent
```

Check what it detected:

```bash
/opt/llmfast/llmfast-agent -hardware -state-dir /var/lib/llmfast-agent
```

```
  node: gpu-a
  cpu:  AMD EPYC 7543 32-Core Processor (64 cores)
  ram:  512.0 GiB
  disk: 3.4 TiB free (NVMe)
  gpu 0: NVIDIA H100 80GB HBM3, 80.0 GiB
  gpu 1: NVIDIA H100 80GB HBM3, 80.0 GiB
  total VRAM: 160.0 GiB across 2 GPU(s)
```

If the GPUs are missing here, they will be missing everywhere: fix the driver
before going further. The agent shells out to `nvidia-smi`, so if that command
does not work for the `llmfast` user, neither will detection.

## Register the node with the gateway

```yaml
# config/config.yaml on the gateway host
server:
  # Models installed through the admin UI are written here, one file each, so
  # config.yaml keeps its structure and comments.
  model_dir: "models.d"

nodes:
  - name: gpu-a
    url: http://10.0.0.11:9900
    token: $LLMFAST_AGENT_TOKEN
    # Never exceed the engine's own --max-num-seqs. This is where the gateway
    # starts shedding load with a 429 instead of queueing.
    max_concurrency: 96
    weight: 2
```

Restart the gateway. **Nodes** in the admin UI should show it as reachable with
its hardware listed.

## Install a model

**Admin UI → Add Model.** Paste a HuggingFace id and press Inspect. The gateway
reads the model's `config.json`, works out its memory needs, and checks it
against every node:

- **Weights** — parameter count times bytes per parameter at the chosen precision
- **KV cache** — `2 × layers × kv_heads × head_dim × 2 bytes` per token of
  context, which is what actually limits concurrency
- **Tensor parallel** — the largest size dividing both the GPU count and the KV
  head count, because vLLM rejects anything else
- **Context** — reduced if the full window would leave room for fewer than 8
  concurrent requests

You get an editable catalog entry with suggested pricing, and one button per
node. Installing:

1. asks the agent to launch the engine,
2. writes `models.d/<model>.yaml`,
3. reloads the catalog with `is_ready: false` so OpenRouter does not route to it,
4. follows the engine's own log output until it is serving.

When it reports ready, test it, then press **Publish**.

The staged-hidden step is not ceremony. A first install downloads tens of
gigabytes; advertising the model before it can answer earns 404s, and 404s count
directly against your uptime at OpenRouter.

## Same thing from the command line

```bash
# Will it fit, and how well?
llmplan Qwen/Qwen3-32B --gpu "H100:80,H100:80"
llmplan Qwen/Qwen3-8B --compare
```

```
1x L40S 48GB                        VIABLE
  engine vllm, fp8 weights, TP=1, context 20480, ~12 concurrent
  weights 8.2 GiB, KV budget 34.9 GiB, disk 18.4 GiB
  competitive for OpenRouter traffic
```

## CPU nodes

A node with no GPU is planned for **llama.cpp** rather than vLLM: vLLM's CPU
backend expects AVX-512, and llama.cpp is both faster on older cores and far
easier to operate. Because llama.cpp cannot read the original safetensors
weights, the gateway also resolves a GGUF conversion for you, preferring the
model owner's own.

Install `llama-server` on the node and it appears as an available engine.

Be clear-eyed about what a CPU node is for. Throughput is bounded by memory
bandwidth divided by weight size — an 8B model at 4-bit on a 40 GB/s server is
roughly 5 tok/s, against 50–100+ tok/s from GPU providers on the same model.
OpenRouter deprioritizes any endpoint more than 1.5 standard deviations below
the peer median, so a CPU endpoint will not hold a routing slot at any price.

It is genuinely useful for validating the whole pipeline — streaming, stats,
billing, your `/v1/models` document — before renting a GPU. The planner says so
plainly rather than letting you find out from a traffic graph.

## Operations

```bash
# What is running where
curl -s -H "Authorization: Bearer $TOKEN" http://10.0.0.11:9900/v1/node/info | jq

# Why did an install fail? The engine's own output is the answer.
curl -s -H "Authorization: Bearer $TOKEN" \
  "http://10.0.0.11:9900/v1/node/logs?served_name=qwen/qwen3-32b&n=200" | jq -r '.lines[]'
```

Behaviour worth knowing:

- **A crashed engine is restarted** with increasing backoff, up to five times,
  then marked failed with its last 20 lines of output attached.
- **Restarting the agent brings its models back.** Stopping a model through the
  UI does not — that is an explicit decision to keep it down.
- **Stopping signals the whole process group.** vLLM forks one worker per GPU,
  and signalling only the parent orphans them still holding VRAM.
- **Engines are stopped on agent shutdown**, so they do not survive as orphans.

## Security

- Bind the agent to a private interface. The token protects it, but there is no
  reason for it to be reachable from the internet.
- The agent needs no inbound access from anywhere except the gateway.
- `HF_TOKEN` lives on the node, not the gateway, so gated model access is not
  reachable from the internet-facing service.
- Engine ports (18000+) must be reachable from the gateway and nowhere else:
  the engines have no authentication of their own.
