# Serving models with vLLM

The same content is available in the admin UI under **Guides**.

## Install

```bash
# Verify the driver first. If this fails, nothing below will work.
nvidia-smi

docker run --gpus all --ipc=host --network=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3-32B \
  --served-model-name qwen/qwen3-32b \
  --port 8000 \
  --tensor-parallel-size 2 \
  --quantization fp8 \
  --max-model-len 131072 \
  --max-num-seqs 128 \
  --enable-prefix-caching \
  --enable-chunked-prefill
```

`--ipc=host` is required. Tensor parallelism communicates through shared memory
between worker processes, and Docker's default 64MB `/dev/shm` will crash it
with an error that does not obviously point at the cause.

Mount the HuggingFace cache. Weights are tens to hundreds of gigabytes, and
re-downloading them on every restart dominates deploy time.

Set `--served-model-name` to the **public** model id. The gateway can rewrite
the echoed model name when it differs, but matching them skips that work on
every response frame.

## Flags that move latency

| Flag | Why it matters |
|---|---|
| `--enable-prefix-caching` | Reuses the KV cache across requests sharing a prefix. System prompts and multi-turn chats are mostly shared prefix, making this the single largest TTFT win available. It is also what produces the `cached_tokens` you bill at a discount. |
| `--enable-chunked-prefill` | Interleaves prefill with decode so one long prompt cannot stall every other stream on the replica. Without it, p99 TTFT collapses as soon as traffic is mixed. |
| `--quantization fp8` | Roughly halves weight memory on Hopper and newer at little quality cost. The freed memory becomes KV cache, which is what actually limits concurrency. |
| `--max-num-seqs` | The real concurrency ceiling. Set the backend's `max_concurrency` in `config.yaml` at or below it, so the gateway sheds load before vLLM queues internally where we cannot observe it. |
| `--max-model-len` | Do not advertise more context than the KV cache can hold at your target concurrency. A 1M context serving one request at a time is worse business than 128k serving sixty. |
| `--gpu-memory-utilization` | Defaults to 0.90. Raising it to 0.95 buys KV cache but leaves less headroom for fragmentation; measure before committing. |

## Hardware sizing

Approximate minimum per replica at FP8. Verify against current vLLM release
notes — memory requirements move with every release.

| Model | Minimum | Notes |
|---|---|---|
| Qwen3-8B, GLM-4-9B | 1× L40S / A100 40GB | Best margin per GPU-hour |
| Qwen3-32B | 1× H100 80GB, or 2× L40S (TP=2) | The workhorse |
| GLM-4.6, Qwen3-72B class | 2× H100 80GB (TP=2) | |
| DeepSeek-V3 / R1 (671B MoE) | 8× H200 (TP=8) | Will not fit 8× H100 80GB without aggressive quantization |
| Kimi-K2 (~1T MoE) | 8× H200 minimum | Expect to tune expert parallelism |

Start with dense mid-size models. They have the best economics and they are what
most routed traffic actually asks for. Add the large MoE models once the smaller
ones are earning.

## SGLang as an alternative

SGLang often beats vLLM on large MoE models thanks to RadixAttention prefix
caching. Because the gateway treats every backend as a generic
OpenAI-compatible upstream, you can mix them freely — vLLM for dense models,
SGLang for DeepSeek and Kimi:

```bash
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3 \
  --served-model-name deepseek/deepseek-v3 \
  --tp 8 --port 8000 --enable-torch-compile
```

Point a `backends:` entry at it and nothing else changes.

## Verify a replica before wiring it up

```bash
curl -s http://10.0.0.11:8000/v1/models | jq '.data[].id'

curl -sN http://10.0.0.11:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen/qwen3-32b","messages":[{"role":"user","content":"hi"}],
       "stream":true,"stream_options":{"include_usage":true}}' | tail -3
```

The final frame must carry a `usage` object. If it does not, the gateway cannot
bill accurately and will fall back to counting content frames — good enough for
throughput charts, not good enough for invoices.

## Measuring what OpenRouter measures

Throughput is output tokens divided by total generation time, *including* the
time before the first token. Benchmark it the same way:

```bash
vllm bench serve \
  --model Qwen/Qwen3-32B \
  --base-url http://localhost:8000 \
  --dataset-name random --random-input-len 2000 --random-output-len 300 \
  --request-rate 20 --num-prompts 500
```

Watch TTFT p99, not the mean. The mean hides exactly the tail that moves you
below the peer median and costs you tool-calling traffic through Auto Exacto.

## Choosing a vLLM version

Three things have to line up, and only the first is obvious:

1. **vLLM implements the architecture.** The checkpoint names it in
   `config.json` -- `Qwen3ForCausalLM`, `Qwen3MoeForCausalLM`,
   `Qwen3_5ForConditionalGeneration` -- and whether vLLM can serve it depends
   on the release. A version older than the model fails *after* the weights
   have downloaded.
2. **vLLM's torch pin has a build for your driver's CUDA.** Each vLLM release
   pins one exact torch, and each torch version is built for particular CUDA
   versions. A driver older than the build refuses to start with "The NVIDIA
   driver on your system is too old".
3. **The driver belongs to the host.** In a container you cannot upgrade it.

That combination is why the newest vLLM is often the wrong answer. On a host
with a CUDA 12.8 driver:

| vLLM | torch pin | cu128 build exists | Qwen3.5 support |
|---|---|---|---|
| 0.11.2 | 2.9.0 | yes | no |
| 0.26.0 | 2.11.0 | yes | **yes** |
| 0.27.1 | 2.13.0 | **no** | yes |

0.26.0 is the newest release that satisfies all three. 0.27.1 is newer and
cannot run at all.

To work it out for a driver:

```bash
# The highest CUDA this host's driver supports
nvidia-smi | grep "CUDA Version"

# Which torch versions have a build for it (substitute cu128)
curl -s https://download.pytorch.org/whl/cu128/torch/ | grep -o 'torch-2\.[0-9]*\.[0-9]*' | sort -uV | tail -3

# Which torch a given vLLM pins
curl -s https://pypi.org/pypi/vllm/0.26.0/json | python3 -c \
  "import json,sys;print([r for r in json.load(sys.stdin)['info']['requires_dist'] if r.startswith('torch')])"

# Whether that vLLM implements the architecture
curl -s https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/vllm/model_executor/models/registry.py \
  | grep Qwen3_5
```

The admin UI checks the first point for you: the agent asks the installed
vLLM for its registry, and Inspect blocks a model the engine cannot implement
rather than letting it fail at launch. `setup-pod.sh` handles the second by
pinning torch to whatever the image ships, since the image is built for the
host it runs on.
