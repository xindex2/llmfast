'use strict';

// Operator documentation, embedded in the binary so it is available on a fresh
// server before anyone has cloned the repo or found the docs directory.
const GUIDES_HTML = `
<h2>Operator guides</h2>

<div class="guide">
<h3>1 — Install vLLM on a GPU node</h3>
<h4>Prerequisites</h4>
<ul>
  <li>NVIDIA driver 550+ and CUDA 12.4+ (<code>nvidia-smi</code> must work)</li>
  <li>Python 3.10–3.12, or use the official container</li>
  <li>Fast local NVMe for the HuggingFace cache — model weights are large and
      re-downloading them on every restart is the slowest part of a deploy</li>
</ul>
<h4>Container (recommended)</h4>
<pre>docker run --gpus all --ipc=host --network=host \\
  -v ~/.cache/huggingface:/root/.cache/huggingface \\
  vllm/vllm-openai:latest \\
  --model Qwen/Qwen3-32B \\
  --served-model-name Qwen/Qwen3-32B \\
  --port 8000 \\
  --tensor-parallel-size 2 \\
  --quantization fp8 \\
  --max-model-len 131072 \\
  --enable-prefix-caching \\
  --enable-chunked-prefill</pre>
<p class="muted"><code>--ipc=host</code> is required: tensor parallelism uses shared
memory between worker processes and the default 64MB container limit will crash it.</p>
</div>

<div class="guide">
<h3>2 — Flags that actually move latency</h3>
<ul>
  <li><strong>--enable-prefix-caching</strong> — reuses the KV cache across requests
      that share a prefix. System prompts and multi-turn chats are mostly shared
      prefix, so this is the single largest TTFT win available and it directly
      feeds the <code>cached_prompt</code> SKU you bill at a discount.</li>
  <li><strong>--enable-chunked-prefill</strong> — interleaves prefill and decode so
      one long prompt cannot stall every other stream on the replica. Without it,
      p99 TTFT collapses under mixed traffic.</li>
  <li><strong>--quantization fp8</strong> — roughly halves weight memory on Hopper and
      newer with little quality loss, which raises the KV cache budget and
      therefore the concurrency each GPU sustains.</li>
  <li><strong>--max-num-seqs</strong> — the real concurrency ceiling. Set the backend's
      <code>max_concurrency</code> in config.yaml at or below it, so this gateway
      sheds load before vLLM starts queueing internally where we cannot see it.</li>
  <li><strong>--max-model-len</strong> — do not advertise more context than the KV cache
      can hold at your target concurrency. A 1M context that only serves one
      request at a time is worse business than 128k that serves sixty.</li>
</ul>
</div>

<div class="guide">
<h3>3 — Model sizing</h3>
<p class="muted">Approximate minimum hardware per replica at FP8. Verify against
current vLLM release notes before buying anything.</p>
<ul>
  <li><strong>Qwen3-8B / GLM-4-9B</strong> — 1× L40S or A100 40GB</li>
  <li><strong>Qwen3-32B</strong> — 1× H100 80GB, or 2× L40S with TP=2</li>
  <li><strong>GLM-4.6 / Qwen3-72B class</strong> — 2× H100 80GB, TP=2</li>
  <li><strong>DeepSeek-V3 / R1</strong> (671B MoE) — 8× H200, TP=8. Will not fit on
      8× H100 80GB without aggressive quantization</li>
  <li><strong>Kimi-K2</strong> (~1T MoE) — 8× H200 minimum, and expect to tune
      expert parallelism</li>
</ul>
<p>Start with the dense mid-size models. They have the best margin per GPU-hour and
they are what most OpenRouter traffic actually asks for.</p>
</div>

<div class="guide">
<h3>4a — Add a model the easy way</h3>
<p>If the node runs <code>llmfast-agent</code>, use the <strong>Add Model</strong> tab.
Paste a HuggingFace id and it will:</p>
<ul>
  <li>read the model's config and compute its weight and KV-cache footprint;</li>
  <li>check each node's real hardware and pick the engine, quantization,
      tensor-parallel size and context length that fit;</li>
  <li>suggest pricing scaled to active parameters;</li>
  <li>launch the engine, write the catalog entry, and follow the engine's own
      log output until it is serving.</li>
</ul>
<p>New models are staged <strong>hidden</strong> (<code>is_ready: false</code>). Publish
once you have tested it: advertising a model whose weights are still downloading
earns 404s, and 404s count against your uptime.</p>
<p class="muted">No agent on the node? Use the manual route below. Both produce
the same result; the agent just does the typing.</p>
</div>

<div class="guide">
<h3>4b — Add a model by hand</h3>
<p>Append to <code>models:</code> in <code>config/config.yaml</code>, then restart:</p>
<pre>models:
  - id: qwen/qwen3-32b            # what OpenRouter calls
    name: "Qwen: Qwen3 32B"
    upstream_model: Qwen/Qwen3-32B  # what vLLM was started with
    backends: [gpu-a]
    hugging_face_id: Qwen/Qwen3-32B
    quantization: fp8
    context_length: 131072
    max_output_tokens: 32768
    pricing:
      prompt: "0.00000010"          # USD per token = $0.10 / M
      completion: "0.00000030"      # $0.30 / M
      cached_prompt: "0.00000002"
    features:
      tools: true
      structured_outputs: true
    datacenters:
      - country_code: DE
        region: eu-central-1
    compliance:
      zdr: true</pre>
<p><strong>Check the Model Doc tab</strong> afterwards — it renders exactly what
OpenRouter will fetch. A model with <code>is_ready: false</code> stays hidden on
their side, which is how you stage one before an announcement.</p>
</div>

<div class="guide">
<h3>4c — Will it even fit?</h3>
<p>Before renting hardware, check what a model actually needs:</p>
<pre>llmplan Qwen/Qwen3-32B --compare
llmplan Qwen/Qwen3-8B --gpu "L40S:48"
llmplan Qwen/Qwen3-8B --cpu 16 --ram 32 --bandwidth 40</pre>
<p>The two numbers that matter:</p>
<ul>
  <li><strong>Weights</strong> — parameters x bytes per parameter. FP8 is one byte,
      4-bit is about 0.6 once scales are counted.</li>
  <li><strong>KV cache</strong> — <code>2 x layers x kv_heads x head_dim x 2</code> bytes
      per token of context. This, not the weights, is what limits concurrency.
      Grouped-query attention is why a modern 32B is cheaper to serve at long
      context than its size suggests.</li>
</ul>
<p>A plan is reported as <em>viable</em> only if it is also competitive. Something
that runs at 5 tok/s technically works and will still lose every routing
decision, so the planner says so rather than letting you find out from a
traffic graph.</p>
</div>

<div class="guide">
<h3>5 — Connect OpenRouter</h3>
<ol>
  <li>Create an API key on the <strong>API Keys</strong> tab, named <code>openrouter</code>.</li>
  <li>Give them your API base URL (<code>https://api.your-domain.com/v1</code>) and
      the models URL (<code>https://api.your-domain.com/v1/models</code>).</li>
  <li>Confirm <code>/v1/models</code> is reachable <em>without</em> credentials — their
      monitor polls it unauthenticated.</li>
  <li>Watch the <strong>Overview</strong> tab. Uptime below 95% moves you to degraded
      routing; below 80% you become fallback-only.</li>
</ol>
</div>

<div class="guide">
<h3>6 — Metrics that decide your traffic</h3>
<ul>
  <li><strong>Uptime</strong> = successes ÷ (requests − user errors). 401/402/404/5xx and
      mid-stream failures count against you. 400/403/413 do not. 429 is tracked
      separately and does not hurt uptime.</li>
  <li><strong>Throughput</strong> = output tokens ÷ total generation time, including the
      time before the first token. Any queueing you do is measured as slowness,
      which is why this gateway returns 429 instead of waiting.</li>
  <li><strong>Tool-call success rate</strong> — feeds Auto Exacto, which reorders providers
      for every request carrying tools. Falling 2σ below the peer median moves
      you to the back of the queue for all tool traffic.</li>
  <li><strong>Keep-alives</strong> — reasoning models can think for a long time before
      emitting a token. This gateway sends SSE comments during those gaps so
      OpenRouter does not time out and fail the request over to a competitor.</li>
</ul>
</div>

<div class="guide">
<h3>7 — Test the endpoint</h3>
<pre>curl -N https://api.your-domain.com/v1/chat/completions \\
  -H "Authorization: Bearer sk-llmfast-..." \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "qwen/qwen3-32b",
    "messages": [{"role":"user","content":"Say hi"}],
    "stream": true
  }'</pre>
<p class="muted"><code>-N</code> disables curl's own buffering. Without it the response
looks batched even when the server streamed it token by token.</p>
</div>
`;
