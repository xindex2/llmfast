# OpenRouter provider application — prepared answers

Form: <https://openrouter.ai/how-to-list>

Fields marked **FILL IN** need your real company details and cannot be drafted
for you. Everything else is ready to paste, but read it first — several answers
make commitments (rate limits, retention, regions) that must match what you
actually run. An application that promises a 30-day log window while the
gateway is configured for 90 will be caught the first time someone checks.

---

## Identity

| Field | Answer |
|---|---|
| **Company Name** | **FILL IN** — the registered legal entity, not the brand |
| **Website** | `https://llmfa.st` |
| **Your Email** | **FILL IN** — must be on your company domain; they invite this address to a Slack Connect channel, so use one a human monitors daily |
| **Display Name** | `LLMFast` |
| **Desired Slug** | `llmfast` |
| **HQ Location** | **FILL IN** — country where the company is registered |
| **Inference Location** | `DE` — update to match your `datacenters` entries in `config/config.yaml` |

> Keep the slug, the `provider.slug` in `config.yaml`, and your domain
> consistent. It is the identifier users see next to every model you serve.

---

## Endpoints

| Field | Answer |
|---|---|
| **API Base URL** | `https://api.llmfa.st/v1` |
| **URL to /models API** | `https://api.llmfa.st/v1/models` |
| **URL to Privacy Policy** | `https://llmfa.st/privacy` |
| **URL to Terms of Service** | `https://llmfa.st/terms` |

Before submitting, verify from a machine outside your network:

```bash
# Must return the schema 2.4 document with no credentials -- their monitor
# polls this unauthenticated.
curl -s https://api.llmfa.st/v1/models | jq '.data[0].schema_version'

# Must stream, and the first token must arrive promptly.
curl -N https://api.llmfa.st/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model":"qwen/qwen3-32b","messages":[{"role":"user","content":"hi"}],"stream":true}'
```

---

## Supported Output Modalities

Check **Text** only.

Do not check Embeddings, Rerank, TTS, Image, Audio or Video unless you are
actually serving them on day one. Claiming a modality you cannot serve produces
404s on routed traffic, and 404 counts directly against your uptime.

---

## Distinguishing Features

Check: **Low Latency**, **High Throughput**, **Low Pricing**.

Leave the rest unchecked, and here is why — each one is a claim they will test:

- **Unique Models** — only if you serve something no other provider does. Open
  weights that five providers already serve are not unique.
- **Unique Infrastructure** — reserve this for custom silicon or a genuinely
  novel serving stack. Well-tuned vLLM on rented H200s is excellent, but it is
  not unique infrastructure.
- **Decentralized** — not applicable.
- **Strategic Partnership** — only if you have one with a model lab or a cloud.

---

## Extra Details

> Paste the following, then edit every number to match your real deployment.

We serve open-weight text models on dedicated NVIDIA GPUs in EU-Central,
fronted by a purpose-built Go gateway. We are optimising for the two metrics
that decide routing — time-to-first-token and sustained throughput — rather
than for headline peak numbers.

**Hardware.** Dedicated bare-metal GPU nodes, not shared or preemptible
capacity. Dense mid-size models (Qwen3-32B, GLM-4.6) run on H100/H200 nodes at
FP8; large MoE models (DeepSeek-V3, Kimi-K2) run on 8×H200 with tensor
parallelism. NVMe-local weight caches, so a replica restart does not mean a
re-download.

**Serving stack.** vLLM with prefix caching, chunked prefill and continuous
batching. Chunked prefill is what keeps a single long prompt from stalling every
other stream on the replica, which is where p99 TTFT usually goes wrong under
mixed traffic.

**Caching.** Automatic prefix caching is enabled on every model, and we pass the
savings through as a separate discounted `cached_prompt` SKU — typically 4-5x
cheaper than full-price prompt tokens. Agentic and multi-turn workloads with
large stable system prompts see the largest benefit. Cache hit rates are visible
to us per model and we will share them on request.

**Gateway.** A single-binary Go edge that streams SSE frames straight through
with per-frame flushing and no intermediate buffering, keeps warm pooled
connections to every replica (so steady-state requests pay no handshake), and
disables compression on the streaming path. It emits SSE keep-alive comments
during long prefills and reasoning phases, so a slow-thinking model is never
mistaken for a hung connection and cancelled.

**Load shedding, not queueing.** We understand that throughput is measured as
output tokens over total generation time, so queueing is indistinguishable from
being slow. Every replica has a hard admission limit set at or below its
`--max-num-seqs`. Past that we return an immediate 429 with `Retry-After` so you
can route elsewhere, rather than accepting a request we would serve slowly. We
would rather lose the request than the metric.

**Rate limits.** Per-model limits are declared in the `capacity` blocks of our
`/v1/models` document and kept current. Current headline figures are
[**FILL IN** — e.g. 3M prompt TPM / 800k completion TPM / 4,000 RPM] on
Qwen3-32B. We will raise these on request as we add replicas; tell us what you
need and we will provision for it.

**Observability.** We track per-request TTFT, throughput, token counts, cache
hit rate and error class, and we compute uptime with your formula — successes
over requests excluding user errors — so our internal numbers and your provider
dashboard should agree. If they diverge we want to know immediately.

**Volume discounts.** Available on committed monthly volume. We can also expose
a discount directly to users through `discount_to_user` on selected models
during launch periods.

**Capacity plan.** [**FILL IN** — how many GPUs today, what you can add, and how
fast. Be specific and be honest; this is the answer that decides whether they
believe you can absorb the traffic they would send.]

---

## Data Policy

> This must match `site/privacy.html` and the `compliance.zdr` flag in
> `config/config.yaml`. Pick **one** of the two options below and delete the other.

### Option A — Zero data retention (recommended)

We operate zero data retention. Prompts and completions are held in volatile
memory only for the lifetime of the request and are never written to disk. We do
not log prompt or completion content. We do not train on, fine-tune with, or
otherwise use customer data for model development, and we do not share it with
any third party.

We retain request *metadata* only: timestamp, model, token counts, latency,
HTTP status and error class. This contains no prompt or completion content and
is kept for 30 days for billing and reliability monitoring, then deleted.

Our `/v1/models` document declares `compliance.zdr: true` on every model.

### Option B — Limited retention

We do not train on prompts or completions, and we do not share them with third
parties. Prompt and completion content is retained for **[N] hours** solely for
abuse investigation and debugging, encrypted at rest, then permanently deleted.
Request metadata (timestamp, model, token counts, latency, status) is retained
for 30 days for billing and reliability monitoring.

If you choose Option B, set `compliance.zdr: false` on every model — declaring
ZDR while retaining content is a misrepresentation that will surface the first
time an enterprise customer audits you.

The same choice has to be reflected in `site/privacy.html`, which offers both
sections and expects you to delete the one you do not operate.

---

## Before you submit — checklist

- [ ] `/v1/models` is reachable from the public internet **without** credentials
- [ ] `schema_version` is `"2.4"` and the document validates (Admin → Model Doc)
- [ ] Every model listed actually responds; no advertised model 404s
- [ ] Streaming works end to end, including through your TLS terminator
- [ ] Your reverse proxy does **not** buffer SSE (see `docs/deploy.md`)
- [ ] Privacy policy and terms are live at the URLs you gave
- [ ] The Data Policy answer matches both the privacy policy and the `zdr` flags
- [ ] A dedicated API key exists for OpenRouter (Admin → API Keys)
- [ ] Billing is arranged — they require auto top-up or invoicing to pay you
- [ ] `is_ready: false` on anything not ready to take live traffic
