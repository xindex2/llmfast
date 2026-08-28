# Running llmfast on a dedicated CPU server

Written for a **Dual Xeon E5-2660v2, 128GB RAM, 480GB SSD + 480GB NVMe + 2TB
NVMe**. Most of it applies to any dedicated Linux box; the parts that depend on
having no GPU are called out.

Read the first section before installing anything. It decides whether this is
worth doing at all, and the answer is narrower than you might expect.

## What this hardware can actually serve

Twenty cores across two sockets, DDR3-1866 in quad channel: about **60 GB/s of
memory bandwidth per socket**. Token generation is bandwidth-bound — every
token reads the model's active weights — so that number, not the core count,
sets the speed.

Planned against real checkpoints at a 8192 context:

| Model | Fits | Single-stream |
|---|---|---|
| Qwen3-1.7B | viable | 33 tok/s |
| Qwen3-4B | fits | 17 tok/s |
| **Qwen3-30B-A3B** (3B active) | fits | **20 tok/s** |
| Qwen3-8B | fits | 8 tok/s |
| Qwen3-14B | fits | 5 tok/s |

The ordering is not a mistake. A **30B mixture-of-experts beats a 14B dense
model by four times**, because it activates only 3B parameters per token. On a
bandwidth-starved machine, active parameters are the only number that matters.

**So: serve a small-active MoE, or serve nothing.** A dense 14B at 5 tok/s is
below every threshold that matters.

## Whether it pays

OpenRouter deprioritises providers below roughly 12 tok/s, and the median for a
30B-class model is 42–50. At 20 tok/s single-stream you are in the bottom
quartile — you will be routed to, but not often, and mostly by price-sensitive
traffic.

Revenue at ~34 tok/s aggregate (four concurrent; CPU batching gains far less
than a GPU's):

| Utilisation | Revenue/month | Server must cost less than |
|---|---|---|
| 25% | $63 | $63/mo |
| 50% | $126 | $126/mo |
| 100% | $251 | $251/mo |

That is for `qwen3-30b-a3b-thinking-2507` at $2.40/M output with one
competitor. On a $0.28/M model with four competitors the ceiling is $188/mo at
*full* utilisation, which no new provider sees.

**The honest summary:** if this server costs you under about $60/month, a 30B
MoE can be marginally profitable. If it costs more, or you were going to serve
a dense model, it will not pay for itself. A $321/month A40 does roughly eight
times the throughput.

Where this box *is* genuinely good: a private endpoint for your own tools, a
staging environment, or batch work where nobody is waiting.

## Install

One command, as root, from a checkout:

```bash
git clone https://github.com/xindex2/llmfast.git /opt/src/llmfast
cd /opt/src/llmfast
sudo bash scripts/setup-server.sh
```

It installs Go, builds the binaries into `/opt/llmfast`, builds llama.cpp from
source, creates a `llmfast` system account, generates `/etc/llmfast/config.yaml`
and a random admin token, and starts both services under systemd.

llama.cpp is built rather than packaged deliberately: distribution builds lag
by months, and a model released this quarter will not load in last year's
build. Expect five to fifteen minutes on twenty cores.

Re-run the same command after a `git pull` to rebuild and restart.

### Where things live

| Path | What |
|---|---|
| `/opt/llmfast/` | binaries: gateway, agent, `llama-server` |
| `/etc/llmfast/config.yaml` | configuration |
| `/etc/llmfast/env` | admin and agent tokens, mode 0640 |
| `/var/lib/llmfast/` | SQLite database, model definitions |
| `/var/lib/llmfast-agent/hf/` | downloaded weights — put this on the 2TB NVMe |

### Put the weights on the fast disk

Weights are read once at load, but a 30B checkpoint is 17GB at 4-bit and you
will hold several. Before installing any model:

```bash
systemctl stop llmfast-agent
mkdir -p /mnt/nvme2/llmfast-hf
rsync -a /var/lib/llmfast-agent/hf/ /mnt/nvme2/llmfast-hf/ 2>/dev/null || true
rm -rf /var/lib/llmfast-agent/hf
ln -s /mnt/nvme2/llmfast-hf /var/lib/llmfast-agent/hf
chown -R llmfast:llmfast /mnt/nvme2/llmfast-hf
systemctl start llmfast-agent
```

Keep the SQLite database on the SSD, not the 2TB drive. It takes small
synchronous writes on every request, which is a latency workload, not a
throughput one.

## Reaching it

Both listeners bind `127.0.0.1` on purpose. Nothing is exposed until you choose
how.

**The API must be public. The admin UI must not be**, or must have identity in
front of it — it holds your API keys, your request history, and the ability to
stop your models.

### Option A — Caddy (simplest real TLS)

```bash
apt-get install -y caddy
cat > /etc/caddy/Caddyfile <<'EOF'
api.example.com {
    reverse_proxy 127.0.0.1:8080 {
        # Streaming must not be buffered: a buffered proxy turns per-token
        # frames into one response at the end, which destroys time-to-first-
        # token and with it your routing position.
        flush_interval -1
    }
}
EOF
systemctl reload caddy
```

Caddy obtains and renews the certificate itself. Admin stays unreachable; get
to it over SSH:

```bash
ssh -L 8090:127.0.0.1:8090 root@your-server
# then http://localhost:8090
```

### Option B — Cloudflare Tunnel

No open ports at all, and it works behind NAT:

```bash
cloudflared tunnel login
cloudflared tunnel create llmfast
bash /opt/src/llmfast/scripts/setup-tunnel.sh llmfast api.example.com
```

Deliberately without an admin hostname. If you want one, add Cloudflare Access
in front of it first — see the "Decide how you reach the admin UI" section of
the [README](../README.md).

### Firewall

```bash
ufw default deny incoming
ufw allow 22/tcp
ufw allow 80,443/tcp     # only with Caddy; omit entirely with a tunnel
ufw enable
```

Verify nothing else listens publicly:

```bash
ss -lptn | grep -v 127.0.0.1
```

## Production checklist

Work through this before you publish a model.

### Services survive reboots

```bash
systemctl is-enabled llmfast llmfast-agent    # both: enabled
systemctl status llmfast llmfast-agent
```

The units restart on failure, wait 45s for in-flight generations to drain
rather than cutting streams mid-response, and raise the file-descriptor limit
to 65535 — streaming holds a connection per in-flight request in each
direction, and the default 1024 is far too low.

The agent runs with `OOMScoreAdjust=-500`. Without it the OOM killer picks the
largest resident process, which is always the engine, so any memory spike
anywhere on the box kills your model rather than the thing that caused it.

### Back up the database

`/var/lib/llmfast/llmfast.db` holds your admin accounts, API key hashes and
request history. Losing it means re-creating accounts and re-issuing keys.

```bash
cat > /etc/cron.daily/llmfast-backup <<'EOF'
#!/bin/sh
d=/var/backups/llmfast; mkdir -p "$d"
# .backup is safe on a live database; copying the file is not, because WAL
# writes can land mid-copy.
/usr/bin/sqlite3 /var/lib/llmfast/llmfast.db ".backup '$d/llmfast-$(date +%F).db'"
find "$d" -name 'llmfast-*.db' -mtime +14 -delete
EOF
chmod +x /etc/cron.daily/llmfast-backup
apt-get install -y sqlite3
```

Also copy `/etc/llmfast/env` somewhere safe, once. It is not regenerable.

### Watch the things that decide your routing

Uptime and throughput are what OpenRouter pays on, so alert on them rather than
on CPU graphs:

```bash
# Is the public endpoint answering?
curl -sf https://api.example.com/v1/models >/dev/null || echo DOWN

# Is a model actually ready, or is the engine restarting?
curl -s http://127.0.0.1:8080/health | grep -q '"status":"ok"' || echo DEGRADED
```

Put those two in cron with an alert you will actually see. A model that dies at
3am costs a day of routing before you notice — and the Requests tab will show
the errors, but only if you look.

Logs are in journald:

```bash
journalctl -u llmfast -f
journalctl -u llmfast-agent --since "1 hour ago"
```

### Set concurrency from measurement, not from the core count

Install a model, then use the **Benchmark** tab at `1, 2, 4, 8`. Find where
aggregate throughput stops climbing and put that number in `max_concurrency` in
`/etc/llmfast/config.yaml`. On CPU the knee is low — usually 2 to 4 — because
batching cannot buy back memory bandwidth that is already saturated.

```bash
systemctl reload llmfast    # re-reads models and pricing without dropping streams
```

### Declare what you can serve

Set the context to what the machine actually holds and no more. A prompt longer
than your declared context is **rejected**, and that is an error against your
uptime; low concurrency merely queues. On 128GB you have room for a large KV
cache, so 32768 is comfortable — but check the Inspect screen rather than
assuming.

## Upgrading

```bash
cd /opt/src/llmfast && git pull
sudo bash scripts/setup-server.sh     # rebuilds and restarts
```

To rebuild llama.cpp as well — worth doing when a model is too new to load:

```bash
sudo REBUILD_LLAMA=1 bash scripts/setup-server.sh
```

## When something is wrong

| Symptom | Where to look |
|---|---|
| Model will not start | `journalctl -u llmfast-agent -n 100`, then the Nodes page — it shows the engine's own error |
| `unreachable` node | `systemctl status llmfast-agent`; the gateway polls `127.0.0.1:9900` |
| Slow first token | A buffering proxy. Caddy needs `flush_interval -1` |
| 401 from the admin UI | `grep LLMFAST_ADMIN_TOKEN /etc/llmfast/env` |
| Model too new to load | `sudo REBUILD_LLAMA=1 bash scripts/setup-server.sh` |
