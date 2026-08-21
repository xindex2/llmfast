# Deployment

## Architecture

```
                    ┌──────────────────────────────────────────┐
   OpenRouter ──────▶  nginx / Caddy  (TLS, SSE buffering OFF) │
                    └───────────────────┬──────────────────────┘
                                        │ :8080  public API
                    ┌───────────────────▼──────────────────────┐
                    │  llmfast gateway (single Go binary)      │
                    │   • auth, admission control, streaming   │
                    │   • /v1/models  (OpenRouter schema 2.4)  │
                    │   • SQLite: keys, request log, rollups   │
                    │   • :8081  admin UI  (bind to localhost) │
                    └───────────────────┬──────────────────────┘
                                        │ private network
              ┌─────────────────────────┼─────────────────────────┐
              ▼                         ▼                         ▼
        vLLM  gpu-a               vLLM  gpu-b               vLLM  gpu-c
        Qwen3-32B                 GLM-4.6                   DeepSeek-V3
        (H100 ×2)                 (H100 ×2)                 (H200 ×8)
```

The gateway is CPU- and network-bound and holds no model state, so it belongs on
a small separate host — not on a GPU node, where it would compete for the CPU
that vLLM needs for tokenization and scheduling.

## 1. Build

The binary is pure Go with no cgo, so it cross-compiles from any machine:

```bash
make build-linux          # -> dist/llmfast-linux-amd64
```

## 2. Install on the gateway host

```bash
sudo useradd --system --no-create-home --shell /usr/sbin/nologin llmfast
sudo mkdir -p /opt/llmfast /etc/llmfast /var/lib/llmfast
sudo chown llmfast:llmfast /var/lib/llmfast

sudo install -m 0755 dist/llmfast-linux-amd64 /opt/llmfast/llmfast
sudo install -m 0640 -o llmfast -g llmfast config/config.yaml /etc/llmfast/config.yaml

# The admin token gates the dashboard. Generate a real one.
sudo sh -c 'echo "LLMFAST_ADMIN_TOKEN=$(openssl rand -hex 32)" > /etc/llmfast/env'
sudo chmod 0600 /etc/llmfast/env

sudo install -m 0644 deploy/llmfast.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now llmfast

# The bootstrap API key is printed once, on first start only.
sudo journalctl -u llmfast | grep sk-llmfast-
```

## 3. TLS and reverse proxy

**This is where provider latency usually dies.** nginx buffers proxied responses
by default, which batches SSE frames and turns a smooth token stream into
periodic bursts. It does not break anything visibly — it just makes your
measured throughput worse than the hardware you paid for.

`deploy/nginx.conf` has the correct settings. The critical lines:

```nginx
proxy_buffering off;        # do not accumulate the response before forwarding
proxy_request_buffering off;
proxy_cache off;
chunked_transfer_encoding on;
proxy_read_timeout 600s;    # long generations must not be cut off
proxy_http_version 1.1;
proxy_set_header Connection "";   # keep upstream connections alive
```

The gateway also sends `X-Accel-Buffering: no` on every stream, which nginx
honours — but set `proxy_buffering off` explicitly rather than relying on it,
because other proxies ignore the header.

Verify after deploying. Frames must arrive spread over time, not all at once:

```bash
curl -N -w '\n' https://api.your-domain.com/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model":"qwen/qwen3-32b","messages":[{"role":"user","content":"count to 20"}],"stream":true}' \
  | while IFS= read -r line; do printf '%s  %s\n' "$(date +%T.%3N)" "${line:0:60}"; done
```

## 4. Firewall

| Port | Exposure |
|---|---|
| 443 | Public. Ideally restricted to OpenRouter's egress ranges — ask them |
| 8080 | Localhost only; nginx proxies to it |
| 8081 | **Localhost only.** Never expose the admin UI |
| 8000 (vLLM) | Private network only. vLLM has no authentication of its own |

Reach the admin UI over an SSH tunnel:

```bash
ssh -L 8081:127.0.0.1:8081 gateway-host
# then open http://localhost:8081
```

## 5. GPU nodes

The recommended route is to run `llmfast-agent` on each node and install models
from the admin UI: see [nodes.md](nodes.md). The agent detects the hardware,
sizes the model against it, and launches the engine with the right flags, so the
list below is handled for you.

To manage engines yourself instead, see the **Guides** tab in the admin UI or
`docs/models.md`. Each node then needs:

- vLLM reachable from the gateway on the private network
- `--served-model-name` set (matching the public model id avoids a rewrite)
- `--max-num-seqs` at or above the backend's `max_concurrency` in `config.yaml`
- A systemd unit so it restarts on failure

## 6. Operations

```bash
# Reload models and pricing without dropping in-flight streams.
sudo systemctl reload llmfast     # sends SIGHUP

# Changing `backends` needs a restart: connection pools and admission
# counters are rebuilt.
sudo systemctl restart llmfast

# Health, including per-backend status.
curl -s localhost:8080/health | jq
```

### Backups

Everything durable is one SQLite file. Back it up with the online backup API,
which is safe against a running writer — copying the file directly while the WAL
is active can produce a corrupt snapshot:

```bash
sqlite3 /var/lib/llmfast/llmfast.db ".backup '/backup/llmfast-$(date +%F).db'"
```

API keys are stored only as hashes, so a stolen backup does not yield working
credentials — but it does contain your full request metadata history.

### When to outgrow SQLite

SQLite handles millions of requests per day on one gateway host. Move the store
to Postgres or ClickHouse when either becomes true:

- You run more than one gateway instance and need shared stats
- Dashboard queries over 30 days start taking more than a second

The `store` package is the only thing that touches SQL, so the change is
contained to it.

## 7. Scaling out

The gateway is stateless apart from its SQLite file, so several instances can
sit behind a load balancer. Two things need attention when you do:

1. **Admission counters are per instance.** Two gateways each allowing 96
   concurrent requests to one replica will send it 192. Divide
   `max_concurrency` by the number of gateway instances.
2. **Stats are per instance.** Each writes its own SQLite file. Either accept
   per-instance dashboards or move the store to a shared database.

Until you are past a few thousand requests per minute, one well-provisioned
gateway host is simpler and faster than a fleet.
