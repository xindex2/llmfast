#!/usr/bin/env bash
# Prepare a fresh RunPod pod to serve models.
#
# Installs Go, vLLM and cloudflared, builds the gateway and agent, and writes a
# config. It does not start anything: read what it prints, then start the pieces
# yourself so you can see each one come up.
#
# Run it from the pod's web terminal or over SSH:
#   curl -fsSL https://raw.githubusercontent.com/xindex2/llmfast/main/scripts/setup-pod.sh | bash
set -euo pipefail

WORKDIR="${WORKDIR:-/workspace}"
GO_VERSION="${GO_VERSION:-1.24.5}"

say() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }

# /workspace is the persistent volume on RunPod. Everything lives there so a
# pod restart does not throw away the weights, which is the slow part.
if [ ! -d "$WORKDIR" ]; then
  echo "No $WORKDIR directory. On RunPod that means no volume is attached."
  echo "Attach one of at least 60GB and mount it at /workspace, then run this again."
  exit 1
fi
cd "$WORKDIR"

say "System packages"
apt-get update -qq
apt-get install -y -qq git curl ca-certificates >/dev/null

say "Go ${GO_VERSION}"
if ! command -v go >/dev/null 2>&1; then
  curl -fsSL "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz" | tar -C /usr/local -xz
fi
export PATH=$PATH:/usr/local/go/bin
grep -q '/usr/local/go/bin' ~/.bashrc 2>/dev/null || echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
go version

say "llmfast"
if [ -d "$WORKDIR/llmfast/.git" ]; then
  cd "$WORKDIR/llmfast" && git pull --ff-only
else
  git clone -q https://github.com/xindex2/llmfast.git "$WORKDIR/llmfast"
  cd "$WORKDIR/llmfast"
fi
make build
echo "built: $(ls dist/)"

say "vLLM"
# RunPod pods cannot normally run Docker inside themselves, so the engine is
# installed directly and the agent runs in native mode.
if command -v vllm >/dev/null 2>&1; then
  echo "already installed: $(vllm --version 2>/dev/null | head -1)"
else
  pip install -q --upgrade pip
  pip install -q vllm
  echo "installed: $(vllm --version 2>/dev/null | head -1)"
fi

say "cloudflared"
if ! command -v cloudflared >/dev/null 2>&1; then
  curl -fsSL -o /usr/local/bin/cloudflared \
    https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
  chmod +x /usr/local/bin/cloudflared
fi
cloudflared --version

say "Tokens and config"
mkdir -p "$WORKDIR/state" "$WORKDIR/hf" "$WORKDIR/models.d"

ENV_FILE="$WORKDIR/llmfast.env"
if [ ! -f "$ENV_FILE" ]; then
  cat > "$ENV_FILE" <<ENVEOF
# Generated once. Keep it; regenerating changes your admin password.
export LLMFAST_AGENT_TOKEN=$(openssl rand -hex 32)
export LLMFAST_ADMIN_TOKEN=$(openssl rand -hex 32)
# Uncomment and fill in if you serve a gated model.
# export HF_TOKEN=hf_...
ENVEOF
  chmod 600 "$ENV_FILE"
  echo "wrote $ENV_FILE"
else
  echo "$ENV_FILE already exists, keeping it"
fi

CONFIG="$WORKDIR/config.yaml"
if [ ! -f "$CONFIG" ]; then
  cat > "$CONFIG" <<'CFGEOF'
provider:
  slug: llmfast
  display_name: LLMFast
  public_url: https://api.llmfa.st

server:
  # Both listeners stay on localhost. The Cloudflare tunnel is what faces the
  # internet, so nothing here needs an open port.
  listen: "127.0.0.1:8080"
  admin_listen: "127.0.0.1:8081"
  admin_token: "$LLMFAST_ADMIN_TOKEN"
  db_path: "/workspace/llmfast.db"
  model_dir: "models.d"
  keepalive_interval: 10s
  raw_retention_days: 30

nodes:
  - name: gpu-a
    url: http://127.0.0.1:9900
    token: $LLMFAST_AGENT_TOKEN
    # Raise or lower this after running the Benchmark tab: set it at the knee
    # where aggregate throughput stops improving.
    max_concurrency: 13

models: []
CFGEOF
  echo "wrote $CONFIG"
else
  echo "$CONFIG already exists, keeping it"
fi

say "Hardware detected"
"$WORKDIR/llmfast/dist/llmfast-agent" -hardware -state-dir "$WORKDIR/state" || true

cat <<'NEXT'

Ready. Three things to start, each in its own terminal tab.

  source /workspace/llmfast.env

  1) the agent — supervises engines
     /workspace/llmfast/dist/llmfast-agent \
       -listen 127.0.0.1:9900 -name gpu-a \
       -state-dir /workspace/state -hf-cache /workspace/hf -mode native

  2) the gateway — the API and admin UI
     cd /workspace && /workspace/llmfast/dist/llmfast -config /workspace/config.yaml

  3) the tunnel — puts api.llmfa.st in front of it
     cloudflared tunnel login
     cloudflared tunnel create llmfast
     cloudflared tunnel route dns llmfast api.llmfa.st
     cloudflared tunnel --url http://127.0.0.1:8080 run llmfast

Your admin token is in /workspace/llmfast.env. Print it with:
  source /workspace/llmfast.env && echo $LLMFAST_ADMIN_TOKEN

NEXT
