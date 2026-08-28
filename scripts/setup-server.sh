#!/usr/bin/env bash
# Install llmfast on a dedicated Linux server, as a system service.
#
#   sudo bash scripts/setup-server.sh
#
# Unlike setup-pod.sh, which sets up a disposable RunPod container, this
# installs into the filesystem locations a long-lived machine expects and runs
# everything under systemd: restart on failure, start on boot, journald logs.
#
# Idempotent. Re-run it after a `git pull` to rebuild and restart.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"

GO_VERSION="${GO_VERSION:-1.24.5}"
USER_NAME="${LLMFAST_USER:-llmfast}"
PREFIX=/opt/llmfast          # binaries
CONFDIR=/etc/llmfast         # config and secrets
STATEDIR=/var/lib/llmfast    # database and model definitions
AGENTDIR=/var/lib/llmfast-agent  # engine state and model weights

say()  { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$1"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$1"; }

[ "$(id -u)" = 0 ] || { echo "Run as root: sudo bash $0"; exit 1; }
[ -f "$REPO/go.mod" ] || { echo "Run this from inside the llmfast checkout."; exit 1; }

# ---------------------------------------------------------------- packages ---

say "System packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq curl ca-certificates git build-essential cmake libcurl4-openssl-dev >/dev/null
ok "build tools"

# -------------------------------------------------------------------- user ---

say "Service account"
if ! id -u "$USER_NAME" >/dev/null 2>&1; then
  # A system account with no login shell: this process faces the internet, and
  # nothing about it needs a home directory or a password.
  useradd --system --no-create-home --shell /usr/sbin/nologin "$USER_NAME"
  ok "created $USER_NAME"
else
  ok "$USER_NAME exists"
fi
install -d -o "$USER_NAME" -g "$USER_NAME" -m 0750 "$STATEDIR" "$AGENTDIR" "$STATEDIR/models.d"
install -d -m 0755 "$PREFIX"
# root owns the config, the service account reads it through the group. The
# directory needs the group too: 0750 root:root would leave the files readable
# in principle and the directory untraversable in practice, which fails as
# "permission denied" on a file whose own mode looks correct.
install -d -o root -g "$USER_NAME" -m 0750 "$CONFDIR"

# --------------------------------------------------------------------- Go ----

say "Go ${GO_VERSION}"
if ! command -v go >/dev/null 2>&1 && [ ! -x /usr/local/go/bin/go ]; then
  curl -fsSL "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz" | tar -C /usr/local -xz
fi
export PATH="$PATH:/usr/local/go/bin"
ok "$(go version)"

say "Building"
( cd "$REPO" && make build )
install -m 0755 "$REPO/dist/llmfast" "$REPO/dist/llmfast-agent" "$REPO/dist/llmplan" "$PREFIX/"
ok "installed into $PREFIX"

# --------------------------------------------------------------- llama.cpp ---

say "llama.cpp"
# A CPU server serves through llama.cpp. It is built from source rather than
# installed from a package because distribution builds lag badly and a model
# released last month will not load in a build from last year.
if [ ! -x "$PREFIX/llama-server" ] || [ "${REBUILD_LLAMA:-0}" = 1 ]; then
  SRC=/usr/local/src/llama.cpp
  if [ -d "$SRC/.git" ]; then
    git -C "$SRC" pull -q
  else
    git clone -q --depth 1 https://github.com/ggml-org/llama.cpp.git "$SRC"
  fi
  cmake -S "$SRC" -B "$SRC/build" -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF >/dev/null
  cmake --build "$SRC/build" --config Release -j"$(nproc)" \
    --target llama-server llama-quantize >/dev/null
  install -m 0755 "$SRC/build/bin/llama-server" "$SRC/build/bin/llama-quantize" "$PREFIX/"
fi
ln -sf "$PREFIX/llama-server" /usr/local/bin/llama-server
ln -sf "$PREFIX/llama-quantize" /usr/local/bin/llama-quantize
ok "$("$PREFIX/llama-server" --version 2>&1 | head -1)"

# ------------------------------------------------------------------ config ---

say "Configuration"
if [ ! -f "$CONFDIR/env" ]; then
  cat > "$CONFDIR/env" <<ENVEOF
# Generated once. Keep it: regenerating changes your admin token.
LLMFAST_ADMIN_TOKEN=$(openssl rand -hex 32)
LLMFAST_AGENT_TOKEN=$(openssl rand -hex 32)
# Uncomment for gated models:
# HF_TOKEN=hf_...
ENVEOF
  chmod 0640 "$CONFDIR/env"; chown root:"$USER_NAME" "$CONFDIR/env"
  ok "wrote $CONFDIR/env"
else
  ok "$CONFDIR/env exists, keeping it"
fi
cp -f "$CONFDIR/env" "$CONFDIR/agent.env"
chmod 0640 "$CONFDIR/agent.env"; chown root:"$USER_NAME" "$CONFDIR/agent.env"

if [ ! -f "$CONFDIR/config.yaml" ]; then
  CORES=$(nproc)
  cat > "$CONFDIR/config.yaml" <<CFGEOF
provider:
  slug: llmfast
  display_name: LLMFast
  public_url: https://api.example.com   # <-- change to your domain

server:
  # Both listeners stay on localhost. A reverse proxy or tunnel is what faces
  # the internet, so nothing here needs an open port.
  listen: "127.0.0.1:8080"
  admin_listen: "127.0.0.1:8090"
  admin_token: "\$LLMFAST_ADMIN_TOKEN"
  db_path: "$STATEDIR/llmfast.db"
  model_dir: "$STATEDIR/models.d"
  keepalive_interval: 10s
  raw_retention_days: 30

nodes:
  - name: cpu-a
    url: http://127.0.0.1:9900
    token: "\$LLMFAST_AGENT_TOKEN"
    # Set this from the Benchmark tab, at the knee where aggregate throughput
    # stops improving. CPU batching gains far less than a GPU's.
    max_concurrency: 4

models: []
CFGEOF
  chmod 0640 "$CONFDIR/config.yaml"; chown root:"$USER_NAME" "$CONFDIR/config.yaml"
  ok "wrote $CONFDIR/config.yaml -- set public_url before publishing"
else
  ok "$CONFDIR/config.yaml exists, keeping it"
fi

# ----------------------------------------------------------------- systemd ---

# Correct permissions on every run: an earlier version of this script created
# $CONFDIR as root:root, which the service account cannot traverse.
chown root:"$USER_NAME" "$CONFDIR"
chmod 0750 "$CONFDIR"
for f in "$CONFDIR"/config.yaml "$CONFDIR"/env "$CONFDIR"/agent.env; do
  [ -f "$f" ] || continue
  chown root:"$USER_NAME" "$f"
  chmod 0640 "$f"
done
ok "permissions on $CONFDIR"

say "systemd units"
sed -e "s|__PREFIX__|$PREFIX|g" -e "s|__CONFDIR__|$CONFDIR|g" \
    -e "s|__STATEDIR__|$STATEDIR|g" -e "s|__USER__|$USER_NAME|g" \
    "$REPO/deploy/llmfast.service" > /etc/systemd/system/llmfast.service
sed -e "s|__PREFIX__|$PREFIX|g" -e "s|__CONFDIR__|$CONFDIR|g" \
    -e "s|__AGENTDIR__|$AGENTDIR|g" -e "s|__USER__|$USER_NAME|g" \
    "$REPO/deploy/llmfast-agent.service" > /etc/systemd/system/llmfast-agent.service
systemctl daemon-reload
systemctl enable --now llmfast-agent.service llmfast.service
sleep 3
for u in llmfast-agent llmfast; do
  if systemctl is-active --quiet "$u"; then ok "$u running"
  else warn "$u failed -- journalctl -u $u -n 40"; fi
done

# ----------------------------------------------------------------- logrotate --

cat > /etc/logrotate.d/llmfast <<'LOGEOF'
/var/log/llmfast/*.log {
    daily
    rotate 14
    compress
    missingok
    notifempty
    copytruncate
}
LOGEOF

say "Done"
cat <<NEXT

  Admin token:
    sudo grep LLMFAST_ADMIN_TOKEN $CONFDIR/env

  Reach the admin UI from your own machine (it is not exposed):
    ssh -L 8090:127.0.0.1:8090 root@this-server
    then open http://localhost:8090

  Service control:
    systemctl status llmfast llmfast-agent
    journalctl -u llmfast -f
    systemctl reload llmfast          # re-read models without dropping streams

  Before publishing, set public_url in $CONFDIR/config.yaml and put a
  reverse proxy or Cloudflare tunnel in front of 127.0.0.1:8080.
  See docs/dedicated-server.md.

NEXT
