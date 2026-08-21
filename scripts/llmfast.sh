#!/usr/bin/env bash
# Start, stop and inspect the three llmfast processes.
#
#   ./scripts/llmfast.sh start    launch everything, detached
#   ./scripts/llmfast.sh status   what is running, and is it healthy
#   ./scripts/llmfast.sh logs     attach to the running session
#   ./scripts/llmfast.sh stop     shut it all down
#
# Everything runs inside one tmux session, so closing your browser or terminal
# does not stop it. Nothing here survives the pod itself restarting — run
# `start` again after that.
set -uo pipefail

WORKDIR="${WORKDIR:-/workspace}"
REPO="${REPO:-$WORKDIR/llmfast}"
SESSION="llmfast"
TUNNEL="${TUNNEL_NAME:-llmfast}"

say() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }

need_tmux() {
  command -v tmux >/dev/null 2>&1 && return 0
  say "Installing tmux"
  apt-get update -qq && apt-get install -y -qq tmux >/dev/null
}

start() {
  need_tmux
  if [ ! -f "$WORKDIR/llmfast.env" ]; then
    echo "No $WORKDIR/llmfast.env. Run scripts/setup-pod.sh first."
    exit 1
  fi
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Already running. Use 'status' to check it, or 'stop' first."
    exit 0
  fi

  # shellcheck disable=SC1090
  source "$WORKDIR/llmfast.env"

  say "Starting"
  # A detached session with one window per process, so `logs` can show any of
  # them and a crash in one is visible rather than silent.
  tmux new-session -d -s "$SESSION" -n agent \
    "source $WORKDIR/llmfast.env; exec $REPO/dist/llmfast-agent \
       -listen 127.0.0.1:9900 -name gpu-a \
       -state-dir $WORKDIR/state -hf-cache $WORKDIR/hf -mode native"

  # The gateway polls the agent, so give the agent a moment to bind first.
  sleep 2
  tmux new-window -t "$SESSION" -n gateway \
    "cd $WORKDIR; source $WORKDIR/llmfast.env; exec $REPO/dist/llmfast -config $WORKDIR/config.yaml"

  sleep 2
  if [ -f /root/.cloudflared/config.yml ]; then
    tmux new-window -t "$SESSION" -n tunnel "exec cloudflared tunnel run $TUNNEL"
  else
    echo "  no /root/.cloudflared/config.yml, skipping the tunnel"
    echo "  run scripts/setup-tunnel.sh first if you want a public hostname"
  fi

  sleep 3
  status
}

status() {
  say "Processes"
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux list-windows -t "$SESSION" -F '  #{window_name}: #{?window_active,active,running} (#{window_panes} pane)'
  else
    echo "  not running"
    return
  fi

  say "Health"
  probe() {
    local name=$1 url=$2
    local code
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$url" 2>/dev/null)
    if [ "$code" = "200" ]; then
      printf '  %-22s ok\n' "$name"
    else
      printf '  %-22s NOT RESPONDING (%s)\n' "$name" "${code:-no answer}"
    fi
  }
  probe "agent    :9900" http://127.0.0.1:9900/health
  probe "gateway  :8080" http://127.0.0.1:8080/health
  probe "admin    :8090" http://127.0.0.1:8090/

  # The public hostname is the one that actually matters to OpenRouter.
  local host
  host=$(grep -m1 'hostname:' /root/.cloudflared/config.yml 2>/dev/null | awk '{print $3}')
  if [ -n "${host:-}" ]; then
    probe "public   $host" "https://$host/v1/models"
  fi

  say "Models"
  curl -s --max-time 5 http://127.0.0.1:8080/v1/models 2>/dev/null \
    | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
except Exception:
    print('  gateway not answering'); raise SystemExit
if not d.get('data'):
    print('  none installed yet')
for m in d.get('data', []):
    print(f\"  {m['id']:24s} ready={str(m.get('is_ready')):5s} ctx={m['input_modalities'][0]['supported_inputs']['max_context_length']['value']:,}\")
" 2>/dev/null || echo "  gateway not answering"
}

case "${1:-status}" in
  start)  start ;;
  stop)   tmux kill-session -t "$SESSION" 2>/dev/null && echo "stopped" || echo "was not running" ;;
  status) status ;;
  logs)   tmux attach -t "$SESSION" ;;
  restart) tmux kill-session -t "$SESSION" 2>/dev/null; sleep 1; start ;;
  *)      echo "usage: $0 {start|stop|restart|status|logs}"; exit 2 ;;
esac
