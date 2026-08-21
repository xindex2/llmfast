#!/usr/bin/env bash
# Start, stop and inspect the llmfast processes.
#
#   ./scripts/llmfast.sh start     launch everything, detached
#   ./scripts/llmfast.sh status    what is running, and is it healthy
#   ./scripts/llmfast.sh logs      follow the logs (Ctrl-C to stop watching)
#   ./scripts/llmfast.sh stop      shut it all down
#   ./scripts/llmfast.sh restart   after a git pull && make build
#
# Processes are detached with setsid, so closing the terminal or the browser
# does not stop them. Nothing survives the pod itself restarting -- run `start`
# again after that.
#
# Everything writes to a log file. An earlier version of this script ran the
# processes inside tmux, which destroyed the session the moment the first one
# exited -- taking the error message with it and reporting only "not running".
# Failures are now kept on disk where they can be read.
set -uo pipefail

WORKDIR="${WORKDIR:-/workspace}"
REPO="${REPO:-$WORKDIR/llmfast}"
LOGDIR="$WORKDIR/logs"
RUNDIR="$WORKDIR/run"
TUNNEL="${TUNNEL_NAME:-llmfast}"

RED=$'\033[31m'; GRN=$'\033[32m'; YEL=$'\033[33m'; BLD=$'\033[1m'; OFF=$'\033[0m'
say()  { printf '\n%s==> %s%s\n' "$BLD" "$1" "$OFF"; }
ok()   { printf '  %s✓%s %s\n' "$GRN" "$OFF" "$1"; }
bad()  { printf '  %s✗%s %s\n' "$RED" "$OFF" "$1"; }
warn() { printf '  %s!%s %s\n' "$YEL" "$OFF" "$1"; }

mkdir -p "$LOGDIR" "$RUNDIR"

# ---------------------------------------------------------------- process ---

pidfile() { echo "$RUNDIR/$1.pid"; }
logfile() { echo "$LOGDIR/$1.log"; }

# alive reports whether a service's recorded pid is still running.
alive() {
  local pf; pf=$(pidfile "$1")
  [ -f "$pf" ] || return 1
  local pid; pid=$(cat "$pf" 2>/dev/null)
  [ -n "$pid" ] || return 1
  kill -0 "$pid" 2>/dev/null
}

# launch starts one service detached, or explains why it could not.
launch() {
  local name=$1; shift
  if alive "$name"; then
    ok "$name already running (pid $(cat "$(pidfile "$name")"))"
    return 0
  fi
  local log; log=$(logfile "$name")
  # Keep the previous run's log; a crash loop is much easier to read with
  # history than without.
  [ -f "$log" ] && mv "$log" "$log.prev"
  # setsid puts the process in its own session, so no terminal hangup can ever
  # reach it and stop can signal the whole group. It is Linux-only; nohup is
  # the portable fallback and is enough to survive a closed terminal.
  if command -v setsid >/dev/null 2>&1; then
    setsid "$@" >"$log" 2>&1 < /dev/null &
  else
    nohup "$@" >"$log" 2>&1 < /dev/null &
  fi
  local pid=$!
  disown 2>/dev/null || true
  echo "$pid" > "$(pidfile "$name")"

  # Give it a moment to fail. Most failures here are immediate -- a missing
  # binary, a port already taken, a config it cannot parse.
  local i
  for i in 1 2 3 4 5 6; do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.5
  done
  if alive "$name"; then
    ok "$name started (pid $pid), logging to $log"
    return 0
  fi
  bad "$name exited immediately. Last lines of $log:"
  echo
  sed 's/^/      /' "$log" | tail -20
  echo
  rm -f "$(pidfile "$name")"
  return 1
}

# --------------------------------------------------------------- preflight ---

preflight() {
  local fail=0
  say "Checking"

  for b in llmfast llmfast-agent; do
    if [ -x "$REPO/dist/$b" ]; then
      ok "dist/$b"
    else
      bad "dist/$b is missing -- run: cd $REPO && make build"
      fail=1
    fi
  done

  if [ -f "$WORKDIR/llmfast.env" ]; then
    ok "llmfast.env"
  else
    bad "$WORKDIR/llmfast.env is missing -- run scripts/setup-pod.sh first"
    fail=1
  fi

  if [ -f "$WORKDIR/config.yaml" ]; then
    ok "config.yaml"
  else
    bad "$WORKDIR/config.yaml is missing -- run scripts/setup-pod.sh first"
    fail=1
  fi

  # A port held by something else is the single most common reason a process
  # dies half a second after it starts, and the least obvious from the outside.
  # It is also what happens when a previous run was left behind: closing a web
  # terminal does not always kill what was started in it.
  local p
  for p in 9900 8080 8090; do
    local holder=""
    if command -v ss >/dev/null 2>&1; then
      holder=$(ss -lptnH "sport = :$p" 2>/dev/null | head -1 | grep -o 'users:(("[^"]*"' | cut -d'"' -f2)
    elif command -v lsof >/dev/null 2>&1; then
      holder=$(lsof -nP -iTCP:"$p" -sTCP:LISTEN -Fc 2>/dev/null | grep '^c' | head -1 | cut -c2-)
    fi
    if [ -n "$holder" ]; then
      case "$holder" in
        *llmfast*)
          bad "port $p is held by an older llmfast process ($holder) -- run: bash $0 stop"
          fail=1 ;;
        *)
          bad "port $p is already taken by $holder"
          fail=1 ;;
      esac
    else
      ok "port $p is free"
    fi
  done

  if command -v cloudflared >/dev/null 2>&1; then
    if [ -f /root/.cloudflared/config.yml ]; then
      ok "cloudflared configured"
    else
      warn "no /root/.cloudflared/config.yml -- run scripts/setup-tunnel.sh to get a public hostname"
    fi
  else
    warn "cloudflared is not installed -- the API will only be reachable on this pod"
  fi

  return $fail
}

# ------------------------------------------------------------------ start ---

start() {
  preflight || { echo; bad "not starting -- fix the above first"; exit 1; }

  # shellcheck disable=SC1090
  set -a; source "$WORKDIR/llmfast.env"; set +a

  say "Starting"
  launch agent "$REPO/dist/llmfast-agent" \
    -listen 127.0.0.1:9900 -name gpu-a \
    -state-dir "$WORKDIR/state" -hf-cache "$WORKDIR/hf" -mode native || exit 1

  # The gateway contacts the agent on startup, so let the agent bind first.
  sleep 2
  ( cd "$WORKDIR" && launch gateway "$REPO/dist/llmfast" -config "$WORKDIR/config.yaml" ) || exit 1

  if [ -f /root/.cloudflared/config.yml ] && command -v cloudflared >/dev/null 2>&1; then
    sleep 1
    launch tunnel cloudflared tunnel run "$TUNNEL" || true
  fi

  sleep 3
  status
}

# ----------------------------------------------------------------- status ---

status() {
  say "Processes"
  local svc any=0
  for svc in agent gateway tunnel; do
    if alive "$svc"; then
      printf '  %s✓%s %-8s pid %s\n' "$GRN" "$OFF" "$svc" "$(cat "$(pidfile "$svc")")"
      any=1
    elif [ -f "$(pidfile "$svc")" ]; then
      # A pidfile with no process behind it means it died on its own. That is
      # the case worth showing the log for, and the case a clean stop is not:
      # stop removes the pidfile, so this cannot be confused with one.
      printf '  %s✗%s %-8s CRASHED -- last lines of %s:\n' "$RED" "$OFF" "$svc" "$(logfile "$svc")"
      sed 's/^/      /' "$(logfile "$svc")" 2>/dev/null | tail -12
    elif [ -f "$(logfile "$svc")" ]; then
      printf '  %s-%s %-8s stopped\n' "$YEL" "$OFF" "$svc"
    else
      printf '  %s-%s %-8s never started\n' "$YEL" "$OFF" "$svc"
    fi
  done
  [ "$any" = 1 ] || { echo; warn "nothing is running -- start it with: bash $0 start"; return; }

  say "Health"
  probe() {
    local name=$1 url=$2 code
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$url" 2>/dev/null)
    case "$code" in
      200) ok "$name" ;;
      # The gateway reports 503 "degraded" until a model is ready. That is the
      # correct answer to a load balancer and not a fault to report here: the
      # process is up and serving, it simply has nothing to route to yet.
      503) warn "$name (up, but no model is ready yet)" ;;
      *)   bad "$name (HTTP ${code:-no answer})" ;;
    esac
  }
  probe "agent    127.0.0.1:9900" http://127.0.0.1:9900/health
  probe "gateway  127.0.0.1:8080" http://127.0.0.1:8080/health
  probe "admin    127.0.0.1:8090" http://127.0.0.1:8090/

  # The public hostname is the one that decides whether OpenRouter can reach us.
  local host
  host=$(grep -m1 -E '^\s*hostname:' /root/.cloudflared/config.yml 2>/dev/null | awk '{print $2}')
  [ -n "${host:-}" ] && probe "public   $host" "https://$host/v1/models"

  # An admin hostname on the tunnel with nothing in front of it is the most
  # damaging misconfiguration available here: that page exposes API keys, the
  # full request history, and the ability to install and stop models. A bearer
  # token is one guess away from all of it, and it is reachable from anywhere.
  local admin_host
  admin_host=$(grep -B1 'service: http://127.0.0.1:8090' /root/.cloudflared/config.yml 2>/dev/null |
               grep -m1 -E 'hostname:' | awk '{print $2}')
  if [ -n "${admin_host:-}" ]; then
    say "Admin exposure"
    local loc
    loc=$(curl -s -o /dev/null -w '%{redirect_url}' --max-time 8 "https://$admin_host/" 2>/dev/null)
    if echo "$loc" | grep -q 'cloudflareaccess.com'; then
      ok "$admin_host is behind Cloudflare Access"
    else
      bad "$admin_host is PUBLIC -- nothing is checking identity in front of it"
      echo "      Anyone who finds this hostname reaches your admin login, and only"
      echo "      a bearer token stands between them and your API keys, your request"
      echo "      history, and the ability to stop your models."
      echo
      echo "      Simplest fix -- stop publishing it, and use SSH instead:"
      echo "        bash $REPO/scripts/setup-tunnel.sh ${TUNNEL} $(grep -m1 -A1 'ingress:' /root/.cloudflared/config.yml 2>/dev/null | grep hostname | awk '{print $3}')"
      echo "        bash $REPO/scripts/llmfast.sh restart"
      echo "      then from your own machine:"
      echo "        ssh -L 8090:127.0.0.1:8090 root@<pod-ip> -p <pod-port>"
      echo "        and open http://localhost:8090"
      echo
      echo "      Or keep the hostname and put Cloudflare Access in front of it."
      echo "      If its one-time-PIN emails never arrive, add Google as an identity"
      echo "      provider instead: Access -> Settings -> Authentication -> Add new."
    fi
  fi

  say "Models"
  curl -s --max-time 5 http://127.0.0.1:8080/v1/models 2>/dev/null | python3 -c '
import json, sys
try:
    data = (json.load(sys.stdin) or {}).get("data") or []
except Exception:
    print("  could not read the model list")
    raise SystemExit
if not data:
    print("  none installed yet -- add one in the admin UI")
for m in data:
    try:
        ctx = m["input_modalities"][0]["supported_inputs"]["max_context_length"]["value"]
        ctx = format(ctx, ",")
    except Exception:
        ctx = "?"
    ready = str(m.get("is_ready"))
    print("  %-28s ready=%-5s ctx=%s" % (m.get("id", "?"), ready, ctx))
' 2>/dev/null || echo "  could not read the model list"
}

# ------------------------------------------------------------------ stop ----

stop() {
  local svc stopped=0
  for svc in tunnel gateway agent; do
    if alive "$svc"; then
      local pid; pid=$(cat "$(pidfile "$svc")")
      # Negative pid signals the whole process group, so engines started by the
      # agent go down with it rather than being orphaned onto the GPU.
      kill -TERM "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
      ok "stopped $svc"
      stopped=1
    fi
    rm -f "$(pidfile "$svc")"
  done

  # Processes started by hand in a terminal that was later closed have no
  # pidfile here, but they still hold the ports and still block a clean start.
  local p
  for p in 9900 8080 8090; do
    local pids=""
    if command -v ss >/dev/null 2>&1; then
      pids=$(ss -lptnH "sport = :$p" 2>/dev/null | grep -o 'pid=[0-9]*' | cut -d= -f2)
    elif command -v lsof >/dev/null 2>&1; then
      pids=$(lsof -nP -iTCP:"$p" -sTCP:LISTEN -t 2>/dev/null)
    fi
    local pid
    for pid in $pids; do
      # Only ever our own binaries: something else on the port is the
      # operator's business, not ours to kill.
      local exe
      exe=$(ps -p "$pid" -o comm= 2>/dev/null)
      case "$exe" in
        *llmfast*|*cloudflared*)
          kill -TERM "$pid" 2>/dev/null && ok "stopped stray $exe on port $p (pid $pid)"
          stopped=1 ;;
      esac
    done
  done

  [ "$stopped" = 1 ] || echo "nothing was running"
}

# show_token prints the admin token the gateway is actually using, resolving it
# exactly the way the gateway does: the value in config.yaml wins, unless it
# names an environment variable, in which case llmfast.env supplies it.
show_token() {
  WORKDIR="$WORKDIR" python3 - <<'PYEOF'
import os, re, sys

work = os.environ["WORKDIR"]

def read(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return ""

def unquote(v):
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        v = v[1:-1]
    return v

tok = ""
for line in read(os.path.join(work, "config.yaml")).splitlines():
    m = re.match(r"\s*admin_token:\s*(.+?)\s*(?:#.*)?$", line)
    if m:
        tok = unquote(m.group(1))
        break

env = {}
for line in read(os.path.join(work, "llmfast.env")).splitlines():
    if "=" in line and not line.lstrip().startswith("#"):
        k, v = line.split("=", 1)
        env[k.strip()] = unquote(v)

# A leading $ means "read this from the environment", which is why a token that
# genuinely begins with $ cannot be written straight into config.yaml.
if tok.startswith("$"):
    tok = env.get(tok.lstrip("${").rstrip("}"), "")
elif not tok:
    tok = env.get("LLMFAST_ADMIN_TOKEN", "")

if not tok:
    print("  no admin token found in config.yaml or llmfast.env")
    sys.exit(1)

print()
print("  " + tok)
print()
print("  Paste exactly that into the admin login box.")
print("  No $ in front of it, no quotes around it.")
PYEOF
}

case "${1:-status}" in
  start)   start ;;
  token)   say "Admin token"; show_token ;;
  stop)    stop ;;
  restart) stop; sleep 2; start ;;
  status)  status ;;
  logs)    tail -n 40 -F "$LOGDIR"/agent.log "$LOGDIR"/gateway.log "$LOGDIR"/tunnel.log 2>/dev/null ;;
  *)       echo "usage: $0 {start|stop|restart|status|logs|token}"; exit 2 ;;
esac
