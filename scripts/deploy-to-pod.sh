#!/usr/bin/env bash
# Build and ship llmfast to a RunPod pod.
#
# RUN THIS ON YOUR OWN COMPUTER, from inside the llmfast repository — not on the
# pod. It cross-compiles the Linux binaries, copies them and the setup script
# up, and then runs the setup remotely.
#
#   ./scripts/deploy-to-pod.sh 69.30.85.5 22105
#
# The address and port are the ones RunPod shows under "Direct TCP ports".
set -euo pipefail

IP="${1:-}"
PORT="${2:-22}"
USER="${SSH_USER:-root}"
REMOTE="${REMOTE_DIR:-/workspace}"

if [ -z "$IP" ]; then
  cat <<'USAGE'
usage: ./scripts/deploy-to-pod.sh <ip> <port>

Take both from the RunPod console, under "Direct TCP ports". For an entry
reading  SSH -> 69.30.85.5:22105 -> :22  the command is:

    ./scripts/deploy-to-pod.sh 69.30.85.5 22105
USAGE
  exit 2
fi

# Guard against the commonest mistake: running this on the pod itself.
if [ -f /.dockerenv ] || [ -d /workspace/llmfast ] && [ ! -f go.mod ]; then
  echo "This looks like the pod, not your own computer."
  echo "Run it from the llmfast repository on the machine you develop on."
  exit 1
fi
if [ ! -f go.mod ] || [ ! -d scripts ]; then
  echo "Run this from inside the llmfast repository (the folder with go.mod)."
  echo "You are currently in: $(pwd)"
  exit 1
fi

say() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }

say "Checking the pod is reachable"
if ! ssh -o BatchMode=yes -o ConnectTimeout=10 -p "$PORT" "$USER@$IP" true 2>/dev/null; then
  cat <<SSHFAIL

Could not SSH to $USER@$IP on port $PORT.

Two usual causes:

  1. Your SSH public key is not on the pod. RunPod injects the keys from your
     account settings when the pod starts. Add one at
     runpod.io/console/user/settings, then restart the pod.

     Your public key is probably:
       $(ls ~/.ssh/*.pub 2>/dev/null | head -1)

  2. The address or port is wrong. Copy them again from the RunPod console
     under "Direct TCP ports".

If you cannot get SSH working, use the deploy-key route from the pod's web
terminal instead — see the README.

SSHFAIL
  exit 1
fi
echo "reachable"

say "Building Linux binaries"
make build-linux
ls -lh dist/*linux-amd64 | awk '{printf "  %-34s %s\n", $9, $5}'

say "Copying to $USER@$IP:$REMOTE"
ssh -p "$PORT" "$USER@$IP" "mkdir -p $REMOTE/llmfast/dist"
scp -P "$PORT" -q dist/llmfast-linux-amd64       "$USER@$IP:$REMOTE/llmfast/dist/llmfast"
scp -P "$PORT" -q dist/llmfast-agent-linux-amd64 "$USER@$IP:$REMOTE/llmfast/dist/llmfast-agent"
scp -P "$PORT" -q dist/llmplan-linux-amd64       "$USER@$IP:$REMOTE/llmfast/dist/llmplan"
scp -P "$PORT" -q scripts/setup-pod.sh           "$USER@$IP:$REMOTE/setup-pod.sh"
scp -P "$PORT" -q scripts/check-streaming.sh     "$USER@$IP:$REMOTE/check-streaming.sh"
ssh -p "$PORT" "$USER@$IP" "chmod +x $REMOTE/llmfast/dist/* $REMOTE/*.sh"
echo "copied"

say "Running setup on the pod"
ssh -p "$PORT" "$USER@$IP" "bash $REMOTE/setup-pod.sh"

cat <<DONE

Copied and set up. Everything from here happens on the pod:

  ssh $USER@$IP -p $PORT

Then follow the three start commands the setup printed above.

To send a new build later, just run this script again.

DONE
