#!/usr/bin/env bash
# Prepare a fresh RunPod pod to serve models.
#
# Installs vLLM and cloudflared, gets the gateway and agent onto the box, and
# writes a config. It does not start anything: read what it prints, then start
# the pieces yourself so you can watch each one come up.
#
# There are three ways the code can arrive here, and the script copes with all
# of them:
#
#   1. Binaries already copied up (no git, no Go needed):
#        # on your machine
#        make build-linux
#        scp -P <port> dist/llmfast-linux-amd64 root@<ip>:/workspace/llmfast/dist/llmfast
#        scp -P <port> dist/llmfast-agent-linux-amd64 root@<ip>:/workspace/llmfast/dist/llmfast-agent
#        scp -P <port> scripts/setup-pod.sh root@<ip>:/workspace/
#        # then on the pod
#        bash /workspace/setup-pod.sh
#
#   2. Already inside a clone:
#        cd /workspace/llmfast && bash scripts/setup-pod.sh
#
#   3. Public repository:
#        curl -fsSL https://raw.githubusercontent.com/xindex2/llmfast/main/scripts/setup-pod.sh | bash
set -euo pipefail

WORKDIR="${WORKDIR:-/workspace}"
GO_VERSION="${GO_VERSION:-1.24.5}"

say() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }

# Resolve where this script lives BEFORE anything changes the working
# directory. Invoked as "bash scripts/setup-pod.sh" from inside the checkout,
# $0 is relative, so resolving it after the cd below looked for
# /workspace/scripts and failed with "cd: scripts: No such file or directory".
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

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

say "llmfast binaries"
REPO="$WORKDIR/llmfast"
# If this script is being run from inside a checkout, use that one.
if [ -f "$SCRIPT_DIR/../go.mod" ]; then
  REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
  echo "using the checkout this script came from: $REPO"
fi

if [ -x "$REPO/dist/llmfast" ] && [ -x "$REPO/dist/llmfast-agent" ]; then
  # Binaries were copied up. Nothing to build, and Go is not needed at all.
  echo "binaries already present, skipping the build"
elif [ -f "$REPO/go.mod" ]; then
  say "Go ${GO_VERSION}"
  if ! command -v go >/dev/null 2>&1; then
    curl -fsSL "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz" | tar -C /usr/local -xz
  fi
  export PATH=$PATH:/usr/local/go/bin
  # /etc/profile.d rather than ~/.bashrc: a web terminal may open a shell that
  # never reads ~/.bashrc, and any tab opened before this ran will not have it.
  echo 'export PATH=$PATH:/usr/local/go/bin' > /etc/profile.d/go.sh
  chmod +x /etc/profile.d/go.sh
  go version
  ( cd "$REPO" && make build )
elif git clone -q https://github.com/xindex2/llmfast.git "$REPO" 2>/dev/null; then
  say "Go ${GO_VERSION}"
  if ! command -v go >/dev/null 2>&1; then
    curl -fsSL "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz" | tar -C /usr/local -xz
  fi
  export PATH=$PATH:/usr/local/go/bin
  go version
  ( cd "$REPO" && make build )
else
  cat <<'NOREPO'

Could not fetch the source, and no binaries were found.

If the repository is private, GitHub answers an unauthenticated request with a
404, so a plain clone or curl cannot see it. Two ways forward:

  Copy the binaries up from your own machine — simplest, and needs no
  credentials on the pod at all:

    make build-linux
    ssh root@<ip> -p <port> 'mkdir -p /workspace/llmfast/dist'
    scp -P <port> dist/llmfast-linux-amd64       root@<ip>:/workspace/llmfast/dist/llmfast
    scp -P <port> dist/llmfast-agent-linux-amd64 root@<ip>:/workspace/llmfast/dist/llmfast-agent
    scp -P <port> scripts/setup-pod.sh           root@<ip>:/workspace/
    ssh root@<ip> -p <port> 'chmod +x /workspace/llmfast/dist/* && bash /workspace/setup-pod.sh'

  Or give the pod read-only access with a deploy key:

    ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519 -C "runpod"
    cat ~/.ssh/id_ed25519.pub
    # paste that at github.com/xindex2/llmfast/settings/keys — leave write access off
    git clone git@github.com:xindex2/llmfast.git /workspace/llmfast
    bash /workspace/llmfast/scripts/setup-pod.sh

NOREPO
  exit 1
fi
echo "binaries: $(ls "$REPO/dist" 2>/dev/null | tr '\n' ' ')"

say "vLLM"
# RunPod pods cannot normally run Docker inside themselves, so the engine is
# installed directly and the agent runs in native mode.
# Python lives on the volume, not in the container.
#
# Editing a pod in the RunPod console -- resizing a disk, changing a port --
# recreates the container from its image. /workspace survives; everything
# installed into the system Python does not. Reinstalling vLLM is a 3GB
# download and twenty minutes, and it has to happen every single time.
#
# A virtualenv under /workspace survives instead, so a reset costs seconds.
VENV="$WORKDIR/venv"
export PATH="$VENV/bin:$WORKDIR/bin:$PATH"

# pip unpacks every wheel through TMPDIR before installing it, and TMPDIR
# defaults to /tmp -- which is the container's disk, not the volume. Installing
# torch means unpacking roughly 2GB of CUDA libraries there, and the failure is
# "OSError: [Errno 28] No space left on device" pointing at nothing in
# particular while /workspace still has tens of gigabytes free.
export TMPDIR="$WORKDIR/tmp"
mkdir -p "$TMPDIR"

if [ ! -x "$VENV/bin/python" ]; then
  say "Creating $VENV"
  # Deliberately NOT --system-site-packages. With the container's packages
  # visible, pip reports the image's nvidia-cublas, nvidia-cuda-runtime and
  # friends as already satisfied and does not install them into the venv --
  # so the venv depends on files that the next pod reset deletes, which is
  # precisely what putting it on the volume was meant to prevent. A clean
  # venv costs about 15GB of volume and is actually self-contained.
  apt-get install -y -qq python3-venv >/dev/null 2>&1 || true
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --no-cache-dir --upgrade pip >/dev/null
fi

if command -v vllm >/dev/null 2>&1 && [ -x "$VENV/bin/vllm" ]; then
  echo "already installed in the venv: $("$VENV/bin/python" -c 'import vllm; print(vllm.__version__)' 2>/dev/null)"
else
  echo "This pulls PyTorch and the CUDA runtime -- several gigabytes."
  echo

  # Install torch first, pinned and from the index that has a build for this
  # driver. Left to resolve everything at once, pip backtracks through dozens
  # of cuda-python versions and is killed by the OOM killer on a container
  # with modest RAM -- which looks like "Killed" and nothing else.
  TORCH_VER="${TORCH_VERSION:-}"
  CUDA_TAG=$(nvidia-smi 2>/dev/null | sed -nE 's/.*CUDA Version:[[:space:]]*([0-9]+)\.([0-9]+).*/cu\1\2/p' | head -1)
  CUDA_TAG=${CUDA_TAG:-cu128}
  if [ -f "$WORKDIR/versions.lock" ]; then
    TORCH_VER=$(sed -nE 's/^torch=(.+)\+.*$/\1/p' "$WORKDIR/versions.lock")
  fi

  # --no-cache-dir keeps a 3GB wheel cache off the container disk.
  PIP="pip install --no-cache-dir"
  if [ -n "$TORCH_VER" ]; then
    echo "installing torch $TORCH_VER for $CUDA_TAG"
    $PIP "torch==$TORCH_VER" --index-url "https://download.pytorch.org/whl/$CUDA_TAG"
  fi
  $PIP "vllm${VLLM_VERSION:+==$VLLM_VERSION}" --extra-index-url "https://download.pytorch.org/whl/$CUDA_TAG"
  $PIP hf_transfer

  # FreeToken serves mixture-of-experts models larger than VRAM by streaming
  # cold experts from host RAM. It needs CUDA 13 and nvcc on PATH, so it is
  # installed only where that holds rather than failing the whole setup.
  if [ "${INSTALL_FREETOKEN:-0}" = 1 ]; then
    DRV_MAJOR=$(printf '%s' "${CUDA_TAG#cu}" | cut -c1-2)
    if [ "${DRV_MAJOR:-0}" -ge 13 ] 2>/dev/null; then
      $PIP "freetoken[accel]" || echo "  freetoken install failed; the other engines still work"
    else
      echo "  skipping freetoken: it needs a CUDA 13 driver and this host reports ${CUDA_TAG}"
    fi
  fi

  echo
  echo "installed: $("$VENV/bin/python" -c 'import vllm; print(vllm.__version__)' 2>/dev/null)"
fi

# RunPod images ship their own PyTorch. If vLLM replaced it with a build for a
# different CUDA version, the engine fails at model load with an error that does
# not mention the cause, so it is worth catching here.
if python3 -c "import torch" 2>/dev/null; then
  TORCH_CUDA=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo unknown)
  TORCH_OK=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo False)
  echo "torch: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null), CUDA $TORCH_CUDA, sees GPU: $TORCH_OK"
  if [ "$TORCH_OK" != "True" ]; then
    echo
    echo "  WARNING: torch cannot see the GPU. vLLM will fail to start."
    echo "  Usually this means the vLLM install replaced torch with a build for a"
    echo "  different CUDA version. Check 'nvidia-smi' works, then reinstall torch"
    echo "  matching the driver's CUDA version before going further."
  fi
fi

# Editing a pod in the RunPod console recreates the container from its image,
# which keeps /workspace and discards everything installed into the system
# Python. Recording the working versions here turns that from an afternoon of
# rediscovery into one command.
{
  echo "# Written by setup-pod.sh. These versions were verified working on this host."
  echo "# After a pod reset, restore the environment with:"
  echo "#   pip install $(python3 -c "import vllm; print('vllm=='+vllm.__version__)" 2>/dev/null || echo vllm) \\"
  echo "#     $(python3 -c "import torch; print('torch=='+torch.__version__.split('+')[0])" 2>/dev/null || echo torch) hf_transfer"
  echo "driver_cuda=$(nvidia-smi 2>/dev/null | sed -nE 's/.*CUDA Version:[[:space:]]*([0-9.]+).*/\1/p' | head -1)"
  echo "torch=$(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null)"
  echo "torch_cuda=$(python3 -c 'import torch; print(torch.version.cuda)' 2>/dev/null)"
  echo "vllm=$(python3 -c 'import vllm; print(vllm.__version__)' 2>/dev/null)"
} > "$WORKDIR/versions.lock" 2>/dev/null || true
echo "recorded working versions in $WORKDIR/versions.lock"

say "cloudflared"
mkdir -p "$WORKDIR/bin"
if [ ! -x "$WORKDIR/bin/cloudflared" ]; then
  curl -fsSL -o "$WORKDIR/bin/cloudflared" \
    https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
  chmod +x "$WORKDIR/bin/cloudflared"
fi
# Its credentials live in /root by default, which is exactly what a reset
# destroys. Keep them on the volume instead.
mkdir -p "$WORKDIR/cloudflared"
ln -sfn "$WORKDIR/cloudflared" /root/.cloudflared
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
  # Not 8081: RunPod images run their own nginx there, and the clash is only
  # visible as a bind error in the log.
  admin_listen: "127.0.0.1:8090"
  admin_token: "$LLMFAST_ADMIN_TOKEN"
  db_path: "/workspace/llmfast.db"
  site_dir: "$REPO/site"
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
"$REPO/dist/llmfast-agent" -hardware -state-dir "$WORKDIR/state" || true

cat <<NEXT

Ready.

Start everything -- detached, so closing this terminal does not stop it:

  bash $REPO/scripts/llmfast.sh start

Then, once, to put api.llmfa.st in front of it:

  cloudflared tunnel login
  cloudflared tunnel create llmfast          # skip if the tunnel already exists
  bash $REPO/scripts/setup-tunnel.sh llmfast api.llmfa.st admin.llmfa.st
  bash $REPO/scripts/llmfast.sh restart

Day to day:

  bash $REPO/scripts/llmfast.sh status       # is everything up, including the public hostname
  bash $REPO/scripts/llmfast.sh logs         # follow all three logs
  bash $REPO/scripts/llmfast.sh token        # print the admin token to sign in with

NEXT
