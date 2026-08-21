#!/usr/bin/env bash
# Point api.llmfa.st and admin.llmfa.st at this pod through one tunnel.
#
# Run on the pod, after `cloudflared tunnel login` and `cloudflared tunnel
# create <name>`:
#
#   bash scripts/setup-tunnel.sh llmfast api.llmfa.st admin.llmfa.st
set -euo pipefail

NAME="${1:-llmfast}"
API_HOST="${2:-api.llmfa.st}"
ADMIN_HOST="${3:-}"

say() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }

ID=$(cloudflared tunnel list --output json 2>/dev/null \
     | python3 -c "
import json,sys
name=sys.argv[1]
for t in json.load(sys.stdin):
    if t.get('name')==name:
        print(t['id']); break
" "$NAME" 2>/dev/null || true)

if [ -z "$ID" ]; then
  echo "No tunnel named '$NAME'. Create it first:"
  echo "  cloudflared tunnel login"
  echo "  cloudflared tunnel create $NAME"
  exit 1
fi
echo "tunnel $NAME is $ID"

say "Writing /root/.cloudflared/config.yml"
mkdir -p /root/.cloudflared
{
  echo "tunnel: $ID"
  echo "credentials-file: /root/.cloudflared/$ID.json"
  echo "retries: 5"
  echo "grace-period: 30s"
  echo ""
  echo "ingress:"
  echo "  - hostname: $API_HOST"
  echo "    service: http://127.0.0.1:8080"
  echo "    originRequest:"
  echo "      connectTimeout: 10s"
  if [ -n "$ADMIN_HOST" ]; then
    echo "  - hostname: $ADMIN_HOST"
    echo "    service: http://127.0.0.1:8090"
    echo "    originRequest:"
    echo "      connectTimeout: 10s"
  fi
  echo "  - service: http_status:404"
} > /root/.cloudflared/config.yml
cat /root/.cloudflared/config.yml | sed 's/^/  /'

say "DNS"
cloudflared tunnel route dns "$NAME" "$API_HOST" 2>&1 | sed 's/^/  /' || true
if [ -n "$ADMIN_HOST" ]; then
  cloudflared tunnel route dns "$NAME" "$ADMIN_HOST" 2>&1 | sed 's/^/  /' || true
fi

cat <<NEXT

Now restart everything so the tunnel comes up with the rest of the stack, as a
detached process that a closed terminal cannot take down:

  bash /workspace/llmfast/scripts/llmfast.sh restart

Then check it end to end, including the public hostname:

  bash /workspace/llmfast/scripts/llmfast.sh status

NEXT

if [ -n "$ADMIN_HOST" ]; then
cat <<ACCESS
BEFORE you use $ADMIN_HOST, put Cloudflare Access in front of it.

That page exposes your API keys, your full request history, and the ability to
install and stop models. The gateway checks a bearer token, but a token is one
secret with no second factor, no expiry and no audit trail. Access adds a login.

  1. Cloudflare dashboard -> Zero Trust -> Access -> Applications
  2. Add an application -> Self-hosted
  3. Subdomain: ${ADMIN_HOST%%.*}   Domain: ${ADMIN_HOST#*.}
  4. Add a policy: Action Allow, Include -> Emails -> your address
  5. Save

You will then get an email code the first time you open it. Until that policy
exists, treat $ADMIN_HOST as public and do not route it.

ACCESS
fi
