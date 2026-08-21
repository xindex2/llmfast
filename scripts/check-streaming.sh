#!/usr/bin/env bash
# Verify that tokens actually stream, rather than arriving in one lump.
#
# A buffering proxy does not fail visibly. The response is correct and the
# request succeeds; the tokens just all arrive at the end. The only symptom is
# that your throughput on OpenRouter is worse than your hardware, which is a
# terrible way to find out. This measures the gap between frames.
#
#   ./scripts/check-streaming.sh https://api.llmfa.st sk-llmfast-... qwen/qwen3.8-27b
set -euo pipefail

BASE="${1:?usage: check-streaming.sh <base-url> <api-key> [model]}"
KEY="${2:?api key required}"
MODEL="${3:-qwen/qwen3.8-27b}"

echo "Streaming from ${BASE}/v1/chat/completions as ${MODEL}"
echo

curl -sN --no-buffer "${BASE}/v1/chat/completions" \
  -H "Authorization: Bearer ${KEY}" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Count slowly from 1 to 40, one number per line.\"}],\"stream\":true,\"max_tokens\":200}" \
| awk -v start="$(python3 -c 'import time;print(time.time())')" '
  /^data: /{
    "python3 -c \"import time;print(time.time())\"" | getline now
    close("python3 -c \"import time;print(time.time())\"")
    n++
    if (n == 1) { first = now - start }
    last = now - start
  }
  END {
    if (n == 0) { print "  no frames received - check the key and model"; exit 1 }
    printf "  frames:            %d\n", n
    printf "  first frame at:    %.2fs\n", first
    printf "  last frame at:     %.2fs\n", last
    printf "  spread:            %.2fs\n", last - first
    print ""
    if (last - first < 0.15) {
      print "  BUFFERED. Every frame arrived at once, so something between you and"
      print "  the engine is accumulating the response. Check:"
      print "    nginx      proxy_buffering off;"
      print "    Caddy      flush_interval -1"
      print "    Cloudflare do not proxy the API subdomain, or disable compression on it"
      exit 1
    }
    print "  STREAMING. Frames arrived spread over time, which is what you want."
  }'
