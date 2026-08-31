# Adding a second GPU, keeping one domain

The stack is already built for this. One **gateway** holds the catalog, the
accounts, the keys, the stats and the routing; each GPU box runs an **agent**
that supervises engines on that machine. Adding hardware means adding a node,
not a second deployment.

Customers keep one endpoint, one dashboard, one set of API keys. They never
learn there is more than one machine.

## How the gateway reaches an engine

This is the fact that decides the whole network design.

A node is configured with a URL, say `http://10.1.2.3:9900`. When the agent
reports an engine is ready on port 18000, the gateway routes to
`http://10.1.2.3:18000/v1` — it takes the host from the node URL and the port
from the engine.

So **both the control port and the engine ports must be reachable from the
gateway**. And here is the part worth stopping on:

> The control API checks a bearer token. **The engines behind it check
> nothing.** An engine port reachable from the internet is free use of your
> GPU for anyone who finds it.

Never expose 18000+ publicly. Put the nodes on a private network instead.

## The shape to build

```
                     customers
                         |
                 https://api.llmfa.st
                         |
                   ┌───────────┐
                   │  gateway  │   catalog, accounts, keys, stats, routing
                   │  + SQLite │
                   └─────┬─────┘
                         │  private mesh (100.x addresses)
              ┌──────────┼──────────┐
          ┌───┴───┐  ┌───┴───┐  ┌───┴───┐
          │ agent │  │ agent │  │ agent │
          │ A40 #1│  │ A40 #2│  │  CPU  │
          └───────┘  └───────┘  └───────┘
```

Put the gateway where the domain already points and where the database can
live on real storage. Your Xeon server is the natural home: it is always on, it
already has TLS and the DNS record, and it is not going to be reallocated to
someone else mid-month.

## Private networking with Tailscale

Free for this size, works inside a RunPod container, and gives every node a
stable `100.x` address that survives the pod being recreated — which matters,
because a RunPod IP does not.

**On each machine, gateway and pods alike:**

```bash
curl -fsSL https://tailscale.com/install.sh | sh
tailscale up --authkey tskey-auth-xxxx --hostname llmfast-gpu-1
tailscale ip -4        # note the 100.x address
```

Use `--hostname` deliberately: reusing the same one when a pod is rebuilt gets
you the same address back, so the gateway config does not need editing.

**On each GPU pod**, bind the agent and its engines to that address:

```bash
TS=$(tailscale ip -4)
/workspace/llmfast/dist/llmfast-agent \
  -listen "$TS:9900" \
  -engine-host "$TS" \
  -name gpu-2 \
  -state-dir /workspace/state -hf-cache /workspace/hf -mode native
```

`-engine-host` is the important one. Without it engines bind `0.0.0.0`, and on
a machine with any public interface that is an open GPU. The agent warns you if
you leave it that way on a non-loopback listener.

**On the gateway**, add the node:

```yaml
nodes:
  - name: gpu-a
    url: http://100.64.0.11:9900
    token: "$LLMFAST_AGENT_TOKEN"
    max_concurrency: 8
  - name: gpu-b                     # the new pod
    url: http://100.64.0.12:9900
    token: "$LLMFAST_AGENT_TOKEN"
    max_concurrency: 8
```

Then `systemctl reload llmfast` — models and nodes are re-read without dropping
in-flight streams.

The **Nodes** page will show both. **Add Model** will plan against each one and
tell you where a given model fits.

## Two ways to use a second GPU

**Different models on each.** The usual choice. Each node serves what fits it,
and the catalog is the union. A customer calling `model-a` reaches pod 1 and
`model-b` reaches pod 2 without knowing either exists.

**The same model on both.** Install it on both nodes with the same model id and
the gateway load-balances across them: least-loaded first, weighted by each
node's `weight`. That doubles throughput for one listing and means a pod dying
degrades rather than fails. Worth doing once a model earns enough to justify
two cards.

## What to check after adding one

```bash
bash /workspace/llmfast/scripts/llmfast.sh status   # on each pod
curl -s https://api.llmfa.st/health | python3 -m json.tool
```

`/health` lists every backend and whether it is healthy. A node that is up but
unreachable from the gateway shows as unreachable on the Nodes page with the
connection error, which is nearly always a firewall or a wrong address rather
than a broken agent.

## Latency is worth measuring, not assuming

Every token crosses gateway → engine. Within one datacenter that is a fraction
of a millisecond. Between a European server and a US pod it can be 80–100ms
added to time-to-first-token — and TTFT is half of what OpenRouter ranks you
on.

Measure it before committing:

```bash
# from the gateway
curl -o /dev/null -s -w 'connect %{time_connect}s  total %{time_total}s\n' \
  http://100.64.0.12:9900/health
```

Under 20ms is fine. Over 50ms and the gateway belongs nearer the GPUs — put it
on one of the pods and keep the Xeon as a node, accepting that a pod restart
takes the gateway with it.

## If you would rather not run a mesh

RunPod can expose TCP ports directly. It works, and it is the wrong default:
you would be publishing unauthenticated inference endpoints. If you do it
anyway, expose **only** 9900, keep engines on the mesh or loopback, and accept
that the agent token is the only thing between the internet and your GPU.
