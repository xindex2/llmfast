'use strict';

const state = { view: 'usage', range: '24h', me: null, signup: false };

const $ = id => document.getElementById(id);
const main = () => $('main');
const esc = s => String(s ?? '').replace(/[&<>"']/g, c =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

const fmtNum = n => (n === null || n === undefined) ? '—' :
  n >= 1e9 ? (n / 1e9).toFixed(2) + 'B' :
  n >= 1e6 ? (n / 1e6).toFixed(2) + 'M' :
  n >= 1e3 ? (n / 1e3).toFixed(1) + 'k' : String(Math.round(n));

// Costs here are fractions of a cent. Two decimal places would render every
// figure as $0.00, so the precision follows the magnitude.
function fmtUSD(v) {
  const n = Number(v) || 0;
  if (n === 0) return '$0';
  if (n < 0.01) return '$' + n.toPrecision(2);
  return '$' + n.toFixed(2);
}
const fmtTime = ts => !ts ? '—' : new Date(ts > 1e12 ? ts : ts * 1000).toLocaleString();

async function api(path, opts = {}) {
  const res = await fetch(path, { credentials: 'same-origin', ...opts });
  if (res.status === 401) { showGate(); throw new Error('unauthorized'); }
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).error.message; } catch (_) {}
    throw new Error(msg);
  }
  return res.json();
}

// ---------------------------------------------------------------- sign in ---

function paintGate() {
  $('gate-title').textContent = state.signup ? 'Create your account' : 'Sign in';
  $('gate-sub').textContent = state.signup
    ? 'Free to start. You only pay for the tokens you use.'
    : 'Your API keys and usage live here.';
  $('gate-go').textContent = state.signup ? 'Create account' : 'Sign in';
  $('switch-text').textContent = state.signup ? 'Already have an account?' : 'New here?';
  $('switch-link').textContent = state.signup ? 'Sign in' : 'Create an account';
  $('password').autocomplete = state.signup ? 'new-password' : 'current-password';
}

function showGate() { $('app').classList.add('hidden'); $('gate').classList.remove('hidden'); paintGate(); }
function showApp() { $('gate').classList.add('hidden'); $('app').classList.remove('hidden'); render(); }

$('switch-link').onclick = e => { e.preventDefault(); state.signup = !state.signup; paintGate(); $('gate-err').textContent = ''; };

$('gate-form').addEventListener('submit', async e => {
  e.preventDefault();
  const err = $('gate-err'), btn = $('gate-go');
  err.textContent = ''; btn.disabled = true;
  try {
    const res = await fetch(state.signup ? '/api/register' : '/api/login', {
      method: 'POST', credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email: $('email').value.trim(), password: $('password').value }),
    });
    if (!res.ok) {
      let m = 'Failed';
      try { m = (await res.json()).error.message; } catch (_) {}
      throw new Error(m);
    }
    $('password').value = '';
    await loadMe();
    showApp();
  } catch (e2) {
    err.textContent = e2.message;
  } finally {
    btn.disabled = false;
  }
});

$('signout').onclick = async () => {
  await fetch('/api/logout', { method: 'POST', credentials: 'same-origin' });
  state.me = null;
  showGate();
};

async function loadMe() {
  const d = await (await fetch('/api/me', { credentials: 'same-origin' })).json();
  state.me = d.signed_in ? d : null;
  $('who').textContent = state.me ? state.me.email : '';
  return state.me;
}

// ------------------------------------------------------------------ views ---

async function viewUsage() {
  const d = await api('/api/usage?range=' + encodeURIComponent(state.range));
  const t = d.total || {};
  const rows = d.by_model || [];
  main().innerHTML = `
    <h2>Usage</h2>
    <p class="sub">Everything your API keys have done in this period.</p>
    <div class="cards">
      ${card('Requests', fmtNum(t.requests), `${fmtNum(t.errors)} failed`)}
      ${card('Input tokens', fmtNum(t.prompt_tokens), `${fmtNum(t.cached_tokens)} cached`)}
      ${card('Output tokens', fmtNum(t.completion_tokens), '')}
      ${card('Spend', fmtUSD(t.cost_usd), 'this period')}
      ${card('Avg speed', (t.tps_avg || 0).toFixed(0) + ' t/s', 'per request')}
    </div>
    <div class="panel">
      <h3>Models you used</h3>
      <p class="hint">Ordered by how often you called them.</p>
      ${rows.length ? `<table><thead><tr>
        <th>Model</th><th class="num">Requests</th><th class="num">Input</th>
        <th class="num">Output</th><th class="num">Avg t/s</th><th class="num">Spend</th><th>Last used</th>
      </tr></thead><tbody>${rows.map(m => `<tr>
        <td class="mono">${esc(m.model)}</td>
        <td class="num">${fmtNum(m.requests)}</td>
        <td class="num">${fmtNum(m.prompt_tokens)}</td>
        <td class="num">${fmtNum(m.completion_tokens)}</td>
        <td class="num">${(m.tps_avg || 0).toFixed(0)}</td>
        <td class="num">${fmtUSD(m.cost_usd)}</td>
        <td class="muted">${fmtTime(m.last_used)}</td>
      </tr>`).join('')}</tbody></table>`
      : '<div class="empty">No requests yet. Create a key and send one — see <b>Get started</b>.</div>'}
    </div>`;
}

const card = (k, v, s) => `<div class="card"><div class="k">${esc(k)}</div>
  <div class="v">${esc(v)}</div><div class="s">${esc(s)}</div></div>`;

async function viewKeys() {
  const d = await api('/api/keys');
  const keys = d.keys || [];
  main().innerHTML = `
    <h2>API keys</h2>
    <p class="sub">A key is shown once, when you create it. Only its hash is stored, so a lost key must be replaced rather than recovered.</p>
    <div class="panel">
      <div class="row">
        <div style="flex:1"><input id="kname" placeholder="What is this key for? e.g. production" style="width:100%"></div>
        <button class="primary" id="knew">Create key</button>
      </div>
      <div id="kout"></div>
    </div>
    <div class="panel">
      ${keys.length ? `<table><thead><tr>
        <th>Name</th><th>Key</th><th>Created</th><th>Status</th><th></th>
      </tr></thead><tbody>${keys.map(k => `<tr>
        <td>${esc(k.name)}</td>
        <td class="mono">${esc(k.prefix)}…</td>
        <td class="muted">${fmtTime(k.created_at)}</td>
        <td>${k.disabled ? '<span class="pill off">disabled</span>' : '<span class="pill ok">active</span>'}</td>
        <td><button class="danger" data-del="${k.id}" data-name="${esc(k.name)}">Revoke</button></td>
      </tr>`).join('')}</tbody></table>`
      : '<div class="empty">No keys yet.</div>'}
    </div>`;

  $('knew').onclick = async () => {
    const out = $('kout');
    try {
      const r = await api('/api/keys', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: $('kname').value.trim() || 'default' }),
      });
      out.innerHTML = `<div class="note ok">Copy this now — it will not be shown again.</div>
        <div class="secret">${esc(r.secret)}</div>`;
      const keep = out.innerHTML;
      await viewKeys();
      $('kout').innerHTML = keep;
    } catch (e) { out.innerHTML = `<div class="note bad">${esc(e.message)}</div>`; }
  };

  main().querySelectorAll('[data-del]').forEach(b => b.onclick = async () => {
    if (!confirm(`Revoke "${b.dataset.name}"?\n\nAnything using this key stops working immediately.`)) return;
    b.disabled = true;
    try { await api('/api/keys/' + b.dataset.del, { method: 'DELETE' }); await viewKeys(); }
    catch (e) { alert(e.message); b.disabled = false; }
  });
}

async function viewModels() {
  const d = await (await fetch('/v1/models')).json();
  const models = (d.data || []).filter(m => m.is_ready !== false);
  main().innerHTML = `
    <h2>Models</h2>
    <p class="sub">What you can call right now, and what it costs.</p>
    <div class="panel">
      ${models.length ? `<table><thead><tr>
        <th>Model</th><th class="num">Context</th><th class="num">Input /M</th>
        <th class="num">Output /M</th><th>Quantization</th>
      </tr></thead><tbody>${models.map(m => {
        const inp = m.input_modalities?.[0] || {};
        const ctx = inp.supported_inputs?.max_context_length?.value;
        const price = t => {
          const p = (inp.pricing || []).find(x => x.type === t)
            || (m.output_modalities?.[0]?.pricing || []).find(x => x.type === t);
          return p ? '$' + (parseFloat(p.cost_usd) * 1e6).toFixed(2) : '—';
        };
        return `<tr>
          <td class="mono">${esc(m.id)}</td>
          <td class="num">${ctx ? fmtNum(ctx) : '—'}</td>
          <td class="num">${price('prompt')}</td>
          <td class="num">${price('completion')}</td>
          <td>${m.quantization ? `<span class="pill off">${esc(m.quantization)}</span>` : '—'}</td>
        </tr>`;
      }).join('')}</tbody></table>`
      : '<div class="empty">No models are published yet.</div>'}
    </div>`;
}

async function viewRequests() {
  const d = await api('/api/requests?limit=100');
  const rows = d.requests || [];
  main().innerHTML = `
    <h2>Requests</h2>
    <p class="sub">Your last 100 calls, newest first.</p>
    <div class="panel">
      ${rows.length ? `<table><thead><tr>
        <th>When</th><th>Model</th><th class="num">Status</th><th class="num">In</th>
        <th class="num">Out</th><th class="num">TTFT</th><th class="num">t/s</th><th class="num">Cost</th>
      </tr></thead><tbody>${rows.map(r => `<tr>
        <td class="muted">${fmtTime(r.ts)}</td>
        <td class="mono">${esc(r.model)}</td>
        <td class="num">${r.status >= 400
          ? `<span class="pill bad">${r.status}</span>`
          : `<span class="pill ok">${r.status}</span>`}</td>
        <td class="num">${fmtNum(r.prompt_tokens)}</td>
        <td class="num">${fmtNum(r.completion_tokens)}</td>
        <td class="num">${r.ttft_ms >= 0 ? r.ttft_ms + 'ms' : '—'}</td>
        <td class="num">${(r.tps || 0).toFixed(0)}</td>
        <td class="num">${fmtUSD(r.cost_usd)}</td>
      </tr>`).join('')}</tbody></table>`
      : '<div class="empty">No requests yet.</div>'}
    </div>`;
}

function viewStart() {
  const base = location.origin;
  main().innerHTML = `
    <h2>Get started</h2>
    <p class="sub">The API is OpenAI-compatible, so any client that speaks to OpenAI speaks to us. Change two lines.</p>
    <div class="panel">
      <h3>1 — Create a key</h3>
      <p class="hint">Go to <b>API keys</b> and make one. It is shown once.</p>
    </div>
    <div class="panel">
      <h3>2 — Point your client here</h3>
      <pre class="code">from openai import OpenAI

client = OpenAI(
    base_url="${esc(base)}/v1",
    api_key="sk-llmfast-...",
)

resp = client.chat.completions.create(
    model="MODEL_ID",          # see the Models tab
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
)
for chunk in resp:
    print(chunk.choices[0].delta.content or "", end="")</pre>
    </div>
    <div class="panel">
      <h3>Or with curl</h3>
      <pre class="code">curl ${esc(base)}/v1/chat/completions \\
  -H "Authorization: Bearer $LLMFAST_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"model":"MODEL_ID","messages":[{"role":"user","content":"Hello"}],"stream":true}'</pre>
      <p class="hint">Streaming is on by default and frames are flushed as they are produced — no buffering.</p>
    </div>`;
}

const VIEWS = { usage: viewUsage, keys: viewKeys, models: viewModels, requests: viewRequests, start: viewStart };

async function render() {
  try { await VIEWS[state.view](); }
  catch (e) {
    if (e.message !== 'unauthorized') {
      main().innerHTML = `<div class="empty">Could not load: ${esc(e.message)}</div>`;
    }
  }
}

document.querySelectorAll('#nav button').forEach(b => b.onclick = () => {
  document.querySelectorAll('#nav button').forEach(x => x.classList.remove('active'));
  b.classList.add('active');
  state.view = b.dataset.view;
  render();
});
$('range').onchange = e => { state.range = e.target.value; render(); };

loadMe().then(me => me ? showApp() : showGate());
