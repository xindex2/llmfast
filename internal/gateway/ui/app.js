'use strict';

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------

// The admin token is exchanged for an HttpOnly cookie at login, so requests
// here carry no credentials of their own -- a 401 simply means the cookie is
// gone or expired and we fall back to the login screen.
async function api(path, opts = {}) {
  const res = await fetch(path, { credentials: 'same-origin', ...opts });
  if (res.status === 401 || res.status === 403) { showLogin(); throw new Error('unauthorized'); }
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).error.message; } catch (_) {}
    throw new Error(msg);
  }
  return res.json();
}

const state = { view: 'overview', range: '24h', timer: null, secret: null };

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

function fmtNum(n) {
  if (n === null || n === undefined) return '—';
  const a = Math.abs(n);
  if (a >= 1e9) return (n / 1e9).toFixed(2) + 'B';
  if (a >= 1e6) return (n / 1e6).toFixed(2) + 'M';
  if (a >= 1e3) return (n / 1e3).toFixed(1) + 'k';
  return String(Math.round(n));
}
// -1 is the "not measured" sentinel (non-streaming requests have no TTFT).
// A measured 0 means sub-millisecond, which is a real value, not a missing one.
function fmtMs(ms) {
  if (ms === null || ms === undefined) return '—';
  if (ms < 0) return '—';
  if (ms === 0) return '<1ms';
  return ms >= 1000 ? (ms / 1000).toFixed(2) + 's' : Math.round(ms) + 'ms';
}
// Costs at these rates span many orders of magnitude: a single playground
// request is a few millionths of a dollar while a month of traffic is tens.
// A fixed number of decimals would render one end as $0.00000 and the other
// with meaningless precision, so significant figures are used below a cent.
function fmtUSD(v) {
  if (!v) return '$0';
  const a = Math.abs(v);
  // toPrecision switches to exponential notation below 1e-7, which reads as
  // noise in a table, so anything smaller is reported as a bound.
  if (a < 1e-7) return '<$0.0000001';
  if (a < 0.01) return '$' + Number(v.toPrecision(2));
  if (a < 1) return '$' + v.toFixed(4);
  return '$' + v.toFixed(2);
}
function fmtTime(unix, bucketSec) {
  const d = new Date(unix * 1000);
  if (bucketSec >= 86400) return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  if (bucketSec >= 3600) return d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
  return d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
}
function fmtFullTime(ms) { return new Date(ms).toLocaleString(); }
function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, c =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
// Requests rejected before a model could be resolved are logged with "-" so
// they still count toward uptime. Label them rather than showing a bare dash.
function modelLabel(m) {
  return m === '-' ? '<span class="muted">(no model)</span>' : esc(m);
}
function statusPill(code) {
  if (code === 200) return '<span class="pill ok">200</span>';
  if (code === 429) return '<span class="pill warn">429</span>';
  if (code === 499) return '<span class="pill off">499</span>';
  if (code >= 400 && code < 500) return `<span class="pill warn">${code}</span>`;
  return `<span class="pill down">${code}</span>`;
}

// ---------------------------------------------------------------------------
// Charts
//
// Hand-rolled SVG rather than a charting library: the admin binary embeds its
// own assets and a strict no-CDN posture is simpler to defend than pinning a
// vendored bundle. These are line and bar charts over a uniform time axis,
// which is little more than a coordinate transform.
// ---------------------------------------------------------------------------

const tip = document.createElement('div');
tip.className = 'tip';
document.body.appendChild(tip);

const W = 600, H = 170, PAD_L = 46, PAD_R = 10, PAD_T = 12, PAD_B = 22;

// The origin tick is always a plain zero. Value formatters carry units and
// qualifiers ("<1ms", "$0") that read as noise on an axis baseline.
function axisLabel(v, opts) {
  return v === 0 ? '0' : (opts.fmt || fmtNum)(v);
}

function niceMax(v) {
  if (v <= 0) return 1;
  const mag = Math.pow(10, Math.floor(Math.log10(v)));
  const norm = v / mag;
  const step = norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10;
  return step * mag;
}

// lineChart renders one or more series sharing a y-axis.
// series: [{ key, label, color, fmt }]
function lineChart(el, points, series, bucketSec, opts = {}) {
  if (!points.length) { el.innerHTML = '<div class="empty">No data in this range</div>'; return; }

  let max = 0;
  for (const p of points) for (const s of series) max = Math.max(max, p[s.key] || 0);
  max = niceMax(max || 1);

  const n = points.length;
  const innerW = W - PAD_L - PAD_R, innerH = H - PAD_T - PAD_B;
  const x = i => PAD_L + (n === 1 ? innerW / 2 : (i / (n - 1)) * innerW);
  const y = v => PAD_T + innerH - (Math.min(v, max) / max) * innerH;

  let svg = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">`;

  // Horizontal gridlines with value labels.
  for (let g = 0; g <= 4; g++) {
    const v = (max / 4) * g, yy = y(v);
    svg += `<line class="grid-line" x1="${PAD_L}" y1="${yy}" x2="${W - PAD_R}" y2="${yy}"/>`;
    svg += `<text class="axis-text" x="${PAD_L - 6}" y="${yy + 3}" text-anchor="end">${axisLabel(v, opts)}</text>`;
  }

  // Time labels: a handful of evenly spaced ticks, never so many they collide.
  const ticks = Math.min(6, n);
  for (let t = 0; t < ticks; t++) {
    const i = Math.round((t / Math.max(1, ticks - 1)) * (n - 1));
    svg += `<text class="axis-text" x="${x(i)}" y="${H - 6}" text-anchor="middle">${fmtTime(points[i].bucket, bucketSec)}</text>`;
  }

  for (const s of series) {
    const pts = points.map((p, i) => `${x(i).toFixed(1)},${y(p[s.key] || 0).toFixed(1)}`).join(' ');
    if (opts.area && series.length === 1) {
      svg += `<polygon points="${PAD_L},${y(0)} ${pts} ${x(n - 1)},${y(0)}" fill="${s.color}" opacity=".13"/>`;
    }
    svg += `<polyline points="${pts}" fill="none" stroke="${s.color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`;
  }

  svg += `<line class="hover-line hidden" x1="0" y1="${PAD_T}" x2="0" y2="${PAD_T + innerH}"/>`;
  svg += `<rect x="${PAD_L}" y="${PAD_T}" width="${innerW}" height="${innerH}" fill="transparent"/>`;
  svg += '</svg>';
  el.innerHTML = svg;

  attachHover(el, points, series, bucketSec, x, n, opts);
}

// barChart shares the axis logic but draws discrete buckets, which reads better
// for counts than a line that implies interpolation between samples.
function barChart(el, points, key, color, bucketSec, opts = {}) {
  if (!points.length) { el.innerHTML = '<div class="empty">No data in this range</div>'; return; }
  let max = 0;
  for (const p of points) max = Math.max(max, p[key] || 0);
  max = niceMax(max || 1);

  const n = points.length;
  const innerW = W - PAD_L - PAD_R, innerH = H - PAD_T - PAD_B;
  const bw = Math.max(1, (innerW / n) - 1);
  const x = i => PAD_L + (i / n) * innerW;
  const y = v => PAD_T + innerH - (Math.min(v, max) / max) * innerH;

  let svg = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">`;
  for (let g = 0; g <= 4; g++) {
    const v = (max / 4) * g, yy = y(v);
    svg += `<line class="grid-line" x1="${PAD_L}" y1="${yy}" x2="${W - PAD_R}" y2="${yy}"/>`;
    svg += `<text class="axis-text" x="${PAD_L - 6}" y="${yy + 3}" text-anchor="end">${axisLabel(v, opts)}</text>`;
  }
  const ticks = Math.min(6, n);
  for (let t = 0; t < ticks; t++) {
    const i = Math.round((t / Math.max(1, ticks - 1)) * (n - 1));
    svg += `<text class="axis-text" x="${x(i) + bw / 2}" y="${H - 6}" text-anchor="middle">${fmtTime(points[i].bucket, bucketSec)}</text>`;
  }
  points.forEach((p, i) => {
    const v = p[key] || 0;
    if (v <= 0) return;
    const h = Math.max(1, PAD_T + innerH - y(v));
    svg += `<rect x="${x(i).toFixed(1)}" y="${y(v).toFixed(1)}" width="${bw.toFixed(1)}" height="${h.toFixed(1)}" fill="${color}" rx="1"/>`;
  });
  svg += `<line class="hover-line hidden" x1="0" y1="${PAD_T}" x2="0" y2="${PAD_T + innerH}"/>`;
  svg += `<rect x="${PAD_L}" y="${PAD_T}" width="${innerW}" height="${innerH}" fill="transparent"/>`;
  svg += '</svg>';
  el.innerHTML = svg;

  attachHover(el, points, [{ key, label: opts.label || key, color, fmt: opts.fmt }], bucketSec,
              i => x(i) + bw / 2, n, opts);
}

// attachHover maps pointer position back to the nearest bucket. The SVG uses a
// viewBox with non-uniform scaling, so screen coordinates are converted through
// the element's own width rather than assumed to match the viewBox.
function attachHover(el, points, series, bucketSec, xOf, n, opts) {
  const svg = el.querySelector('svg');
  if (!svg) return;
  const vline = svg.querySelector('.hover-line');

  svg.addEventListener('mousemove', ev => {
    const rect = svg.getBoundingClientRect();
    const vx = ((ev.clientX - rect.left) / rect.width) * W;
    let best = 0, bestD = Infinity;
    for (let i = 0; i < n; i++) {
      const d = Math.abs(xOf(i) - vx);
      if (d < bestD) { bestD = d; best = i; }
    }
    const p = points[best];
    vline.classList.remove('hidden');
    vline.setAttribute('x1', xOf(best)); vline.setAttribute('x2', xOf(best));

    const lines = [fmtTime(p.bucket, bucketSec)];
    for (const s of series) lines.push(`${s.label}: ${(s.fmt || opts.fmt || fmtNum)(p[s.key] || 0)}`);
    tip.textContent = lines.join('\n');
    tip.style.opacity = '1';
    tip.style.left = Math.min(ev.clientX + 14, window.innerWidth - 190) + 'px';
    tip.style.top = (ev.clientY + 14) + 'px';
  });
  svg.addEventListener('mouseleave', () => {
    tip.style.opacity = '0';
    if (vline) vline.classList.add('hidden');
  });
}

// ---------------------------------------------------------------------------
// Views
// ---------------------------------------------------------------------------

const main = () => document.getElementById('main');

async function renderOverview() {
  const [ov, se] = await Promise.all([
    api('/admin/api/overview?range=' + state.range),
    api('/admin/api/series?range=' + state.range),
  ]);
  const t = ov.totals, pts = se.points || [], bs = se.bucket_sec;

  // Uptime is reported as unknown below OpenRouter's 100-request floor rather
  // than shown as a misleading ratio computed from a handful of calls.
  let uptimeHTML;
  if (t.uptime < 0) {
    uptimeHTML = `<div class="value">—</div><div class="sub">needs 100+ requests</div>`;
  } else {
    const pct = t.uptime * 100;
    const cls = pct >= 95 ? 'good' : pct >= 80 ? 'warn' : 'bad';
    const note = pct >= 95 ? 'normal routing' : pct >= 80 ? 'degraded at OpenRouter' : 'down at OpenRouter';
    uptimeHTML = `<div class="value ${cls}">${pct.toFixed(2)}%</div><div class="sub">${note}</div>`;
  }

  const totalTok = t.prompt_tokens + t.completion_tokens;
  const cards = [
    ['Requests', fmtNum(t.requests), `${fmtNum(t.errors)} err · ${fmtNum(t.rate_limited)} 429`],
    ['Tokens', fmtNum(totalTok), `${fmtNum(t.prompt_tokens)} in · ${fmtNum(t.completion_tokens)} out`],
    ['Cached prompt', fmtNum(t.cached_tokens),
      t.prompt_tokens ? ((t.cached_tokens / t.prompt_tokens) * 100).toFixed(1) + '% hit rate' : 'no prompt tokens'],
    ['TTFT p50', fmtMs(t.ttft_p50), `p95 ${fmtMs(t.ttft_p95)} · p99 ${fmtMs(t.ttft_p99)}`],
    ['Throughput', t.tps_avg ? t.tps_avg.toFixed(1) + ' tok/s' : '—', 'output ÷ total time'],
    ['Revenue', fmtUSD(t.cost_usd), 'at listed prices'],
  ].map(([label, value, sub]) =>
    `<div class="card"><div class="label">${label}</div><div class="value">${value}</div><div class="sub">${sub}</div></div>`
  ).join('');

  main().innerHTML = `
    <div class="cards">
      ${cards}
      <div class="card"><div class="label">Uptime</div>${uptimeHTML}</div>
    </div>
    <div class="charts">
      <div class="chart"><h3>Requests</h3>
        <div class="legend"><span><i style="background:var(--accent)"></i>requests</span></div>
        <div id="c-req"></div></div>
      <div class="chart"><h3>Errors &amp; rate limits</h3>
        <div class="legend">
          <span><i style="background:var(--bad)"></i>errors (uptime-affecting)</span>
          <span><i style="background:var(--warn)"></i>429</span>
          <span><i style="background:var(--muted)"></i>user errors</span></div>
        <div id="c-err"></div></div>
      <div class="chart"><h3>Tokens</h3>
        <div class="legend">
          <span><i style="background:var(--accent)"></i>prompt</span>
          <span><i style="background:var(--good)"></i>completion</span></div>
        <div id="c-tok"></div></div>
      <div class="chart"><h3>Time to first token</h3>
        <div class="legend">
          <span><i style="background:var(--good)"></i>p50</span>
          <span><i style="background:var(--warn)"></i>p95</span>
          <span><i style="background:var(--bad)"></i>p99</span></div>
        <div id="c-ttft"></div></div>
      <div class="chart"><h3>Throughput</h3>
        <div class="legend"><span><i style="background:var(--good)"></i>tokens/sec</span></div>
        <div id="c-tps"></div></div>
      <div class="chart"><h3>Revenue</h3>
        <div class="legend"><span><i style="background:var(--accent)"></i>USD</span></div>
        <div id="c-cost"></div></div>
    </div>
    <h2>By model</h2>
    <div class="tablewrap">${modelTable(ov.by_model || [])}</div>
    <h2 style="margin-top:22px">Backends</h2>
    <div class="tablewrap">${backendTable(ov.backends || [])}</div>`;

  barChart(document.getElementById('c-req'), pts, 'requests', 'var(--accent)', bs, { label: 'requests' });
  lineChart(document.getElementById('c-err'), pts, [
    { key: 'errors', label: 'errors', color: 'var(--bad)' },
    { key: 'rate_limited', label: '429', color: 'var(--warn)' },
    { key: 'user_errors', label: 'user errors', color: 'var(--muted)' },
  ], bs);
  lineChart(document.getElementById('c-tok'), pts, [
    { key: 'prompt_tokens', label: 'prompt', color: 'var(--accent)' },
    { key: 'completion_tokens', label: 'completion', color: 'var(--good)' },
  ], bs);
  lineChart(document.getElementById('c-ttft'), pts, [
    { key: 'ttft_p50', label: 'p50', color: 'var(--good)' },
    { key: 'ttft_p95', label: 'p95', color: 'var(--warn)' },
    { key: 'ttft_p99', label: 'p99', color: 'var(--bad)' },
  ], bs, { fmt: fmtMs });
  lineChart(document.getElementById('c-tps'), pts,
    [{ key: 'tps_avg', label: 'tok/s', color: 'var(--good)' }], bs,
    { area: true, fmt: v => v.toFixed(0) });
  barChart(document.getElementById('c-cost'), pts, 'cost_usd', 'var(--accent)', bs,
    { label: 'revenue', fmt: fmtUSD });
}

function modelTable(rows) {
  if (!rows.length) return '<div class="empty">No traffic in this range</div>';
  const max = Math.max(...rows.map(r => r.requests), 1);
  return `<table><thead><tr>
    <th>Model</th><th class="num">Requests</th><th></th>
    <th class="num">Prompt</th><th class="num">Completion</th>
    <th class="num">TTFT p50</th><th class="num">tok/s</th>
    <th class="num">Errors</th><th class="num">429</th><th class="num">Revenue</th>
  </tr></thead><tbody>${rows.map(r => `<tr>
    <td class="mono">${modelLabel(r.model)}</td>
    <td class="num">${fmtNum(r.requests)}</td>
    <td><div class="bar"><span style="width:${(r.requests / max) * 100}%"></span></div></td>
    <td class="num">${fmtNum(r.prompt_tokens)}</td>
    <td class="num">${fmtNum(r.completion_tokens)}</td>
    <td class="num">${fmtMs(r.ttft_p50)}</td>
    <td class="num">${r.tps_avg ? r.tps_avg.toFixed(1) : '—'}</td>
    <td class="num">${r.errors ? `<span class="pill down">${r.errors}</span>` : '0'}</td>
    <td class="num">${r.rate_limited ? `<span class="pill warn">${r.rate_limited}</span>` : '0'}</td>
    <td class="num">${fmtUSD(r.cost_usd)}</td>
  </tr>`).join('')}</tbody></table>`;
}

function backendTable(rows) {
  if (!rows.length) return '<div class="empty">No backends configured</div>';
  return `<table><thead><tr>
    <th>Backend</th><th>Status</th><th class="num">In flight</th><th>Load</th><th>Last error</th>
  </tr></thead><tbody>${rows.map(b => {
    const pct = b.max_concurrency ? (b.inflight / b.max_concurrency) * 100 : 0;
    return `<tr>
      <td class="mono">${esc(b.name)}</td>
      <td>${b.healthy ? '<span class="pill ok">healthy</span>' : '<span class="pill down">down</span>'}</td>
      <td class="num">${b.inflight} / ${b.max_concurrency}</td>
      <td><div class="bar"><span style="width:${Math.min(100, pct)}%;background:${pct > 85 ? 'var(--bad)' : 'var(--accent)'}"></span></div></td>
      <td class="mono muted">${esc(b.last_error || '')}</td>
    </tr>`;
  }).join('')}</tbody></table>`;
}

async function renderModels() {
  const d = await api('/admin/api/models');
  const rows = d.models || [];
  main().innerHTML = `<h2>Model catalog <span class="muted">(${rows.length})</span></h2>
    <p class="muted" style="margin-top:-8px;margin-bottom:14px">
      Edit <code>config/config.yaml</code> and restart to change this. Prices are USD per token.</p>
    <div class="tablewrap">${rows.length ? `<table><thead><tr>
      <th>ID</th><th>Upstream</th><th>Backends</th><th class="num">Context</th>
      <th class="num">Max out</th><th>Quant</th>
      <th class="num">Prompt $/M</th><th class="num">Completion $/M</th>
      <th>Caps</th><th>State</th>
    </tr></thead><tbody>${rows.map(m => `<tr>
      <td class="mono">${esc(m.id)}</td>
      <td class="mono muted">${esc(m.upstream_model)}</td>
      <td class="mono muted">${esc((m.backends || []).join(', '))}</td>
      <td class="num">${fmtNum(m.context_length)}</td>
      <td class="num">${fmtNum(m.max_output_tokens)}</td>
      <td>${m.quantization ? `<span class="pill off">${esc(m.quantization)}</span>` : '—'}</td>
      <td class="num">${perMillion(m.prompt_usd)}</td>
      <td class="num">${perMillion(m.completion_usd)}</td>
      <td>${m.tools ? '<span class="pill ok">tools</span> ' : ''}${m.reasoning ? '<span class="pill ok">reasoning</span>' : ''}</td>
      <td>${m.is_free ? '<span class="pill warn">free</span> ' : ''}${m.ready ? '<span class="pill ok">ready</span>' : '<span class="pill off">hidden</span>'}</td>
    </tr>`).join('')}</tbody></table>` : '<div class="empty">No models configured</div>'}</div>`;
}

// Per-token prices are unreadable at these magnitudes, so the table shows the
// per-million figure that OpenRouter displays to users.
function perMillion(s) {
  const v = parseFloat(s);
  if (!v) return '<span class="muted">free</span>';
  return '$' + (v * 1e6).toFixed(2);
}

async function renderBackends() {
  const d = await api('/admin/api/backends');
  main().innerHTML = `<h2>Backends</h2>
    <p class="muted" style="margin-top:-8px;margin-bottom:14px">
      Health is probed every 10s against each replica's <code>/v1/models</code>.
      A replica at its concurrency cap sheds new requests with a 429 rather than queueing them.</p>
    <div class="tablewrap">${backendTableFull(d.backends || [])}</div>`;
}

function backendTableFull(rows) {
  if (!rows.length) return '<div class="empty">No backends configured</div>';
  return `<table><thead><tr>
    <th>Backend</th><th>Base URL</th><th>Status</th>
    <th class="num">In flight</th><th>Load</th><th>Last error</th>
  </tr></thead><tbody>${rows.map(b => {
    const pct = b.max_concurrency ? (b.inflight / b.max_concurrency) * 100 : 0;
    return `<tr>
      <td class="mono">${esc(b.name)}</td>
      <td class="mono muted">${esc(b.base_url)}</td>
      <td>${b.healthy ? '<span class="pill ok">healthy</span>' : '<span class="pill down">down</span>'}</td>
      <td class="num">${b.inflight} / ${b.max_concurrency}</td>
      <td><div class="bar"><span style="width:${Math.min(100, pct)}%;background:${pct > 85 ? 'var(--bad)' : 'var(--accent)'}"></span></div></td>
      <td class="mono muted">${esc(b.last_error || '')}</td>
    </tr>`;
  }).join('')}</tbody></table>`;
}

async function renderRequests() {
  const d = await api('/admin/api/requests?limit=200');
  const rows = d.requests || [];
  main().innerHTML = `<h2>Recent requests</h2>
    <p class="muted" style="margin-top:-8px;margin-bottom:14px">
      Last 200 requests. Status 499 means the client disconnected before completion — not counted as an error.</p>
    <div class="tablewrap">${rows.length ? `<table><thead><tr>
      <th>Time</th><th>Request</th><th>Model</th><th>Backend</th><th>Status</th><th>Mode</th>
      <th class="num">Prompt</th><th class="num">Completion</th>
      <th class="num">TTFT</th><th class="num">Total</th><th class="num">tok/s</th>
      <th class="num">Cost</th><th>Error</th>
    </tr></thead><tbody>${rows.map(r => `<tr>
      <td class="mono muted">${fmtFullTime(r.ts)}</td>
      <td class="mono muted">${esc(r.request_id.slice(0, 12))}</td>
      <td class="mono">${modelLabel(r.model)}</td>
      <td class="mono muted">${esc(r.backend || '—')}</td>
      <td>${statusPill(r.status)}</td>
      <td class="muted">${r.streamed ? 'stream' : 'json'}</td>
      <td class="num">${fmtNum(r.prompt_tokens)}</td>
      <td class="num">${fmtNum(r.completion_tokens)}</td>
      <td class="num">${r.ttft_ms >= 0 ? fmtMs(r.ttft_ms) : '—'}</td>
      <td class="num">${fmtMs(r.total_ms)}</td>
      <td class="num">${r.tps ? r.tps.toFixed(1) : '—'}</td>
      <td class="num">${fmtUSD(r.cost_usd)}</td>
      <td class="mono muted">${esc((r.error || '').slice(0, 60))}</td>
    </tr>`).join('')}</tbody></table>` : '<div class="empty">No requests yet</div>'}</div>`;
}

async function renderKeys() {
  const d = await api('/admin/api/keys');
  const rows = d.keys || [];
  main().innerHTML = `<h2>API keys</h2>
    ${state.secret ? `<div class="secret"><strong>Copy this now — it is not stored and cannot be shown again:</strong><br>${esc(state.secret)}</div>` : ''}
    <div class="toolbar">
      <input id="k-name" placeholder="Key name (e.g. openrouter)" style="min-width:240px">
      <input id="k-rpm" type="number" min="0" placeholder="Requests/min (0 = unlimited)" style="min-width:220px">
      <button class="primary" id="k-create">Create key</button>
    </div>
    <div class="tablewrap">${rows.length ? `<table><thead><tr>
      <th>Name</th><th>Prefix</th><th>Created</th><th class="num">RPM limit</th><th>State</th><th></th>
    </tr></thead><tbody>${rows.map(k => `<tr>
      <td>${esc(k.name)}</td>
      <td class="mono muted">${esc(k.prefix)}…</td>
      <td class="muted">${new Date(k.created_at * 1000).toLocaleDateString()}</td>
      <td class="num">${k.rpm_limit || '∞'}</td>
      <td>${k.disabled ? '<span class="pill off">disabled</span>' : '<span class="pill ok">active</span>'}</td>
      <td>
        <button data-toggle="${k.id}" data-disabled="${k.disabled ? 0 : 1}">${k.disabled ? 'Enable' : 'Disable'}</button>
        <button class="danger" data-del="${k.id}">Delete</button>
      </td>
    </tr>`).join('')}</tbody></table>` : '<div class="empty">No keys yet — create one for OpenRouter to authenticate with</div>'}</div>`;

  state.secret = null; // shown once, then cleared on the next render

  document.getElementById('k-create').onclick = async () => {
    const name = document.getElementById('k-name').value.trim();
    const rpm = parseInt(document.getElementById('k-rpm').value || '0', 10);
    const res = await api('/admin/api/keys', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, rpm_limit: rpm }),
    });
    state.secret = res.secret;
    renderKeys();
  };
  main().querySelectorAll('[data-del]').forEach(b => b.onclick = async () => {
    if (!confirm('Delete this key? Any client using it stops working immediately.')) return;
    await api('/admin/api/keys/' + b.dataset.del, { method: 'DELETE' });
    renderKeys();
  });
  main().querySelectorAll('[data-toggle]').forEach(b => b.onclick = async () => {
    await api('/admin/api/keys/' + b.dataset.toggle + '/toggle', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ disabled: b.dataset.disabled === '1' }),
    });
    renderKeys();
  });
}

async function renderModelDoc() {
  const doc = await api('/admin/api/modeldoc');
  main().innerHTML = `<h2>Provider model document</h2>
    <p class="muted" style="margin-top:-8px;margin-bottom:14px">
      This is exactly what OpenRouter fetches from <code>/v1/models</code>, in schema 2.4 format.
      Verify it before submitting your application.</p>
    <pre class="doc">${esc(JSON.stringify(doc, null, 2))}</pre>`;
}

function renderGuides() {
  main().innerHTML = GUIDES_HTML;
}


// ---------------------------------------------------------------------------
// Nodes
// ---------------------------------------------------------------------------

function gib(bytes) {
  if (!bytes) return '—';
  return (bytes / (1024 ** 3)).toFixed(1) + ' GiB';
}

function hardwareLine(hw) {
  if (!hw) return '';
  const parts = [];
  if (hw.gpus && hw.gpus.length) {
    const names = {};
    for (const g of hw.gpus) names[g.name] = (names[g.name] || 0) + 1;
    parts.push(Object.entries(names).map(([n, c]) => `${c}x ${n}`).join(', '));
    parts.push(gib(hw.gpus.reduce((a, g) => a + g.vram_bytes, 0)) + ' VRAM');
  } else {
    parts.push('no GPU');
    if (hw.mem_bandwidth_gbs) parts.push(`~${hw.mem_bandwidth_gbs} GB/s mem`);
  }
  parts.push(`${hw.cpu_cores} cores`, gib(hw.ram_bytes) + ' RAM', gib(hw.disk_free_bytes) + ' free');
  if (!hw.has_nvme) parts.push('no NVMe');
  return parts.join(' · ');
}

function engineStatePill(state) {
  if (state === 'ready') return '<span class="pill ok">ready</span>';
  if (state === 'starting') return '<span class="pill warn"><span class="spinner"></span>starting</span>';
  if (state === 'failed') return '<span class="pill down">failed</span>';
  return `<span class="pill off">${esc(state)}</span>`;
}

async function renderNodes() {
  const d = await api('/admin/api/nodes');
  const nodes = d.nodes || [];
  if (!nodes.length) {
    main().innerHTML = `<h2>Nodes</h2>
      <div class="step">
        <h3>No nodes configured</h3>
        <p class="hint">A node is an inference host running <code>llmfast-agent</code>. Add one to
           <code>config/config.yaml</code> and restart:</p>
        <pre class="logs">nodes:
  - name: gpu-a
    url: http://10.0.0.11:9900
    token: $LLMFAST_AGENT_TOKEN
    max_concurrency: 96</pre>
      </div>`;
    return;
  }

  main().innerHTML = `<h2>Nodes</h2>
    <p class="muted" style="margin-top:-8px;margin-bottom:14px">
      Inference hosts running <code>llmfast-agent</code>. Engines they report as ready are routed to automatically.</p>
    ${nodes.map(n => {
      const info = n.info || {};
      const inst = info.instances || [];
      return `<div class="nodecard ${n.reachable ? 'viable' : 'blocked'}">
        <div class="head">
          <span class="name">${esc(n.name)}</span>
          ${n.reachable ? '<span class="pill ok">reachable</span>' : '<span class="pill down">unreachable</span>'}
          <span class="muted mono" style="font-size:12px">${esc(n.url)}</span>
          ${(info.engines_available || []).map(e => `<span class="pill off">${esc(e)}</span>`).join(' ')}
        </div>
        ${n.reachable ? `<div class="hw">${esc(hardwareLine(info.node))}</div>` : ''}
        ${n.last_error ? `<div class="note block">${esc(n.last_error)}</div>` : ''}
        ${inst.length ? `<table style="margin-top:6px"><thead><tr>
            <th>Model</th><th>Engine</th><th>State</th><th class="num">Port</th>
            <th class="num">Restarts</th><th></th></tr></thead><tbody>
          ${inst.map(i => `<tr>
            <td class="mono">${esc(i.spec.served_name)}</td>
            <td class="muted">${esc(i.spec.engine)} ${esc(i.spec.quantization || '')}</td>
            <td>${engineStatePill(i.state)}</td>
            <td class="num">${i.port}</td>
            <td class="num">${i.restarts || 0}</td>
            <td>
              <button data-logs="${esc(i.spec.served_name)}" data-node="${esc(n.name)}">Logs</button>
              <button class="danger" data-stopengine="${esc(i.spec.served_name)}" data-node="${esc(n.name)}">Stop</button>
            </td>
          </tr>${i.error ? `<tr><td colspan="6"><div class="note block">${esc(i.error)}</div></td></tr>` : ''}`).join('')}
        </tbody></table>` : '<div class="note info">No engines running on this node.</div>'}
        <div id="logs-${cssId(n.name)}"></div>
      </div>`;
    }).join('')}`;

  main().querySelectorAll('[data-logs]').forEach(b => b.onclick = async () => {
    const target = document.getElementById('logs-' + cssId(b.dataset.node));
    target.innerHTML = '<div class="note info"><span class="spinner"></span>loading logs…</div>';
    try {
      const r = await api(`/admin/api/nodes/${encodeURIComponent(b.dataset.node)}/logs?served_name=${encodeURIComponent(b.dataset.logs)}&n=200`);
      target.innerHTML = `<pre class="logs">${esc((r.lines || []).join('\n') || '(no output captured yet)')}</pre>`;
    } catch (e) {
      target.innerHTML = `<div class="note block">${esc(e.message)}</div>`;
    }
  });
  main().querySelectorAll('[data-stopengine]').forEach(b => b.onclick = async () => {
    if (!confirm(`Stop ${b.dataset.stopengine} on ${b.dataset.node}? It will not restart by itself.`)) return;
    b.disabled = true;
    try {
      await api(`/admin/api/nodes/${encodeURIComponent(b.dataset.node)}/stop`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ served_name: b.dataset.stopengine }),
      });
    } catch (e) { alert(e.message); }
    renderNodes();
  });
}

// cssId makes an element id safe from a node name.
function cssId(s) { return String(s).replace(/[^a-zA-Z0-9_-]/g, '_'); }

// ---------------------------------------------------------------------------
// Add Model
//
// The flow is deliberately two steps with a confirmation in between. Installing
// a model downloads tens of gigabytes and occupies a GPU, so the operator sees
// exactly what will run, where, and at what price before anything starts.
// ---------------------------------------------------------------------------

function renderAddModel() {
  main().innerHTML = `<div class="wizard">
    <h2>Add a model</h2>
    <div class="step">
      <h3>1 — Which model?</h3>
      <p class="hint">Paste a HuggingFace repository id. We read its config to work out what it needs,
         then check it against each node's hardware.</p>
      <div class="row">
        <input type="text" id="hfid" placeholder="Qwen/Qwen3-8B" spellcheck="false" autocapitalize="off">
        <button class="primary" id="inspect-btn">Inspect</button>
      </div>
      <div class="row" style="margin-top:8px">
        <span class="muted" style="font-size:12px">Try:</span>
        ${['Qwen/Qwen3-8B', 'Qwen/Qwen3-32B', 'zai-org/GLM-4.6', 'deepseek-ai/DeepSeek-V3']
          .map(m => `<button data-example="${m}" style="font-size:12px;padding:4px 9px">${m}</button>`).join('')}
      </div>
      <div id="inspect-err" class="err"></div>
    </div>
    <div id="inspect-out"></div>
  </div>`;

  const go = () => inspectModel(document.getElementById('hfid').value.trim());
  document.getElementById('inspect-btn').onclick = go;
  document.getElementById('hfid').addEventListener('keydown', e => { if (e.key === 'Enter') go(); });
  main().querySelectorAll('[data-example]').forEach(b => b.onclick = () => {
    document.getElementById('hfid').value = b.dataset.example;
    go();
  });
}

async function inspectModel(hfID) {
  const err = document.getElementById('inspect-err');
  const out = document.getElementById('inspect-out');
  err.textContent = '';
  if (!hfID) { err.textContent = 'Enter a model id, e.g. Qwen/Qwen3-8B'; return; }

  out.innerHTML = `<div class="step"><span class="spinner"></span>Reading ${esc(hfID)} from HuggingFace…</div>`;
  let d;
  try {
    d = await api('/admin/api/inspect', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ hf_id: hfID }),
    });
  } catch (e) {
    out.innerHTML = '';
    err.textContent = e.message;
    return;
  }
  renderInspectResult(d);
}

function renderInspectResult(d) {
  const i = d.info, out = document.getElementById('inspect-out');
  const kvKB = (i.num_layers && i.num_kv_heads && i.head_dim)
    ? (2 * i.num_layers * i.num_kv_heads * i.head_dim * 2 / 1024) : 0;

  const perM = s => '$' + (parseFloat(s) * 1e6).toFixed(3);

  out.innerHTML = `
    <div class="step">
      <h3>2 — ${esc(i.id)}</h3>
      ${d.already_installed ? `<div class="note warn">A model with id <code>${esc(d.suggested_id)}</code>
         is already in the catalog. Installing again will replace its entry.</div>` : ''}
      ${i.gated ? `<div class="note warn">This repository is gated. Accept its licence on HuggingFace and
         set <code>HF_TOKEN</code> on the node agent, or the download will fail.</div>` : ''}
      <div class="specs">
        <div class="spec"><div class="k">Parameters</div><div class="v">${esc(i.params_label)}${i.is_moe ? ' MoE' : ''}</div></div>
        <div class="spec"><div class="k">Precision</div><div class="v">${esc(i.dtype || '—')}</div></div>
        <div class="spec"><div class="k">Max context</div><div class="v">${fmtNum(i.max_positions)}</div></div>
        <div class="spec"><div class="k">Layers</div><div class="v">${i.num_layers}</div></div>
        <div class="spec"><div class="k">KV heads</div><div class="v">${i.num_kv_heads} / ${i.num_heads}</div></div>
        <div class="spec"><div class="k">KV cache</div><div class="v">${kvKB.toFixed(0)} KB<span class="muted" style="font-size:11px">/tok</span></div></div>
      </div>
      <div class="hint" style="margin-top:12px">${esc(i.architecture || '')} · downloads ${fmtNum(i.downloads)}</div>
    </div>

    <div class="step">
      <h3>3 — Catalog entry</h3>
      <p class="hint">How this model will appear to OpenRouter. Prices are USD per million tokens;
         the suggestion scales with active parameters and is a starting point, not advice.</p>
      <div class="fields">
        <div class="field"><label>Model id</label><input type="text" id="f-id" value="${esc(d.suggested_id)}"></div>
        <div class="field"><label>Display name</label><input type="text" id="f-name" value="${esc(d.suggested_name)}"></div>
        <div class="field"><label>Prompt $/M</label><input type="number" step="0.001" id="f-prompt" value="${(parseFloat(d.pricing.prompt) * 1e6).toFixed(3)}"></div>
        <div class="field"><label>Completion $/M</label><input type="number" step="0.001" id="f-completion" value="${(parseFloat(d.pricing.completion) * 1e6).toFixed(3)}"></div>
        <div class="field"><label>Cached $/M</label><input type="number" step="0.0001" id="f-cached" value="${(parseFloat(d.pricing.cached_prompt) * 1e6).toFixed(4)}"></div>
      </div>
      <div class="row" style="margin-top:14px">
        <label class="live"><input type="checkbox" id="f-tools" checked> Supports tools</label>
        <label class="live"><input type="checkbox" id="f-reasoning"> Reasoning model</label>
        <label class="live"><input type="checkbox" id="f-zdr" checked> Zero data retention</label>
      </div>
      <p class="hint" style="margin-top:8px">
        This gateway never writes prompt or completion content to disk, so zero data
        retention is accurate by default. It is published as <code>compliance.zdr</code>
        and must match what your privacy policy says — uncheck it if you log prompts
        anywhere in your stack.
      </p>
    </div>

    ${d.quant_candidates && d.quant_candidates.length ? `
    <div class="step">
      <h3>This model needs a smaller checkpoint</h3>
      <p class="hint">It does not fit in VRAM at full precision. Quantization is stored in the
         weights themselves, so there is no flag that shrinks it — you install a different
         repository. These are published versions of this same model, the owner's own first.</p>
      ${d.quant_candidates.map(q => `
        <div class="row" style="align-items:center;gap:10px;margin-top:6px">
          <button class="btn" data-requant="${esc(q.repo)}">Inspect this instead</button>
          <code>${esc(q.repo)}</code>
          <span class="pill ${q.quant === 'fp8' ? 'ok' : 'warn'}">${esc(q.quant)}</span>
          ${q.official ? '<span class="pill ok">official</span>' : ''}
          <span class="hint">${fmtNum(q.downloads)} downloads</span>
        </div>`).join('')}
      <p class="hint" style="margin-top:10px">
        fp8 is near-lossless and usually the right choice. 4-bit (AWQ/GPTQ) frees more VRAM
        for concurrency but costs some output quality — worth measuring in the playground
        before you publish.</p>
    </div>` : (d.quant_error ? `<div class="step"><div class="note warn">${esc(d.quant_error)}</div></div>` : '')}

    ${d.gguf_candidates && d.gguf_candidates.length ? `
    <div class="step">
      <h3>GGUF conversion</h3>
      <p class="hint">llama.cpp cannot read the original weights, so a CPU node needs a converted repo.
         The model owner's own conversion is listed first.</p>
      <select id="f-gguf" style="min-width:340px">
        ${d.gguf_candidates.map(g => `<option value="${esc(g.repo)}">${esc(g.repo)}${g.official ? '  (official)' : ''} — ${fmtNum(g.downloads)} downloads</option>`).join('')}
      </select>
    </div>` : (d.gguf_error ? `<div class="step"><div class="note warn">${esc(d.gguf_error)}</div></div>` : '')}

    <div class="step">
      <h3>4 — Where should it run?</h3>
      <p class="hint">Each node is checked against what this model actually needs.</p>
      ${(d.plans || []).map(p => nodePlanCard(p)).join('') || '<div class="note info">No nodes configured.</div>'}
    </div>
    <div id="install-out"></div>`;

  main().querySelectorAll('[data-install]').forEach(b => b.onclick = () => installOnNode(b.dataset.install, d));
  main().querySelectorAll('[data-requant]').forEach(b => b.onclick = () => {
    const box = document.getElementById('hfid');
    if (box) box.value = b.dataset.requant;
    inspectModel(b.dataset.requant);
  });
}

function nodePlanCard(p) {
  if (!p.plan) {
    return `<div class="nodecard blocked">
      <div class="head"><span class="name">${esc(p.node)}</span>
        <span class="pill down">unreachable</span></div>
      <div class="note block">${esc(p.reason || 'no information')}</div></div>`;
  }
  const pl = p.plan;
  const cls = pl.viable ? 'viable' : (pl.fits ? '' : 'blocked');
  const badge = pl.viable ? '<span class="pill ok">recommended</span>'
    : pl.fits ? '<span class="pill warn">runs, but slow</span>'
    : '<span class="pill down">will not fit</span>';

  return `<div class="nodecard ${cls}">
    <div class="head">
      <span class="name">${esc(p.node)}</span>${badge}
      <span class="pill off">${esc(pl.engine)}</span>
      <span class="pill off">${esc(pl.quantization)}</span>
      ${pl.kv_cache_dtype === 'fp8' ? '<span class="pill ok">fp8 KV</span>' : ''}
      ${pl.tensor_parallel > 1 ? `<span class="pill off">TP=${pl.tensor_parallel}</span>` : ''}
    </div>
    <div class="hw">${esc(hardwareLine(p.hardware))}</div>
    <div class="specs">
      <div class="spec"><div class="k">Context</div><div class="v">${fmtNum(pl.max_model_len)}</div></div>
      <div class="spec"><div class="k">Concurrent</div><div class="v">${pl.max_num_seqs}</div></div>
      <div class="spec"><div class="k">Weights</div><div class="v">${gib(pl.weight_bytes)}</div></div>
      <div class="spec"><div class="k">KV budget</div><div class="v">${gib(pl.kv_budget_bytes)}</div></div>
      <div class="spec"><div class="k">Disk needed</div><div class="v">${gib(pl.disk_bytes)}</div></div>
      <div class="spec"><div class="k">Est. speed</div><div class="v">${pl.est_tokens_per_sec ? pl.est_tokens_per_sec.toFixed(0) + ' t/s' : '—'}</div></div>
    </div>
    ${pl.viability_note ? `<div class="note ${pl.viable ? 'ok' : 'warn'}">${esc(pl.viability_note)}</div>` : ''}
    ${(pl.warnings || []).map(w => `<div class="note warn">${esc(w)}</div>`).join('')}
    ${(pl.blockers || []).map(b => `<div class="note block">${esc(b)}</div>`).join('')}
    <div class="row" style="margin-top:10px">
      <button class="${pl.viable ? 'primary' : ''}" data-install="${esc(p.node)}" ${pl.fits ? '' : 'disabled'}>
        Install on ${esc(p.node)}</button>
      ${pl.fits && !pl.viable ? '<span class="muted" style="font-size:12px">You can still install it — useful for testing the pipeline.</span>' : ''}
    </div>
  </div>`;
}

async function installOnNode(node, d) {
  const plan = (d.plans || []).find(p => p.node === node)?.plan;
  if (!plan) return;

  const modelID = document.getElementById('f-id').value.trim();
  const perTok = id => (parseFloat(document.getElementById(id).value || '0') / 1e6).toFixed(12).replace(/0+$/, '') || '0';

  const body = {
    node, hf_id: d.info.id, model_id: modelID,
    name: document.getElementById('f-name').value.trim(),
    engine: plan.engine, quantization: plan.quantization,
    kv_cache_dtype: plan.kv_cache_dtype,
    tensor_parallel: plan.tensor_parallel, max_model_len: plan.max_model_len,
    max_num_seqs: plan.max_num_seqs,
    gguf_repo: document.getElementById('f-gguf')?.value || '',
    prompt_usd: perTok('f-prompt'), completion_usd: perTok('f-completion'), cached_usd: perTok('f-cached'),
    tools: document.getElementById('f-tools').checked,
    reasoning: document.getElementById('f-reasoning').checked,
    zdr: document.getElementById('f-zdr').checked,
  };

  const out = document.getElementById('install-out');
  out.innerHTML = `<div class="step"><span class="spinner"></span>Asking ${esc(node)} to start the engine…</div>`;

  let res;
  try {
    res = await api('/admin/api/install', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    });
  } catch (e) {
    out.innerHTML = `<div class="step"><div class="note block">${esc(e.message)}</div></div>`;
    return;
  }

  out.innerHTML = `<div class="step">
    <h3>Installing ${esc(modelID)} on ${esc(node)}</h3>
    <div class="note ok">Engine launched. ${esc(res.note || '')}</div>
    <div class="hint" style="margin-top:10px">Command running on the node:</div>
    <pre class="logs">${esc(res.agent?.command || '')}</pre>
    <div id="install-progress"></div>
  </div>`;
  pollInstall(node, modelID);
}

// pollInstall follows the engine until it is serving or has failed. A first
// download can take a long time, so progress is shown from the engine's own
// output rather than leaving the operator staring at a spinner.
async function pollInstall(node, modelID) {
  const el = document.getElementById('install-progress');
  if (!el) return;

  for (let attempt = 0; attempt < 2400; attempt++) {
    let state = 'starting', lines = [], error = '';
    try {
      const r = await api(`/admin/api/nodes/${encodeURIComponent(node)}/logs?served_name=${encodeURIComponent(modelID)}&n=40`);
      state = r.state; lines = r.lines || []; error = r.error || '';
    } catch (e) {
      el.innerHTML = `<div class="note block">${esc(e.message)}</div>`;
      return;
    }
    if (!document.getElementById('install-progress')) return; // navigated away

    if (state === 'ready') {
      el.innerHTML = `<div class="note ok"><strong>${esc(modelID)} is serving.</strong>
        It is staged hidden so OpenRouter will not route to it yet.</div>
        <div class="row" style="margin-top:10px">
          <button class="primary" id="publish-now">Publish to OpenRouter</button>
          <span class="muted" style="font-size:12px">Test it first, then publish.</span>
        </div>
        <div id="publish-out"></div>`;
      document.getElementById('publish-now').onclick = async () => {
        try {
          await api('/admin/api/publish', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_id: modelID, ready: true }),
          });
          document.getElementById('publish-out').innerHTML =
            `<div class="note ok" style="margin-top:10px">Published. It now appears in
             <code>/v1/models</code> with <code>is_ready: true</code>.</div>`;
        } catch (e) {
          document.getElementById('publish-out').innerHTML = `<div class="note block">${esc(e.message)}</div>`;
        }
      };
      return;
    }
    if (state === 'failed') {
      el.innerHTML = `<div class="note block"><strong>Install failed.</strong><br>${esc(error)}</div>
        <pre class="logs">${esc(lines.join('\n'))}</pre>`;
      return;
    }
    el.innerHTML = `<div class="note info"><span class="spinner"></span>
      ${esc(state)} — downloading weights and loading the model. This can take a long time on a first install.</div>
      <pre class="logs">${esc(lines.join('\n') || '(waiting for output)')}</pre>`;
    await new Promise(r => setTimeout(r, 3000));
  }
}


// ---------------------------------------------------------------------------
// Playground
//
// Sends a real completion through the gateway's own inference path, so what is
// measured here is what a customer would experience: the same admission
// control, the same SSE relay, the same token accounting.
// ---------------------------------------------------------------------------

const pgState = { abort: null, lastModel: null };

async function renderPlayground() {
  const d = await api('/admin/api/playground/models');
  const models = d.models || [];
  const usable = models.filter(m => m.available);

  main().innerHTML = `<h2>Playground</h2>
    <p class="muted" style="margin-top:-8px;margin-bottom:14px">
      Runs through the real <code>/v1/chat/completions</code> path — same streaming, admission control
      and accounting a customer gets. Requests appear in your stats.</p>
    ${!models.length ? `<div class="step"><h3>No models in the catalog</h3>
       <p class="hint">Add one from the <strong>Add Model</strong> tab first.</p></div>` : `
    <div class="pg">
      <div class="pg-panel">
        <h3>Request</h3>
        <div class="field">
          <label>Model</label>
          <select id="pg-model">
            ${models.map(m => `<option value="${esc(m.id)}" ${m.available ? '' : 'disabled'}
               data-max="${m.max_output_tokens}" data-prompt="${esc(m.prompt_usd || '0')}"
               data-completion="${esc(m.completion_usd || '0')}">
               ${esc(m.id)}${m.available ? '' : ' — not running'}</option>`).join('')}
          </select>
        </div>
        <div class="field">
          <label>System prompt</label>
          <textarea id="pg-system" rows="3" placeholder="You are a helpful assistant."></textarea>
        </div>
        <div class="field">
          <label>Temperature</label>
          <div class="range-row">
            <input type="range" id="pg-temp" min="0" max="2" step="0.05" value="0.7">
            <span class="val" id="pg-temp-v">0.70</span>
          </div>
        </div>
        <div class="field">
          <label>Top P</label>
          <div class="range-row">
            <input type="range" id="pg-topp" min="0.05" max="1" step="0.05" value="1">
            <span class="val" id="pg-topp-v">1.00</span>
          </div>
        </div>
        <div class="field">
          <label>Max tokens</label>
          <input type="number" id="pg-max" value="512" min="1">
        </div>
        <label class="live"><input type="checkbox" id="pg-stream" checked> Stream the response</label>
      </div>

      <div>
        <div class="pg-panel">
          <div class="field" style="margin-bottom:0">
            <label>Message</label>
            <textarea id="pg-input" rows="3" placeholder="Ask it something…">Explain what a KV cache is, in two sentences.</textarea>
          </div>
          <div class="pg-bar">
            <button class="primary" id="pg-send" ${usable.length ? '' : 'disabled'}>Send</button>
            <button id="pg-stop" class="danger hidden">Stop</button>
            <button id="pg-clear">Clear</button>
            ${usable.length ? '' : '<span class="muted" style="font-size:12px">No model is currently running.</span>'}
          </div>
        </div>

        <div class="pg-out" style="margin-top:14px">
          <div class="pg-msg" id="pg-msg"><span class="muted">The response will stream here.</span></div>
          <div class="pg-metrics" id="pg-metrics">
            ${metricTile('TTFT', 'pg-ttft')}
            ${metricTile('Tokens out', 'pg-tokens')}
            ${metricTile('Throughput', 'pg-tps')}
            ${metricTile('Total time', 'pg-total')}
            ${metricTile('Prompt', 'pg-prompt')}
            ${metricTile('Cost', 'pg-cost')}
          </div>
        </div>
        <div id="pg-err" class="err"></div>
      </div>
    </div>`}`;

  if (!models.length) return;

  if (pgState.lastModel && models.some(m => m.id === pgState.lastModel && m.available)) {
    document.getElementById('pg-model').value = pgState.lastModel;
  } else if (usable.length) {
    document.getElementById('pg-model').value = usable[0].id;
  }

  const bind = (id, out, fmt) => {
    const el = document.getElementById(id);
    const show = () => document.getElementById(out).textContent = fmt(el.value);
    el.addEventListener('input', show); show();
  };
  bind('pg-temp', 'pg-temp-v', v => (+v).toFixed(2));
  bind('pg-topp', 'pg-topp-v', v => (+v).toFixed(2));

  document.getElementById('pg-send').onclick = runPlayground;
  document.getElementById('pg-stop').onclick = () => pgState.abort?.abort();
  document.getElementById('pg-clear').onclick = () => {
    document.getElementById('pg-msg').innerHTML = '<span class="muted">The response will stream here.</span>';
    resetMetrics();
    document.getElementById('pg-err').textContent = '';
  };
  document.getElementById('pg-input').addEventListener('keydown', e => {
    // Enter sends; Shift+Enter inserts a newline, as in every chat UI.
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); runPlayground(); }
  });
}

function metricTile(label, id) {
  return `<div class="m"><div class="k">${label}</div><div class="v" id="${id}">—</div></div>`;
}

function resetMetrics() {
  for (const id of ['pg-ttft', 'pg-tokens', 'pg-tps', 'pg-total', 'pg-prompt', 'pg-cost']) {
    const el = document.getElementById(id);
    if (el) el.textContent = '—';
  }
}

async function runPlayground() {
  const modelSel = document.getElementById('pg-model');
  const model = modelSel.value;
  if (!model) return;
  pgState.lastModel = model;

  const text = document.getElementById('pg-input').value.trim();
  if (!text) return;

  const system = document.getElementById('pg-system').value.trim();
  const messages = system ? [{ role: 'system', content: system }, { role: 'user', content: text }]
                          : [{ role: 'user', content: text }];

  const body = {
    model, messages,
    temperature: parseFloat(document.getElementById('pg-temp').value),
    top_p: parseFloat(document.getElementById('pg-topp').value),
    max_tokens: parseInt(document.getElementById('pg-max').value || '512', 10),
    stream: document.getElementById('pg-stream').checked,
    // Ask for usage so cost and token counts are the engine's real numbers
    // rather than something counted client-side.
    stream_options: { include_usage: true },
  };

  const msg = document.getElementById('pg-msg');
  const err = document.getElementById('pg-err');
  const sendBtn = document.getElementById('pg-send');
  const stopBtn = document.getElementById('pg-stop');
  err.textContent = '';
  msg.innerHTML = '<span class="cursor"></span>';
  resetMetrics();

  sendBtn.disabled = true;
  stopBtn.classList.remove('hidden');
  pgState.abort = new AbortController();

  const priceIn = parseFloat(modelSel.selectedOptions[0]?.dataset.prompt || '0');
  const priceOut = parseFloat(modelSel.selectedOptions[0]?.dataset.completion || '0');
  const t0 = performance.now();
  let ttft = null, content = '', reasoning = '', usage = null, chunks = 0;

  const paint = () => {
    // Reasoning traces are shown separately so they read as the model's
    // thinking rather than part of its answer.
    const think = reasoning ? `<span class="think">${esc(reasoning)}</span>` : '';
    msg.innerHTML = think + esc(content) + '<span class="cursor"></span>';
    msg.scrollTop = msg.scrollHeight;
  };

  try {
    const res = await fetch('/admin/api/playground', {
      method: 'POST', credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body), signal: pgState.abort.signal,
    });

    if (!res.ok) {
      let m = res.statusText;
      try { m = (await res.json()).error.message; } catch (_) {}
      throw new Error(`${res.status} — ${m}`);
    }

    if (!body.stream) {
      const j = await res.json();
      content = j.choices?.[0]?.message?.content || '';
      usage = j.usage;
      msg.innerHTML = esc(content);
    } else {
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });

        // SSE frames are separated by a blank line; hold any partial tail.
        const frames = buf.split('\n\n');
        buf = frames.pop();
        for (const frame of frames) {
          for (const line of frame.split('\n')) {
            if (!line.startsWith('data: ')) continue;   // skip keep-alive comments
            const payload = line.slice(6).trim();
            if (payload === '[DONE]') continue;
            let j;
            try { j = JSON.parse(payload); } catch (_) { continue; }

            if (j.usage) usage = j.usage;
            const delta = j.choices?.[0]?.delta;
            if (!delta) continue;
            const piece = delta.content || '';
            const think = delta.reasoning_content || delta.reasoning || '';
            if (piece || think) {
              if (ttft === null) {
                ttft = performance.now() - t0;
                document.getElementById('pg-ttft').textContent = fmtMs(Math.round(ttft));
              }
              chunks++;
              content += piece;
              reasoning += think;
              paint();
            }
          }
        }
      }
      msg.innerHTML = (reasoning ? `<span class="think">${esc(reasoning)}</span>` : '') + esc(content);
    }

    const total = performance.now() - t0;
    const outTok = usage?.completion_tokens ?? chunks;
    const inTok = usage?.prompt_tokens ?? 0;
    const cachedTok = usage?.prompt_tokens_details?.cached_tokens ?? 0;

    document.getElementById('pg-total').textContent = fmtMs(Math.round(total));
    document.getElementById('pg-tokens').textContent = fmtNum(outTok);
    // Throughput matches how OpenRouter measures it: output tokens over the
    // whole generation, including the wait for the first one.
    document.getElementById('pg-tps').textContent = outTok ? (outTok / (total / 1000)).toFixed(1) : '—';
    document.getElementById('pg-prompt').textContent =
      inTok ? fmtNum(inTok) + (cachedTok ? ` (${fmtNum(cachedTok)} cached)` : '') : '—';
    document.getElementById('pg-cost').textContent =
      fmtUSD(Math.max(0, inTok - cachedTok) * priceIn + outTok * priceOut);
    if (ttft === null && !body.stream) document.getElementById('pg-ttft').textContent = 'n/a';

  } catch (e) {
    if (e.name === 'AbortError') {
      msg.innerHTML = esc(content) + '<span class="muted"> [stopped]</span>';
    } else {
      err.textContent = e.message;
      msg.innerHTML = esc(content) || '<span class="muted">No response.</span>';
    }
  } finally {
    sendBtn.disabled = false;
    stopBtn.classList.add('hidden');
    pgState.abort = null;
  }
}


// ---------------------------------------------------------------------------
// Benchmark
//
// Every figure the planner produces is an estimate from published
// specifications. This measures the endpoint through the real request path, so
// the decision to keep or return a GPU rests on what it actually does.
// ---------------------------------------------------------------------------

const benchState = { result: null, abort: null };

async function renderBenchmark() {
  const d = await api('/admin/api/playground/models');
  const usable = (d.models || []).filter(m => m.available);

  main().innerHTML = `<h2>Benchmark</h2>
    <p class="muted" style="margin-top:-8px;margin-bottom:14px">
      Sweeps concurrency through the real <code>/v1/chat/completions</code> path and reports what
      the endpoint actually delivers. Run it before you publish a model — and before you decide
      whether the GPU is worth keeping.</p>
    ${!usable.length ? `<div class="step"><h3>No model is running</h3>
      <p class="hint">Install one from <strong>Add Model</strong> first.</p></div>` : `
    <div class="bench-grid">
      <div class="pg-panel">
        <h3>Load shape</h3>
        <div class="field">
          <label>Model</label>
          <select id="b-model">${usable.map(m => `<option value="${esc(m.id)}">${esc(m.id)}</option>`).join('')}</select>
        </div>
        <div class="field">
          <label>Concurrency levels</label>
          <input type="text" id="b-conc" value="1, 4, 8, 16">
        </div>
        <div class="field">
          <label>Requests per level</label>
          <input type="number" id="b-reqs" value="8" min="1" max="64">
        </div>
        <div class="field">
          <label>Prompt tokens</label>
          <input type="number" id="b-prompt" value="512" min="16">
        </div>
        <div class="field">
          <label>Output tokens</label>
          <input type="number" id="b-max" value="128" min="8">
        </div>
        <button class="primary" id="b-run" style="width:100%">Run benchmark</button>
        <p class="hint" style="margin:12px 0 0">
          This puts real load on the GPU. Do not run it against a live endpoint that is
          serving customers.</p>
        <div id="b-err" class="err"></div>
      </div>
      <div>
        <div class="pg-panel"><h3>Results</h3><div id="b-out">
          <span class="muted">Run a sweep to see measured latency and throughput.</span>
        </div></div>
        <div id="b-econ"></div>
      </div>
    </div>`}`;

  if (!usable.length) return;
  document.getElementById('b-run').onclick = runBenchmark;
}

async function runBenchmark() {
  const conc = document.getElementById('b-conc').value
    .split(',').map(x => parseInt(x.trim(), 10)).filter(n => n > 0);
  const body = {
    model: document.getElementById('b-model').value,
    concurrency: conc,
    requests_per_level: parseInt(document.getElementById('b-reqs').value || '8', 10),
    prompt_tokens: parseInt(document.getElementById('b-prompt').value || '512', 10),
    max_tokens: parseInt(document.getElementById('b-max').value || '128', 10),
  };

  const out = document.getElementById('b-out');
  const err = document.getElementById('b-err');
  const btn = document.getElementById('b-run');
  err.textContent = '';
  btn.disabled = true;
  out.innerHTML = '<div class="note info"><span class="spinner"></span>starting…</div>';

  const levels = [];
  try {
    const res = await fetch('/admin/api/benchmark', {
      method: 'POST', credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    });
    if (!res.ok) {
      let m = res.statusText;
      try { m = (await res.json()).error.message; } catch (_) {}
      throw new Error(m);
    }

    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const frames = buf.split('\n\n');
      buf = frames.pop();
      for (const frame of frames) {
        for (const line of frame.split('\n')) {
          if (!line.startsWith('data: ')) continue;
          const payload = line.slice(6).trim();
          if (payload === '[DONE]') continue;
          let msg;
          try { msg = JSON.parse(payload); } catch (_) { continue; }

          if (msg.type === 'progress') {
            out.innerHTML = renderBenchTable(levels) +
              `<div class="note info" style="margin-top:10px"><span class="spinner"></span>
               running ${msg.data.concurrency} concurrent…</div>`;
          } else if (msg.type === 'level') {
            levels.push(msg.data);
            out.innerHTML = renderBenchTable(levels);
          } else if (msg.type === 'result') {
            benchState.result = msg.data;
            out.innerHTML = renderBenchTable(msg.data.levels, msg.data);
            renderEconomics(msg.data);
          }
        }
      }
    }
  } catch (e) {
    err.textContent = e.message;
  } finally {
    btn.disabled = false;
  }
}

function renderBenchTable(levels, result) {
  if (!levels.length) return '<span class="muted">waiting for the first level…</span>';
  const peak = Math.max(...levels.map(l => l.aggregate_tps), 1);
  return `<table><thead><tr>
      <th class="num">Concurrent</th><th class="num">TTFT p50</th><th class="num">p95</th>
      <th class="num">Per request</th><th class="num">Aggregate</th><th></th><th class="num">Errors</th>
    </tr></thead><tbody>
    ${levels.map(l => `<tr class="${result && l.concurrency === result.best_concurrency ? 'peak' : ''}">
      <td class="num">${l.concurrency}</td>
      <td class="num">${fmtMs(l.ttft_p50)}</td>
      <td class="num">${fmtMs(l.ttft_p95)}</td>
      <td class="num">${l.per_request_tps.toFixed(1)} t/s</td>
      <td class="num"><strong>${l.aggregate_tps.toFixed(0)} t/s</strong></td>
      <td><div class="bench-bar"><span style="width:${(l.aggregate_tps / peak) * 100}%"></span></div></td>
      <td class="num">${l.errors ? `<span class="pill down">${l.errors}</span>` : '0'}</td>
    </tr>`).join('')}
    </tbody></table>
    ${result ? `<div class="note ${result.note.includes('still climbing') ? 'warn' : 'ok'}"
       style="margin-top:12px">${esc(result.note)}</div>
      <p class="hint" style="margin-top:10px">
        <strong>Per request</strong> is what a single client experiences, and is what OpenRouter
        records against you. <strong>Aggregate</strong> is what the GPU produces in total, and is
        what determines revenue. They move in opposite directions as concurrency rises.</p>` : ''}`;
}

// ---------------------------------------------------------------------------
// Economics — driven by the measured number, not an estimate
// ---------------------------------------------------------------------------

function renderEconomics(bench) {
  const el = document.getElementById('b-econ');
  el.innerHTML = `<div class="pg-panel" style="margin-top:14px">
    <h3>What would this earn?</h3>
    <p class="hint">Uses the aggregate throughput just measured (${bench.peak_aggregate_tps.toFixed(0)} tok/s)
       and this model's configured prices.</p>
    <div class="fields">
      <div class="field"><label>GPU cost $/month</label>
        <input type="number" id="e-cost" value="336" step="1"></div>
      <div class="field"><label>Utilisation %</label>
        <input type="number" id="e-util" value="25" min="1" max="100"></div>
      <div class="field"><label>Input : output ratio</label>
        <input type="number" id="e-ratio" value="5" step="0.5" min="0.1"></div>
      <div class="field"><label>Cache hit %</label>
        <input type="number" id="e-cache" value="0" min="0" max="100"></div>
    </div>
    <div class="cta" style="margin-top:14px">
      <button class="primary" id="e-run">Calculate</button>
    </div>
    <div id="e-out"></div>
  </div>`;

  const go = async () => {
    const body = {
      model: bench.model,
      gpu_cost_per_month: parseFloat(document.getElementById('e-cost').value || '0'),
      aggregate_tps: bench.peak_aggregate_tps,
      utilisation: parseFloat(document.getElementById('e-util').value || '25') / 100,
      input_ratio: parseFloat(document.getElementById('e-ratio').value || '5'),
      cache_hit_rate: parseFloat(document.getElementById('e-cache').value || '0') / 100,
    };
    const o = document.getElementById('e-out');
    try {
      const r = await api('/admin/api/economics', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
      });
      const sign = v => v >= 0 ? 'pos' : 'neg';
      o.innerHTML = `<div class="econ" style="margin-top:16px">
        ${econTile('Revenue / month', fmtUSD(r.revenue_total))}
        ${econTile('Cost / month', fmtUSD(r.cost))}
        ${econTile('Margin / month', fmtUSD(r.margin), sign(r.margin))}
        ${econTile('Margin / day', fmtUSD(r.margin_per_day), sign(r.margin_per_day))}
        ${econTile('Output tokens / mo', fmtNum(r.output_tokens_per_month))}
        ${econTile('Break-even util.', (r.break_even_utilisation * 100).toFixed(0) + '%')}
      </div>
      <div class="verdict ${r.break_even_utilisation > 1 ? 'bad' :
                            r.break_even_utilisation > 0.5 ? 'warn' : 'good'}">
        ${esc(r.verdict)}</div>
      ${(r.notes || []).map(n => `<p class="hint" style="margin-top:10px">${esc(n)}</p>`).join('')}`;
    } catch (e) {
      o.innerHTML = `<div class="note block" style="margin-top:12px">${esc(e.message)}</div>`;
    }
  };
  document.getElementById('e-run').onclick = go;
  go();
}

function econTile(k, v, cls) {
  return `<div class="e"><div class="k">${k}</div><div class="v ${cls || ''}">${v}</div></div>`;
}

const VIEWS = {
  overview: renderOverview,
  benchmark: renderBenchmark,
  playground: renderPlayground,
  models: renderModels,
  addmodel: renderAddModel,
  nodes: renderNodes,
  backends: renderBackends,
  requests: renderRequests,
  keys: renderKeys,
  modeldoc: renderModelDoc,
  guides: renderGuides,
};

// ---------------------------------------------------------------------------
// Shell
// ---------------------------------------------------------------------------

async function render() {
  try {
    await VIEWS[state.view]();
  } catch (e) {
    if (e.message !== 'unauthorized') {
      main().innerHTML = `<div class="empty">Failed to load: ${esc(e.message)}</div>`;
    }
  }
}

function showLogin() {
  stopTimer();
  document.getElementById('app').classList.add('hidden');
  document.getElementById('login').classList.remove('hidden');
}

function showApp() {
  document.getElementById('login').classList.add('hidden');
  document.getElementById('app').classList.remove('hidden');
  render();
  startTimer();
}

// Only the live-updating views are polled; the static ones would just churn.
function startTimer() {
  stopTimer();
  if (!document.getElementById('auto').checked) return;
  state.timer = setInterval(() => {
    if (['overview', 'requests', 'backends', 'nodes'].includes(state.view)) render();
  }, 10000);
}
function stopTimer() { if (state.timer) { clearInterval(state.timer); state.timer = null; } }

document.getElementById('login-form').addEventListener('submit', async ev => {
  ev.preventDefault();
  const err = document.getElementById('login-err');
  err.textContent = '';
  try {
    const res = await fetch('/admin/api/login', {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ token: document.getElementById('token').value }),
    });
    if (!res.ok) throw new Error('Invalid admin token');
    showApp();
  } catch (e) {
    err.textContent = e.message;
  }
});

document.getElementById('nav').addEventListener('click', ev => {
  const btn = ev.target.closest('button[data-view]');
  if (!btn) return;
  document.querySelectorAll('#nav button').forEach(b => b.classList.toggle('active', b === btn));
  state.view = btn.dataset.view;
  render();
});

document.getElementById('range').addEventListener('change', ev => {
  state.range = ev.target.value;
  if (state.view === 'overview') render();
});

document.getElementById('auto').addEventListener('change', startTimer);

// Probe for an existing session cookie so a reload does not force a re-login.
(async function boot() {
  try {
    await api('/admin/api/overview?range=1h');
    showApp();
  } catch (_) {
    showLogin();
  }
})();
