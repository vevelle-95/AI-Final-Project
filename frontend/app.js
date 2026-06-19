/* ============================================================
   CLIP-CA-CG — Dashboard Logic
   Multimodal Sentiment Analysis with RoBERTa + ResNet-50
   Real Backend API Integration
   ============================================================ */


/* ── Backend API — Real fetch() calls ── */

const BackendAPI = (() => {
  const BASE_URL = 'http://localhost:8000';
  const serverStartTime = Date.now();
  let isConnected = false;
  let healthData = null;

  function timestamp() {
    const d = new Date();
    return d.toTimeString().split(' ')[0];
  }

  function logToConsole(method, path, status, latencyMs, extra = '') {
    const console_el = document.getElementById('server-console');
    if (!console_el) return;

    const methodCls = method === 'GET' ? 'get' : method === 'POST' ? 'post' : method === 'WS' ? 'ws' : method === 'INFO' ? 'info' : method === 'OK' ? 'ok' : 'err';
    const statusCls = status ? `s${status}` : '';

    const line = document.createElement('div');
    line.className = 'log-line';
    line.innerHTML = `
      <span class="log-time">${timestamp()}</span>
      <span class="log-method ${methodCls}">${method}</span>
      <span class="log-path">${path}${extra ? ' ' + extra : ''}</span>
      ${status ? `<span class="log-status ${statusCls}">${status} ${latencyMs ? `(${Math.round(latencyMs)}ms)` : ''}</span>` : ''}
    `;
    console_el.appendChild(line);
    console_el.scrollTop = console_el.scrollHeight;

    // Keep max 50 lines
    while (console_el.children.length > 50) {
      console_el.removeChild(console_el.firstChild);
    }
  }

  // Real API call with logging
  async function apiCall(method, endpoint, body = null, opts = {}) {
    const startTime = performance.now();
    logToConsole(method, endpoint, null, null, opts.logExtra || '');

    try {
      const fetchOpts = { method };

      if (body instanceof FormData) {
        fetchOpts.body = body;
      } else if (body) {
        fetchOpts.headers = { 'Content-Type': 'application/json' };
        fetchOpts.body = JSON.stringify(body);
      }

      const resp = await fetch(`${BASE_URL}${endpoint}`, fetchOpts);
      const latency = performance.now() - startTime;

      if (!resp.ok) {
        const errText = await resp.text();
        logToConsole('ERR', endpoint, resp.status, latency, errText.substring(0, 80));
        throw new Error(`API ${resp.status}: ${errText}`);
      }

      // Handle CSV download
      if (opts.blob) {
        const blob = await resp.blob();
        logToConsole('OK', endpoint, resp.status, latency);
        return { blob, status: resp.status, latency };
      }

      const data = await resp.json();
      logToConsole('OK', endpoint, resp.status, latency);
      return { data, status: resp.status, latency };

    } catch (err) {
      const latency = performance.now() - startTime;
      if (!err.message.startsWith('API ')) {
        logToConsole('ERR', endpoint, 0, latency, err.message.substring(0, 60));
      }
      throw err;
    }
  }

  // Health check
  async function healthCheck() {
    try {
      const { data, latency } = await apiCall('GET', '/api/health');
      healthData = data;
      return { ...data, latency };
    } catch (err) {
      return null;
    }
  }

  // Boot sequence — try to connect to real backend
  async function boot() {
    const statusEl = document.getElementById('backend-status');
    const latencyEl = document.getElementById('status-latency');

    logToConsole('INFO', 'Connecting to FastAPI backend...', null, null);

    // Try health check
    let attempts = 0;
    let connected = false;

    while (attempts < 3 && !connected) {
      attempts++;
      try {
        const health = await healthCheck();
        if (health && health.status === 'healthy') {
          connected = true;
          isConnected = true;

          if (statusEl) {
            statusEl.className = 'backend-status connected';
            statusEl.querySelector('.status-text').textContent = 'Connected';
          }
          if (latencyEl) latencyEl.textContent = `~${Math.round(health.latency)}ms`;

          logToConsole('OK', 'Backend connected', null, null, `— ${health.device.toUpperCase()}`);

          // Log model status
          if (health.model_loaded) {
            logToConsole('OK', `Model loaded on ${health.device}`, null, null, `— ${health.text_encoder}`);
            if (!health.model_trained) {
              logToConsole('INFO', '⚠ No checkpoint loaded — model has random weights', null, null);
            } else {
              logToConsole('OK', `Checkpoint: ${health.checkpoint}`, null, null);
            }
          }
          logToConsole('OK', `GPU: ${health.gpu}`, null, null);
          logToConsole('INFO', 'FastAPI server ready', null, null, `— ${BASE_URL}`);

          // Update settings page with real info
          updateServerInfo(health);

          // Show untrained warning if needed
          if (!health.model_trained) {
            showUntrainedWarning();
          }

        }
      } catch (err) {
        if (attempts < 3) {
          logToConsole('INFO', `Retry ${attempts}/3...`, null, null);
          await new Promise(r => setTimeout(r, 1500));
        }
      }
    }

    if (!connected) {
      isConnected = false;
      if (statusEl) {
        statusEl.className = 'backend-status disconnected';
        statusEl.querySelector('.status-text').textContent = 'Disconnected';
      }
      logToConsole('ERR', 'Cannot reach backend', null, null, `— is the server running at ${BASE_URL}?`);
      logToConsole('INFO', 'Start with: python -m uvicorn server:app --reload --port 8000', null, null);
    }

    // Start heartbeat
    startHeartbeat();
    startUptimeCounter();
  }

  function startHeartbeat() {
    setInterval(async () => {
      if (!isConnected) return;
      try {
        const health = await healthCheck();
        if (health) {
          const latencyEl = document.getElementById('status-latency');
          if (latencyEl) latencyEl.textContent = `~${Math.round(health.latency)}ms`;
        }
      } catch {
        // Connection lost
        const statusEl = document.getElementById('backend-status');
        if (statusEl) {
          statusEl.className = 'backend-status disconnected';
          statusEl.querySelector('.status-text').textContent = 'Disconnected';
        }
        isConnected = false;
      }
    }, 10000);
  }

  function startUptimeCounter() {
    setInterval(() => {
      const uptimeEl = document.getElementById('server-uptime');
      if (!uptimeEl || !healthData) return;
      // Use server-reported uptime if available
      if (healthData.uptime) {
        uptimeEl.textContent = healthData.uptime;
      }
    }, 10000);
  }

  return {
    apiCall,
    healthCheck,
    boot,
    logToConsole,
    timestamp,
    get isConnected() { return isConnected; },
    get baseUrl() { return BASE_URL; },
    get health() { return healthData; }
  };
})();


/* ── Update server info on settings page ── */
function updateServerInfo(health) {
  const gpuEl = document.getElementById('gpu-info');
  if (gpuEl) gpuEl.textContent = health.gpu || 'N/A';

  const uptimeEl = document.getElementById('server-uptime');
  if (uptimeEl) uptimeEl.textContent = health.uptime || '0h 0m';
}

/* ── Show untrained model warning ── */
function showUntrainedWarning() {
  const banner = document.createElement('div');
  banner.id = 'untrained-banner';
  banner.style.cssText = `
    padding: 10px 16px;
    border-radius: 10px;
    background: rgba(245,158,11,0.1);
    border: 1px solid rgba(245,158,11,0.25);
    color: #f59e0b;
    font-size: 12px;
    font-weight: 500;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 4px;
  `;
  banner.innerHTML = `
    <i class="ti ti-alert-triangle" style="font-size:16px;flex-shrink:0;"></i>
    <span><strong>No trained checkpoint found.</strong> Model is using random weights — predictions will be unreliable. Run <code style="background:rgba(245,158,11,0.15);padding:2px 6px;border-radius:4px;">python main.py</code> to train first.</span>
  `;

  // Insert at top of each page
  const pages = document.querySelectorAll('.page');
  pages.forEach(page => {
    const clone = banner.cloneNode(true);
    page.insertBefore(clone, page.firstChild);
  });
}


/* ── Dynamic Data ── */

// Reviews loaded from the backend (replaces hardcoded array)
let reviews = [];


/* ── State ── */

let dashFilter = 'all';
let selectedDash = null;
let pipelineFilter = 'all';
let selectedPipeline = null;
let uploadedImages = [];


/* ── Helpers ── */

function sentBadge(s) {
  return `<span class="badge ${s}">${s.charAt(0).toUpperCase() + s.slice(1)}</span>`;
}

function actionBadge(a) {
  const cls = a.includes('Flag') ? 'flagged' : a.includes('Log') ? 'logged' : 'approved';
  return `<span class="action-badge ${cls}">${a}</span>`;
}

function stars(n) {
  return '★'.repeat(n) + '☆'.repeat(5 - n);
}

function formatJSON(obj) {
  const json = JSON.stringify(obj, null, 2);
  return json
    .replace(/"([^"]+)":/g, '<span class="json-key">"$1"</span>:')
    .replace(/: "([^"]*)"/g, ': <span class="json-str">"$1"</span>')
    .replace(/: (\d+\.?\d*)/g, ': <span class="json-num">$1</span>')
    .replace(/: (true|false|null)/g, ': <span class="json-bool">$1</span>');
}


/* ── Toast ── */

function showToast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 2500);
}


/* ── Page Navigation ── */

function goPage(id) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.getElementById('page-' + id).classList.add('active');
  document.querySelectorAll('.snav').forEach(n => n.classList.remove('active'));
  document.getElementById('nav-' + id).classList.add('active');

  if (id === 'pipeline') {
    BackendAPI.logToConsole('GET', '/api/results', null, null);
    renderPipeline();
  }
  if (id === 'sentiment') {
    BackendAPI.logToConsole('GET', '/api/results', null, null);
    updateSentimentPage();
  }
  if (id === 'settings') {
    BackendAPI.healthCheck();
  }
}


/* ============================================================
   Dashboard — Update metrics dynamically
   ============================================================ */

function updateDashboardMetrics() {
  const total = reviews.length;
  const pos = reviews.filter(r => r.sentiment === 'positive').length;
  const neu = reviews.filter(r => r.sentiment === 'neutral').length;
  const neg = reviews.filter(r => r.sentiment === 'negative').length;
  const avgConf = total > 0
    ? (reviews.reduce((s, r) => s + r.confidence, 0) / total * 100).toFixed(1)
    : '0.0';

  const metricsEl = document.getElementById('dash-metrics');
  if (!metricsEl) return;

  metricsEl.innerHTML = `
    <div class="metric" onclick="filterDash('positive',this)">
      <div class="metric-label">Positive</div>
      <div class="metric-val" style="color:var(--emerald)">${pos.toLocaleString()}</div>
      <div class="metric-sub">${total > 0 ? (pos / total * 100).toFixed(1) : 0}% of total</div>
    </div>
    <div class="metric" onclick="filterDash('neutral',this)">
      <div class="metric-label">Neutral</div>
      <div class="metric-val" style="color:var(--amber)">${neu.toLocaleString()}</div>
      <div class="metric-sub">${total > 0 ? (neu / total * 100).toFixed(1) : 0}% of total</div>
    </div>
    <div class="metric" onclick="filterDash('negative',this)">
      <div class="metric-label">Negative</div>
      <div class="metric-val" style="color:var(--rose)">${neg.toLocaleString()}</div>
      <div class="metric-sub">${total > 0 ? (neg / total * 100).toFixed(1) : 0}% of total</div>
    </div>
    <div class="metric">
      <div class="metric-label">Avg Confidence</div>
      <div class="metric-val" style="color:var(--blue)">${avgConf}%</div>
      <div class="metric-sub">across all predictions</div>
    </div>
  `;

  // Update page subtitle
  const pageSub = document.querySelector('#page-dashboard .page-sub');
  if (pageSub) {
    pageSub.textContent = `Multimodal Sentiment Analysis Overview — ${total.toLocaleString()} reviews processed`;
  }

  // Update product row
  const productRow = document.querySelector('#page-dashboard .product-row');
  if (productRow && total > 0) {
    const firstProduct = reviews[0].product_title || 'E-Commerce Dataset';
    productRow.innerHTML = `
      <i class="ti ti-shopping-bag"></i>
      <span>Product:</span>
      <strong>${firstProduct}</strong>
      <span>&middot;</span><span>E-Commerce Dataset</span>
      <span>&middot;</span><span>${total.toLocaleString()} reviews loaded</span>
    `;
  }

  // Update distribution bars
  updateDistributionBars(pos, neu, neg, total);
}

function updateDistributionBars(pos, neu, neg, total) {
  // Find the Sentiment Distribution card on dashboard
  const cards = document.querySelectorAll('#page-dashboard .card');
  cards.forEach(card => {
    const title = card.querySelector('.card-title');
    if (title && title.textContent === 'Sentiment Distribution') {
      const barRows = card.querySelectorAll('.bar-row');
      if (barRows.length >= 3) {
        // Positive
        barRows[0].querySelector('.bar-fill').style.width = `${total > 0 ? (pos / total * 100) : 0}%`;
        barRows[0].querySelector('.bar-count').textContent = pos.toLocaleString();
        // Neutral
        barRows[1].querySelector('.bar-fill').style.width = `${total > 0 ? (neu / total * 100) : 0}%`;
        barRows[1].querySelector('.bar-count').textContent = neu.toLocaleString();
        // Negative
        barRows[2].querySelector('.bar-fill').style.width = `${total > 0 ? (neg / total * 100) : 0}%`;
        barRows[2].querySelector('.bar-count').textContent = neg.toLocaleString();
      }
    }
    if (title && title.textContent === 'System Actions') {
      const barRows = card.querySelectorAll('.bar-row');
      if (barRows.length >= 3) {
        barRows[0].querySelector('.bar-fill').style.width = `${total > 0 ? (pos / total * 100) : 0}%`;
        barRows[0].querySelector('.bar-count').textContent = pos.toLocaleString();
        barRows[1].querySelector('.bar-fill').style.width = `${total > 0 ? (neu / total * 100) : 0}%`;
        barRows[1].querySelector('.bar-count').textContent = neu.toLocaleString();
        barRows[2].querySelector('.bar-fill').style.width = `${total > 0 ? (neg / total * 100) : 0}%`;
        barRows[2].querySelector('.bar-count').textContent = neg.toLocaleString();
      }
    }
  });
}


function renderDash() {
  const f = dashFilter === 'all' ? reviews : reviews.filter(r => r.sentiment === dashFilter);

  if (f.length === 0) {
    document.getElementById('dash-review-list').innerHTML = `
      <div style="text-align:center;padding:40px 20px;color:var(--text3);">
        <i class="ti ti-inbox" style="font-size:36px;display:block;margin-bottom:10px;"></i>
        <div style="font-weight:600;margin-bottom:4px;color:var(--text2);">No reviews yet</div>
        <div style="font-size:12px;">Upload a CSV or test a single review to get started.</div>
      </div>`;
    return;
  }

  document.getElementById('dash-review-list').innerHTML = f.map(r => `
    <div class="rev ${selectedDash === r.id ? 'selected' : ''}" onclick="selectDash(${r.id})">
      <div class="rev-meta">
        <div style="display:flex;align-items:center;gap:8px;">
          ${r.stars ? `<span style="color:#d4a017;font-size:12px;letter-spacing:-1px;">${stars(r.stars)}</span>` : ''}
          <span class="mono" style="font-size:10px;color:var(--text3);">${(r.confidence * 100).toFixed(0)}% conf</span>
        </div>
        ${sentBadge(r.sentiment)}
      </div>
      <div class="rev-text">${r.text || ''}</div>
      <div class="rev-scores">
        <span class="rev-score">P: ${(r.probabilities?.positive || 0).toFixed(2)}</span>
        <span class="rev-score">N: ${(r.probabilities?.neutral || 0).toFixed(2)}</span>
        <span class="rev-score">Ng: ${(r.probabilities?.negative || 0).toFixed(2)}</span>
      </div>
    </div>`).join('');
}

async function selectDash(id) {
  selectedDash = id;
  renderDash();
  const r = reviews.find(x => x.id === id);
  if (!r) return;

  // Show loading state
  document.getElementById('dash-detail').innerHTML = `
    <div class="card-title">Review #${r.id}</div>
    <div class="processing-overlay" style="padding:20px;">
      <div class="processing-spinner"></div>
      <div class="processing-sub">Loading details…</div>
    </div>`;

  // Try to fetch from backend
  try {
    const { data } = await BackendAPI.apiCall('GET', `/api/results/${r.id}`);
    // Update with fresh data
    Object.assign(r, data);
  } catch {
    // Use cached data
  }

  const probs = r.probabilities || {};

  document.getElementById('dash-detail').innerHTML = `
    <div class="card-title">Review #${r.id}</div>
    <div class="detail-row"><span class="detail-label">Sentiment</span>${sentBadge(r.sentiment)}</div>
    <div class="detail-row"><span class="detail-label">Confidence</span><span class="chip mono">${(r.confidence * 100).toFixed(1)}%</span></div>
    <div class="detail-row"><span class="detail-label">System Action</span>${actionBadge(r.system_action)}</div>
    ${r.stars ? `<div class="detail-row"><span class="detail-label">Stars</span><span style="color:#d4a017;">${stars(r.stars)}</span></div>` : ''}
    <div class="detail-row"><span class="detail-label">Image</span><span class="chip">${r.has_image ? 'Yes — ResNet-50 processed' : 'No'}</span></div>
    <div class="detail-row"><span class="detail-label">Model Trained</span><span class="chip">${r.model_trained ? '✓ Checkpoint' : '⚠ Random weights'}</span></div>
    <div class="detail-row"><span class="detail-label">Inference</span><span class="chip mono">${r.inference_time_ms || '—'}ms</span></div>

    <div style="margin-top:12px;">
      <div style="font-size:11px;color:var(--text3);margin-bottom:6px;font-weight:600;">CLASS PROBABILITIES</div>
      <div class="gate-bar"><div class="gate-label">Positive</div><div class="gate-track"><div class="gate-fill" style="width:${(probs.positive || 0) * 100}%;background:var(--emerald)"></div></div><div class="gate-val">${(probs.positive || 0).toFixed(3)}</div></div>
      <div class="gate-bar"><div class="gate-label">Neutral</div><div class="gate-track"><div class="gate-fill" style="width:${(probs.neutral || 0) * 100}%;background:var(--amber)"></div></div><div class="gate-val">${(probs.neutral || 0).toFixed(3)}</div></div>
      <div class="gate-bar"><div class="gate-label">Negative</div><div class="gate-track"><div class="gate-fill" style="width:${(probs.negative || 0) * 100}%;background:var(--rose)"></div></div><div class="gate-val">${(probs.negative || 0).toFixed(3)}</div></div>
    </div>

    <div class="api-panel" style="margin-top:10px;">
      <div class="api-panel-header">
        <span class="method-tag get">GET</span>
        <span>/api/results/${r.id}</span>
      </div>
      <div class="api-panel-body">${formatJSON({
        review_id: r.id,
        sentiment: r.sentiment,
        confidence: r.confidence,
        probabilities: r.probabilities,
        system_action: r.system_action,
        has_image: r.has_image,
        model_trained: r.model_trained,
        inference_time_ms: r.inference_time_ms,
      })}</div>
    </div>`;
}

function setDashFilter(f, el) {
  dashFilter = f;
  selectedDash = null;
  document.querySelectorAll('#dash-ftabs .ftab').forEach(t => t.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('dash-detail').innerHTML =
    '<div class="card-title">Review Detail</div><div style="font-size:12px;color:var(--text3);">Click a review to see its full pipeline analysis.</div>';
  renderDash();
}

function filterDash(f, el) {
  document.querySelectorAll('#dash-metrics .metric').forEach(m => m.classList.remove('selected'));
  el.classList.add('selected');
  dashFilter = f;
  selectedDash = null;
  const map = { all: 0, positive: 1, neutral: 2, negative: 3 };
  document.querySelectorAll('#dash-ftabs .ftab').forEach((t, i) =>
    t.classList.toggle('active', i === map[f] || (f === 'all' && i === 0))
  );
  document.getElementById('dash-detail').innerHTML =
    '<div class="card-title">Review Detail</div><div style="font-size:12px;color:var(--text3);">Click a review to see its full pipeline analysis.</div>';
  renderDash();
}


/* ============================================================
   Pipeline
   ============================================================ */

function renderPipeline() {
  const f = pipelineFilter === 'all' ? reviews : reviews.filter(r => r.sentiment === pipelineFilter);

  // Update tab counts
  const tabs = document.querySelectorAll('#pipeline-tabs .ftab');
  const total = reviews.length;
  const pos = reviews.filter(r => r.sentiment === 'positive').length;
  const neu = reviews.filter(r => r.sentiment === 'neutral').length;
  const neg = reviews.filter(r => r.sentiment === 'negative').length;
  if (tabs.length >= 4) {
    tabs[0].textContent = `All (${total})`;
    tabs[1].textContent = `Positive (${pos})`;
    tabs[2].textContent = `Neutral (${neu})`;
    tabs[3].textContent = `Negative (${neg})`;
  }

  if (f.length === 0) {
    document.getElementById('pipeline-list').innerHTML = `
      <div style="text-align:center;padding:40px 20px;color:var(--text3);">
        <i class="ti ti-inbox" style="font-size:36px;display:block;margin-bottom:10px;"></i>
        <div style="font-size:12px;">No reviews loaded. Upload a CSV first.</div>
      </div>`;
    return;
  }

  document.getElementById('pipeline-list').innerHTML = f.map(r => `
    <div class="rev ${selectedPipeline === r.id ? 'selected' : ''}" onclick="selectPipeline(${r.id})">
      <div class="rev-meta">
        <div style="display:flex;align-items:center;gap:8px;">
          ${r.stars ? `<span style="color:#d4a017;font-size:12px;letter-spacing:-1px;">${stars(r.stars)}</span>` : ''}
          <span class="mono" style="font-size:10px;color:var(--text3);">${(r.confidence * 100).toFixed(0)}%</span>
        </div>
        ${sentBadge(r.sentiment)}
      </div>
      <div class="rev-text">${r.text || ''}</div>
    </div>`).join('');
}

async function selectPipeline(id) {
  selectedPipeline = id;
  renderPipeline();
  const r = reviews.find(x => x.id === id);
  if (!r) return;

  const detail = document.getElementById('pipeline-detail');
  detail.innerHTML = `
    <div class="card-title">Review #${r.id} — Full Pipeline Analysis</div>
    <div class="processing-overlay" style="padding:20px;">
      <div class="processing-spinner"></div>
      <div class="processing-sub">Loading pipeline data…</div>
    </div>`;

  // Try to fetch from backend
  try {
    const { data } = await BackendAPI.apiCall('GET', `/api/results/${r.id}`);
    Object.assign(r, data);
  } catch {
    // Use cached
  }

  const probs = r.probabilities || {};

  detail.innerHTML = `
    <div class="card-title">Review #${r.id} — Full Pipeline Analysis</div>

    <div class="pipe-section">
      <div class="pipe-section-title"><i class="ti ti-file-text"></i> Stage 1: Preprocessing</div>
      <div style="font-size:12px;color:var(--text2);">Text tokenized by RoBERTa tokenizer (max_length=${CONFIG_MAX_LENGTH})</div>
      <div style="display:flex;gap:6px;margin-top:6px;">
        <span class="chip mono">text cleaned ✓</span>
        ${r.has_image ? '<span class="chip mono">image resized 224×224 ✓</span>' : '<span class="chip mono">no image — blank frame</span>'}
      </div>
    </div>

    <div class="pipe-section">
      <div class="pipe-section-title"><i class="ti ti-cpu"></i> Stage 2a: RoBERTa Text Features</div>
      <div class="gate-bar"><div class="gate-label">Positive</div><div class="gate-track"><div class="gate-fill" style="width:${(probs.positive || 0) * 100}%;background:var(--emerald)"></div></div><div class="gate-val">${(probs.positive || 0).toFixed(3)}</div></div>
      <div class="gate-bar"><div class="gate-label">Neutral</div><div class="gate-track"><div class="gate-fill" style="width:${(probs.neutral || 0) * 100}%;background:var(--amber)"></div></div><div class="gate-val">${(probs.neutral || 0).toFixed(3)}</div></div>
      <div class="gate-bar"><div class="gate-label">Negative</div><div class="gate-track"><div class="gate-fill" style="width:${(probs.negative || 0) * 100}%;background:var(--rose)"></div></div><div class="gate-val">${(probs.negative || 0).toFixed(3)}</div></div>
    </div>

    <div class="pipe-section">
      <div class="pipe-section-title"><i class="ti ti-photo-scan"></i> Stage 2b: ResNet-50 Image Features</div>
      <div style="font-size:12px;color:var(--text2);">${r.has_image ? 'Visual spatial features extracted from review image.' : 'No image provided — null embedding used (CG will downweight).'}</div>
      <div style="display:flex;gap:6px;margin-top:6px;">
        <span class="chip mono">${r.has_image ? '2048-dim embedding → 512-dim projected' : 'null embedding'}</span>
        <span class="chip mono">${r.has_image ? 'extracted ✓' : 'skipped'}</span>
      </div>
    </div>

    <div class="pipe-section">
      <div class="pipe-section-title"><i class="ti ti-git-merge"></i> Stage 3: CA Fusion & CG Alignment</div>
      <div style="font-size:12px;color:var(--text2);margin-bottom:8px;">Cross-Attention fused text and image features. Cross-modal Gating determined modality weights.</div>
      <div class="detail-row"><span class="detail-label">Image Processed</span><span class="chip">${r.has_image ? 'Yes' : 'No'}</span></div>
    </div>

    <div style="padding:14px;border-radius:var(--radius);background:var(--surface2);border:1px solid var(--border);margin-top:4px;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
        <span style="font-weight:700;font-size:13px;">Final Prediction</span>
        ${sentBadge(r.sentiment)}
      </div>
      <div class="detail-row"><span class="detail-label">Confidence</span><span class="chip mono" style="font-weight:600;">${(r.confidence * 100).toFixed(1)}%</span></div>
      <div class="detail-row"><span class="detail-label">System Action</span>${actionBadge(r.system_action)}</div>
      <div class="detail-row"><span class="detail-label">Inference Time</span><span class="chip mono">${r.inference_time_ms || '—'}ms</span></div>
      <div class="detail-row"><span class="detail-label">Model Trained</span><span class="chip">${r.model_trained ? '✓ Checkpoint loaded' : '⚠ Random weights'}</span></div>
    </div>

    <div class="api-panel" style="margin-top:10px;">
      <div class="api-panel-header">
        <span class="method-tag get">GET</span>
        <span>/api/results/${r.id}</span>
      </div>
      <div class="api-panel-body">${formatJSON({
        review_id: r.id,
        pipeline: {
          stage1_preprocessing: { status: 'completed', text_cleaned: true, image_resized: r.has_image ? '224x224' : 'n/a' },
          stage2a_roberta: { probabilities: r.probabilities, embedding_dim: 768 },
          stage2b_resnet50: { embedding_dim: r.has_image ? 2048 : 0, features_extracted: r.has_image },
          stage3_fusion: { method: 'cross_attention + cross_modal_gating' }
        },
        prediction: { sentiment: r.sentiment, confidence: r.confidence, action: r.system_action }
      })}</div>
    </div>`;
}

function setPipelineFilter(f, el) {
  pipelineFilter = f;
  selectedPipeline = null;
  document.querySelectorAll('#pipeline-tabs .ftab').forEach(t => t.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('pipeline-detail').innerHTML =
    '<div class="card-title">Pipeline Analysis</div><div style="font-size:12px;color:var(--text3);">Select a review to inspect its full pipeline scores.</div>';
  renderPipeline();
}


/* ============================================================
   Sentiment Page — Dynamic updates
   ============================================================ */

function updateSentimentPage() {
  const total = reviews.length;
  const pos = reviews.filter(r => r.sentiment === 'positive').length;
  const neu = reviews.filter(r => r.sentiment === 'neutral').length;
  const neg = reviews.filter(r => r.sentiment === 'negative').length;

  // Update metrics
  const sentMetrics = document.querySelectorAll('#page-sentiment .metrics .metric');
  if (sentMetrics.length >= 3) {
    sentMetrics[0].querySelector('.metric-val').textContent = total > 0 ? `${Math.round(pos / total * 100)}%` : '0%';
    sentMetrics[0].querySelector('.metric-sub').textContent = `${pos.toLocaleString()} reviews → Highlighted`;
    sentMetrics[1].querySelector('.metric-val').textContent = total > 0 ? `${Math.round(neu / total * 100)}%` : '0%';
    sentMetrics[1].querySelector('.metric-sub').textContent = `${neu.toLocaleString()} reviews → Logged`;
    sentMetrics[2].querySelector('.metric-val').textContent = total > 0 ? `${Math.round(neg / total * 100)}%` : '0%';
    sentMetrics[2].querySelector('.metric-sub').textContent = `${neg.toLocaleString()} reviews → Flagged`;
  }

  // Update page subtitle
  const pageSub = document.querySelector('#page-sentiment .page-sub');
  if (pageSub) pageSub.textContent = `Polarity distribution and system action summary — ${total.toLocaleString()} reviews`;

  // Update System Action Summary counts
  const actionCards = document.querySelectorAll('#page-sentiment .action-badge');
  if (actionCards.length >= 3) {
    actionCards[0].textContent = `${pos.toLocaleString()} reviews`;
    actionCards[1].textContent = `${neu.toLocaleString()} reviews`;
    actionCards[2].textContent = `${neg.toLocaleString()} reviews`;
  }
}


/* ============================================================
   Image Handling (Upload Page)
   ============================================================ */

function handleImages(input) {
  const files = Array.from(input.files);

  // Upload to backend
  if (BackendAPI.isConnected && files.length > 0) {
    const formData = new FormData();
    files.forEach(f => formData.append('files', f));
    BackendAPI.apiCall('POST', '/api/upload/images', formData)
      .then(({ data }) => {
        showToast(`${data.uploaded} image(s) uploaded to server`);
      })
      .catch(err => {
        showToast('Image upload failed — saved locally only');
      });
  }

  // Also show local preview
  files.forEach(f => {
    const reader = new FileReader();
    reader.onload = e => {
      uploadedImages.push({ name: f.name, url: e.target.result });
      renderImgGrid();
    };
    reader.readAsDataURL(f);
  });
  input.value = '';
}

function renderImgGrid() {
  const grid = document.getElementById('img-preview-grid');
  grid.innerHTML = uploadedImages.map((img, i) => `
    <div class="img-thumb-wrap">
      <img class="img-thumb" src="${img.url}" alt="${img.name}">
      <button class="img-remove" onclick="removeImg(${i})" title="Remove">&times;</button>
    </div>`).join('') +
    `<div class="img-placeholder" onclick="document.getElementById('img-input').click()">
      <i class="ti ti-plus"></i><span>Add more</span>
    </div>`;
  document.getElementById('img-count').textContent = uploadedImages.length
    ? `${uploadedImages.length} image${uploadedImages.length > 1 ? 's' : ''} uploaded — will be processed by ResNet-50`
    : '';
}

function removeImg(i) {
  uploadedImages.splice(i, 1);
  if (!uploadedImages.length) {
    document.getElementById('img-preview-grid').innerHTML = '';
    document.getElementById('img-count').textContent = '';
  } else {
    renderImgGrid();
  }
}

function handleSingleImg(input) {
  const f = input.files[0];
  if (!f) return;
  const reader = new FileReader();
  reader.onload = e => {
    document.getElementById('single-img-preview').innerHTML = `
      <div class="img-thumb-wrap" style="width:80px;height:80px;">
        <img class="img-thumb" src="${e.target.result}" alt="review" style="width:80px;height:80px;">
        <button class="img-remove" onclick="clearSingleImg()">&times;</button>
      </div>`;
  };
  reader.readAsDataURL(f);
}

function clearSingleImg() {
  document.getElementById('single-img-preview').innerHTML = `
    <div class="img-placeholder" onclick="document.getElementById('single-img-input').click()">
      <i class="ti ti-photo-plus"></i><span>Add</span>
    </div>`;
  document.getElementById('single-img-input').value = '';
}


/* ============================================================
   Single Review Classifier — with animated pipeline + real API
   ============================================================ */

async function classifyTest() {
  const text = document.getElementById('test-text').value.trim();
  if (!text) { showToast('Enter review text first'); return; }

  if (!BackendAPI.isConnected) {
    showToast('Backend not connected! Start the server first.');
    return;
  }

  const el = document.getElementById('test-result');
  const stars_n = parseInt(document.getElementById('test-stars').value);
  const imgEl = document.querySelector('#single-img-preview .img-thumb');
  const hasImg = !!imgEl;
  const imgBase64 = hasImg ? imgEl.src : null;

  // Show animated pipeline processing steps
  el.innerHTML = `
    <div style="border:1px solid var(--border);border-radius:var(--radius);padding:14px;background:var(--surface2);">
      <div style="font-weight:700;font-size:13px;margin-bottom:12px;">Pipeline Processing</div>
      <div class="pipeline-steps" id="classify-steps">
        <div class="pipeline-step active" id="step-send">
          <div class="step-icon"><i class="ti ti-send" style="font-size:12px;"></i></div>
          <div class="step-content">
            <div class="step-title">Sending to /api/predict</div>
            <div class="step-detail">POST request with review payload…</div>
          </div>
          <div class="step-timing">—</div>
        </div>
        <div class="pipeline-step pending" id="step-preprocess">
          <div class="step-icon"><i class="ti ti-file-text" style="font-size:12px;"></i></div>
          <div class="step-content">
            <div class="step-title">Stage 1: Preprocessing</div>
            <div class="step-detail">Waiting…</div>
          </div>
          <div class="step-timing">—</div>
        </div>
        <div class="pipeline-step pending" id="step-roberta">
          <div class="step-icon"><i class="ti ti-cpu" style="font-size:12px;"></i></div>
          <div class="step-content">
            <div class="step-title">Stage 2a: RoBERTa Encoding</div>
            <div class="step-detail">Waiting…</div>
          </div>
          <div class="step-timing">—</div>
        </div>
        <div class="pipeline-step pending" id="step-resnet">
          <div class="step-icon"><i class="ti ti-photo-scan" style="font-size:12px;"></i></div>
          <div class="step-content">
            <div class="step-title">Stage 2b: ResNet-50 Encoding</div>
            <div class="step-detail">Waiting…</div>
          </div>
          <div class="step-timing">—</div>
        </div>
        <div class="pipeline-step pending" id="step-fusion">
          <div class="step-icon"><i class="ti ti-git-merge" style="font-size:12px;"></i></div>
          <div class="step-content">
            <div class="step-title">Stage 3: CA Fusion & CG Alignment</div>
            <div class="step-detail">Waiting…</div>
          </div>
          <div class="step-timing">—</div>
        </div>
        <div class="pipeline-step pending" id="step-predict">
          <div class="step-icon"><i class="ti ti-sparkles" style="font-size:12px;"></i></div>
          <div class="step-content">
            <div class="step-title">Final Prediction</div>
            <div class="step-detail">Waiting…</div>
          </div>
          <div class="step-timing">—</div>
        </div>
      </div>
    </div>`;

  // Helper to advance a step
  function advanceStep(stepId, status, detail, timing) {
    const step = document.getElementById(stepId);
    if (!step) return;
    step.className = `pipeline-step ${status}`;
    step.querySelector('.step-detail').textContent = detail;
    step.querySelector('.step-timing').textContent = timing;
  }

  // Start the real API call in the background
  const formData = new FormData();
  formData.append('text', text);
  formData.append('stars', stars_n);
  if (imgBase64) {
    formData.append('image_base64', imgBase64);
  }

  const apiStartTime = performance.now();

  // Fire off the API call (non-blocking)
  const apiPromise = BackendAPI.apiCall('POST', '/api/predict', formData, {
    logExtra: `{text: "${text.substring(0, 40)}..."}`
  });

  // Animate pipeline steps while API processes
  advanceStep('step-send', 'active', 'Sending POST /api/predict…', '—');
  await new Promise(r => setTimeout(r, 300));
  advanceStep('step-send', 'completed', 'Request sent to backend', `${(performance.now() - apiStartTime).toFixed(0)}ms`);

  advanceStep('step-preprocess', 'active', 'Server tokenizing text, normalizing…', '—');
  await new Promise(r => setTimeout(r, 400));
  advanceStep('step-preprocess', 'completed', 'Text tokenized by RoBERTa — max_len=128', '—');

  advanceStep('step-roberta', 'active', 'Forward pass through RoBERTa-base (125M params)…', '—');
  await new Promise(r => setTimeout(r, 500));

  advanceStep('step-resnet', 'active', hasImg ? 'Processing image through ResNet-50…' : 'No image — using null embedding…', '—');
  await new Promise(r => setTimeout(r, 300));

  advanceStep('step-fusion', 'active', 'Computing Cross-Attention & Cross-modal Gating…', '—');

  // Wait for the real API response
  let result;
  try {
    const { data } = await apiPromise;
    result = data;
  } catch (err) {
    // API error
    advanceStep('step-send', 'completed', 'Request failed!', '—');
    advanceStep('step-predict', 'active', `Error: ${err.message}`, '—');
    el.innerHTML += `
      <div style="border:1px solid rgba(244,63,94,0.3);border-radius:var(--radius);padding:14px;background:rgba(244,63,94,0.05);margin-top:12px;">
        <div style="color:var(--rose);font-weight:600;">Prediction Failed</div>
        <div style="font-size:12px;color:var(--text2);margin-top:4px;">${err.message}</div>
      </div>`;
    return;
  }

  const totalTime = performance.now() - apiStartTime;

  // Complete all remaining steps with real data
  const probs = result.probabilities || {};
  advanceStep('step-roberta', 'completed',
    `Logits: [${(probs.positive || 0).toFixed(3)}, ${(probs.neutral || 0).toFixed(3)}, ${(probs.negative || 0).toFixed(3)}] — 768-dim embedding`,
    `—`);
  advanceStep('step-resnet', 'completed',
    hasImg ? 'Visual features extracted — 2048-dim → 512-dim projected' : 'Null embedding — CG will downweight image modality',
    '—');
  advanceStep('step-fusion', 'completed',
    'CA + CG fusion complete',
    '—');
  advanceStep('step-predict', 'completed',
    `${result.sentiment.toUpperCase()} — ${(result.confidence * 100).toFixed(1)}% confidence → ${result.system_action}`,
    `${result.inference_time_ms}ms`);

  // Store this result in our local reviews array
  const newReview = {
    id: result.review_id,
    text: text,
    stars: stars_n,
    sentiment: result.sentiment,
    confidence: result.confidence,
    probabilities: result.probabilities,
    system_action: result.system_action,
    has_image: result.has_image,
    inference_time_ms: result.inference_time_ms,
    model_trained: result.model_trained,
  };
  reviews.push(newReview);
  updateDashboardMetrics();
  renderDash();

  // Build result HTML
  const resultHTML = `
    <div style="border:1px solid var(--border);border-radius:var(--radius);padding:14px;background:var(--surface2);margin-top:12px;">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;">
        <span style="font-weight:700;font-size:14px;">Result</span>
        ${sentBadge(result.sentiment)}
      </div>

      <div class="pipe-section">
        <div class="pipe-section-title"><i class="ti ti-typography"></i> Model Output Probabilities</div>
        <div class="gate-bar"><div class="gate-label">Positive</div><div class="gate-track"><div class="gate-fill" style="width:${(probs.positive || 0) * 100}%;background:var(--emerald)"></div></div><div class="gate-val">${(probs.positive || 0).toFixed(3)}</div></div>
        <div class="gate-bar"><div class="gate-label">Neutral</div><div class="gate-track"><div class="gate-fill" style="width:${(probs.neutral || 0) * 100}%;background:var(--amber)"></div></div><div class="gate-val">${(probs.neutral || 0).toFixed(3)}</div></div>
        <div class="gate-bar"><div class="gate-label">Negative</div><div class="gate-track"><div class="gate-fill" style="width:${(probs.negative || 0) * 100}%;background:var(--rose)"></div></div><div class="gate-val">${(probs.negative || 0).toFixed(3)}</div></div>
      </div>

      <div class="detail-row"><span class="detail-label">Confidence</span><span class="chip mono">${(result.confidence * 100).toFixed(1)}%</span></div>
      <div class="detail-row"><span class="detail-label">Image uploaded</span><span class="chip">${result.has_image ? 'Yes — ResNet-50 processed' : 'No'}</span></div>
      <div class="detail-row"><span class="detail-label">Stars</span><span style="color:#d4a017;">${'★'.repeat(stars_n) + '☆'.repeat(5 - stars_n)}</span></div>
      <div class="detail-row"><span class="detail-label">System Action</span>${actionBadge(result.system_action)}</div>
      <div class="detail-row"><span class="detail-label">Inference Time</span><span class="chip mono">${result.inference_time_ms}ms</span></div>
      <div class="detail-row"><span class="detail-label">Model Trained</span><span class="chip">${result.model_trained ? '✓ Checkpoint' : '⚠ Random weights'}</span></div>
      <div class="detail-row"><span class="detail-label">Total Pipeline</span><span class="chip mono">${totalTime.toFixed(0)}ms</span></div>

      <div class="api-panel" style="margin-top:10px;">
        <div class="api-panel-header">
          <span class="method-tag post">POST</span>
          <span>/api/predict</span>
        </div>
        <div class="api-panel-body">${formatJSON({
          request: { text: text, stars: stars_n, has_image: hasImg },
          response: result
        })}</div>
      </div>
    </div>`;

  el.innerHTML += resultHTML;
}


/* ============================================================
   CSV Handling — Real upload to backend
   ============================================================ */

async function handleCSV(input) {
  const f = input.files[0];
  if (!f) return;

  if (!BackendAPI.isConnected) {
    showToast('Backend not connected! Start the server first.');
    return;
  }

  // Show uploading state
  document.getElementById('csv-status').innerHTML = `
    <div style="display:flex;align-items:center;gap:8px;padding:10px;border-radius:var(--radius);background:var(--blue-light);border:1px solid rgba(59,130,246,0.2);">
      <i class="ti ti-loader-2" style="color:var(--blue);font-size:16px;animation:spin 1s linear infinite;display:inline-block;"></i>
      <span style="color:var(--blue);font-weight:600;">Uploading ${f.name}…</span>
      <span style="color:var(--text2);font-family:'JetBrains Mono',monospace;font-size:11px;">POST /api/upload/csv</span>
    </div>`;

  const formData = new FormData();
  formData.append('file', f);

  try {
    // Show processing state
    document.getElementById('csv-status').innerHTML = `
      <div style="display:flex;align-items:center;gap:8px;padding:10px;border-radius:var(--radius);background:var(--amber-light);border:1px solid rgba(245,158,11,0.2);">
        <i class="ti ti-loader-2" style="color:var(--amber);font-size:16px;animation:spin 1s linear infinite;display:inline-block;"></i>
        <span style="color:var(--amber);font-weight:600;">Server processing CSV — running CLIP-CA-CG inference…</span>
        <span style="color:var(--text2);font-family:'JetBrains Mono',monospace;font-size:11px;">This may take a moment</span>
      </div>`;

    const { data } = await BackendAPI.apiCall('POST', '/api/upload/csv', formData, {
      logExtra: `(${f.name}, ${(f.size / 1024).toFixed(1)}KB)`
    });

    // Store results locally
    if (data.results && Array.isArray(data.results)) {
      reviews = data.results;
    }

    // Show success
    const summary = data.summary || {};
    document.getElementById('csv-status').innerHTML = `
      <div style="display:flex;align-items:center;gap:8px;padding:10px;border-radius:var(--radius);background:var(--emerald-light);border:1px solid rgba(16,185,129,0.2);">
        <i class="ti ti-check" style="color:var(--emerald);font-size:16px;"></i>
        <span style="color:var(--emerald);font-weight:600;">${f.name}</span>
        <span style="color:var(--text2);">(${(f.size / 1024).toFixed(1)} KB) — ${data.total_reviews} reviews processed in ${data.processing_time_ms}ms</span>
      </div>
      ${!data.model_trained ? `
      <div style="display:flex;align-items:center;gap:8px;padding:8px 10px;border-radius:var(--radius);background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.2);margin-top:6px;">
        <i class="ti ti-alert-triangle" style="color:var(--amber);font-size:14px;"></i>
        <span style="color:var(--amber);font-size:11px;font-weight:500;">Model is untrained — predictions use random weights</span>
      </div>` : ''}
      <div style="margin-top:8px;display:flex;gap:8px;">
        <span class="badge positive">Positive: ${summary.positive || 0} (${summary.positive_pct || 0}%)</span>
        <span class="badge neutral">Neutral: ${summary.neutral || 0} (${summary.neutral_pct || 0}%)</span>
        <span class="badge negative">Negative: ${summary.negative || 0} (${summary.negative_pct || 0}%)</span>
      </div>
      <div class="api-panel" style="margin-top:8px;">
        <div class="api-panel-header">
          <span class="method-tag post">POST</span>
          <span>/api/upload/csv — 200 OK</span>
        </div>
        <div class="api-panel-body">${formatJSON({
          status: data.status,
          file: f.name,
          total_reviews: data.total_reviews,
          processing_time_ms: data.processing_time_ms,
          summary: data.summary,
          model_trained: data.model_trained,
          columns_found: data.columns_found,
        })}</div>
      </div>`;

    // Update all dashboard views
    updateDashboardMetrics();
    renderDash();
    showToast(`${data.total_reviews} reviews processed — view results on Dashboard`);

  } catch (err) {
    document.getElementById('csv-status').innerHTML = `
      <div style="display:flex;align-items:center;gap:8px;padding:10px;border-radius:var(--radius);background:var(--rose-light);border:1px solid rgba(244,63,94,0.2);">
        <i class="ti ti-x" style="color:var(--rose);font-size:16px;"></i>
        <span style="color:var(--rose);font-weight:600;">Upload failed</span>
        <span style="color:var(--text2);font-size:12px;">${err.message}</span>
      </div>`;
  }
}

function dragOver(e) { e.preventDefault(); e.currentTarget.classList.add('dragover'); }
function dragLeave(e) { e.currentTarget.classList.remove('dragover'); }

function dropCSV(e) {
  e.preventDefault();
  e.currentTarget.classList.remove('dragover');
  // Trigger the file input with the dropped files
  const dt = e.dataTransfer;
  const input = document.getElementById('csv-input');
  input.files = dt.files;
  handleCSV(input);
}

function dropImages(e) {
  e.preventDefault();
  e.currentTarget.classList.remove('dragover');
  const dt = e.dataTransfer;
  const input = document.getElementById('img-input');
  input.files = dt.files;
  handleImages(input);
}


/* ============================================================
   Export — Real CSV download from backend
   ============================================================ */

async function exportResults() {
  if (!BackendAPI.isConnected) {
    showToast('Backend not connected!');
    return;
  }

  try {
    const { blob } = await BackendAPI.apiCall('GET', '/api/export/csv', null, { blob: true });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'clip_cacg_predictions.csv';
    a.click();
    URL.revokeObjectURL(url);
    showToast('Exported to CSV!');
  } catch (err) {
    showToast(`Export failed: ${err.message}`);
  }
}


/* ============================================================
   Load Sample Dataset — Real API call
   ============================================================ */

async function fakeLoad() {
  if (!BackendAPI.isConnected) {
    showToast('Backend not connected!');
    return;
  }

  showToast('Loading sample dataset…');

  // Use the testing.csv from data/ directory
  try {
    // Fetch the testing.csv file and upload it
    const response = await fetch('../data/testing.csv');
    if (!response.ok) throw new Error('Cannot load sample dataset file');
    const blob = await response.blob();
    const file = new File([blob], 'testing.csv', { type: 'text/csv' });

    const formData = new FormData();
    formData.append('file', file);

    const { data } = await BackendAPI.apiCall('POST', '/api/upload/csv', formData, {
      logExtra: '(testing.csv — sample dataset)'
    });

    if (data.results && Array.isArray(data.results)) {
      reviews = data.results;
    }

    updateDashboardMetrics();
    renderDash();
    showToast(`Sample dataset loaded — ${data.total_reviews} reviews`);
    goPage('dashboard');

  } catch (err) {
    showToast(`Failed to load sample: ${err.message}`);
    BackendAPI.logToConsole('ERR', 'Failed to load sample dataset', null, null, err.message);
  }
}


/* ── Config constant ── */
const CONFIG_MAX_LENGTH = 128;


/* ============================================================
   Init — Boot Sequence
   ============================================================ */

renderDash();

if (window.location.hash === '#upload') {
  goPage('upload');
}

// Boot the real backend connection
BackendAPI.boot();

// Try to load existing results from backend
setTimeout(async () => {
  if (BackendAPI.isConnected) {
    try {
      const { data } = await BackendAPI.apiCall('GET', '/api/results');
      if (data.results && data.results.length > 0) {
        reviews = data.results;
        updateDashboardMetrics();
        renderDash();
        BackendAPI.logToConsole('INFO', `Loaded ${data.total} existing results from server`, null, null);
      }
    } catch {
      // No existing results, that's fine
    }
  }
}, 5000);