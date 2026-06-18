/* ============================================================
   CLIP-CA-CG — Dashboard Logic
   Multimodal Sentiment Analysis with RoBERTa + ResNet-50
   Simulated Backend API Layer
   ============================================================ */


/* ── Simulated Backend API ── */

const BackendAPI = (() => {
  const BASE_URL = 'http://localhost:8000';
  const serverStartTime = Date.now();
  let requestCounter = 0;
  let isConnected = false;

  // Generate realistic latency
  function latency(min = 80, max = 300) {
    return min + Math.random() * (max - min);
  }

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

  // Simulate an API call with realistic delay
  async function simulateRequest(method, endpoint, body = null, opts = {}) {
    const reqId = ++requestCounter;
    const lat = latency(opts.minLat || 80, opts.maxLat || 350);

    // Log request
    logToConsole(method, endpoint, null, null, body ? `(${JSON.stringify(body).substring(0, 60)}...)` : '');

    await new Promise(r => setTimeout(r, lat));

    // Log response
    const status = opts.status || 200;
    logToConsole(method === 'POST' ? 'OK' : 'OK', `${endpoint}`, status, lat);

    return { status, latency: lat, requestId: reqId };
  }

  // Health check / heartbeat
  async function healthCheck() {
    const lat = latency(15, 60);
    await new Promise(r => setTimeout(r, lat));
    return { status: 'healthy', latency: lat, gpu: 'NVIDIA RTX 3060', cuda: true, models_loaded: true, uptime: Date.now() - serverStartTime };
  }

  // Simulate boot sequence
  async function boot() {
    const statusEl = document.getElementById('backend-status');
    const latencyEl = document.getElementById('status-latency');

    logToConsole('INFO', 'Attempting connection to FastAPI backend...', null, null);
    await new Promise(r => setTimeout(r, 600));

    logToConsole('GET', '/api/health', null, null);
    await new Promise(r => setTimeout(r, 400));
    logToConsole('OK', '/api/health', 200, 42);

    await new Promise(r => setTimeout(r, 300));
    logToConsole('INFO', 'Loading RoBERTa tokenizer (roberta-base)...', null, null);
    await new Promise(r => setTimeout(r, 800));
    logToConsole('OK', 'RoBERTa tokenizer loaded', null, null, '— 50265 tokens');

    await new Promise(r => setTimeout(r, 200));
    logToConsole('INFO', 'Loading model checkpoint final_model/...', null, null);
    await new Promise(r => setTimeout(r, 1200));
    logToConsole('OK', 'Model loaded to GPU (CUDA:0)', null, null, '— 125M params');

    await new Promise(r => setTimeout(r, 200));
    logToConsole('INFO', 'Loading ResNet-50 image encoder...', null, null);
    await new Promise(r => setTimeout(r, 600));
    logToConsole('OK', 'ResNet-50 loaded', null, null, '— 25.6M params');

    await new Promise(r => setTimeout(r, 200));
    logToConsole('INFO', 'Initializing Cross-Attention & CG modules...', null, null);
    await new Promise(r => setTimeout(r, 400));
    logToConsole('OK', 'CA + CG fusion pipeline ready', null, null);

    await new Promise(r => setTimeout(r, 200));
    logToConsole('INFO', 'FastAPI server ready', null, null, '— http://localhost:8000');

    isConnected = true;

    if (statusEl) {
      statusEl.className = 'backend-status connected';
      statusEl.querySelector('.status-text').textContent = 'Connected';
    }
    if (latencyEl) latencyEl.textContent = '~32ms';

    // Load demo dataset
    await new Promise(r => setTimeout(r, 500));
    logToConsole('GET', '/api/dataset/demo', 200, 128);
    logToConsole('INFO', 'Loaded 2,104 reviews from demo dataset', null, null);

    // Start heartbeat
    startHeartbeat();
    startUptimeCounter();
  }

  function startHeartbeat() {
    setInterval(async () => {
      if (!isConnected) return;
      const lat = latency(12, 45);
      const latencyEl = document.getElementById('status-latency');
      if (latencyEl) latencyEl.textContent = `~${Math.round(lat)}ms`;
    }, 5000);
  }

  function startUptimeCounter() {
    setInterval(() => {
      const uptimeEl = document.getElementById('server-uptime');
      if (!uptimeEl) return;
      const elapsed = Date.now() - serverStartTime;
      const mins = Math.floor(elapsed / 60000);
      const hours = Math.floor(mins / 60);
      const m = mins % 60;
      uptimeEl.textContent = `${hours}h ${m}m`;
    }, 10000);
  }

  return {
    simulateRequest,
    healthCheck,
    boot,
    logToConsole,
    timestamp,
    get isConnected() { return isConnected; },
    get baseUrl() { return BASE_URL; }
  };
})();


/* ── Demo Data ── */

const reviews = [
  {
    id: 1,
    text: 'Ergonomic sa kamay, smooth ang scroll wheel at maganda ang clicks. Worth it talaga para sa presyo.',
    stars: 5,
    sentiment: 'positive',
    confidence: 0.94,
    cg_alignment: 0.91,
    cg_gates: { text: 0.72, image: 0.85 },
    ca_weights: { t2i: 0.78, i2t: 0.84 },
    roberta_conf: [0.88, 0.09, 0.03],
    system_action: 'Highlighted / Approved',
    note: 'High confidence positive with strong text-image CG alignment (0.91)'
  },
  {
    id: 2,
    text: 'Panget ng mouse, hindi gumagana yung bluetooth agad nung pagbukas ko! Wag kayo umorder dito sayang pera.',
    stars: 1,
    sentiment: 'negative',
    confidence: 0.92,
    cg_alignment: 0.87,
    cg_gates: { text: 0.88, image: 0.72 },
    ca_weights: { t2i: 0.82, i2t: 0.68 },
    roberta_conf: [0.04, 0.06, 0.90],
    system_action: 'Flagged for Support',
    note: 'High confidence negative — bluetooth defect complaint with visual evidence reinforcing text'
  },
  {
    id: 3,
    text: 'Okay lang naman, gumagana siya. Normal mouse, walang special.',
    stars: 3,
    sentiment: 'neutral',
    confidence: 0.81,
    cg_alignment: 0.65,
    cg_gates: { text: 0.78, image: 0.42 },
    ca_weights: { t2i: 0.55, i2t: 0.48 },
    roberta_conf: [0.15, 0.72, 0.13],
    system_action: 'Logged',
    note: 'Neutral factual description — CG gives more weight to text modality (low image signal)'
  },
  {
    id: 4,
    text: 'Solid ang build quality, medyo mahal pero sulit. Battery life okay lang, sana type-C na.',
    stars: 4,
    sentiment: 'positive',
    confidence: 0.79,
    cg_alignment: 0.86,
    cg_gates: { text: 0.68, image: 0.81 },
    ca_weights: { t2i: 0.74, i2t: 0.79 },
    roberta_conf: [0.71, 0.19, 0.10],
    system_action: 'Highlighted / Approved',
    note: 'Mixed but overall positive — CA fusion detects praise for build quality outweighing minor complaints'
  },
  {
    id: 5,
    text: 'Sira ang box pagdating, may gasgas na yung mouse mismo. Nagrereklamo na ako sa seller.',
    stars: 1,
    sentiment: 'negative',
    confidence: 0.96,
    cg_alignment: 0.93,
    cg_gates: { text: 0.85, image: 0.92 },
    ca_weights: { t2i: 0.91, i2t: 0.88 },
    roberta_conf: [0.02, 0.04, 0.94],
    system_action: 'Flagged for Support',
    note: 'Very high confidence negative — text describes damage AND image shows scratches (strong CG alignment 0.93)'
  },
  {
    id: 6,
    text: 'Best mouse ever purchased! Maganda sa kamay, magaan, mabilis mag-connect. Highly recommend!',
    stars: 5,
    sentiment: 'positive',
    confidence: 0.91,
    cg_alignment: 0.88,
    cg_gates: { text: 0.80, image: 0.76 },
    ca_weights: { t2i: 0.72, i2t: 0.70 },
    roberta_conf: [0.92, 0.05, 0.03],
    system_action: 'Highlighted / Approved',
    note: 'Strong positive text signal — RoBERTa detects enthusiastic praise, image shows product in good condition'
  },
  {
    id: 7,
    text: 'Di ko pa nagagamit pero maganda packaging. Sana worth it.',
    stars: 4,
    sentiment: 'neutral',
    confidence: 0.68,
    cg_alignment: 0.55,
    cg_gates: { text: 0.82, image: 0.35 },
    ca_weights: { t2i: 0.40, i2t: 0.38 },
    roberta_conf: [0.28, 0.58, 0.14],
    system_action: 'Logged',
    note: 'Low confidence neutral — review only discusses packaging, CG low image gate (no product use visible)'
  },
  {
    id: 8,
    text: 'Maingay ang click, hindi silent tulad ng sinabi sa listing. Misleading product description.',
    stars: 2,
    sentiment: 'negative',
    confidence: 0.88,
    cg_alignment: 0.78,
    cg_gates: { text: 0.90, image: 0.60 },
    ca_weights: { t2i: 0.65, i2t: 0.58 },
    roberta_conf: [0.05, 0.10, 0.85],
    system_action: 'Flagged for Support',
    note: 'Negative — complaint about misleading listing, CG text gate dominates (0.90 vs 0.60 image)'
  },
  {
    id: 9,
    text: 'Gumagana naman pero parang walang difference sa old mouse ko. Okay na rin.',
    stars: 3,
    sentiment: 'neutral',
    confidence: 0.74,
    cg_alignment: 0.62,
    cg_gates: { text: 0.75, image: 0.45 },
    ca_weights: { t2i: 0.52, i2t: 0.50 },
    roberta_conf: [0.18, 0.65, 0.17],
    system_action: 'Logged',
    note: 'Neutral — no strong sentiment either way, CG alignment moderate'
  },
  {
    id: 10,
    text: 'Mabilis mag-pair sa laptop at phone. Love yung side buttons para sa productivity!',
    stars: 5,
    sentiment: 'positive',
    confidence: 0.93,
    cg_alignment: 0.84,
    cg_gates: { text: 0.77, image: 0.79 },
    ca_weights: { t2i: 0.80, i2t: 0.76 },
    roberta_conf: [0.90, 0.07, 0.03],
    system_action: 'Highlighted / Approved',
    note: 'High confidence positive — specific functional praise detected by RoBERTa'
  },
];


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
    BackendAPI.simulateRequest('GET', '/api/results/pipeline', null, { minLat: 60, maxLat: 180 });
    renderPipeline();
  }
  if (id === 'sentiment') {
    BackendAPI.simulateRequest('GET', '/api/results/sentiment-summary', null, { minLat: 40, maxLat: 120 });
  }
  if (id === 'settings') {
    BackendAPI.simulateRequest('GET', '/api/health', null, { minLat: 15, maxLat: 50 });
  }
}


/* ============================================================
   Dashboard
   ============================================================ */

function renderDash() {
  const f = dashFilter === 'all' ? reviews : reviews.filter(r => r.sentiment === dashFilter);
  document.getElementById('dash-review-list').innerHTML = f.map(r => `
    <div class="rev ${selectedDash === r.id ? 'selected' : ''}" onclick="selectDash(${r.id})">
      <div class="rev-meta">
        <div style="display:flex;align-items:center;gap:8px;">
          <span style="color:#d4a017;font-size:12px;letter-spacing:-1px;">${stars(r.stars)}</span>
          <span class="mono" style="font-size:10px;color:var(--text3);">${(r.confidence * 100).toFixed(0)}% conf</span>
        </div>
        ${sentBadge(r.sentiment)}
      </div>
      <div class="rev-text">${r.text}</div>
      <div class="rev-scores">
        <span class="rev-score">CG: ${r.cg_alignment.toFixed(2)}</span>
        <span class="rev-score">T→I: ${r.ca_weights.t2i.toFixed(2)}</span>
        <span class="rev-score">I→T: ${r.ca_weights.i2t.toFixed(2)}</span>
      </div>
    </div>`).join('');
}

async function selectDash(id) {
  selectedDash = id;
  renderDash();
  const r = reviews.find(x => x.id === id);

  // Show loading state
  document.getElementById('dash-detail').innerHTML = `
    <div class="card-title">Review #${r.id}</div>
    <div class="processing-overlay" style="padding:20px;">
      <div class="processing-spinner"></div>
      <div class="processing-sub">GET /api/results/${r.id}</div>
    </div>`;

  // Simulate API call
  await BackendAPI.simulateRequest('GET', `/api/results/${r.id}`, null, { minLat: 120, maxLat: 350 });

  document.getElementById('dash-detail').innerHTML = `
    <div class="card-title">Review #${r.id}</div>
    <div class="detail-row"><span class="detail-label">Sentiment</span>${sentBadge(r.sentiment)}</div>
    <div class="detail-row"><span class="detail-label">Confidence</span><span class="chip mono">${(r.confidence * 100).toFixed(1)}%</span></div>
    <div class="detail-row"><span class="detail-label">CG Alignment</span><span class="chip mono">${r.cg_alignment.toFixed(2)}</span></div>
    <div class="detail-row"><span class="detail-label">System Action</span>${actionBadge(r.system_action)}</div>
    <div class="detail-row"><span class="detail-label">Stars</span><span style="color:#d4a017;">${stars(r.stars)}</span></div>
    <div style="margin-top:10px;font-size:11px;padding:8px 10px;border-radius:var(--radius);background:var(--surface2);color:var(--text2);border:1px solid var(--border);">${r.note}</div>
    <div class="api-panel" style="margin-top:10px;">
      <div class="api-panel-header">
        <span class="method-tag get">GET</span>
        <span>/api/results/${r.id}</span>
      </div>
      <div class="api-panel-body">${formatJSON({
        review_id: r.id,
        sentiment: r.sentiment,
        confidence: r.confidence,
        cg_alignment: r.cg_alignment,
        cg_gates: r.cg_gates,
        ca_weights: r.ca_weights,
        roberta_logits: r.roberta_conf,
        system_action: r.system_action
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
  document.getElementById('pipeline-list').innerHTML = f.map(r => `
    <div class="rev ${selectedPipeline === r.id ? 'selected' : ''}" onclick="selectPipeline(${r.id})">
      <div class="rev-meta">
        <div style="display:flex;align-items:center;gap:8px;">
          <span style="color:#d4a017;font-size:12px;letter-spacing:-1px;">${stars(r.stars)}</span>
          <span class="mono" style="font-size:10px;color:var(--text3);">CG: ${r.cg_alignment.toFixed(2)}</span>
          <span class="mono" style="font-size:10px;color:var(--text3);">${(r.confidence * 100).toFixed(0)}%</span>
        </div>
        ${sentBadge(r.sentiment)}
      </div>
      <div class="rev-text">${r.text}</div>
    </div>`).join('');
}

async function selectPipeline(id) {
  selectedPipeline = id;
  renderPipeline();
  const r = reviews.find(x => x.id === id);

  // Show loading
  const detail = document.getElementById('pipeline-detail');
  detail.innerHTML = `
    <div class="card-title">Review #${r.id} — Full Pipeline Analysis</div>
    <div class="processing-overlay" style="padding:20px;">
      <div class="processing-spinner"></div>
      <div class="processing-sub">GET /api/results/${r.id}/pipeline</div>
    </div>`;

  await BackendAPI.simulateRequest('GET', `/api/results/${r.id}/pipeline`, null, { minLat: 150, maxLat: 400 });

  detail.innerHTML = `
    <div class="card-title">Review #${r.id} — Full Pipeline Analysis</div>

    <div class="pipe-section">
      <div class="pipe-section-title"><i class="ti ti-typography"></i> Stage 2a: RoBERTa Text Features</div>
      <div class="gate-bar"><div class="gate-label">Positive</div><div class="gate-track"><div class="gate-fill" style="width:${r.roberta_conf[0] * 100}%;background:var(--emerald)"></div></div><div class="gate-val">${r.roberta_conf[0].toFixed(2)}</div></div>
      <div class="gate-bar"><div class="gate-label">Neutral</div><div class="gate-track"><div class="gate-fill" style="width:${r.roberta_conf[1] * 100}%;background:var(--amber)"></div></div><div class="gate-val">${r.roberta_conf[1].toFixed(2)}</div></div>
      <div class="gate-bar"><div class="gate-label">Negative</div><div class="gate-track"><div class="gate-fill" style="width:${r.roberta_conf[2] * 100}%;background:var(--rose)"></div></div><div class="gate-val">${r.roberta_conf[2].toFixed(2)}</div></div>
    </div>

    <div class="pipe-section">
      <div class="pipe-section-title"><i class="ti ti-photo-scan"></i> Stage 2b: ResNet-50 Image Features</div>
      <div style="font-size:12px;color:var(--text2);">Visual spatial features extracted from review image.</div>
      <div style="display:flex;gap:6px;margin-top:6px;">
        <span class="chip mono">512-dim embedding</span>
        <span class="chip mono">extracted ✓</span>
      </div>
    </div>

    <div class="pipe-section">
      <div class="pipe-section-title"><i class="ti ti-git-merge"></i> Stage 3: CG Alignment & Gate Values</div>
      <div class="detail-row" style="border-bottom:1px solid var(--border);">
        <span class="detail-label">CG Alignment Score</span>
        <span class="chip mono" style="background:${r.cg_alignment > 0.7 ? 'var(--emerald-light)' : 'var(--amber-light)'}; color:${r.cg_alignment > 0.7 ? 'var(--emerald)' : 'var(--amber)'}">${r.cg_alignment.toFixed(2)} — ${r.cg_alignment > 0.7 ? 'strong agreement' : 'moderate'}</span>
      </div>
      <div style="margin-top:10px;">
        <div style="font-size:11px;color:var(--text3);margin-bottom:6px;font-weight:600;">MODALITY GATE VALUES</div>
        <div class="gate-bar"><div class="gate-label">Text Gate</div><div class="gate-track"><div class="gate-fill" style="width:${r.cg_gates.text * 100}%;background:var(--blue)"></div></div><div class="gate-val">${r.cg_gates.text.toFixed(2)}</div></div>
        <div class="gate-bar"><div class="gate-label">Image Gate</div><div class="gate-track"><div class="gate-fill" style="width:${r.cg_gates.image * 100}%;background:var(--cyan)"></div></div><div class="gate-val">${r.cg_gates.image.toFixed(2)}</div></div>
      </div>
    </div>

    <div class="pipe-section">
      <div class="pipe-section-title"><i class="ti ti-arrows-cross"></i> Cross-Attention Weights</div>
      <div class="gate-bar"><div class="gate-label">Text→Image</div><div class="gate-track"><div class="gate-fill" style="width:${r.ca_weights.t2i * 100}%;background:var(--purple)"></div></div><div class="gate-val">${r.ca_weights.t2i.toFixed(2)}</div></div>
      <div class="gate-bar"><div class="gate-label">Image→Text</div><div class="gate-track"><div class="gate-fill" style="width:${r.ca_weights.i2t * 100}%;background:var(--purple)"></div></div><div class="gate-val">${r.ca_weights.i2t.toFixed(2)}</div></div>
    </div>

    <div style="padding:14px;border-radius:var(--radius);background:var(--surface2);border:1px solid var(--border);margin-top:4px;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
        <span style="font-weight:700;font-size:13px;">Final Prediction</span>
        ${sentBadge(r.sentiment)}
      </div>
      <div class="detail-row"><span class="detail-label">Confidence</span><span class="chip mono" style="font-weight:600;">${(r.confidence * 100).toFixed(1)}%</span></div>
      <div class="detail-row"><span class="detail-label">System Action</span>${actionBadge(r.system_action)}</div>
      <div style="margin-top:8px;font-size:11px;padding:8px;border-radius:6px;background:var(--bg2);color:var(--text2);border:1px solid var(--border);">${r.note}</div>
    </div>

    <div class="api-panel" style="margin-top:10px;">
      <div class="api-panel-header">
        <span class="method-tag get">GET</span>
        <span>/api/results/${r.id}/pipeline</span>
      </div>
      <div class="api-panel-body">${formatJSON({
        review_id: r.id,
        pipeline: {
          stage1_preprocessing: { status: 'completed', text_cleaned: true, image_resized: '224x224' },
          stage2a_roberta: { logits: r.roberta_conf, embedding_dim: 768 },
          stage2b_resnet50: { embedding_dim: 512, features_extracted: true },
          stage3_fusion: { cg_alignment: r.cg_alignment, cg_gates: r.cg_gates, ca_weights: r.ca_weights }
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
   Image Handling (Upload Page)
   ============================================================ */

function handleImages(input) {
  Array.from(input.files).forEach(f => {
    const reader = new FileReader();
    reader.onload = e => {
      uploadedImages.push({ name: f.name, url: e.target.result });
      renderImgGrid();
    };
    reader.readAsDataURL(f);
  });
  input.value = '';
  BackendAPI.logToConsole('POST', '/api/upload/images', 201, 180 + Math.random() * 200);
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
   Single Review Classifier — with animated pipeline
   ============================================================ */

async function classifyTest() {
  const text = document.getElementById('test-text').value.trim();
  if (!text) { showToast('Enter review text first'); return; }

  const el = document.getElementById('test-result');
  const stars_n = parseInt(document.getElementById('test-stars').value);
  const hasImg = !!document.querySelector('#single-img-preview .img-thumb');

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
    step.className = `pipeline-step ${status}`;
    step.querySelector('.step-detail').textContent = detail;
    step.querySelector('.step-timing').textContent = timing;
  }

  // Step 1: Send request
  BackendAPI.logToConsole('POST', '/api/predict', null, null, `{text: "${text.substring(0, 40)}..."}`);
  await new Promise(r => setTimeout(r, 300 + Math.random() * 200));
  advanceStep('step-send', 'completed', 'Request accepted — 202 Accepted', `${(180 + Math.random() * 120).toFixed(0)}ms`);
  BackendAPI.logToConsole('OK', '/api/predict', 202, 195);

  // Step 2: Preprocess
  advanceStep('step-preprocess', 'active', 'Tokenizing text, normalizing…', '—');
  await new Promise(r => setTimeout(r, 400 + Math.random() * 300));
  const tokenCount = 12 + Math.floor(Math.random() * 30);
  advanceStep('step-preprocess', 'completed', `Cleaned & tokenized — ${tokenCount} tokens, max_len=128`, `${(85 + Math.random() * 60).toFixed(0)}ms`);

  // Step 3: RoBERTa
  advanceStep('step-roberta', 'active', 'Forward pass through RoBERTa-base (125M params)…', '—');
  BackendAPI.logToConsole('INFO', `RoBERTa inference — ${tokenCount} tokens`, null, null);
  await new Promise(r => setTimeout(r, 600 + Math.random() * 400));

  // Simulated classification
  const posWords = /(maganda|solid|worth|sulit|love|smooth|ganda|recommend|best|satisfied|great|good|nice)/i;
  const negWords = /(panget|sira|hindi|wag|sayang|broken|defect|gasgas|maingay|misleading|laggy|disconnect)/i;
  const hasPos = posWords.test(text);
  const hasNeg = negWords.test(text);

  let sentiment = 'neutral';
  let roberta = [0.20, 0.60, 0.20];
  if (hasPos && !hasNeg) { sentiment = 'positive'; roberta = [0.80 + Math.random() * 0.12, 0.10, 0.03]; }
  else if (hasNeg && !hasPos) { sentiment = 'negative'; roberta = [0.03, 0.07, 0.82 + Math.random() * 0.10]; }
  else if (hasPos && hasNeg) { sentiment = 'positive'; roberta = [0.55 + Math.random() * 0.15, 0.25, 0.12]; }

  advanceStep('step-roberta', 'completed', `Logits: [${roberta.map(v => v.toFixed(3)).join(', ')}] — 768-dim embedding`, `${(220 + Math.random() * 180).toFixed(0)}ms`);

  // Step 4: ResNet-50
  advanceStep('step-resnet', 'active', hasImg ? 'Processing review image through ResNet-50…' : 'No image provided — generating null embedding…', '—');
  if (hasImg) {
    BackendAPI.logToConsole('INFO', 'ResNet-50 inference — 224×224 image', null, null);
  }
  await new Promise(r => setTimeout(r, hasImg ? 500 + Math.random() * 300 : 150));
  advanceStep('step-resnet', 'completed', hasImg ? 'Visual features extracted — 512-dim embedding' : 'Null embedding (no image) — CG will downweight', `${hasImg ? (180 + Math.random() * 150).toFixed(0) : '45'}ms`);

  // Step 5: Fusion
  advanceStep('step-fusion', 'active', 'Computing Cross-Attention & Cross-modal Gating…', '—');
  BackendAPI.logToConsole('INFO', 'CA Fusion + CG alignment pass', null, null);
  await new Promise(r => setTimeout(r, 400 + Math.random() * 300));

  const cg_align = hasImg ? 0.60 + Math.random() * 0.35 : 0.40 + Math.random() * 0.25;
  const text_gate = 0.65 + Math.random() * 0.25;
  const img_gate = hasImg ? 0.50 + Math.random() * 0.40 : 0.15 + Math.random() * 0.20;
  const t2i = hasImg ? 0.55 + Math.random() * 0.35 : 0.20 + Math.random() * 0.20;
  const i2t = hasImg ? 0.50 + Math.random() * 0.35 : 0.15 + Math.random() * 0.20;

  advanceStep('step-fusion', 'completed', `CG: ${cg_align.toFixed(3)} | Gates: T=${text_gate.toFixed(2)} I=${img_gate.toFixed(2)} | CA: T→I=${t2i.toFixed(2)} I→T=${i2t.toFixed(2)}`, `${(145 + Math.random() * 100).toFixed(0)}ms`);

  // Step 6: Final prediction
  advanceStep('step-predict', 'active', 'Computing final classification…', '—');
  await new Promise(r => setTimeout(r, 200 + Math.random() * 150));

  const confidence = 0.65 + Math.random() * 0.30;
  const sysAction = { positive: 'Highlighted / Approved', neutral: 'Logged', negative: 'Flagged for Support' }[sentiment];

  advanceStep('step-predict', 'completed', `${sentiment.toUpperCase()} — ${(confidence * 100).toFixed(1)}% confidence → ${sysAction}`, `${(35 + Math.random() * 30).toFixed(0)}ms`);

  BackendAPI.logToConsole('OK', '/api/predict — result ready', 200, 1450 + Math.random() * 500);

  // Now show the full result below the pipeline
  const resultHTML = `
    <div style="border:1px solid var(--border);border-radius:var(--radius);padding:14px;background:var(--surface2);margin-top:12px;">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;">
        <span style="font-weight:700;font-size:14px;">Result</span>
        ${sentBadge(sentiment)}
      </div>

      <div class="pipe-section">
        <div class="pipe-section-title"><i class="ti ti-typography"></i> RoBERTa Features</div>
        <div class="gate-bar"><div class="gate-label">Positive</div><div class="gate-track"><div class="gate-fill" style="width:${roberta[0] * 100}%;background:var(--emerald)"></div></div><div class="gate-val">${roberta[0].toFixed(2)}</div></div>
        <div class="gate-bar"><div class="gate-label">Neutral</div><div class="gate-track"><div class="gate-fill" style="width:${roberta[1] * 100}%;background:var(--amber)"></div></div><div class="gate-val">${roberta[1].toFixed(2)}</div></div>
        <div class="gate-bar"><div class="gate-label">Negative</div><div class="gate-track"><div class="gate-fill" style="width:${roberta[2] * 100}%;background:var(--rose)"></div></div><div class="gate-val">${roberta[2].toFixed(2)}</div></div>
      </div>

      <div class="detail-row"><span class="detail-label">CG Alignment</span><span class="chip mono">${cg_align.toFixed(2)} — ${cg_align > 0.7 ? 'strong' : 'moderate'}</span></div>
      <div class="detail-row"><span class="detail-label">CG Text Gate</span><span class="chip mono">${text_gate.toFixed(2)}</span></div>
      <div class="detail-row"><span class="detail-label">CG Image Gate</span><span class="chip mono">${img_gate.toFixed(2)}</span></div>
      <div class="detail-row"><span class="detail-label">Confidence</span><span class="chip mono">${(confidence * 100).toFixed(1)}%</span></div>
      <div class="detail-row"><span class="detail-label">Image uploaded</span><span class="chip">${hasImg ? 'Yes — ResNet-50 processed' : 'No'}</span></div>
      <div class="detail-row"><span class="detail-label">Stars</span><span style="color:#d4a017;">${'★'.repeat(stars_n) + '☆'.repeat(5 - stars_n)}</span></div>
      <div class="detail-row"><span class="detail-label">System Action</span>${actionBadge(sysAction)}</div>

      <div style="margin-top:10px;font-size:12px;padding:10px;border-radius:var(--radius);background:var(--bg2);color:var(--text2);border:1px solid var(--border);">${{
        positive: 'Cross-Attention weighted text praise signals with image features. CG confirmed modality agreement and elevated positive class probability.',
        neutral: 'No strong sentiment signal detected. CG found moderate modality alignment. Review logged for standard processing.',
        negative: 'Negative text features dominate via CG text gate. Cross-Attention identified complaint patterns. Review flagged for customer support.'
      }[sentiment]}</div>

      <div class="api-panel" style="margin-top:10px;">
        <div class="api-panel-header">
          <span class="method-tag post">POST</span>
          <span>/api/predict</span>
        </div>
        <div class="api-panel-body">${formatJSON({
          request: { text: text, stars: stars_n, has_image: hasImg },
          response: {
            sentiment,
            confidence: parseFloat(confidence.toFixed(4)),
            roberta_logits: roberta.map(v => parseFloat(v.toFixed(4))),
            cg: { alignment: parseFloat(cg_align.toFixed(4)), text_gate: parseFloat(text_gate.toFixed(4)), image_gate: parseFloat(img_gate.toFixed(4)) },
            ca: { text_to_image: parseFloat(t2i.toFixed(4)), image_to_text: parseFloat(i2t.toFixed(4)) },
            system_action: sysAction,
            processing_time_ms: (1450 + Math.random() * 500).toFixed(0)
          }
        })}</div>
      </div>
    </div>`;

  el.innerHTML += resultHTML;
}


/* ============================================================
   CSV Handling
   ============================================================ */

async function handleCSV(input) {
  const f = input.files[0];
  if (!f) return;

  // Show uploading state
  document.getElementById('csv-status').innerHTML = `
    <div style="display:flex;align-items:center;gap:8px;padding:10px;border-radius:var(--radius);background:var(--blue-light);border:1px solid rgba(59,130,246,0.2);">
      <i class="ti ti-loader-2" style="color:var(--blue);font-size:16px;animation:spin 1s linear infinite;display:inline-block;"></i>
      <span style="color:var(--blue);font-weight:600;">Uploading ${f.name}…</span>
      <span style="color:var(--text2);font-family:'JetBrains Mono',monospace;font-size:11px;">POST /api/upload/csv</span>
    </div>`;

  BackendAPI.logToConsole('POST', '/api/upload/csv', null, null, `(${f.name}, ${(f.size / 1024).toFixed(1)}KB)`);

  await new Promise(r => setTimeout(r, 800 + Math.random() * 600));
  BackendAPI.logToConsole('OK', '/api/upload/csv', 201, 650);

  // Show processing state
  document.getElementById('csv-status').innerHTML = `
    <div style="display:flex;align-items:center;gap:8px;padding:10px;border-radius:var(--radius);background:var(--amber-light);border:1px solid rgba(245,158,11,0.2);">
      <i class="ti ti-loader-2" style="color:var(--amber);font-size:16px;animation:spin 1s linear infinite;display:inline-block;"></i>
      <span style="color:var(--amber);font-weight:600;">Server processing CSV…</span>
      <span style="color:var(--text2);font-family:'JetBrains Mono',monospace;font-size:11px;">Validating columns, parsing rows</span>
    </div>`;

  BackendAPI.logToConsole('INFO', 'Parsing CSV — validating schema…', null, null);
  await new Promise(r => setTimeout(r, 600 + Math.random() * 400));
  BackendAPI.logToConsole('OK', 'CSV validated', null, null, '— columns: product_id, product_title, review_text, review_image_urls, labels');

  // Show success
  document.getElementById('csv-status').innerHTML = `
    <div style="display:flex;align-items:center;gap:8px;padding:10px;border-radius:var(--radius);background:var(--emerald-light);border:1px solid rgba(16,185,129,0.2);">
      <i class="ti ti-check" style="color:var(--emerald);font-size:16px;"></i>
      <span style="color:var(--emerald);font-weight:600;">${f.name}</span>
      <span style="color:var(--text2);">(${(f.size / 1024).toFixed(1)} KB) — uploaded & validated</span>
    </div>
    <div class="api-panel" style="margin-top:8px;">
      <div class="api-panel-header">
        <span class="method-tag post">POST</span>
        <span>/api/upload/csv — 201 Created</span>
      </div>
      <div class="api-panel-body">${formatJSON({
        status: 'success',
        file: f.name,
        size_bytes: f.size,
        rows_parsed: Math.floor(Math.random() * 200) + 50,
        columns_validated: ['product_id', 'product_title', 'review_text', 'review_image_urls', 'labels'],
        processing_time_ms: (1200 + Math.random() * 800).toFixed(0)
      })}</div>
    </div>`;
}

function dragOver(e) { e.preventDefault(); e.currentTarget.classList.add('dragover'); }
function dragLeave(e) { e.currentTarget.classList.remove('dragover'); }

function dropCSV(e) {
  e.preventDefault();
  e.currentTarget.classList.remove('dragover');
  BackendAPI.logToConsole('POST', '/api/upload/csv', 201, 420);
  showToast('CSV file received!');
}

function dropImages(e) {
  e.preventDefault();
  e.currentTarget.classList.remove('dragover');
  BackendAPI.logToConsole('POST', '/api/upload/images', 201, 380);
  showToast('Images received!');
}

async function fakeLoad() {
  BackendAPI.logToConsole('GET', '/api/dataset/demo', null, null);
  showToast('Loading sample dataset…');
  await new Promise(r => setTimeout(r, 800));
  BackendAPI.logToConsole('OK', '/api/dataset/demo', 200, 340);
  BackendAPI.logToConsole('INFO', 'Loaded 2,104 reviews from demo dataset', null, null);
  showToast('Sample dataset loaded — 2,104 reviews');
  goPage('dashboard');
}


/* ============================================================
   Init — Boot Sequence
   ============================================================ */

renderDash();

if (window.location.hash === '#upload') {
  goPage('upload');
}

// Boot the simulated backend
BackendAPI.boot();