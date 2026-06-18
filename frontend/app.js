/* ============================================================
   CLIP-CA-CG — Dashboard Logic
   Multimodal Sentiment Analysis with RoBERTa + ResNet-50
   ============================================================ */

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
  if (id === 'pipeline') renderPipeline();
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

function selectDash(id) {
  selectedDash = id;
  renderDash();
  const r = reviews.find(x => x.id === id);
  document.getElementById('dash-detail').innerHTML = `
    <div class="card-title">Review #${r.id}</div>
    <div class="detail-row"><span class="detail-label">Sentiment</span>${sentBadge(r.sentiment)}</div>
    <div class="detail-row"><span class="detail-label">Confidence</span><span class="chip mono">${(r.confidence * 100).toFixed(1)}%</span></div>
    <div class="detail-row"><span class="detail-label">CG Alignment</span><span class="chip mono">${r.cg_alignment.toFixed(2)}</span></div>
    <div class="detail-row"><span class="detail-label">System Action</span>${actionBadge(r.system_action)}</div>
    <div class="detail-row"><span class="detail-label">Stars</span><span style="color:#d4a017;">${stars(r.stars)}</span></div>
    <div style="margin-top:10px;font-size:11px;padding:8px 10px;border-radius:var(--radius);background:var(--surface2);color:var(--text2);border:1px solid var(--border);">${r.note}</div>`;
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

function selectPipeline(id) {
  selectedPipeline = id;
  renderPipeline();
  const r = reviews.find(x => x.id === id);
  document.getElementById('pipeline-detail').innerHTML = `
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
   Single Review Classifier
   ============================================================ */

function classifyTest() {
  const text = document.getElementById('test-text').value.trim();
  if (!text) { showToast('Enter review text first'); return; }

  const el = document.getElementById('test-result');
  el.innerHTML = '<div style="text-align:center;padding:16px;color:var(--blue);"><i class="ti ti-loader-2" style="font-size:24px;animation:spin 1s linear infinite;display:inline-block;"></i><div style="font-size:12px;margin-top:6px;">Running through pipeline…</div></div>';

  setTimeout(() => {
    const stars_n = parseInt(document.getElementById('test-stars').value);
    const hasImg = !!document.querySelector('#single-img-preview .img-thumb');

    // Simulated pipeline
    const posWords = /(maganda|solid|worth|sulit|love|smooth|ganda|recommend|best|satisfied|great|good|nice)/i;
    const negWords = /(panget|sira|hindi|wag|sayang|broken|defect|gasgas|maingay|misleading|laggy|disconnect)/i;
    const hasPos = posWords.test(text);
    const hasNeg = negWords.test(text);

    let sentiment = 'neutral';
    let roberta = [0.20, 0.60, 0.20];
    if (hasPos && !hasNeg) { sentiment = 'positive'; roberta = [0.80 + Math.random() * 0.12, 0.10, 0.03]; }
    else if (hasNeg && !hasPos) { sentiment = 'negative'; roberta = [0.03, 0.07, 0.82 + Math.random() * 0.10]; }
    else if (hasPos && hasNeg) { sentiment = 'positive'; roberta = [0.55 + Math.random() * 0.15, 0.25, 0.12]; }

    const cg_align = hasImg ? 0.60 + Math.random() * 0.35 : 0.40 + Math.random() * 0.25;
    const text_gate = 0.65 + Math.random() * 0.25;
    const img_gate = hasImg ? 0.50 + Math.random() * 0.40 : 0.15 + Math.random() * 0.20;
    const confidence = 0.65 + Math.random() * 0.30;

    const sysAction = { positive: 'Highlighted / Approved', neutral: 'Logged', negative: 'Flagged for Support' }[sentiment];

    el.innerHTML = `
      <div style="border:1px solid var(--border);border-radius:var(--radius);padding:14px;background:var(--surface2);">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;">
          <span style="font-weight:700;font-size:14px;">Pipeline Result</span>
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
      </div>`;
  }, 1200);
}


/* ============================================================
   CSV Handling
   ============================================================ */

function handleCSV(input) {
  const f = input.files[0];
  if (!f) return;
  document.getElementById('csv-status').innerHTML = `
    <div style="display:flex;align-items:center;gap:8px;padding:10px;border-radius:var(--radius);background:var(--emerald-light);border:1px solid rgba(16,185,129,0.2);">
      <i class="ti ti-check" style="color:var(--emerald);font-size:16px;"></i>
      <span style="color:var(--emerald);font-weight:600;">${f.name}</span>
      <span style="color:var(--text2);">(${(f.size / 1024).toFixed(1)} KB)</span>
    </div>`;
}

function dragOver(e) { e.preventDefault(); e.currentTarget.classList.add('dragover'); }
function dragLeave(e) { e.currentTarget.classList.remove('dragover'); }

function dropCSV(e) {
  e.preventDefault();
  e.currentTarget.classList.remove('dragover');
  showToast('CSV file received!');
}

function dropImages(e) {
  e.preventDefault();
  e.currentTarget.classList.remove('dragover');
  showToast('Images received!');
}

function fakeLoad() {
  showToast('Sample dataset loaded — 2,104 reviews');
  goPage('dashboard');
}


/* ============================================================
   Init
   ============================================================ */

renderDash();

if (window.location.hash === '#upload') {
  goPage('upload');
}