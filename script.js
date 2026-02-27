// =============================================
//   MULTIVARIATE REGRESSION — script.js
//   Manual slope-intercept method (PDF style)
// =============================================

const FEATURES = [
  'Pregnancies','Glucose','BloodPressure','SkinThickness',
  'Insulin','BMI','DiabetesPedigreeFunction','Age'
];
const SUB  = ['₁','₂','₃','₄','₅','₆','₇','₈'];
const XBAR = ['x̄₁','x̄₂','x̄₃','x̄₄','x̄₅','x̄₆','x̄₇','x̄₈'];

let dataset       = [];
let modelSlopes   = [];
let modelIntercepts = [];
let modelB        = 0;
let modelTrained  = false;

// Sigmoid function — squeezes any value into 0–1 range
function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

// We center raw scores around 0 before sigmoid so predictions spread across 0 and 1
// rawMean is computed after model training
let rawMean = 0;
let rawStd  = 1;

// ── per-feature full row data (for show-more) ──
let featureRows = [];   // featureRows[fi] = array of row objects

// ───────────────────────────────────────────
//  CSV UPLOAD
// ───────────────────────────────────────────
const fileInput  = document.getElementById('csvFile');
const uploadArea = document.getElementById('uploadArea');

fileInput.addEventListener('change', e => handleFile(e.target.files[0]));
uploadArea.addEventListener('dragover',  e => { e.preventDefault(); uploadArea.classList.add('dragover'); });
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
uploadArea.addEventListener('drop', e => { e.preventDefault(); uploadArea.classList.remove('dragover'); handleFile(e.dataTransfer.files[0]); });

function handleFile(file) {
  if (!file) return;
  if (!file.name.endsWith('.csv')) { showMsg('err','❌ Please upload a .csv file.'); return; }
  const reader = new FileReader();
  reader.onload = e => parseCSV(e.target.result, file.name);
  reader.readAsText(file);
}

function parseCSV(text, name) {
  const lines = text.trim().split('\n').map(l => l.trim()).filter(Boolean);
  if (lines.length < 2) { showMsg('err','❌ CSV file is empty or has no data rows.'); return; }
  const header = lines[0].split(',').map(h => h.trim());
  if (header.length < 9) { showMsg('err',`❌ Expected 9 columns, found ${header.length}.`); return; }

  dataset = [];
  for (let i = 1; i < lines.length; i++) {
    const vals = lines[i].split(',').map(v => parseFloat(v.trim()));
    if (vals.length < 9 || vals.some(isNaN)) continue;
    const row = {};
    header.forEach((h,j) => row[h] = vals[j]);
    dataset.push(row);
  }
  if (dataset.length === 0) { showMsg('err','❌ No valid numeric rows found.'); return; }

  showMsg('ok', `✔ "${name}" loaded — ${dataset.length} rows, ${header.length} columns.`);
  show('sec-table'); show('sec-train');
  renderDataTable();
  document.getElementById('lblRowCount').textContent = `${dataset.length} total rows`;

  // Reset
  modelTrained = false;
  hide('sec-predict');
  document.getElementById('computationArea').innerHTML = '';
  document.getElementById('finalModelArea').innerHTML = '';
  hide('finalModelArea');
  document.getElementById('accuracyArea').innerHTML = '';
  hide('accuracyArea');
}

function showMsg(type, msg) {
  const el = document.getElementById('uploadMsg');
  el.className = `upload-msg ${type}`;
  el.textContent = msg;
  el.classList.remove('hidden');
}

// ───────────────────────────────────────────
//  HISTORICAL DATA TABLE (10 rows default)
// ───────────────────────────────────────────
function renderDataTable() {
  const sel  = document.getElementById('selRows').value;
  const rows = sel === 'all' ? dataset : dataset.slice(0, parseInt(sel));
  const means = computeMeans();

  let html = `<table class="data-tbl"><thead><tr><th>#</th>`;
  FEATURES.forEach((f,i) => {
    html += `<th>${f}<br><small>(x${SUB[i]})</small></th>`;
  });
  html += `<th class="col-y">Outcome<br><small>(y)</small></th></tr></thead><tbody>`;

  rows.forEach((r, i) => {
    html += `<tr><td>${i+1}</td>`;
    FEATURES.forEach(f => html += `<td>${r[f]}</td>`);
    html += `<td class="col-y">${r['Outcome']}</td></tr>`;
  });

  // Mean row (always shown regardless of filter)
  html += `<tr class="row-mean"><td>Mean</td>`;
  FEATURES.forEach(f => html += `<td>${means[f].toFixed(4)}</td>`);
  html += `<td class="col-y">${means['Outcome'].toFixed(4)}</td></tr>`;
  html += `</tbody></table>`;

  document.getElementById('dataTableArea').innerHTML = html;
}

// ───────────────────────────────────────────
//  HELPERS
// ───────────────────────────────────────────
function computeMeans() {
  const cols = [...FEATURES, 'Outcome'];
  const means = {};
  cols.forEach(c => { means[c] = dataset.reduce((s,r) => s + r[c], 0) / dataset.length; });
  return means;
}
function fmt(n)  { return n.toFixed(4); }
function fmt2(n) { return n.toFixed(4); }

// ───────────────────────────────────────────
//  COMPUTE MODEL
// ───────────────────────────────────────────
function computeModel() {
  if (dataset.length === 0) return;
  document.getElementById('btnTrain').disabled = true;

  const means = computeMeans();
  const yBar  = means['Outcome'];
  const area  = document.getElementById('computationArea');
  area.innerHTML = '';

  modelSlopes      = [];
  modelIntercepts  = [];
  featureRows      = [];

  FEATURES.forEach((feat, fi) => {
    const xBar = means[feat];
    const xSub = `x${SUB[fi]}`;
    const xBarSub = XBAR[fi];

    let sumXY = 0, sumX2 = 0;
    const rows = dataset.map(r => {
      const x    = r[feat];
      const y    = r['Outcome'];
      const dX   = x - xBar;
      const dY   = y - yBar;
      const dXdY = dX * dY;
      const dX2  = dX * dX;
      sumXY += dXdY;
      sumX2 += dX2;
      return { x, y, dX, dY, dXdY, dX2 };
    });

    featureRows.push(rows);

    const m = sumX2 !== 0 ? sumXY / sumX2 : 0;
    const b = yBar - m * xBar;
    modelSlopes.push(m);
    modelIntercepts.push(b);

    // Render table showing first 10 rows only
    area.innerHTML += buildFeatureTable(fi, feat, xSub, xBarSub, xBar, yBar, rows, sumXY, sumX2, m, b, 10);
  });

  // Overall intercept — average of all b's
  modelB = modelIntercepts.reduce((s,v) => s+v, 0) / modelIntercepts.length;

  // Overall intercept box
  const bLabels = modelIntercepts.map((_,i) => `b${SUB[i]}`).join(' + ');
  const bValues = modelIntercepts.map(b => fmt(b)).join(' + ');

  let finalHtml = `
  <div class="overall-box">
    <div class="overall-title">Computation of the Overall Y-Intercept</div>
    <div class="overall-inner">
      <div class="formula-box">
        b = (${bLabels}) / ${FEATURES.length}<br>
        b = (${bValues}) / ${FEATURES.length}<br>
        <strong>b = ${fmt(modelB)}</strong>
      </div>
    </div>
  </div>`;

  // Final model
  const eqParts = FEATURES.map((f,i) => `(${fmt(modelSlopes[i])})x${SUB[i]}`).join(' + ');
  finalHtml += `
  <div class="final-model-box" style="margin-top:20px">
    <div class="final-model-title">Final Multi-Variate Model</div>
    <div class="final-model-inner">
      <div class="formula-box">
        ŷ = m₁x₁ + m₂x₂ + m₃x₃ + m₄x₄ + m₅x₅ + m₆x₆ + m₇x₇ + m₈x₈ + b<br><br>
        ŷ = ${eqParts} + ${fmt(modelB)}<br><br>
        <small style="color:#555">Sigmoid applied to ŷ → σ(ŷ) = 1 / (1 + e<sup>−ŷ</sup>) — if σ(ŷ) ≥ 0.5 → Predicted 1 (Diabetes), else → 0</small>
      </div>
    </div>
  </div>`;

  const finalArea = document.getElementById('finalModelArea');
  finalArea.innerHTML = finalHtml;
  show('finalModelArea');

  // Compute mean and std of raw scores so we can center before sigmoid
  const allRaw = dataset.map(r => FEATURES.reduce((s,f,i) => s + modelSlopes[i] * r[f], 0) + modelB);
  rawMean = allRaw.reduce((s,v) => s+v, 0) / allRaw.length;
  rawStd  = Math.sqrt(allRaw.reduce((s,v) => s + (v - rawMean)**2, 0) / allRaw.length) || 1;

  buildAccuracyTable();

  modelTrained = true;
  show('sec-predict');
  document.getElementById('btnTrain').disabled = false;
}

// ───────────────────────────────────────────
//  BUILD ONE FEATURE TABLE (PDF columns)
// ───────────────────────────────────────────
function buildFeatureTable(fi, feat, xSub, xBarSub, xBar, yBar, rows, sumXY, sumX2, m, b, limit) {
  const showing = limit === 'all' ? rows : rows.slice(0, limit);
  const hasMore = rows.length > 10;
  const btnId   = `btn-more-${fi}`;

  let html = `<div class="comp-block" id="comp-block-${fi}">`;
  html += `<div class="comp-title">
    <span>Feature: <strong>${feat}</strong> &nbsp;(${xSub})</span>
    <span class="slope-badge">m${SUB[fi]} = ${fmt(m)}</span>
  </div>`;

  // TABLE — columns exactly like PDF
  html += `<table class="comp-tbl"><thead><tr>
    <th>${xSub}</th>
    <th>y</th>
    <th>${xSub} − ${xBarSub}</th>
    <th>y − ȳ</th>
    <th>(${xSub} − ${xBarSub})(y − ȳ)</th>
    <th>(${xSub} − ${xBarSub})²</th>
  </tr></thead><tbody id="comp-tbody-${fi}">`;

  showing.forEach(r => {
    html += `<tr>
      <td>${fmt2(r.x)}</td>
      <td>${fmt2(r.y)}</td>
      <td>${fmt(r.dX)}</td>
      <td>${fmt(r.dY)}</td>
      <td>${fmt(r.dXdY)}</td>
      <td>${fmt(r.dX2)}</td>
    </tr>`;
  });

  // Mean / Total row
  html += `<tr class="row-mean">
    <td>${fmt2(xBar)}<br><small>${xBarSub}</small></td>
    <td>${fmt2(yBar)}<br><small>ȳ</small></td>
    <td></td><td></td>
    <td><strong>${fmt(sumXY)}</strong><br><small>Total</small></td>
    <td><strong>${fmt(sumX2)}</strong><br><small>Total</small></td>
  </tr>`;
  html += `</tbody></table>`;

  // Show more/less button
  if (hasMore) {
    const isAll = limit === 'all';
    html += `<div class="show-more-wrap">
      <button class="btn-show-more" id="${btnId}" onclick="toggleFeatureRows(${fi}, this)">
        ${isAll ? `▲ Show Less (10 rows)` : `▼ Show All ${rows.length} Rows`}
      </button>
    </div>`;
  }

  // Y-Intercept line
  html += `<div class="intercept-row">
    Y-Intercept: &nbsp; b${SUB[fi]} = ȳ − m${SUB[fi]} · ${xBarSub}
    &nbsp;=&nbsp; ${fmt2(yBar)} − (${fmt(m)})(${fmt2(xBar)})
    &nbsp;=&nbsp; <strong>${fmt(b)}</strong>
  </div>`;
  html += `</div>`;

  return html;
}

// ───────────────────────────────────────────
//  TOGGLE SHOW MORE / LESS ROWS
// ───────────────────────────────────────────
function toggleFeatureRows(fi, btn) {
  const feat    = FEATURES[fi];
  const xSub    = `x${SUB[fi]}`;
  const xBarSub = XBAR[fi];
  const means   = computeMeans();
  const xBar    = means[feat];
  const yBar    = means['Outcome'];
  const rows    = featureRows[fi];
  const m       = modelSlopes[fi];
  const b       = modelIntercepts[fi];
  let   sumXY   = rows.reduce((s,r) => s+r.dXdY, 0);
  let   sumX2   = rows.reduce((s,r) => s+r.dX2, 0);

  const isExpanded = btn.textContent.includes('Less');
  const newLimit   = isExpanded ? 10 : 'all';

  // Replace the entire comp-block
  const block = document.getElementById(`comp-block-${fi}`);
  const tmp   = document.createElement('div');
  tmp.innerHTML = buildFeatureTable(fi, feat, xSub, xBarSub, xBar, yBar, rows, sumXY, sumX2, m, b, newLimit);
  block.replaceWith(tmp.firstElementChild);
}

// ───────────────────────────────────────────
//  ACCURACY TABLE (dynamic threshold)
// ───────────────────────────────────────────
function buildAccuracyTable() {
  const preds = dataset.map(r => {
    const raw    = FEATURES.reduce((s,f,i) => s + modelSlopes[i] * r[f], 0) + modelB;
    const prob   = sigmoid((raw - rawMean) / rawStd);
    const pred   = prob >= 0.5 ? 1 : 0;
    const actual = r['Outcome'];
    return { raw, prob, pred, actual, match: pred === actual };
  });

  window._preds = preds;
  const correct = preds.filter(p => p.match).length;
  const acc     = (correct / preds.length * 100).toFixed(2);

  let html = `<div class="accuracy-block">
    <div class="accuracy-header">
      <span>Prediction Results vs Actual Outcome</span>
      <span>Accuracy: ${acc}% &nbsp;(${correct}/${preds.length} correct)</span>
    </div>
    <div class="accuracy-stats">
      <div class="stat-item"><div class="stat-label">Total Samples</div><div class="stat-value">${preds.length}</div></div>
      <div class="stat-item"><div class="stat-label">Correct</div><div class="stat-value">${correct}</div></div>
      <div class="stat-item"><div class="stat-label">Incorrect</div><div class="stat-value">${preds.length - correct}</div></div>
      <div class="stat-item"><div class="stat-label">Accuracy</div><div class="stat-value">${acc}%</div></div>
      <div class="stat-item"><div class="stat-label">Threshold</div><div class="stat-value" style="font-size:1rem">σ ≥ 0.5</div></div>
    </div>
    <div class="tbl-show-ctrl">
      Show rows: <select onchange="filterPredTable(this.value)">
        <option value="10">10</option>
        <option value="20">20</option>
        <option value="50">50</option>
        <option value="100">100</option>
        <option value="all">All</option>
      </select>
    </div>
    <div id="predTableInner"></div>
  </div>`;

  const accArea = document.getElementById('accuracyArea');
  accArea.innerHTML = html;
  show('accuracyArea');
  renderPredTable(preds.slice(0, 10));
}

function filterPredTable(val) {
  const data = window._preds;
  const rows = val === 'all' ? data : data.slice(0, parseInt(val));
  renderPredTable(rows);
}

function renderPredTable(rows) {
  // Column headers: #, x₁, x₂ ... x₈, Actual (y), Raw Score, Predicted, Match
  let html = `<table class="pred-tbl"><thead><tr>
    <th>#</th>`;
  FEATURES.forEach((f,i) => html += `<th>${f}<br><small>(x${SUB[i]})</small></th>`);
  html += `<th>Actual<br><small>(y)</small></th><th>Raw Score<br><small>(ŷ)</small></th><th>Sigmoid<br><small>σ(ŷ)</small></th><th>Predicted</th><th>Match</th>
  </tr></thead><tbody>`;

  rows.forEach((p) => {
    const idx = window._preds.indexOf(p);
    html += `<tr>`;
    html += `<td>${idx + 1}</td>`;
    FEATURES.forEach(f => html += `<td>${dataset[idx][f]}</td>`);
    html += `<td><strong>${p.actual}</strong></td>`;
    html += `<td class="raw-score">${p.raw.toFixed(4)}</td>`;
    html += `<td class="raw-score">${p.prob.toFixed(4)}</td>`;
    html += `<td><strong>${p.pred}</strong></td>`;
    html += `<td class="${p.match ? 'match-yes' : 'match-no'}">${p.match ? '✔ Yes' : '✗ No'}</td>`;
    html += `</tr>`;
  });
  html += '</tbody></table>';
  document.getElementById('predTableInner').innerHTML = html;
}

// ───────────────────────────────────────────
//  MANUAL PREDICTION
// ───────────────────────────────────────────
function makePrediction() {
  if (!modelTrained) { alert('Please compute the model first.'); return; }
  const ids  = ['p1','p2','p3','p4','p5','p6','p7','p8'];
  const vals = ids.map(id => parseFloat(document.getElementById(id).value));
  if (vals.some(isNaN)) { alert('Please fill in all 8 input fields.'); return; }

  const raw     = FEATURES.reduce((s,_,i) => s + modelSlopes[i] * vals[i], 0) + modelB;
  const prob    = sigmoid((raw - rawMean) / rawStd);
  const outcome = prob >= 0.5 ? 1 : 0;

  const parts   = FEATURES.map((f,i) => `(${fmt(modelSlopes[i])})(${vals[i]})`);
  const numeric = FEATURES.map((_,i) => (modelSlopes[i] * vals[i]).toFixed(4));

  let formulaHtml = `<strong>Solution:</strong><br><br>`;
  formulaHtml += `ŷ = m₁x₁ + m₂x₂ + m₃x₃ + m₄x₄ + m₅x₅ + m₆x₆ + m₇x₇ + m₈x₈ + b<br><br>`;
  formulaHtml += `ŷ = ${parts.join(' + ')} + ${fmt(modelB)}<br><br>`;
  formulaHtml += `ŷ = ${numeric.join(' + ')} + ${fmt(modelB)}<br><br>`;
  formulaHtml += `<strong>ŷ = ${raw.toFixed(4)}</strong><br><br>`;
  formulaHtml += `Centered z = (${raw.toFixed(4)} − ${rawMean.toFixed(4)}) / ${rawStd.toFixed(4)} = <strong>${((raw-rawMean)/rawStd).toFixed(4)}</strong><br><br>`;
  formulaHtml += `σ(z) = 1 / (1 + e<sup>−z</sup>) = <strong>${prob.toFixed(4)}</strong><br><br>`;
  formulaHtml += `<small>Since σ(z) = ${prob.toFixed(4)} ${prob >= 0.5 ? '≥' : '<'} 0.5 → Predicted = <strong>${outcome}</strong></small>`;

  document.getElementById('predFormula').innerHTML = formulaHtml;
  document.getElementById('predAnswer').innerHTML  = `
    <span class="ans-number">${outcome}</span>
    <span class="ans-label">${outcome === 0 ? '= No Diabetes' : '= Has Diabetes'}</span>
  `;
  show('predResult');
}

// ───────────────────────────────────────────
//  HELPERS
// ───────────────────────────────────────────
function show(id) { document.getElementById(id).classList.remove('hidden'); }
function hide(id) { document.getElementById(id).classList.add('hidden'); }