

const FEATURES = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'];
const SUB  = ['₁','₂','₃','₄','₅','₆','₇','₈'];

let dataset = [];
let modelSlopes = [];
let modelIntercepts = [];
let modelB = 0;
let modelTrained = false;
let rawMean = 0;
let rawStd = 1;


function show(id) { document.getElementById(id).classList.remove('hidden'); }
function fmt(n) { return n.toFixed(4); }

// MANUAL CSV LOADING 
document.getElementById('csvFile').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function(event) {
        const lines = event.target.result.trim().split('\n').map(l => l.trim()).filter(Boolean);
        const header = lines[0].split(',').map(h => h.trim());
        dataset = [];

        for (let i = 1; i < lines.length; i++) {
            const vals = lines[i].split(',').map(v => parseFloat(v.trim()));
            if (vals.length < 9 || vals.some(isNaN)) continue;
            const row = {};
            header.forEach((h, j) => row[h] = vals[j]);
            dataset.push(row);
        }

        document.getElementById('uploadMsg').innerHTML = `<div class="upload-msg ok">✔ ${dataset.length} rows loaded.</div>`;
        document.getElementById('uploadMsg').classList.remove('hidden');
        show('sec-table'); show('sec-train');
        renderDataTable();
        document.getElementById('lblRowCount').textContent = `${dataset.length} total rows`;
    };
    reader.readAsText(file);
});

// HISTORICAL DATA TABLE 
function renderDataTable() {
    const sel = document.getElementById('selRows').value;
    const rows = sel === 'all' ? dataset : dataset.slice(0, parseInt(sel));
    
    const means = {};
    [...FEATURES, 'Outcome'].forEach(col => {
        const sum = dataset.reduce((a, b) => a + (b[col] || 0), 0);
        means[col] = sum / dataset.length;
    });

    let html = `<table class="data-tbl"><thead><tr><th>#</th>`;
    FEATURES.forEach((f, i) => html += `<th>${f}<br><small>(x${SUB[i]})</small></th>`);
    html += `<th class="col-y">Outcome<br><small>(y)</small></th></tr></thead><tbody>`;

    rows.forEach((r, i) => {
        html += `<tr><td>${i+1}</td>`;
        FEATURES.forEach(f => html += `<td>${r[f]}</td>`);
        html += `<td class="col-y">${r['Outcome']}</td></tr>`;
    });

    html += `<tr class="row-mean"><td>Mean</td>`;
    FEATURES.forEach(f => html += `<td>${means[f].toFixed(4)}</td>`);
    html += `<td class="col-y">${means['Outcome'].toFixed(4)}</td></tr></tbody></table>`;

    document.getElementById('dataTableArea').innerHTML = html;
}

// FEATURE COMPUTATION
function computeModel() {
    if (dataset.length === 0) return;
    const area = document.getElementById('computationArea');
    area.innerHTML = '';
    
    const yVals = dataset.map(r => r['Outcome']);
    const yBar = yVals.reduce((a, b) => a + b, 0) / yVals.length;
    
    modelSlopes = [];
    modelIntercepts = [];

    FEATURES.forEach((feat, fi) => {
        const xVals = dataset.map(r => r[feat]);
        const xBar = xVals.reduce((a, b) => a + b, 0) / xVals.length;
        
        let sumXY = 0;
        let sumX2 = 0;
        let rowsHtml = '';

        for (let i = 0; i < dataset.length; i++) {
            const dx = xVals[i] - xBar;
            const dy = yVals[i] - yBar;
            sumXY += dx * dy;
            sumX2 += dx * dx;
            
            if (i < 10) {
                rowsHtml += `<tr>
                    <td>${fmt(xVals[i])}</td><td>${fmt(yVals[i])}</td>
                    <td>${fmt(dx)}</td><td>${fmt(dy)}</td>
                    <td>${fmt(dx * dy)}</td><td>${fmt(dx * dx)}</td>
                </tr>`;
            }
        }
        
        const m = sumX2 !== 0 ? sumXY / sumX2 : 0;
        const b = yBar - (m * xBar);
        modelSlopes.push(m);
        modelIntercepts.push(b);

        area.innerHTML += `
            <div class="comp-block">
                <div class="comp-title">
                    <span>Feature: <strong>${feat}</strong> (x${SUB[fi]})</span>
                    <div class="slope-badge">m${SUB[fi]} = ${fmt(m)}</div>
                </div>
                <table class="comp-tbl">
                    <thead><tr><th>x${SUB[fi]}</th><th>y</th><th>x-x̄</th><th>y-ȳ</th><th>(x-x̄)(y-ȳ)</th><th>(x-x̄)²</th></tr></thead>
                    <tbody>
                        ${rowsHtml}
                        <tr class="row-mean">
                            <td>${fmt(xBar)}<br><small>x̄</small></td>
                            <td>${fmt(yBar)}<br><small>ȳ</small></td>
                            <td></td><td></td>
                            <td>${fmt(sumXY)}<br><small>Total</small></td>
                            <td>${fmt(sumX2)}<br><small>Total</small></td>
                        </tr>
                    </tbody>
                </table>
                <div class="intercept-row">Y-Intercept b${SUB[fi]} = ${fmt(b)}</div>
            </div>`;
    });

    modelB = modelIntercepts.reduce((a, b) => a + b, 0) / modelIntercepts.length;
    

    const rawScores = dataset.map(r => FEATURES.reduce((acc, f, i) => acc + (modelSlopes[i] * r[f]), 0) + modelB);
    rawMean = rawScores.reduce((a, b) => a + b, 0) / rawScores.length;
    rawStd = Math.sqrt(rawScores.map(v => Math.pow(v - rawMean, 2)).reduce((a, b) => a + b, 0) / dataset.length) || 1;

    renderFinalModelUI();
    buildAccuracyTable();
    show('sec-predict');
    show('finalModelArea');
    modelTrained = true;
}

// OVERALL MODEL & ACCURACY 
function renderFinalModelUI() {
    const bSumStr = modelIntercepts.map(b => fmt(b)).join(' + ');
    const eqParts = FEATURES.map((f, i) => `(${fmt(modelSlopes[i])})x${SUB[i]}`).join(' + ');

    document.getElementById('finalModelArea').innerHTML = `
        <div class="overall-box">
            <div class="overall-title">Computation of the Overall Y-Intercept</div>
            <div class="overall-inner"><div class="formula-box">b = (${bSumStr}) / 8 = ${fmt(modelB)}</div></div>
        </div>
        <div class="final-model-box" style="margin-top:24px">
            <div class="final-model-title">Final Multivariate Model (ŷ)</div>
            <div class="final-model-inner"><div class="formula-box">ŷ = ${eqParts} + ${fmt(modelB)}</div></div>
        </div>`;
}

function buildAccuracyTable() {
    let correct = 0;
    const total = dataset.length;
    
    const results = dataset.map((r) => {
        let raw = FEATURES.reduce((acc, f, j) => acc + (r[f] * modelSlopes[j]), 0) + modelB;
        let z = (raw - rawMean) / rawStd;
        let sigmoid = 1 / (1 + Math.exp(-z));
        let pred = sigmoid >= 0.5 ? 1 : 0;
        if (pred === parseInt(r['Outcome'])) correct++;
        return { ...r, raw, sigmoid, pred, match: pred === parseInt(r['Outcome']) };
    });

    const accuracy = (correct / total) * 100;

    let html = `
        <div class="accuracy-header" style="background:var(--navy); color:white; padding:12px 20px; display:flex; justify-content:space-between;">
            <span>Prediction Results vs Actual Outcome</span>
            <span>Accuracy: ${accuracy.toFixed(2)}% (${correct}/${total} correct)</span>
        </div>
        <div class="accuracy-stats" style="display:flex; justify-content:space-around; padding:20px; background:#fafafa; border:1px solid #ccc; border-top:none;">
            <div style="text-align:center;"><div style="font-size:11px;">TOTAL</div><div style="font-size:22px; font-weight:700;">${total}</div></div>
            <div style="text-align:center;"><div style="font-size:11px;">CORRECT</div><div style="font-size:22px; font-weight:700;">${correct}</div></div>
            <div style="text-align:center;"><div style="font-size:11px;">ACCURACY</div><div style="font-size:22px; font-weight:700;">${accuracy.toFixed(2)}%</div></div>
        </div>
        <div class="tbl-scroll" style="overflow-x:auto; margin-top:10px;"><table class="data-tbl" style="min-width:1200px;">
            <thead><tr><th>#</th>${FEATURES.map(f => `<th>${f.toUpperCase()}</th>`).join('')}<th>ACTUAL</th><th>ŷ SCORE</th><th>SIGMOID</th><th>PRED</th></tr></thead>
            <tbody>${results.slice(0, 10).map((r, i) => `<tr><td>${i+1}</td>${FEATURES.map(f => `<td>${r[f]}</td>`).join('')}<td>${r.Outcome}</td><td>${fmt(r.raw)}</td><td>${fmt(r.sigmoid)}</td><td>${r.pred}</td></tr>`).join('')}</tbody>
        </table></div>`;
    document.getElementById('accuracyArea').innerHTML = html;
    show('accuracyArea');
}

// FINAL PREDICTION
function makePrediction() {
    if (!modelTrained) return;
    const inputs = [];
    for (let i = 1; i <= 8; i++) {
        let v = parseFloat(document.getElementById(`p${i}`).value);
        if (isNaN(v)) { alert("Please fill all fields"); return; }
        inputs.push(v);
    }

    let subScores = FEATURES.map((f, i) => inputs[i] * modelSlopes[i]);
    let rawScore = subScores.reduce((a, b) => a + b, 0) + modelB;
    let z = (rawScore - rawMean) / rawStd;
    let sigmoid = 1 / (1 + Math.exp(-z));
    let finalPred = sigmoid >= 0.5 ? 1 : 0;

    document.getElementById('predFormula').innerHTML = `
        <div style="font-family: monospace; line-height: 1.6;">
            <strong>Solution Breakdown:</strong><br><br>
            ŷ = Σ(mₙ * xₙ) + b<br>
            ŷ = ${subScores.map(s => fmt(s)).join(' + ')} + ${fmt(modelB)}<br>
            <strong>ŷ = ${fmt(rawScore)}</strong><br><br>
            Centered z = ${fmt(z)}<br>
            σ(z) = ${fmt(sigmoid)}<br>
            Result: <strong>${finalPred}</strong>
        </div>
    `;

    document.getElementById('predAnswer').innerHTML = `
        <div style="text-align:center; padding: 20px; font-size: 24px; font-weight: bold; background: #e6f4f1; border: 2px solid #0d6e56; color: #0d6e56;">
            ${finalPred}<br>
            <span style="font-size: 16px;">= ${finalPred === 1 ? 'Diabetes' : 'No Diabetes'}</span>
        </div>
    `;
    show('predResult');
}
