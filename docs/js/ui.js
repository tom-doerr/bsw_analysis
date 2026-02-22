// UI controls and info panel
import { PARTIES, METRICS } from './data.js';
import { PARTY_COLORS } from './colors.js';

export function initUI(onChange) {
  const sel = document.getElementById('party-select');
  PARTIES.forEach(p => {
    const opt = document.createElement('option');
    opt.value = p; opt.textContent = p;
    sel.appendChild(opt);
  });
  sel.addEventListener('change', onChange);

  // Metric radios
  const radDiv = document.getElementById('metric-radios');
  radDiv.className = 'radio-group';
  METRICS.forEach((m, i) => {
    const label = document.createElement('label');
    const inp = document.createElement('input');
    inp.type = 'radio'; inp.name = 'metric';
    inp.value = m.id;
    if (i === 0) inp.checked = true;
    inp.addEventListener('change', onChange);
    label.appendChild(inp);
    label.append(' ' + m.label);
    radDiv.appendChild(label);
  });
  // Extrude toggle
  document.getElementById('extrude-toggle')
    .addEventListener('change', onChange);
}

export function getState() {
  return {
    party: document.getElementById('party-select').value,
    metric: document.querySelector(
      'input[name="metric"]:checked')?.value || 'share',
    extrude: document.getElementById('extrude-toggle').checked,
  };
}

export function showTooltip(x, y, wkrData, metric, party) {
  const el = document.getElementById('tooltip');
  if (!wkrData) { el.classList.add('hidden'); return; }
  let val;
  if (metric === 'share') val = (wkrData.shares[party]||0).toFixed(1)+'%';
  else if (metric === 'resid') val = (wkrData.resid[party]||0).toFixed(2)+'pp';
  else val = wkrData.turnout.toFixed(1) + '%';
  const lbl = metric === 'turnout' ? 'Turnout' : party;
  el.innerHTML = `<b>${wkrData.name}</b><br>${lbl}: ${val}`;
  el.style.left = (x + 15) + 'px';
  el.style.top = (y + 15) + 'px';
  el.classList.remove('hidden');
}

export function showInfoPanel(d) {
  const el = document.getElementById('info-panel');
  if (!d) { el.classList.add('hidden'); return; }
  let html = `<h2>WKR ${d.wkr}</h2>`;
  html += `<div class="land">${d.name} (${d.land})</div>`;
  html += `<p style="margin:8px 0;font-size:12px">`;
  html += `Turnout: ${d.turnout.toFixed(1)}% | `;
  html += `Precincts: ${d.n_precincts}</p>`;
  html += '<h3 style="font-size:13px;margin:10px 0 6px">';
  html += 'Zweitstimme Shares</h3>';
  const sorted = Object.entries(d.shares)
    .sort((a,b) => b[1] - a[1]);
  const maxShare = sorted[0]?.[1] || 1;
  for (const [p, v] of sorted) {
    if (v < 0.1) continue;
    const pct = (v / maxShare * 100).toFixed(0);
    const c = PARTY_COLORS[p] || '#666';
    html += `<div class="bar-row">`;
    html += `<span class="bar-label">${p}</span>`;
    html += `<span class="bar-bg"><span class="bar-fill" `;
    html += `style="width:${pct}%;background:${c}">`;
    html += `</span></span>`;
    html += `<span class="bar-val">${v.toFixed(1)}%</span>`;
    html += `</div>`;
  }
  // Residuals section
  html += '<h3 style="font-size:13px;margin:10px 0 6px">';
  html += 'Model Residuals (pp)</h3>';
  for (const [p, v] of Object.entries(d.resid)) {
    if (Math.abs(v) < 0.01) continue;
    const sign = v > 0 ? '+' : '';
    html += `<div style="font-size:12px">${p}: `;
    html += `${sign}${v.toFixed(2)}</div>`;
  }
  el.innerHTML = html;
  el.classList.remove('hidden');
}

export function updateLegend(min, max) {
  const el = document.getElementById('legend');
  const isDiverging = min < 0;
  let cvs = el.querySelector('canvas');
  if (!cvs) {
    cvs = document.createElement('canvas');
    cvs.width = 200; cvs.height = 20;
    el.appendChild(cvs);
  }
  const ctx = cvs.getContext('2d');
  for (let i = 0; i < cvs.width; i++) {
    const t = i / (cvs.width - 1);
    ctx.fillStyle = isDiverging
      ? _divCSS(t*2-1) : _seqCSS(t);
    ctx.fillRect(i, 0, 1, 20);
  }
  let labels = el.querySelector('.legend-labels');
  if (!labels) {
    labels = document.createElement('div');
    labels.className = 'legend-labels';
    el.appendChild(labels);
  }
  labels.innerHTML =
    `<span>${min.toFixed(1)}</span>` +
    `<span>${max.toFixed(1)}</span>`;
}

function _seqCSS(t) {
  const r = Math.round(255*(1-t*0.7));
  const g = Math.round(255*(1-t*0.85));
  const b = Math.round(255*(1-t*0.3));
  return `rgb(${r},${g},${b})`;
}

function _divCSS(t) {
  if (t < 0) {
    const s = -t;
    const r = Math.round(255*(1-s*0.67));
    const g = Math.round(255*(1-s*0.6));
    return `rgb(${r},${g},255)`;
  }
  const g = Math.round(255*(1-t*0.87));
  const b = Math.round(255*(1-t*0.84));
  return `rgb(255,${g},${b})`;
}

export function initChips(summary) {
  const el = document.getElementById('chips');
  const bsw = summary.parties.BSW;
  const chips = [
    `R²=${bsw.xgb_r2} (XGB)`,
    `${summary.n_precincts.toLocaleString()} precincts`,
    `${summary.n_features_xgb} features`,
  ];
  el.innerHTML = chips.map(
    c => `<span class="chip">${c}</span>`).join('');
}
