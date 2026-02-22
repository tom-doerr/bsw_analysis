// Data loading and lookup
let wkrData = [];
let geoData = null;
let summary = null;
let wkrMap = {};

let shapData = null;

export async function loadAll() {
  const [wd, geo, sum] = await Promise.all([
    fetch('data/wkr_data.json').then(r => r.json()),
    fetch('data/wahlkreise.json').then(r => r.json()),
    fetch('data/summary.json').then(r => r.json()),
  ]);
  wkrData = wd;
  geoData = geo;
  summary = sum;
  wkrData.forEach(d => { wkrMap[d.wkr] = d; });
  try {
    const r = await fetch('data/shap_summary.json');
    if (r.ok) shapData = await r.json();
  } catch(e) {}
  return { wkrData, geoData, summary };
}

export function getShap() { return shapData; }

export function getWkr(wkrNr) { return wkrMap[wkrNr]; }
export function getAllWkr() { return wkrData; }
export function getSummary() { return summary; }
export function getGeo() { return geoData; }

export function getValue(wkrNr, metric, party) {
  const d = wkrMap[wkrNr];
  if (!d) return 0;
  if (metric === 'share') return d.shares[party] || 0;
  if (metric === 'resid') return d.resid[party] || 0;
  if (metric === 'swing') return (d.swing||{})[party] || 0;
  if (metric === 'turnout') return d.turnout || 0;
  return 0;
}

export const PARTIES = [
  'BSW', 'AfD', 'CDU', 'SPD', 'GRÜNE',
  'CSU', 'FDP', 'Die Linke', 'FREIE WÄHLER',
];

export const METRICS = [
  { id: 'share', label: 'Vote Share (%)' },
  { id: 'resid', label: 'Model Residual (pp)' },
  { id: 'swing', label: 'EW24→BTW25 Swing (pp)' },
  { id: 'turnout', label: 'Turnout (%)' },
];
