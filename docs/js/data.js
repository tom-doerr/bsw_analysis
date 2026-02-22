// Data loading and lookup
let wkrData = [];
let geoData = null;
let summary = null;
let wkrMap = {};

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
  return { wkrData, geoData, summary };
}

export function getWkr(wkrNr) { return wkrMap[wkrNr]; }
export function getAllWkr() { return wkrData; }
export function getSummary() { return summary; }
export function getGeo() { return geoData; }

export function getValue(wkrNr, metric, party) {
  const d = wkrMap[wkrNr];
  if (!d) return 0;
  if (metric === 'share') return d.shares[party] || 0;
  if (metric === 'resid') return d.resid[party] || 0;
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
  { id: 'turnout', label: 'Turnout (%)' },
];
