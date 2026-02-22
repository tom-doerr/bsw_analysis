// Color scale utilities
export const PARTY_COLORS = {
  BSW: '#b5179e', AfD: '#009ee0',
  CDU: '#000000', SPD: '#e3000f',
  'GRÜNE': '#64a12d', CSU: '#008ac5',
  FDP: '#ffed00', 'Die Linke': '#be3075',
  'FREIE WÄHLER': '#f7a800',
};

// Interpolate between two hex colors
function lerpColor(a, b, t) {
  const pa = [
    parseInt(a.slice(1,3),16),
    parseInt(a.slice(3,5),16),
    parseInt(a.slice(5,7),16)];
  const pb = [
    parseInt(b.slice(1,3),16),
    parseInt(b.slice(3,5),16),
    parseInt(b.slice(5,7),16)];
  const r = pa.map((v,i) =>
    Math.round(v + (pb[i]-v) * t));
  return (r[0]<<16) | (r[1]<<8) | r[2];
}

// Sequential scale (white → party color)
export function seqScale(t, party) {
  const hex = PARTY_COLORS[party] || '#b5179e';
  return lerpColor('#ffffff', hex, Math.max(0, Math.min(1, t)));
}

// Diverging scale (blue → white → red)
export function divScale(t) {
  // t in [-1, 1], 0 = white
  if (t < 0) return lerpColor('#ffffff', '#2166ac', -t);
  return lerpColor('#ffffff', '#b2182b', t);
}

// Get min/max for legend
export function getRange(data, key, party) {
  const vals = data.map(d => {
    if (key === 'share') return d.shares[party] || 0;
    if (key === 'resid') return d.resid[party] || 0;
    if (key === 'turnout') return d.turnout;
    return 0;
  });
  return [Math.min(...vals), Math.max(...vals)];
}
