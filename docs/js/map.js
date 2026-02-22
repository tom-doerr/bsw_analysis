// GeoJSON to Three.js mesh conversion
import * as THREE from 'three';

// Mercator projection centered on Germany
const CENTER = [10.4, 51.1];
const SCALE = 6;

function project(lon, lat) {
  const x = (lon - CENTER[0]) * SCALE *
    Math.cos(CENTER[1] * Math.PI / 180);
  const z = -(lat - CENTER[1]) * SCALE;
  return [x, z];
}

function ringToShape(ring) {
  const shape = new THREE.Shape();
  ring.forEach(([lon, lat], i) => {
    const [x, z] = project(lon, lat);
    if (i === 0) shape.moveTo(x, z);
    else shape.lineTo(x, z);
  });
  return shape;
}

function polyToShapes(geometry) {
  const shapes = [];
  let polys;
  if (geometry.type === 'Polygon') {
    polys = [geometry.coordinates];
  } else if (geometry.type === 'MultiPolygon') {
    polys = geometry.coordinates;
  } else return shapes;

  for (const poly of polys) {
    const outer = ringToShape(poly[0]);
    // Add holes
    for (let h = 1; h < poly.length; h++) {
      const hole = new THREE.Path();
      poly[h].forEach(([lon, lat], i) => {
        const [x, z] = project(lon, lat);
        if (i === 0) hole.moveTo(x, z);
        else hole.lineTo(x, z);
      });
      outer.holes.push(hole);
    }
    shapes.push(outer);
  }
  return shapes;
}

export function buildMap(geoData) {
  const group = new THREE.Group();
  const meshes = {};

  for (const feat of geoData.features) {
    const wkr = feat.properties.wkr;
    const shapes = polyToShapes(feat.geometry);
    if (shapes.length === 0) continue;

    const extArgs = { depth: 0.1, bevelEnabled: false };
    const geom = new THREE.ExtrudeGeometry(shapes, extArgs);
    geom.rotateX(-Math.PI / 2);
    const mat = new THREE.MeshPhongMaterial({
      color: 0x444466,
      side: THREE.DoubleSide,
    });
    const mesh = new THREE.Mesh(geom, mat);
    mesh.userData.wkr = wkr;
    mesh.userData.name = feat.properties.name;
    meshes[wkr] = mesh;
    group.add(mesh);
  }
  return { group, meshes };
}

export function updateMap(meshes, geoData, wkrData,
    party, metric, extrude) {
  // Compute value range
  const vals = wkrData.map(d => {
    if (metric === 'share') return d.shares[party] || 0;
    if (metric === 'resid') return d.resid[party] || 0;
    if (metric === 'swing') return (d.swing||{})[party] || 0;
    if (metric === 'turnout') return d.turnout || 0;
    return 0;
  });
  const mn = Math.min(...vals);
  const mx = Math.max(...vals);
  const range = mx - mn || 1;
  const isDiverging = metric === 'resid' || metric === 'swing';

  // Rebuild each mesh
  for (const feat of geoData.features) {
    const wkr = feat.properties.wkr;
    const mesh = meshes[wkr];
    if (!mesh) continue;
    const d = wkrData.find(w => w.wkr === wkr);
    let v = 0;
    if (d) {
      if (metric === 'share') v = d.shares[party] || 0;
      else if (metric === 'resid') v = d.resid[party] || 0;
      else if (metric === 'swing') v = (d.swing||{})[party] || 0;
      else if (metric === 'turnout') v = d.turnout || 0;
    }
    const t = (v - mn) / range;
    // Color
    let color;
    if (isDiverging) {
      const absMax = Math.max(Math.abs(mn), Math.abs(mx));
      const tn = v / (absMax || 1);
      color = divColor(tn);
    } else {
      color = seqColor(t);
    }
    mesh.material.color.set(color);
    // Height
    const h = extrude ? 0.1 + t * 3 : 0.1;
    mesh.scale.set(1, h / 0.1, 1);
    mesh.position.y = 0;
  }
  return { min: mn, max: mx };
}

function seqColor(t) {
  const r = Math.round(255 * (1 - t * 0.7));
  const g = Math.round(255 * (1 - t * 0.85));
  const b = Math.round(255 * (1 - t * 0.3));
  return (r << 16) | (g << 8) | b;
}

function divColor(t) {
  // t in [-1,1]: blue(-) white(0) red(+)
  if (t < 0) {
    const s = -t;
    const r = Math.round(255 * (1 - s * 0.67));
    const g = Math.round(255 * (1 - s * 0.6));
    return (r << 16) | (g << 8) | 255;
  }
  const r = 255;
  const g = Math.round(255 * (1 - t * 0.87));
  const b = Math.round(255 * (1 - t * 0.84));
  return (r << 16) | (g << 8) | b;
}
