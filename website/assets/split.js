// A model too big for the printer, cut into pieces that each fit, with holes for pins (dowels) across every cut so
// the pieces line up when glued. All in the browser, with Manifold (assets/vendor/manifold-3.5.4, Apache-2.0): it
// cuts and drills a closed surface and keeps it closed, so every piece is a solid a slicer prints as it is.
(function () {
  const LIB = new URL("vendor/manifold-3.5.4/manifold.js", document.currentScript.src).href;
  const MAX_CUTS = 8; // pieces along one side; more would be a puzzle, not a print
  let lib = null;

  function load() {
    return lib || (lib = import(LIB).then(async (m) => {
      const w = await m.default();
      w.setup();
      return w;
    }).catch((err) => { lib = null; throw err; }));
  }

  // Pins: thick enough not to snap, holes a little wider so they slide in. Walls: plastic left around a hole.
  const PIN = {
    fdm: { d: 5, dBig: 6, depth: 8, play: 0.3, wall: 2 },
    resin: { d: 4, dBig: 4, depth: 6, play: 0.2, wall: 1.5 },
  };

  // How many pieces along x and y for a footprint: as few as possible, each fitting the plate (or turned a quarter).
  function grid(w, d, plate) {
    let best = null;
    for (let nx = 1; nx <= MAX_CUTS; nx++) for (let ny = 1; ny <= MAX_CUTS; ny++) {
      const a = w / nx, b = d / ny;
      if (!((a <= plate.x && b <= plate.y) || (a <= plate.y && b <= plate.x))) continue;
      if (!best || nx * ny < best[0] * best[1]) best = [nx, ny];
    }
    return best;
  }

  // The model's box in mm at this size.
  function dims(model, size) {
    const e = [0, 1, 2].map((k) => model.max[k] - model.min[k]), s = size / Math.max(...e);
    return e.map((v) => v * s);
  }

  const OTHER = [[1, 2], [2, 0], [0, 1]];
  // Axis a turned to z, so a cut across it is a slice: (u, v, w) = the other two axes, then a. Column-major.
  function toZ(a) {
    const [u, v] = OTHER[a], m = new Array(16).fill(0);
    m[u * 4] = 1; m[v * 4 + 1] = 1; m[a * 4 + 2] = 1; m[15] = 1;
    return m;
  }

  function inside(polys, x, y) {
    let c = false;
    for (const poly of polys) for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const [xi, yi] = poly[i], [xj, yj] = poly[j];
      if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) c = !c;
    }
    return c;
  }

  // Up to two spots in a region of the cut (where a pin fits with plastic all around): the middle, or for a long
  // region two spots a quarter in from each end, so the pieces cannot twist.
  function spots(island) {
    const polys = island.toPolygons(), b = island.bounds();
    const w = b.max[0] - b.min[0], h = b.max[1] - b.min[1], step = Math.max(0.5, Math.max(w, h) / 60);
    const pts = [];
    for (let x = b.min[0]; x <= b.max[0]; x += step) for (let y = b.min[1]; y <= b.max[1]; y += step)
      if (inside(polys, x, y)) pts.push([x, y]);
    if (!pts.length) return [];
    const cx = (b.min[0] + b.max[0]) / 2, cy = (b.min[1] + b.max[1]) / 2;
    const near = (tx, ty) => pts.reduce((p, q) => (Math.hypot(q[0] - tx, q[1] - ty) < Math.hypot(p[0] - tx, p[1] - ty) ? q : p));
    if (Math.max(w, h) < 30) return [near(cx, cy)];
    const two = w >= h ? [near(b.min[0] + w / 4, cy), near(b.max[0] - w / 4, cy)] : [near(cx, b.min[1] + h / 4), near(cx, b.max[1] - h / 4)];
    return Math.hypot(two[0][0] - two[1][0], two[0][1] - two[1][1]) > 10 ? two : [two[0]];
  }

  // The plan, without cutting yet (quick, for the page to show): the model in mm (Supports.mm: centred on x and y,
  // standing on z = 0) cut into equal layers that fit the printer's height, then each layer into as few pieces as its
  // own footprint needs (a thin stem under a wide top stays whole).
  function layoutFor(model, size, plate) {
    const P = Supports.mm(model, size), T = model.tri;
    let top = 0;
    for (let i = 2; i < P.length; i += 3) top = Math.max(top, P[i]);
    const nz = Math.ceil(top / plate.z - 1e-9);
    if (nz > MAX_CUTS) return { count: 0 };
    const zs = Array.from({ length: nz + 1 }, (_, k) => (top * k) / nz);
    const slabs = [];
    for (let k = 0; k < nz; k++) {
      const z0 = zs[k], z1 = zs[k + 1], lo = [Infinity, Infinity], hi = [-Infinity, -Infinity];
      // Every corner of a triangle reaching into the layer: a little more than the cut piece, never less.
      for (let t = 0; t < T.length; t += 3) {
        const a = T[t] * 3, b = T[t + 1] * 3, c = T[t + 2] * 3;
        if (Math.max(P[a + 2], P[b + 2], P[c + 2]) < z0 || Math.min(P[a + 2], P[b + 2], P[c + 2]) > z1) continue;
        for (const v of [a, b, c]) for (let q = 0; q < 2; q++) { lo[q] = Math.min(lo[q], P[v + q]); hi[q] = Math.max(hi[q], P[v + q]); }
      }
      if (lo[0] > hi[0]) continue;
      const size2 = [hi[0] - lo[0], hi[1] - lo[1]], n = grid(size2[0], size2[1], plate);
      if (!n) return { count: 0 };
      const cuts = n.map((m, q) => Array.from({ length: m - 1 }, (_, i) => lo[q] + (size2[q] * (i + 1)) / m));
      slabs.push({ k, z0, z1, lo, hi, n, cuts });
    }
    return { top, zcuts: zs.slice(1, -1), slabs, count: slabs.reduce((n, s) => n + s.n[0] * s.n[1], 0) };
  }

  async function split(model, size, plate, kindName) {
    const w = await load();
    const { Manifold, Mesh, CrossSection } = w;
    const L = layoutFor(model, size, plate);
    if (!L.count) throw Object.assign(new Error("too many pieces"), { code: "many" });
    const P = PIN[kindName] || PIN.fdm, pinD = size >= 250 ? P.dBig : P.d;
    const r = pinD / 2, holeR = r + P.play / 2, keep = holeR + P.wall;
    const trash = [];
    const gc = (m) => (trash.push(m), m);

    let solid;
    try {
      const mesh = new Mesh({ numProp: 3, vertProperties: Supports.mm(model, size), triVerts: model.tri });
      mesh.merge();
      solid = gc(new Manifold(mesh));
    } catch { solid = null; }
    if (!solid || solid.status() !== "NoError" || solid.isEmpty())
      throw Object.assign(new Error("not a closed surface"), { code: "open" });

    // Where pins go: on each cut, in each share of it between two pieces (edges: the piece edges along the other
    // two axes), where there is plastic all around the hole for the whole depth on both sides.
    const dowels = [], turned = [];
    const edgesOf = (s, q) => (q === 2 ? [s.z0, s.z1] : [s.lo[q], ...s.cuts[q], s.hi[q]]);
    const merge = (a, b) => [...new Set([...a, ...b].map((v) => Math.round(v * 1000) / 1000))].sort((x, y) => x - y);
    // The pieces' lengths on either side of cut c (edges include the cuts themselves).
    const width = (edges, c) => [c - Math.max(...edges.filter((e) => e < c - 1e-6)), Math.min(...edges.filter((e) => e > c + 1e-6)) - c];
    function place(a, c, edges, room) {
      const depth = Math.min(P.depth, room - P.wall);
      if (depth < 4) return;
      const t = turned[a] || (turned[a] = gc(solid.transform(toZ(a)))), [u, v] = OTHER[a];
      let area = null;
      for (const z of [c, c - depth / 2, c + depth / 2, c - depth - P.wall / 2, c + depth + P.wall / 2]) {
        const sec = gc(gc(t.slice(z)).offset(-keep, "Round"));
        area = area ? gc(area.intersect(sec)) : sec;
      }
      for (let i = 0; i + 1 < edges[u].length; i++) for (let j = 0; j + 1 < edges[v].length; j++) {
        const w0 = edges[u][i + 1] - edges[u][i] - 2 * keep, h0 = edges[v][j + 1] - edges[v][j] - 2 * keep;
        if (w0 <= 0 || h0 <= 0) continue;
        const box = gc(gc(CrossSection.square([w0, h0])).translate([edges[u][i] + keep, edges[v][j] + keep]));
        for (const island of gc(area.intersect(box)).decompose()) {
          gc(island);
          if (island.area() < 1) continue;
          for (const [x, y] of spots(island)) {
            const p = [0, 0, 0];
            p[u] = x; p[v] = y; p[a] = c;
            dowels.push({ axis: a, at: p, depth });
          }
        }
      }
    }
    for (const sl of L.slabs) for (const a of [0, 1]) for (const c of sl.cuts[a]) {
      const e = [edgesOf(sl, 0), edgesOf(sl, 1), edgesOf(sl, 2)];
      place(a, c, e, Math.min(...width(e[a], c)) / 2);
    }
    L.zcuts.forEach((c, k) => {
      const below = L.slabs.find((s) => s.k === k), above = L.slabs.find((s) => s.k === k + 1);
      if (!below || !above) return;
      const e = [merge(edgesOf(below, 0), edgesOf(above, 0)), merge(edgesOf(below, 1), edgesOf(above, 1)), [below.z0, above.z1]];
      place(2, c, e, Math.min(below.z1 - below.z0, above.z1 - above.z0) / 2);
    });

    // Holes: one cylinder through each cut, half in each piece.
    if (dowels.length) {
      const holes = dowels.map(({ axis, at, depth }) => {
        let h = gc(Manifold.cylinder(2 * depth, holeR, holeR, 24, true));
        if (axis === 0) h = gc(h.rotate([0, 90, 0]));
        if (axis === 1) h = gc(h.rotate([90, 0, 0]));
        return gc(h.translate(at));
      });
      solid = gc(solid.subtract(gc(Manifold.union(holes))));
    }

    // Cut: into layers, then each layer along x, then each strip along y.
    const along = (m, a, cuts) => {
      const out = [];
      let rest = m;
      for (const c of cuts) {
        const normal = [0, 0, 0];
        normal[a] = 1;
        const [above, below] = rest.splitByPlane(normal, c);
        out.push(gc(below));
        rest = gc(above);
      }
      out.push(rest);
      return out;
    };
    const pieces = [];
    along(solid, 2, L.zcuts).forEach((layer, k) => {
      const sl = L.slabs.find((s) => s.k === k);
      if (!sl) return;
      along(layer, 0, sl.cuts[0]).forEach((strip, i) =>
        along(strip, 1, sl.cuts[1]).forEach((m, j) => pieces.push({ m, idx: [i, j, k], n: [sl.n[0], sl.n[1], L.slabs.length] })));
    });

    const parts = [];
    for (const p of pieces) {
      if (p.m.isEmpty() || p.m.volume() < 1) continue;
      const g = p.m.getMesh(), np = g.numProp, nv = g.vertProperties.length / np;
      const pos = new Float32Array(nv * 3);
      for (let i = 0; i < nv; i++) for (let k = 0; k < 3; k++) pos[i * 3 + k] = g.vertProperties[i * np + k];
      const b = p.m.boundingBox();
      parts.push({ idx: p.idx, n: p.n, pos, tri: Uint32Array.from(g.triVerts), min: Array.from(b.min), max: Array.from(b.max), volume: p.m.volume() });
    }
    // Bottom layer first, then front to back, left to right.
    parts.sort((p, q) => p.idx[2] - q.idx[2] || p.idx[1] - q.idx[1] || p.idx[0] - q.idx[0]);
    parts.forEach((p, i) => (p.number = i + 1));

    const pins = dowels.length ? pinMesh(w, dowels, r, P) : null;
    for (const m of trash) try { m.delete(); } catch {}
    return { ...L, dims: dims(model, size), parts, dowels, pins, pinD, holeR };
  }

  // The pins, lying down in rows (stronger than standing: no layer lines across them), with a flat strip along the
  // bottom so they stay put, chamfered ends to find the hole, and two spares.
  function pinMesh(w, dowels, r, P) {
    const { Manifold } = w;
    const trash = [], gc = (m) => (trash.push(m), m);
    const len = Math.min(...dowels.map((d) => d.depth)) * 2 - 1, ch = 0.6, flat = 0.4;
    const body = gc(Manifold.cylinder(len - 2 * ch, r, r, 24, true));
    const tip = gc(Manifold.cylinder(ch, r, r - ch, 24));
    const pin = gc(Manifold.union([body, gc(tip.translate([0, 0, len / 2 - ch])), gc(gc(tip.mirror([0, 0, 1])).translate([0, 0, -(len / 2 - ch)]))]));
    const lying = gc(gc(gc(pin.rotate([0, 90, 0])).trimByPlane([0, 0, 1], -(r - flat))).translate([0, 0, r - flat]));
    const count = dowels.length + 2, cols = Math.ceil(Math.sqrt(count * 3)), gapX = len + 4, gapY = 2 * r + 4;
    const all = gc(Manifold.compose(Array.from({ length: count }, (_, i) =>
      gc(lying.translate([(i % cols) * gapX, Math.floor(i / cols) * gapY, 0])))));
    const g = all.getMesh(), np = g.numProp, nv = g.vertProperties.length / np;
    const pos = new Float32Array(nv * 3);
    for (let i = 0; i < nv; i++) for (let k = 0; k < 3; k++) pos[i * 3 + k] = g.vertProperties[i * np + k];
    for (const m of trash) try { m.delete(); } catch {}
    return { pos, tri: Uint32Array.from(g.triVerts), count, length: len };
  }

  // The six sides a piece can lie on: rotations (row by row) taking that side down. Proper rotations only, so the
  // faces still face out.
  const TURNS = [
    [1, 0, 0, 0, 1, 0, 0, 0, 1], // as scanned
    [1, 0, 0, 0, -1, 0, 0, 0, -1], // upside down
    [0, 0, 1, 0, 1, 0, -1, 0, 0], // its +x side down
    [0, 0, -1, 0, 1, 0, 1, 0, 0], // its -x side down
    [1, 0, 0, 0, 0, 1, 0, -1, 0], // its +y side down
    [1, 0, 0, 0, 0, -1, 0, 1, 0], // its -y side down
  ];

  // How much support a way of lying needs, roughly: the area facing down more steeply than the printer manages,
  // each bit counted by its height above the plate (a branch that long under it), plus a little for every tip.
  function overhang(pos, tri, R, angle) {
    const lim = -Math.sin((angle * Math.PI) / 180), Z = (i) => R[6] * pos[i] + R[7] * pos[i + 1] + R[8] * pos[i + 2];
    let floor = Infinity;
    for (let i = 0; i < pos.length; i += 3) floor = Math.min(floor, Z(i));
    let cost = 0;
    for (let t = 0; t < tri.length; t += 3) {
      const a = tri[t] * 3, b = tri[t + 1] * 3, c = tri[t + 2] * 3;
      const ux = pos[b] - pos[a], uy = pos[b + 1] - pos[a + 1], uz = pos[b + 2] - pos[a + 2];
      const vx = pos[c] - pos[a], vy = pos[c + 1] - pos[a + 1], vz = pos[c + 2] - pos[a + 2];
      const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx, len = Math.hypot(nx, ny, nz);
      if (!len || (R[6] * nx + R[7] * ny + R[8] * nz) / len >= lim) continue;
      const height = (Z(a) + Z(b) + Z(c)) / 3 - floor;
      if (height > 0.3) cost += (len / 2) * (height + 5); // on the plate itself needs nothing
    }
    return cost;
  }

  // A piece laid on its best side, moved to the middle of the plate and standing on z = 0, as Supports.plan wants it
  // (its frame is then the identity), with plan's skip for the pin holes. A side only wins when it saves a good part
  // of the supports, so pieces stay as scanned when it makes little difference.
  function forPrint(part, dowels, holeR, plate, angle) {
    const fits = (R) => {
      const e = [0, 1, 2].map((k) => {
        let lo = Infinity, hi = -Infinity;
        for (const x of [part.min[0], part.max[0]]) for (const y of [part.min[1], part.max[1]]) for (const z of [part.min[2], part.max[2]]) {
          const v = R[k * 3] * x + R[k * 3 + 1] * y + R[k * 3 + 2] * z;
          lo = Math.min(lo, v); hi = Math.max(hi, v);
        }
        return hi - lo;
      });
      return e[2] <= plate.z + 0.5 && ((e[0] <= plate.x + 0.5 && e[1] <= plate.y + 0.5) || (e[0] <= plate.y + 0.5 && e[1] <= plate.x + 0.5));
    };
    let R = TURNS[0], best = overhang(part.pos, part.tri, R, angle), turned = 0;
    TURNS.forEach((T, i) => {
      if (!i || !fits(T)) return;
      const c = overhang(part.pos, part.tri, T, angle);
      if (c < best * 0.7) { best = c; R = T; turned = i; }
    });
    const n = part.pos.length / 3, pos = new Float32Array(part.pos.length);
    const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < n; i++) {
      const x = part.pos[i * 3], y = part.pos[i * 3 + 1], z = part.pos[i * 3 + 2];
      for (let k = 0; k < 3; k++) {
        const v = R[k * 3] * x + R[k * 3 + 1] * y + R[k * 3 + 2] * z;
        pos[i * 3 + k] = v;
        if (v < min[k]) min[k] = v;
        if (v > max[k]) max[k] = v;
      }
    }
    const shift = [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, min[2]];
    for (let i = 0; i < pos.length; i++) pos[i] -= shift[i % 3];
    for (let k = 0; k < 3; k++) { min[k] -= shift[k]; max[k] -= shift[k]; }
    // A spot on the plate, back where it was in the piece (R turned back: its transpose), against each hole.
    const R1 = holeR + 1;
    const skip = (x, y, z) => {
      const q = [x + shift[0], y + shift[1], z + shift[2]];
      const p = [0, 1, 2].map((k) => R[k] * q[0] + R[3 + k] * q[1] + R[6 + k] * q[2]);
      return dowels.some(({ axis, at, depth }) => {
        const d = [p[0] - at[0], p[1] - at[1], p[2] - at[2]];
        return Math.abs(d[axis]) <= depth + 1 && Math.hypot(...d.filter((_, k) => k !== axis)) <= R1;
      });
    };
    return { model: { pos, tri: part.tri, min, max }, longest: Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2]), skip, turned };
  }

  // --- 3D preview ---------------------------------------------------------------------------------------------------
  // The pieces pulled apart a little, each its own colour, as a GLB (Y-up, metres); the label spots for the viewer.

  const COLORS = [[0.39, 0.4, 0.95], [0.96, 0.62, 0.04], [0.06, 0.73, 0.51], [0.93, 0.27, 0.6], [0.23, 0.51, 0.96],
    [0.94, 0.33, 0.27], [0.55, 0.36, 0.96], [0.08, 0.72, 0.65], [0.85, 0.65, 0.13], [0.39, 0.45, 0.55]];

  function preview(result) {
    const gap = Math.max(...result.dims) * 0.12;
    const doc = { asset: { version: "2.0", generator: "Orbitize" }, scene: 0, scenes: [{ nodes: [] }], nodes: [], meshes: [],
      materials: [], accessors: [], bufferViews: [], buffers: [{ byteLength: 0 }] };
    const chunks = [], labels = [];
    let offset = 0;
    const add = (bytes, target) => {
      doc.bufferViews.push({ buffer: 0, byteOffset: offset, byteLength: bytes.length, target });
      chunks.push(bytes);
      offset += bytes.length;
      return doc.bufferViews.length - 1;
    };
    for (const p of result.parts) {
      const shift = [(p.idx[0] - (p.n[0] - 1) / 2) * gap, (p.idx[1] - (p.n[1] - 1) / 2) * gap, p.idx[2] * gap];
      const nv = p.pos.length / 3, pos = new Float32Array(p.pos.length);
      const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
      for (let i = 0; i < nv; i++) {
        const x = (p.pos[i * 3] + shift[0]) / 1000, y = (p.pos[i * 3 + 1] + shift[1]) / 1000, z = (p.pos[i * 3 + 2] + shift[2]) / 1000;
        const v = [x, z, -y];
        for (let k = 0; k < 3; k++) { pos[i * 3 + k] = v[k]; min[k] = Math.min(min[k], v[k]); max[k] = Math.max(max[k], v[k]); }
      }
      const pv = add(new Uint8Array(pos.buffer), 34962), iv = add(new Uint8Array(p.tri.buffer.slice(0)), 34963);
      doc.accessors.push({ bufferView: pv, componentType: 5126, count: nv, type: "VEC3", min, max },
        { bufferView: iv, componentType: 5125, count: p.tri.length, type: "SCALAR" });
      const c = COLORS[(p.number - 1) % COLORS.length];
      doc.materials.push({ pbrMetallicRoughness: { baseColorFactor: [...c, 1], metallicFactor: 0, roughnessFactor: 0.55 } });
      doc.meshes.push({ primitives: [{ attributes: { POSITION: doc.accessors.length - 2 }, indices: doc.accessors.length - 1,
        material: doc.materials.length - 1, mode: 4 }] });
      doc.nodes.push({ mesh: doc.meshes.length - 1 });
      doc.scenes[0].nodes.push(doc.nodes.length - 1);
      labels.push({ number: p.number, at: [(min[0] + max[0]) / 2, max[1], (min[2] + max[2]) / 2] });
    }
    doc.buffers[0].byteLength = offset;
    return { blob: Supports.pack(doc, chunks, offset), labels };
  }

  window.Split = { load, layoutFor, split, forPrint, preview };
})();
