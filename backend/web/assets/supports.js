// Print supports, made in the browser from the scan's GLB (the one the 3D viewer already loaded).
// Finds the parts that hang over air and grows tree supports under them, as slicers do: each spot gets a small tip
// (so it snaps off after printing), the branches lean towards each other and join into thicker trunks, go around
// the model, and stand on the plate on a wider foot. Writes the model with them as STL, 3MF or OBJ. Sizes in mm.
(function () {
  // step: spacing of the tips; r, grow, join: a branch's radius, how much it thickens per mm down and per branch
  // joining it; angle: overhangs steeper than this (from upright) get support; lean: the most a
  // branch leans from upright; reach: how far a branch looks for another to join; clear: gap kept to the model;
  // cell: detail of the model map the branches steer by; dz: height of each step down.
  const KINDS = {
    fdm: { step: 3, angle: 35, lean: 40, minGap: 1.2, tip: 2, contact: 0.4, r: 1, rMax: 6, grow: 0.05, join: 0.25, foot: 2, footH: 1.2,
      sink: 0.25, reach: 15, clear: 1, cell: 1.2, dz: 1 },
    resin: { step: 2.25, angle: 30, lean: 35, minGap: 1, tip: 1.4, contact: 0.22, r: 0.5, rMax: 3, grow: 0.025, join: 0.12, foot: 1.2, footH: 0.6,
      sink: 0.15, reach: 10, clear: 0.8, cell: 1, dz: 0.8 },
  };
  const SIDES = 10;
  const MAX_COLUMNS = 4000; // more than this: spaced further apart

  // --- Reading the GLB ------------------------------------------------------------------------------------------
  // The server's GLB is glTF's Y-up; printing is Z-up with the flat base at z = 0 (as the STL from the server).

  function readGLB(buf) {
    const dv = new DataView(buf);
    if (dv.getUint32(0, true) !== 0x46546c67) throw new Error("not a GLB");
    const jsonLen = dv.getUint32(12, true);
    const doc = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 20, jsonLen)));
    const binStart = 20 + jsonLen + 8;
    const binLen = dv.getUint32(20 + jsonLen, true);
    return { doc, bin: new Uint8Array(buf, binStart, binLen) };
  }

  function accessorData(doc, bin, index) {
    const a = doc.accessors[index], v = doc.bufferViews[a.bufferView];
    const per = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 }[a.type];
    const Type = { 5126: Float32Array, 5125: Uint32Array, 5123: Uint16Array, 5121: Uint8Array }[a.componentType];
    const start = bin.byteOffset + (v.byteOffset || 0) + (a.byteOffset || 0);
    const stride = v.byteStride ? v.byteStride / Type.BYTES_PER_ELEMENT : per;
    const out = new Type(a.count * per);
    const src = new Type(bin.buffer, start, (a.count - 1) * stride + per);
    for (let i = 0; i < a.count; i++) for (let k = 0; k < per; k++) out[i * per + k] = src[i * stride + k];
    return out;
  }

  // Every triangle of the GLB as one mesh, Z-up, with points shared between triangles welded back together
  // (the textured GLB repeats them along the texture's seams, which slicers would see as holes).
  function fromGLB(buf) {
    const { doc, bin } = readGLB(buf);
    const pos = [], tri = [];
    let base = 0;
    for (const mesh of doc.meshes || []) for (const p of mesh.primitives) {
      if (p.mode != null && p.mode !== 4) continue;
      const P = accessorData(doc, bin, p.attributes.POSITION);
      const n = P.length / 3;
      for (let i = 0; i < n; i++) pos.push(P[i * 3], -P[i * 3 + 2], P[i * 3 + 1]);
      const I = p.indices != null ? accessorData(doc, bin, p.indices) : Uint32Array.from({ length: n }, (_, i) => i);
      for (let i = 0; i < I.length; i++) tri.push(I[i] + base);
      base += n;
    }
    return orient(weld(new Float32Array(pos), Uint32Array.from(tri)));
  }

  function weld(pos, tri) {
    const n = pos.length / 3, bits = new Uint32Array(pos.buffer);
    let size = 1;
    while (size < n * 2) size <<= 1;
    const table = new Int32Array(size).fill(-1), remap = new Uint32Array(n), out = new Float32Array(pos.length);
    const outBits = new Uint32Array(out.buffer);
    let count = 0;
    for (let i = 0; i < n; i++) {
      const a = bits[i * 3], b = bits[i * 3 + 1], c = bits[i * 3 + 2];
      let h = (Math.imul(a, 73856093) ^ Math.imul(b, 19349663) ^ Math.imul(c, 83492791)) & (size - 1);
      for (;;) {
        const j = table[h];
        if (j < 0) {
          table[h] = count;
          outBits[count * 3] = a; outBits[count * 3 + 1] = b; outBits[count * 3 + 2] = c;
          remap[i] = count++;
          break;
        }
        if (outBits[j * 3] === a && outBits[j * 3 + 1] === b && outBits[j * 3 + 2] === c) { remap[i] = j; break; }
        h = (h + 1) & (size - 1);
      }
    }
    const t = new Uint32Array(tri.length);
    for (let i = 0; i < tri.length; i++) t[i] = remap[tri[i]];
    return { pos: out.slice(0, count * 3), tri: t };
  }

  // Triangles facing outwards (positive volume); a mesh turned inside out would put supports on top.
  function orient(m) {
    const { pos: p, tri: t } = m;
    let vol = 0, area = 0;
    for (let i = 0; i < t.length; i += 3) {
      const a = t[i] * 3, b = t[i + 1] * 3, c = t[i + 2] * 3;
      const ux = p[b] - p[a], uy = p[b + 1] - p[a + 1], uz = p[b + 2] - p[a + 2];
      const vx = p[c] - p[a], vy = p[c + 1] - p[a + 1], vz = p[c + 2] - p[a + 2];
      area += Math.hypot(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx) / 2;
      vol += p[a] * (p[b + 1] * p[c + 2] - p[b + 2] * p[c + 1]) - p[a + 1] * (p[b] * p[c + 2] - p[b + 2] * p[c])
        + p[a + 2] * (p[b] * p[c + 1] - p[b + 1] * p[c]);
    }
    if (vol < 0) for (let i = 0; i < t.length; i += 3) { const k = t[i + 1]; t[i + 1] = t[i + 2]; t[i + 2] = k; }
    const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < p.length; i += 3) for (let k = 0; k < 3; k++) {
      if (p[i + k] < min[k]) min[k] = p[i + k];
      if (p[i + k] > max[k]) max[k] = p[i + k];
    }
    return { ...m, min, max, volume: Math.abs(vol) / 6, area }; // volume and surface in the GLB's units
  }

  // From the GLB's units to printing millimetres: longest side = longestMm, centred, standing on z = 0.
  function frame(model, longestMm) {
    const { min, max } = model;
    const s = longestMm / Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2]);
    return { s, cx: (min[0] + max[0]) / 2, cy: (min[1] + max[1]) / 2, z0: min[2] };
  }

  function scaled(model, f) {
    const p = model.pos, out = new Float32Array(p.length);
    for (let i = 0; i < p.length; i += 3) {
      out[i] = (p[i] - f.cx) * f.s; out[i + 1] = (p[i + 1] - f.cy) * f.s; out[i + 2] = (p[i + 2] - f.z0) * f.s;
    }
    return out;
  }

  // --- Where supports go ----------------------------------------------------------------------------------------
  // A grid of vertical lines under the model. Along each line, every surface that faces down more steeply than the
  // kind's angle and has air under it gets a tip; the trees then grow down from the tips.

  // skip(x, y, z): spots that must not get a support (the holes for pins in split pieces).
  function plan(model, longestMm, kindName, skip) {
    const kind = KINDS[kindName] || KINDS.fdm;
    const f = frame(model, longestMm), P = scaled(model, f), T = model.tri;
    const limit = -Math.sin((kind.angle * Math.PI) / 180);
    let step = kind.step, columns;
    for (let tries = 0; tries < 6; tries++, step *= 1.3) {
      columns = cast(P, T, step, limit, kind.minGap);
      if (skip) columns = columns.filter((c) => !skip(c.x, c.y, c.top));
      if (columns.length <= MAX_COLUMNS) break;
    }
    const trees = grow(columns, solidMap(P, T, kind), kind);
    return { columns, trees, stands: trees.stands, kind: kindName, size: longestMm, frame: f, step, ...amount(trees) };
  }

  // The supports' plastic in mm³ and their surface in mm², from the branches as round tubes.
  function amount(trees) {
    let volume = 0, area = 0;
    for (const path of trees) for (let i = 1; i < path.length; i++) {
      const [x0, y0, z0, r0] = path[i - 1], [x1, y1, z1, r1] = path[i];
      const len = Math.hypot(x1 - x0, y1 - y0, z1 - z0), r = (r0 + r1) / 2;
      volume += Math.PI * r * r * len;
      area += 2 * Math.PI * r * len;
    }
    return { volume, area };
  }

  function bounds(P) {
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (let i = 0; i < P.length; i += 3) {
      x0 = Math.min(x0, P[i]); x1 = Math.max(x1, P[i]); y0 = Math.min(y0, P[i + 1]); y1 = Math.max(y1, P[i + 1]);
    }
    return { x0, x1, y0, y1 };
  }

  // Where each vertical line of the grid (gx + i·step, gy + j·step) crosses the model: cell -> [z, facing, ...],
  // facing < 0 for a surface facing down.
  function raster(P, T, gx, gy, nx, ny, step) {
    const hits = new Map();
    for (let t = 0; t < T.length; t += 3) {
      const a = T[t] * 3, b = T[t + 1] * 3, c = T[t + 2] * 3;
      const ax = P[a], ay = P[a + 1], az = P[a + 2], bx = P[b], by = P[b + 1], bz = P[b + 2], cx = P[c], cy = P[c + 1], cz = P[c + 2];
      const ux = bx - ax, uy = by - ay, uz = bz - az, vx = cx - ax, vy = cy - ay, vz = cz - az;
      const nxv = uy * vz - uz * vy, nyv = uz * vx - ux * vz, nzv = ux * vy - uy * vx;
      const len = Math.hypot(nxv, nyv, nzv);
      if (!len || Math.abs(nzv) < 1e-12) continue; // upright triangles: a vertical line never crosses them
      const facing = nzv / len, det = nzv; // det: the triangle's area on the plate, signed
      const i0 = Math.max(0, Math.ceil((Math.min(ax, bx, cx) - gx) / step)), i1 = Math.min(nx - 1, Math.floor((Math.max(ax, bx, cx) - gx) / step));
      const j0 = Math.max(0, Math.ceil((Math.min(ay, by, cy) - gy) / step)), j1 = Math.min(ny - 1, Math.floor((Math.max(ay, by, cy) - gy) / step));
      for (let i = i0; i <= i1; i++) for (let j = j0; j <= j1; j++) {
        const px = gx + i * step - ax, py = gy + j * step - ay;
        const u = (px * vy - py * vx) / det, v = (ux * py - uy * px) / det;
        if (u < -1e-9 || v < -1e-9 || u + v > 1 + 1e-9) continue;
        const key = i * ny + j;
        let list = hits.get(key);
        if (!list) hits.set(key, (list = []));
        list.push(az + u * uz + v * vz, facing);
      }
    }
    return hits;
  }

  function cast(P, T, step, limit, minGap) {
    const { x0, x1, y0, y1 } = bounds(P);
    // Kept half a step inside the outline, so no tip sits on the very edge.
    const nx = Math.max(1, Math.floor((x1 - x0) / step)), ny = Math.max(1, Math.floor((y1 - y0) / step));
    const gx = x0 + ((x1 - x0) - (nx - 1) * step) / 2, gy = y0 + ((y1 - y0) - (ny - 1) * step) / 2;
    const columns = [];
    for (const [key, list] of raster(P, T, gx, gy, nx, ny, step)) {
      const order = [];
      for (let k = 0; k < list.length; k += 2) order.push(k);
      order.sort((p, q) => list[q] - list[p]); // top to bottom
      let last = Infinity;
      for (let n = 0; n < order.length; n++) {
        const z = list[order[n]], facing = list[order[n] + 1];
        if (facing >= limit || last - z < 1e-3) continue; // not steep, or the same spot on a shared edge
        last = z;
        let below = 0, onModel = false, touching = false;
        for (let m = n - 1; m >= 0 && list[order[m]] - z < 1e-3; m--) if (list[order[m] + 1] > 0) touching = true;
        if (touching) continue; // resting on a surface facing up at the same height (sorted just above it)
        for (let m = n + 1; m < order.length; m++) {
          const zb = list[order[m]];
          if (z - zb < 1e-3 && list[order[m] + 1] < 0) continue; // the same downward surface again
          below = zb; onModel = true; break; // a surface facing up just below (or touching: no gap)
        }
        if (z - below < minGap) continue; // too close to the plate or the model to need one
        columns.push({ x: gx + Math.floor(key / ny) * step, y: gy + (key % ny) * step, top: z, bottom: below, onModel });
      }
    }
    return columns;
  }

  // --- The model as a map the branches steer by -------------------------------------------------------------------
  // A fine grid of vertical lines (with room around the model for branches to go out), each with the heights where
  // it is inside the model: [bottom, top, bottom, top, ...].

  function solidMap(P, T, k) {
    const b = bounds(P), m = k.reach + 10, c = k.cell;
    const gx = b.x0 - m, gy = b.y0 - m;
    const nx = Math.ceil((b.x1 - b.x0 + 2 * m) / c) + 1, ny = Math.ceil((b.y1 - b.y0 + 2 * m) / c) + 1;
    const cells = new Map();
    for (const [key, list] of raster(P, T, gx, gy, nx, ny, c)) {
      const order = [];
      for (let i = 0; i < list.length; i += 2) order.push(i);
      order.sort((p, q) => list[p] - list[q]); // bottom to top
      const spans = [];
      let from = null;
      for (const i of order) {
        if (list[i + 1] < 0) { if (from == null) from = list[i]; } // into the model from below
        else if (from != null) { spans.push(from, list[i]); from = null; }
      }
      if (from != null) spans.push(from, Infinity);
      if (spans.length) cells.set(key, spans);
    }
    const at = (x, y) => {
      const i = Math.round((x - gx) / c), j = Math.round((y - gy) / c);
      return i < 0 || j < 0 || i >= nx || j >= ny ? null : cells.get(i * ny + j);
    };
    return {
      // Is a branch of radius r at (x, y, z) closer to the model than the clearance?
      hits(x, y, z, r) {
        const R = r + k.clear, n = Math.ceil(R / c);
        for (let i = -n; i <= n; i++) for (let j = -n; j <= n; j++) {
          if ((i * i + j * j) * c * c > R * R + c * c) continue;
          const s = at(x + i * c, y + j * c);
          if (s) for (let q = 0; q < s.length; q += 2) if (z > s[q] - k.clear && z < s[q + 1] + k.clear) return true;
        }
        return false;
      },
      // The top of the model right under (x, y), below z; -Infinity over the plate.
      floor(x, y, z) {
        const s = at(x, y);
        let best = -Infinity;
        if (s) for (let q = 1; q < s.length; q += 2) if (s[q] <= z + 1e-6 && s[q] > best) best = s[q];
        return best;
      },
      // A sideways direction away from the model near (x, y, z).
      away(x, y, z, r) {
        const R = r + k.clear + 3, n = Math.ceil(R / c);
        let ax = 0, ay = 0;
        for (let i = -n; i <= n; i++) for (let j = -n; j <= n; j++) {
          const s = at(x + i * c, y + j * c), d2 = (i * i + j * j) * c * c;
          if (!s || !d2) continue;
          for (let q = 0; q < s.length; q += 2) if (z > s[q] - R && z < s[q + 1] + R) { ax -= (i * c) / d2; ay -= (j * c) / d2; break; }
        }
        const l = Math.hypot(ax, ay);
        return l ? [ax / l, ay / l] : null;
      },
    };
  }

  // --- Growing the trees ----------------------------------------------------------------------------------------
  // Step by step downwards. Each branch leans towards the nearest one (at most the kind's lean), keeps clear of the
  // model (going around it), and joins another when they meet, becoming thicker. It ends on the plate, or with a
  // small tip on the model where it cannot get past. Spots too low for a tree get a short column.

  function grow(columns, map, k) {
    const shift = Math.tan((k.lean * Math.PI) / 180) * k.dz;
    const waiting = [], trees = [];
    let stands = 0; // trunks on the plate, branches on the model and short columns: what stands on something
    for (const c of columns) {
      const start = c.top - k.tip;
      if (start - c.bottom < k.tip + 1 || start < k.footH + 1) { trees.push(column(c, k)); stands++; continue; }
      waiting.push({ x: c.x, y: c.y, z: start, r: k.r, path: [[c.x, c.y, c.top + k.sink, k.contact], [c.x, c.y, start, k.r]] });
    }
    waiting.sort((a, b) => b.z - a.z);
    let live = [], w = 0;
    const done = (b) => trees.push(b.path);
    for (let z = waiting.length ? waiting[0].z : 0; w < waiting.length || live.length; z -= k.dz) {
      while (w < waiting.length && waiting[w].z >= z) { const b = waiting[w++]; b.z = z; live.push(b); }
      const next = z - k.dz;
      // Branches that end here: on the plate, or landing on the model just below.
      live = live.filter((b) => {
        const floor = map.floor(b.x, b.y, b.z);
        if (floor > -Infinity && b.z - floor <= k.tip + k.dz) {
          b.path.push([b.x, b.y, floor - k.sink, k.contact]);
          stands++;
          return done(b), false;
        }
        if (floor === -Infinity && next <= k.footH) {
          b.path.push([b.x, b.y, k.footH, b.r], [b.x, b.y, 0, b.r + Math.max(k.foot, b.r * 0.8)]);
          stands++;
          return done(b), false;
        }
        return true;
      });
      // Each one's nearest neighbour within reach (looked up in a grid of reach-sized squares).
      const grid = new Map(), cellOf = (v) => Math.floor(v / k.reach);
      for (const b of live) {
        const key = cellOf(b.x) + "," + cellOf(b.y);
        if (!grid.has(key)) grid.set(key, []);
        grid.get(key).push(b);
      }
      for (const b of live) {
        let best = null, bd = k.reach * k.reach;
        for (let i = -1; i <= 1; i++) for (let j = -1; j <= 1; j++) {
          for (const o of grid.get(cellOf(b.x) + i + "," + (cellOf(b.y) + j)) || []) {
            const d = (o.x - b.x) ** 2 + (o.y - b.y) ** 2;
            if (o !== b && d < bd) { bd = d; best = o; }
          }
        }
        b.to = best;
      }
      for (const b of live) {
        // Thicker after a join, but gradually, so the branch has no step in it.
        let r = Math.min(k.rMax, b.r + k.grow * k.dz);
        if (b.goal > r) r = Math.min(b.goal, r + 0.3 * k.dz);
        const tries = [];
        if (b.to) {
          const dx = b.to.x - b.x, dy = b.to.y - b.y, d = Math.hypot(dx, dy), s = Math.min(shift, d / 2);
          if (d > 1e-6) tries.push([b.x + (dx / d) * s, b.y + (dy / d) * s]);
        }
        tries.push([b.x, b.y]);
        const out = map.away(b.x, b.y, next, r);
        if (out) tries.push([b.x + out[0] * shift, b.y + out[1] * shift]);
        // Straight down is the last resort even when close to the model: it lands on what is below.
        const pick = tries.find(([x, y]) => !map.hits(x, y, next, r)) || [b.x, b.y];
        b.x = pick[0]; b.y = pick[1]; b.z = next; b.r = r;
        b.path.push([b.x, b.y, b.z, b.r]);
      }
      // Branches that met become one, as thick as both together.
      const gone = new Set();
      for (const b of live) {
        if (gone.has(b) || !b.to || gone.has(b.to)) continue;
        const o = b.to;
        if (Math.hypot(o.x - b.x, o.y - b.y) > Math.max(0.5, (b.r + o.r) * 0.35)) continue;
        const [keep, end] = b.r >= o.r ? [b, o] : [o, b];
        end.path.push([keep.x, keep.y, keep.z, end.r]);
        done(end);
        gone.add(end);
        keep.goal = Math.min(k.rMax, Math.max(keep.goal || 0, keep.r, end.r) + k.join);
      }
      if (gone.size) live = live.filter((b) => !gone.has(b));
    }
    const out = trees.map(simplify);
    out.stands = stands;
    return out;
  }

  // A short column for a spot too low for a tree: (x, y, z, radius) points from the tip down.
  function column(c, k) {
    const h = c.top - c.bottom, mid = (c.top + c.bottom) / 2;
    const r = Math.max(k.contact, Math.min(1.5 * k.r, h / 3));
    return c.onModel
      ? [[c.x, c.y, c.top + k.sink, k.contact], [c.x, c.y, mid, r], [c.x, c.y, c.bottom - k.sink, k.contact]]
      : [[c.x, c.y, c.top + k.sink, k.contact], [c.x, c.y, Math.max(mid, k.footH * 0.6), r], [c.x, c.y, 0, r + k.foot * 0.6]];
  }

  // Smoothed (the steps sideways become curves; the tip and the ends stay put), then the points along straight
  // stretches dropped: they add nothing.
  function simplify(path) {
    if (path.length < 5) return path;
    for (let pass = 0; pass < 4; pass++) {
      const prev = path.map((p) => p.slice());
      for (let i = 2; i < path.length - 2; i++) for (const a of [0, 1]) path[i][a] = (prev[i - 1][a] + 2 * prev[i][a] + prev[i + 1][a]) / 4;
    }
    const out = [path[0], path[1]];
    for (let i = 2; i < path.length - 2; i++) {
      const a = out[out.length - 1], p = path[i], n = path[i + 1];
      const ux = p[0] - a[0], uy = p[1] - a[1], uz = p[2] - a[2], vx = n[0] - p[0], vy = n[1] - p[1], vz = n[2] - p[2];
      const cos = (ux * vx + uy * vy + uz * vz) / (Math.hypot(ux, uy, uz) * Math.hypot(vx, vy, vz) || 1);
      if (cos < 0.9985) out.push(p); // bends more than about 3°
    }
    out.push(path[path.length - 2], path[path.length - 1]);
    return out;
  }

  // --- Branch shapes --------------------------------------------------------------------------------------------
  // A round tube along each branch's points, its rings turned to follow the bends.

  function addTube(pos, tri, path) {
    const pts = path.slice().reverse(); // bottom to top
    const start = pos.length / 3, n = pts.length;
    for (let i = 0; i < n; i++) {
      const a = pts[Math.max(0, i - 1)], b = pts[Math.min(n - 1, i + 1)];
      let tx = b[0] - a[0], ty = b[1] - a[1], tz = b[2] - a[2];
      const tl = Math.hypot(tx, ty, tz) || 1;
      tx /= tl; ty /= tl; tz /= tl;
      if (tz < 0.2) { tz = 0.2; const l = Math.hypot(tx, ty, tz); tx /= l; ty /= l; tz /= l; }
      // u = (0, 1, 0) × t, v = t × u: rings that face the same way all along the tube.
      let ux = tz, uy = 0, uz = -tx;
      const ul = Math.hypot(ux, uz) || 1;
      ux /= ul; uz /= ul;
      const vx = ty * uz - tz * uy, vy = tz * ux - tx * uz, vz = tx * uy - ty * ux;
      const [x, y, z, r] = pts[i];
      for (let j = 0; j < SIDES; j++) {
        const t = (j / SIDES) * Math.PI * 2, c = Math.cos(t) * r, s = Math.sin(t) * r;
        pos.push(x + c * ux + s * vx, y + c * uy + s * vy, z + c * uz + s * vz);
      }
    }
    for (let i = 0; i + 1 < n; i++) for (let j = 0; j < SIDES; j++) {
      const a = start + i * SIDES + j, b = start + i * SIDES + ((j + 1) % SIDES), c = b + SIDES, d = a + SIDES;
      tri.push(a, b, c, a, c, d);
    }
    const bottom = pos.length / 3, top = bottom + 1, last = start + (n - 1) * SIDES;
    pos.push(pts[0][0], pts[0][1], pts[0][2], pts[n - 1][0], pts[n - 1][1], pts[n - 1][2]);
    for (let j = 0; j < SIDES; j++) {
      tri.push(bottom, start + ((j + 1) % SIDES), start + j);
      tri.push(top, last + j, last + ((j + 1) % SIDES));
    }
  }

  function supportMesh(p) {
    const pos = [], tri = [];
    for (const path of p.trees) addTube(pos, tri, path);
    return { pos: new Float32Array(pos), tri: Uint32Array.from(tri) };
  }

  // The model in millimetres with its supports, as one mesh.
  function build(model, p) {
    const M = scaled(model, p.frame), S = supportMesh(p), n = M.length / 3;
    const pos = new Float32Array(M.length + S.pos.length), tri = new Uint32Array(model.tri.length + S.tri.length);
    pos.set(M); pos.set(S.pos, M.length);
    tri.set(model.tri);
    for (let i = 0; i < S.tri.length; i++) tri[model.tri.length + i] = S.tri[i] + n;
    return { pos, tri };
  }

  // --- Files ----------------------------------------------------------------------------------------------------

  function stl(m) {
    const n = m.tri.length / 3, buf = new ArrayBuffer(84 + n * 50), dv = new DataView(buf);
    new Uint8Array(buf, 0, 80).set(new TextEncoder().encode("Orbitize model with supports"));
    dv.setUint32(80, n, true);
    const p = m.pos, t = m.tri;
    for (let i = 0, o = 84; i < n; i++, o += 50) {
      const a = t[i * 3] * 3, b = t[i * 3 + 1] * 3, c = t[i * 3 + 2] * 3;
      const ux = p[b] - p[a], uy = p[b + 1] - p[a + 1], uz = p[b + 2] - p[a + 2];
      const vx = p[c] - p[a], vy = p[c + 1] - p[a + 1], vz = p[c + 2] - p[a + 2];
      let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
      const l = Math.hypot(nx, ny, nz) || 1;
      dv.setFloat32(o, nx / l, true); dv.setFloat32(o + 4, ny / l, true); dv.setFloat32(o + 8, nz / l, true);
      for (const [k, v] of [[12, a], [24, b], [36, c]]) {
        dv.setFloat32(o + k, p[v], true); dv.setFloat32(o + k + 4, p[v + 1], true); dv.setFloat32(o + k + 8, p[v + 2], true);
      }
    }
    return new Blob([buf], { type: "model/stl" });
  }

  const num = (v) => String(Math.round(v * 1000) / 1000);

  function obj(m) {
    const parts = ["# Orbitize model with supports, millimetres\n"], p = m.pos, t = m.tri;
    for (let i = 0; i < p.length; i += 3) parts.push(`v ${num(p[i])} ${num(p[i + 1])} ${num(p[i + 2])}\n`);
    for (let i = 0; i < t.length; i += 3) parts.push(`f ${t[i] + 1} ${t[i + 1] + 1} ${t[i + 2] + 1}\n`);
    return new Blob(parts, { type: "model/obj" });
  }

  // 3MF: a zip with the mesh as XML. bed: the plate size, so the model lands in its middle.
  async function threeMF(m, bed) {
    const p = m.pos, t = m.tri, xml = [];
    xml.push('<?xml version="1.0" encoding="UTF-8"?>\n<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">',
      '<metadata name="Application">Orbitize</metadata><resources><object id="1" type="model"><mesh><vertices>');
    for (let i = 0; i < p.length; i += 3) xml.push(`<vertex x="${num(p[i])}" y="${num(p[i + 1])}" z="${num(p[i + 2])}"/>`);
    xml.push("</vertices><triangles>");
    for (let i = 0; i < t.length; i += 3) xml.push(`<triangle v1="${t[i]}" v2="${t[i + 1]}" v3="${t[i + 2]}"/>`);
    const at = bed ? `${num(bed.x / 2)} ${num(bed.y / 2)} 0` : "0 0 0";
    xml.push(`</triangles></mesh></object></resources><build><item objectid="1" transform="1 0 0 0 1 0 0 0 1 ${at}"/></build></model>`);
    const enc = new TextEncoder();
    return zip([
      ["[Content_Types].xml", enc.encode('<?xml version="1.0" encoding="UTF-8"?>\n<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/></Types>')],
      ["_rels/.rels", enc.encode('<?xml version="1.0" encoding="UTF-8"?>\n<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')],
      ["3D/3dmodel.model", new Uint8Array(await new Blob(xml).arrayBuffer())],
    ], "model/3mf");
  }

  let CRC;
  function crc32(data) {
    if (!CRC) {
      CRC = new Uint32Array(256);
      for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; CRC[n] = c; }
    }
    let c = -1;
    for (let i = 0; i < data.length; i++) c = CRC[(c ^ data[i]) & 255] ^ (c >>> 8);
    return (c ^ -1) >>> 0;
  }

  // Compressed where the browser can (CompressionStream), stored as is otherwise.
  async function zip(files, type) {
    const parts = [], central = [];
    let offset = 0;
    for (const [name, data] of files) {
      let body = data, method = 0;
      if (typeof CompressionStream === "function") {
        try {
          body = new Uint8Array(await new Response(new Blob([data]).stream().pipeThrough(new CompressionStream("deflate-raw"))).arrayBuffer());
          method = 8;
        } catch { body = data; }
      }
      const fname = new TextEncoder().encode(name), crc = crc32(data);
      const head = new DataView(new ArrayBuffer(30));
      head.setUint32(0, 0x04034b50, true); head.setUint16(4, 20, true); head.setUint16(6, 0x800, true); head.setUint16(8, method, true);
      head.setUint16(12, 0x21, true); head.setUint32(14, crc, true);
      head.setUint32(18, body.length, true); head.setUint32(22, data.length, true); head.setUint16(26, fname.length, true);
      parts.push(head, fname, body);
      const cd = new DataView(new ArrayBuffer(46));
      cd.setUint32(0, 0x02014b50, true); cd.setUint16(4, 20, true); cd.setUint16(6, 20, true); cd.setUint16(8, 0x800, true); // names in UTF-8
      cd.setUint16(10, method, true);
      cd.setUint16(14, 0x21, true); cd.setUint32(16, crc, true); cd.setUint32(20, body.length, true);
      cd.setUint32(24, data.length, true); cd.setUint16(28, fname.length, true); cd.setUint32(42, offset, true);
      central.push(cd, fname);
      offset += 30 + fname.length + body.length;
    }
    const size = central.reduce((s, p) => s + p.byteLength, 0);
    const end = new DataView(new ArrayBuffer(22));
    end.setUint32(0, 0x06054b50, true); end.setUint16(8, files.length, true); end.setUint16(10, files.length, true);
    end.setUint32(12, size, true); end.setUint32(16, offset, true);
    return new Blob([...parts, ...central, end], { type });
  }

  // --- Preview --------------------------------------------------------------------------------------------------
  // The viewer's GLB with the columns added as one more (violet) part, in the GLB's own units and Y-up.

  function preview(buf, p) {
    const { doc, bin } = readGLB(buf);
    const S = supportMesh(p), f = p.frame, n = S.pos.length / 3;
    const pos = new Float32Array(S.pos.length), min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < n; i++) {
      const X = S.pos[i * 3] / f.s + f.cx, Y = S.pos[i * 3 + 1] / f.s + f.cy, Z = S.pos[i * 3 + 2] / f.s + f.z0;
      const v = [X, Z, -Y];
      for (let k = 0; k < 3; k++) { pos[i * 3 + k] = v[k]; min[k] = Math.min(min[k], v[k]); max[k] = Math.max(max[k], v[k]); }
    }
    const pad = (len) => (4 - (len % 4)) % 4;
    const start = bin.length + pad(bin.length), posBytes = new Uint8Array(pos.buffer), triBytes = new Uint8Array(S.tri.buffer);
    const d = JSON.parse(JSON.stringify(doc));
    d.bufferViews.push({ buffer: 0, byteOffset: start, byteLength: posBytes.length, target: 34962 },
      { buffer: 0, byteOffset: start + posBytes.length, byteLength: triBytes.length, target: 34963 });
    d.accessors.push({ bufferView: d.bufferViews.length - 2, componentType: 5126, count: n, type: "VEC3", min, max },
      { bufferView: d.bufferViews.length - 1, componentType: 5125, count: S.tri.length, type: "SCALAR" });
    d.materials = d.materials || [];
    d.materials.push({ pbrMetallicRoughness: { baseColorFactor: [0.55, 0.45, 0.98, 1], metallicFactor: 0, roughnessFactor: 0.6 } });
    d.meshes.push({ primitives: [{ attributes: { POSITION: d.accessors.length - 2 }, indices: d.accessors.length - 1, material: d.materials.length - 1, mode: 4 }] });
    d.nodes.push({ mesh: d.meshes.length - 1 });
    d.scenes[d.scene || 0].nodes.push(d.nodes.length - 1);
    const binLen = start + posBytes.length + triBytes.length;
    d.buffers[0].byteLength = binLen;
    return pack(d, [bin, new Uint8Array(start - bin.length), posBytes, triBytes], binLen);
  }

  // A GLB from its JSON and its binary chunk (in parts, binLen bytes in all).
  function pack(d, parts, binLen) {
    const pad = (len) => (4 - (len % 4)) % 4;
    const text = new TextEncoder().encode(JSON.stringify(d));
    const json = new Uint8Array(text.length + pad(text.length)).fill(32);
    json.set(text);
    const head = new DataView(new ArrayBuffer(20));
    head.setUint32(0, 0x46546c67, true); head.setUint32(4, 2, true);
    head.setUint32(8, 12 + 8 + json.length + 8 + binLen, true);
    head.setUint32(12, json.length, true); head.setUint32(16, 0x4e4f534a, true);
    const binHead = new DataView(new ArrayBuffer(8));
    binHead.setUint32(0, binLen, true); binHead.setUint32(4, 0x004e4942, true);
    return new Blob([head, json, binHead, ...parts], { type: "model/gltf-binary" });
  }

  // --- The GLB at the chosen size ---------------------------------------------------------------------------------
  // The viewer's GLB (colours and all) with every point scaled so the longest side is longestMm, as the server's
  // own download is; made here, so there is nothing to download again.

  function scaledGLB(buf, longestMm) {
    const { doc, bin } = readGLB(buf);
    const out = bin.slice(), d = JSON.parse(JSON.stringify(doc));
    const ids = [...new Set(d.meshes.flatMap((m) => m.primitives.map((p) => p.attributes.POSITION)))];
    const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
    const points = ids.map((i) => {
      const a = d.accessors[i], v = d.bufferViews[a.bufferView], stride = v.byteStride ? v.byteStride / 4 : 3;
      const p = new Float32Array(out.buffer, (v.byteOffset || 0) + (a.byteOffset || 0), (a.count - 1) * stride + 3);
      for (let j = 0; j < a.count; j++) for (let k = 0; k < 3; k++) {
        const x = p[j * stride + k];
        if (x < min[k]) min[k] = x;
        if (x > max[k]) max[k] = x;
      }
      return { a, p, stride };
    });
    const f = longestMm / Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2]);
    for (const { a, p, stride } of points) {
      for (let j = 0; j < a.count; j++) for (let k = 0; k < 3; k++) p[j * stride + k] *= f;
      if (a.min) a.min = a.min.map((x) => x * f);
      if (a.max) a.max = a.max.map((x) => x * f);
    }
    return pack(d, [out], out.length);
  }

  // The model's points in printing millimetres (longest side longestMm, centred, standing on z = 0).
  const mm = (model, longestMm) => scaled(model, frame(model, longestMm));

  window.Supports = { fromGLB, plan, build, stl, obj, threeMF, preview, scaledGLB, mm, pack, zip };
})();
