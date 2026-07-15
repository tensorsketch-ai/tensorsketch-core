// TensorSketch Studio — a visual projection of the code. Renders the GraphIR the bridge extracts and
// writes edits straight back (reconstruct). Vanilla canvas, hand-drawn (Excalidraw) aesthetic,
// no build step and no dependencies. The blocks and behavior are TensorSketch's; only the look is borrowed.

"use strict";

const START = "__start__";
const END = "__end__";

const PALETTE = {
  ink: "#1e1e1e",
  paper: "#fafafa",
  node: "#ffffff",
  nodeTint: "#f8f5ff",
  hole: "#fff4e6",
  holeInk: "#e8590c",
  inPort: "#2f9e44",
  outPort: "#6741d9",
  seq: "#343a40",
  cond: "#1971c2",
  terminal: "#e9ecef",
  selected: "#e03131",
  ranOk: "#2f9e44",
  ranErr: "#e03131",
  ranFill: "#ebfbee",
  ranErrFill: "#fff0f0",
  badge: "#2b8a3e",
};

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const palette = document.getElementById("palette");

const IDENT = /^[A-Za-z_][A-Za-z0-9_]*$/;

const view = {
  ir: null,
  path: "",
  holes: [], // project-wide unfilled nodes, from /api/holes
  layout: new Map(), // name -> { x, y, w, h, node, kind }
  layoutOverrides: {}, // name -> { x, y } — manual positions, persisted in the sidecar
  pan: { x: 0, y: 0 },
  scale: 1,
  selectedEdge: null,
  // Live trace overlay — a read-only projection of the run's spans onto the nodes. Ephemeral:
  // the code stays the source of truth; this just paints what a running agent is doing.
  trace: { on: false, timer: null, cursor: 0, traceId: null, spans: [], byNode: new Map() },
};

const drag = { mode: null, from: null, sx: 0, sy: 0, mx: 0, my: 0, moved: false, hover: null };

// ---------------------------------------------------------------- data + bridge

async function load() {
  const res = await fetch("/api/graph");
  setGraph(await res.json());
  fit();
  render();
  loadHoles();
}

async function save(ir, note) {
  setStatus("saving…", "");
  try {
    const res = await fetch("/api/graph", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ir }),
    });
    const data = await res.json();
    if (data.error) {
      setStatus("write-back failed: " + data.error, "err");
      return;
    }
    setGraph(data);
    render();
    setStatus(note || "written to code ✓", "ok");
    loadHoles(); // an edit may have created or filled a hole
  } catch (err) {
    setStatus("bridge error: " + err.message, "err");
  }
}

function setGraph(data) {
  view.ir = data.ir;
  view.path = data.path;
  view.layoutOverrides = (data.layout && data.layout.nodes) || {};
  view.selectedEdge = null;
  document.getElementById("file").textContent = data.path;
  computeLayout();
}

async function saveLayout() {
  try {
    await fetch("/api/layout", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ nodes: view.layoutOverrides }),
    });
    setStatus("layout saved", "ok");
  } catch (_err) {
    /* positions are cosmetic — a failed save just means the arrangement isn't remembered */
  }
}

// ---------------------------------------------------------------- project-wide holes

// "What's left to implement?" — every node still stubbed with a `Hole`, across the whole project
// (not just this file). The bridge scans the source dir; the badge counts them and lists them.

async function loadHoles() {
  try {
    const res = await fetch("/api/holes");
    view.holes = (await res.json()).holes || [];
  } catch (_err) {
    view.holes = [];
  }
  renderHolesBadge();
}

function renderHolesBadge() {
  const el = document.getElementById("holes");
  const n = view.holes.length;
  el.textContent = n ? `⚠ ${n} node${n > 1 ? "s" : ""} need code` : "";
  el.classList.toggle("clickable", n > 0);
  if (!n) document.getElementById("holes-panel").hidden = true;
}

function toggleHolesPanel() {
  const panel = document.getElementById("holes-panel");
  if (!panel.hidden || !view.holes.length) {
    panel.hidden = true;
    return;
  }
  const byFile = new Map();
  for (const h of view.holes) {
    if (!byFile.has(h.file)) byFile.set(h.file, []);
    byFile.get(h.file).push(h);
  }
  const n = view.holes.length;
  let html = `<h3>${n} node${n > 1 ? "s" : ""} need code</h3>`;
  for (const [file, hs] of byFile) {
    const short = file.split("/").slice(-2).join("/");
    const here = file === view.path ? ' <span class="here">· this file</span>' : "";
    html += `<div class="hole-file">${esc(short)}${here}</div>`;
    for (const h of hs) {
      const spec = h.spec ? ` — <span class="hole-spec">${esc(h.spec)}</span>` : "";
      html += `<div class="hole-row"><span class="hole-line">${h.line}</span><span class="hole-node">${esc(h.node)}</span>${spec}</div>`;
    }
  }
  panel.innerHTML = html;
  panel.hidden = false;
}

function esc(s) {
  return String(s).replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c],
  );
}

// ---------------------------------------------------------------- live trace overlay

// The overlay is a *read-only projection* of a run's spans onto the graph — the same code⇄canvas
// philosophy applied to observability. Your agent runs in its own process and ships each finished
// span to the bridge (`http_span_sink`); we poll for them and paint status / latency / cost onto
// the matching nodes. Nothing is persisted here; the trace's real home is whatever sink you export
// to. So statelessness holds: Studio reads code and reads telemetry, and owns neither.

function toggleLive() {
  const t = view.trace;
  t.on = !t.on;
  const btn = document.getElementById("live");
  btn.classList.toggle("on", t.on);
  btn.textContent = t.on ? "⏸ live" : "▶ live";
  if (t.on) {
    pollTrace();
    t.timer = setInterval(pollTrace, 600);
    setStatus("live overlay on — run your agent with http_span_sink to this URL", "");
  } else {
    clearInterval(t.timer);
    t.timer = null;
    render();
    setStatus("live overlay off", "");
  }
}

async function pollTrace() {
  const t = view.trace;
  try {
    const res = await fetch("/api/trace?since=" + t.cursor);
    const data = await res.json();
    t.cursor = data.cursor;
    if (!data.records.length) return;
    for (const r of data.records) t.spans.push({ ...r.span, seq: r.seq });
    t.traceId = t.spans[t.spans.length - 1].trace_id; // follow the newest run
    aggregateTrace();
    render();
    const n = t.byNode.size;
    if (n) setStatus(`live: ${n} node${n > 1 ? "s" : ""} traced`, "ok");
  } catch (_err) {
    /* bridge unreachable or no run yet — stay quiet and keep polling */
  }
}

// Fold the flat span list into one aggregate per node: status, wall time, cost, tokens, call count.
// Model/tool spans nest under their node span, so we walk parent links up to the nearest node.
function aggregateTrace() {
  const t = view.trace;
  const spans = t.spans.filter((s) => s.trace_id === t.traceId);
  const byId = new Map(spans.map((s) => [s.span_id, s]));
  const nodeOf = (s) => {
    let cur = s;
    while (cur) {
      if (cur.kind === "node") return (cur.attributes && cur.attributes["tensorsketch.node"]) || cur.name;
      cur = cur.parent_id ? byId.get(cur.parent_id) : null;
    }
    return null;
  };
  const map = new Map();
  for (const s of spans) {
    const name = nodeOf(s);
    if (!name) continue;
    let agg = map.get(name);
    if (!agg) {
      agg = { status: "ok", ms: 0, cost: 0, tokens: 0, calls: 0, order: Infinity };
      map.set(name, agg);
    }
    const attr = s.attributes || {};
    if (s.kind === "node") {
      agg.status = s.status;
      agg.ms = s.duration_ms || 0;
      agg.order = Math.min(agg.order, s.seq);
    }
    if (s.kind === "model" || s.kind === "tool") agg.calls += 1;
    if (attr["gen_ai.usage.cost_usd"]) agg.cost += Number(attr["gen_ai.usage.cost_usd"]);
    if (attr["gen_ai.usage.output_tokens"]) agg.tokens += Number(attr["gen_ai.usage.output_tokens"]);
    if (s.status === "error") agg.status = "error";
  }
  t.byNode = map;
}

function traceBadge(agg) {
  const bits = [];
  if (agg.ms) bits.push(agg.ms >= 1 ? `${Math.round(agg.ms)}ms` : `${agg.ms.toFixed(2)}ms`);
  if (agg.cost) bits.push(`$${agg.cost.toFixed(5)}`);
  if (agg.calls) bits.push(`${agg.calls}×`);
  return bits.join(" · ");
}

function drawTraceOverlay(name, box) {
  if (!view.trace.on) return;
  const agg = view.trace.byNode.get(name);
  if (!agg) return;
  const ok = agg.status !== "error";
  const color = ok ? PALETTE.ranOk : PALETTE.ranErr;

  // a colored hand-drawn ring just outside the node = "this executed"
  ctx.lineWidth = 2.4;
  ctx.strokeStyle = color;
  ctx.setLineDash([]);
  roughRect(box.x - 4, box.y - 4, box.w + 8, box.h + 8, seedOf(box.x, box.y) + 99);

  // a status dot in the top-left corner
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(box.x + 3, box.y + 3, 4, 0, Math.PI * 2);
  ctx.fill();

  // a badge pill beneath the node: latency · cost · call count
  const text = traceBadge(agg);
  if (!text) return;
  ctx.font = "600 11px " + handFont();
  const w = ctx.measureText(text).width + 14;
  const bx = box.x + box.w / 2 - w / 2;
  const by = box.y + box.h + 6;
  fillRoundRect(bx, by, w, 17, 8, ok ? PALETTE.ranFill : PALETTE.ranErrFill);
  ctx.lineWidth = 1.2;
  ctx.strokeStyle = color;
  roughRect(bx, by, w, 17, seedOf(bx, by));
  ctx.fillStyle = ok ? PALETTE.badge : PALETTE.ranErr;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, bx + w / 2, by + 9);
}

// ---------------------------------------------------------------- layered layout

function nodeSize(node, kind) {
  if (kind === "terminal") return { w: 74, h: 40 };
  const rows = Math.max(node ? node.inputs.length : 0, node ? node.outputs.length : 0, 1);
  ctx.font = "600 15px " + handFont();
  const titleW = node ? ctx.measureText(node.name).width : 60;
  return { w: Math.max(168, Math.min(300, titleW + 60)), h: 40 + rows * 22 + 12 };
}

function computeLayout() {
  const ir = view.ir;
  const present = new Set(ir.added);
  const targets = ir.edges.filter((e) => e.target != null);
  for (const e of targets) {
    present.add(e.source);
    present.add(e.target);
  }

  // Longest-path layering from START; cap growth so cycles (loops/routers) can't diverge.
  const layer = new Map([[START, 0]]);
  const cap = present.size + 2;
  const relax = () => {
    for (let i = 0; i < cap; i++) {
      let changed = false;
      for (const e of targets) {
        if (!layer.has(e.source)) continue; // only propagate from placed nodes
        const cand = Math.min(layer.get(e.source) + 1, cap);
        if (cand > (layer.has(e.target) ? layer.get(e.target) : -1)) {
          layer.set(e.target, cand);
          changed = true;
        }
      }
      if (!changed) break;
    }
  };
  relax();
  // Nodes with no incoming static edge (e.g. targets of a dynamic router) are disconnected
  // roots — seat them just below START, then relax again so their successors drop below them.
  for (const n of present) if (!layer.has(n)) layer.set(n, 1);
  relax();

  // Group by layer, preserving a stable within-layer order (added order, then edge order).
  const byLayer = new Map();
  const orderKey = (n) => (n === START ? -1 : n === END ? 1e9 : ir.added.indexOf(n));
  for (const n of new Set([START, ...present, ...(layer.has(END) ? [END] : [])])) {
    const l = layer.get(n) ?? 1;
    if (!byLayer.has(l)) byLayer.set(l, []);
    byLayer.get(l).push(n);
  }
  for (const arr of byLayer.values()) arr.sort((a, b) => orderKey(a) - orderKey(b));

  const HGAP = 46;
  const VGAP = 64;
  view.layout = new Map();
  let y = 0;
  for (const l of [...byLayer.keys()].sort((a, b) => a - b)) {
    const names = byLayer.get(l);
    const sized = names.map((name) => {
      const kind = name === START || name === END ? "terminal" : "node";
      const node = ir.nodes.find((nd) => nd.name === name) || null;
      return { name, kind, node, ...nodeSize(node, kind) };
    });
    const rowH = Math.max(...sized.map((s) => s.h));
    const totalW = sized.reduce((a, s) => a + s.w, 0) + HGAP * (sized.length - 1);
    let x = -totalW / 2;
    for (const s of sized) {
      view.layout.set(s.name, { x, y: y + (rowH - s.h) / 2, w: s.w, h: s.h, node: s.node, kind: s.kind });
      x += s.w + HGAP;
    }
    y += rowH + VGAP;
  }

  // Manual positions win over the automatic layout; unmoved nodes keep their computed spot.
  for (const [name, pos] of Object.entries(view.layoutOverrides)) {
    const box = view.layout.get(name);
    if (box) {
      box.x = pos.x;
      box.y = pos.y;
    }
  }
}

// ---------------------------------------------------------------- hand-drawn primitives

function handFont() {
  return '"Virgil", "Segoe Print", "Comic Sans MS", "Comic Neue", cursive';
}

function prng(seed) {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let x = Math.imul(t ^ (t >>> 15), 1 | t);
    x ^= x + Math.imul(x ^ (x >>> 7), 61 | x);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

function roughLine(x1, y1, x2, y2, seed, j = 1.3) {
  const rng = prng(seed);
  for (let p = 0; p < 2; p++) {
    const o = () => (rng() * 2 - 1) * j;
    ctx.beginPath();
    ctx.moveTo(x1 + o(), y1 + o());
    ctx.quadraticCurveTo((x1 + x2) / 2 + o() * 1.5, (y1 + y2) / 2 + o() * 1.5, x2 + o(), y2 + o());
    ctx.stroke();
  }
}

function seedOf(x, y) {
  return (Math.round(x) * 73856093) ^ (Math.round(y) * 19349663);
}

function roughRect(x, y, w, h, seed) {
  roughLine(x, y, x + w, y, seed + 1);
  roughLine(x + w, y, x + w, y + h, seed + 2);
  roughLine(x + w, y + h, x, y + h, seed + 3);
  roughLine(x, y + h, x, y, seed + 4);
}

function fillRoundRect(x, y, w, h, r, fill) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
  ctx.fillStyle = fill;
  ctx.fill();
}

// ---------------------------------------------------------------- rendering

function render() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = canvas.clientWidth * dpr;
  canvas.height = canvas.clientHeight * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);

  ctx.save();
  ctx.translate(view.pan.x, view.pan.y);
  ctx.scale(view.scale, view.scale);

  for (const e of view.ir.edges) drawEdge(e);
  if (drag.mode === "link") drawPendingLink();
  for (const [name, box] of view.layout) drawNode(name, box);

  ctx.restore();
}

function borderPoint(box, tx, ty) {
  const cx = box.x + box.w / 2;
  const cy = box.y + box.h / 2;
  const dx = tx - cx;
  const dy = ty - cy;
  const hw = box.w / 2 + 2;
  const hh = box.h / 2 + 2;
  const sx = dx === 0 ? Infinity : hw / Math.abs(dx);
  const sy = dy === 0 ? Infinity : hh / Math.abs(dy);
  const s = Math.min(sx, sy);
  return { x: cx + dx * s, y: cy + dy * s };
}

function drawEdge(e) {
  const a = view.layout.get(e.source);
  const b = e.target != null ? view.layout.get(e.target) : null;
  if (!a) return;
  const selected = view.selectedEdge && sameEdge(view.selectedEdge, e);
  const cond = e.kind === "conditional";

  if (!b) {
    // Dynamic conditional with no static targets: a short dangling stub with a "?" label.
    const p = { x: a.x + a.w / 2, y: a.y + a.h + 26 };
    strokeArrow(a.x + a.w / 2, a.y + a.h, p.x, p.y, cond, selected, seedOf(a.x, a.y));
    label(e.condition ? e.condition + " → ?" : "?", p.x, p.y + 6, PALETTE.cond);
    return;
  }
  if (e.source === e.target) {
    // A self-loop (from `loop(...)`): a little arc off the node's right edge, back into itself.
    drawSelfLoop(a, e, selected);
    return;
  }
  const from = borderPoint(a, b.x + b.w / 2, b.y + b.h / 2);
  const to = borderPoint(b, a.x + a.w / 2, a.y + a.h / 2);
  strokeArrow(from.x, from.y, to.x, to.y, cond, selected, seedOf(from.x + e.source.length, to.y));
  e._mid = { x: (from.x + to.x) / 2, y: (from.y + to.y) / 2 };
  if (cond) {
    const lbl = [e.condition, e.key].filter(Boolean).join(": ");
    if (lbl) label(lbl, e._mid.x, e._mid.y, PALETTE.cond);
  }
}

function drawSelfLoop(box, e, selected) {
  // A rounded loop arc anchored on the node's right edge, returning to itself with an arrowhead.
  const x = box.x + box.w;
  const yTop = box.y + box.h * 0.34;
  const yBot = box.y + box.h * 0.66;
  const r = 30;
  ctx.lineWidth = selected ? 2.6 : 1.7;
  ctx.strokeStyle = selected ? PALETTE.selected : PALETTE.cond;
  ctx.setLineDash([7, 5]);
  ctx.beginPath();
  ctx.moveTo(x - 2, yTop);
  ctx.bezierCurveTo(x + r * 1.7, yTop - r * 0.5, x + r * 1.7, yBot + r * 0.5, x - 2, yBot);
  ctx.stroke();
  ctx.setLineDash([]);
  // arrowhead pointing back into the node at the lower anchor
  const ang = Math.atan2(yBot - (yBot + r * 0.5), x - 2 - (x + r * 1.7));
  const h = 10;
  roughLine(x - 2, yBot, x - 2 - h * Math.cos(ang - 0.5), yBot - h * Math.sin(ang - 0.5), 71, 0.7);
  roughLine(x - 2, yBot, x - 2 - h * Math.cos(ang + 0.5), yBot - h * Math.sin(ang + 0.5), 73, 0.7);
  const lbl = [e.condition, e.key].filter(Boolean).join(": ") || "loop";
  label(lbl, x + r * 1.7 + 6, (yTop + yBot) / 2, PALETTE.cond);
  e._mid = { x: x + r * 1.4, y: (yTop + yBot) / 2 }; // clickable region for selection/delete
}

function strokeArrow(x1, y1, x2, y2, dashed, selected, seed) {
  ctx.lineWidth = selected ? 2.6 : 1.7;
  ctx.strokeStyle = selected ? PALETTE.selected : dashed ? PALETTE.cond : PALETTE.seq;
  ctx.setLineDash(dashed ? [7, 5] : []);
  roughLine(x1, y1, x2, y2, seed, 1.1);
  ctx.setLineDash([]);
  const ang = Math.atan2(y2 - y1, x2 - x1);
  const h = 11;
  roughLine(x2, y2, x2 - h * Math.cos(ang - 0.4), y2 - h * Math.sin(ang - 0.4), seed + 7, 0.7);
  roughLine(x2, y2, x2 - h * Math.cos(ang + 0.4), y2 - h * Math.sin(ang + 0.4), seed + 9, 0.7);
}

function drawPendingLink() {
  const a = view.layout.get(drag.from);
  if (!a) return;
  ctx.lineWidth = 1.7;
  ctx.strokeStyle = PALETTE.outPort;
  ctx.setLineDash([6, 5]);
  const w = screenToWorld(drag.mx, drag.my);
  roughLine(a.x + a.w, a.y + a.h / 2, w.x, w.y, 42, 1.0);
  ctx.setLineDash([]);
}

function drawNode(name, box) {
  const seed = seedOf(box.x, box.y);
  if (box.kind === "terminal") {
    fillRoundRect(box.x, box.y, box.w, box.h, box.h / 2, PALETTE.terminal);
    ctx.lineWidth = 1.6;
    ctx.strokeStyle = PALETTE.ink;
    roughRect(box.x, box.y, box.w, box.h, seed);
    ctx.fillStyle = PALETTE.ink;
    ctx.font = "600 14px " + handFont();
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(name === START ? "start" : "end", box.x + box.w / 2, box.y + box.h / 2 + 1);
    return;
  }

  const hole = box.node && box.node.has_hole;
  fillRoundRect(box.x, box.y, box.w, box.h, 8, hole ? PALETTE.hole : PALETTE.nodeTint);
  ctx.lineWidth = 1.8;
  ctx.strokeStyle = hole ? PALETTE.holeInk : PALETTE.ink;
  ctx.setLineDash(hole ? [6, 4] : []);
  roughRect(box.x, box.y, box.w, box.h, seed);
  ctx.setLineDash([]);

  // title
  ctx.fillStyle = PALETTE.ink;
  ctx.font = "700 15px " + handFont();
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.fillText(box.node ? box.node.name : name, box.x + 12, box.y + 20);
  if (hole) {
    ctx.fillStyle = PALETTE.holeInk;
    ctx.font = "600 11px " + handFont();
    ctx.textAlign = "right";
    ctx.fillText("needs code", box.x + box.w - 10, box.y + 20);
  }
  ctx.strokeStyle = PALETTE.line || "#e9ecef";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(box.x + 8, box.y + 32);
  ctx.lineTo(box.x + box.w - 8, box.y + 32);
  ctx.strokeStyle = "#e9ecef";
  ctx.stroke();

  const ins = box.node ? box.node.inputs : [];
  const outs = box.node ? box.node.outputs : [];
  ins.forEach((p, i) => drawPort(box.x, box.y + 44 + i * 22, p, "in"));
  outs.forEach((p, i) => drawPort(box.x + box.w, box.y + 44 + i * 22, p, "out"));

  // right-edge affordance for wiring
  ctx.fillStyle = PALETTE.outPort;
  ctx.globalAlpha = 0.5;
  ctx.beginPath();
  ctx.arc(box.x + box.w, box.y + box.h / 2, 4, 0, Math.PI * 2);
  ctx.fill();
  ctx.globalAlpha = 1;

  drawTraceOverlay(name, box);
}

function drawPort(px, py, port, side) {
  ctx.fillStyle = side === "in" ? PALETTE.inPort : PALETTE.outPort;
  ctx.beginPath();
  ctx.arc(px, py, 3.2, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "#495057";
  ctx.font = "12px " + handFont();
  ctx.textBaseline = "middle";
  if (side === "in") {
    ctx.textAlign = "left";
    ctx.fillText(portLabel(port), px + 8, py);
  } else {
    ctx.textAlign = "right";
    ctx.fillText(portLabel(port), px - 8, py);
  }
}

function portLabel(p) {
  const t = p.type.length > 16 ? p.type.slice(0, 15) + "…" : p.type;
  return `${p.name}: ${t}`;
}

function label(text, x, y, color) {
  ctx.font = "12px " + handFont();
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  const w = ctx.measureText(text).width + 10;
  ctx.fillStyle = "rgba(250,250,250,0.9)";
  ctx.fillRect(x - w / 2, y - 9, w, 18);
  ctx.fillStyle = color;
  ctx.fillText(text, x, y + 1);
}

// ---------------------------------------------------------------- interaction

function screenToWorld(sx, sy) {
  return { x: (sx - view.pan.x) / view.scale, y: (sy - view.pan.y) / view.scale };
}

function nodeAt(w) {
  for (const [name, b] of view.layout) {
    if (w.x >= b.x && w.x <= b.x + b.w && w.y >= b.y && w.y <= b.y + b.h) return name;
  }
  return null;
}

function edgeAt(w) {
  for (const e of view.ir.edges) {
    if (e._mid && Math.hypot(e._mid.x - w.x, e._mid.y - w.y) < 14) return e;
  }
  return null;
}

canvas.addEventListener("mousedown", (ev) => {
  const w = screenToWorld(ev.offsetX, ev.offsetY);
  drag.sx = ev.offsetX;
  drag.sy = ev.offsetY;
  drag.mx = ev.offsetX;
  drag.my = ev.offsetY;
  drag.moved = false;
  const node = nodeAt(w);
  if (node && node !== END && nearWireEdge(node, w)) {
    drag.mode = "maybe-link"; // grabbing the right edge starts a wire
    drag.from = node;
  } else if (node) {
    const box = view.layout.get(node); // grabbing the body starts a move
    drag.mode = "maybe-move";
    drag.from = node;
    drag.moveOX = box.x;
    drag.moveOY = box.y;
  } else {
    const edge = edgeAt(w);
    if (edge) {
      drag.mode = "select";
      view.selectedEdge = edge;
      render();
    } else {
      drag.mode = "pan";
      canvas.classList.add("panning");
    }
  }
});

canvas.addEventListener("mousemove", (ev) => {
  drag.mx = ev.offsetX;
  drag.my = ev.offsetY;
  if (!drag.mode) return;
  const dx = ev.offsetX - drag.sx;
  const dy = ev.offsetY - drag.sy;
  if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;

  if (drag.mode === "pan") {
    view.pan.x += ev.movementX;
    view.pan.y += ev.movementY;
    render();
  } else if (drag.mode === "maybe-link" || drag.mode === "link") {
    drag.mode = "link";
    canvas.classList.add("linking");
    drag.hover = nodeAt(screenToWorld(ev.offsetX, ev.offsetY));
    render();
  } else if (drag.mode === "maybe-move" || drag.mode === "move") {
    drag.mode = "move";
    canvas.classList.add("moving");
    const w0 = screenToWorld(drag.sx, drag.sy);
    const w1 = screenToWorld(ev.offsetX, ev.offsetY);
    const box = view.layout.get(drag.from);
    box.x = drag.moveOX + (w1.x - w0.x);
    box.y = drag.moveOY + (w1.y - w0.y);
    render();
  }
});

window.addEventListener("mouseup", (ev) => {
  if (drag.mode === "link") {
    const target = nodeAt(screenToWorld(drag.mx, drag.my));
    if (target && target !== drag.from && target !== START) {
      addEdge(drag.from, target);
    }
  } else if (drag.mode === "move") {
    const box = view.layout.get(drag.from);
    view.layoutOverrides[drag.from] = { x: box.x, y: box.y };
    saveLayout();
  } else if (drag.mode === "select" && !drag.moved) {
    // selection already set on mousedown
  } else if ((drag.mode === "maybe-link" || drag.mode === "maybe-move") && !drag.moved) {
    view.selectedEdge = null; // a click that didn't drag just clears the selection
    render();
  }
  drag.mode = null;
  drag.from = null;
  drag.hover = null;
  canvas.classList.remove("panning", "linking", "moving");
});

// The right edge of a node is the wiring handle; the rest of the body is the move handle.
function nearWireEdge(name, w) {
  const b = view.layout.get(name);
  return !!b && w.x >= b.x + b.w - 16;
}

canvas.addEventListener("wheel", (ev) => {
  ev.preventDefault();
  const factor = Math.exp(-ev.deltaY * 0.0016);
  const w = screenToWorld(ev.offsetX, ev.offsetY);
  view.scale = Math.max(0.25, Math.min(2.5, view.scale * factor));
  view.pan.x = ev.offsetX - w.x * view.scale;
  view.pan.y = ev.offsetY - w.y * view.scale;
  render();
}, { passive: false });

window.addEventListener("keydown", (ev) => {
  if (!palette.hidden) {
    // While the create-node dialog is open, the keyboard belongs to it (not edge deletion).
    if (ev.key === "Escape") closePalette();
    else if (ev.key === "Enter") createNode();
    return;
  }
  if ((ev.key === "Backspace" || ev.key === "Delete") && view.selectedEdge) {
    ev.preventDefault();
    deleteEdge(view.selectedEdge);
  }
});

document.getElementById("reset").addEventListener("click", () => {
  fit();
  render();
});

document.getElementById("live").addEventListener("click", toggleLive);
document.getElementById("add-node").addEventListener("click", openPalette);
document.getElementById("holes").addEventListener("click", toggleHolesPanel);
document.addEventListener("click", (ev) => {
  // dismiss the holes list on any click outside it (the badge toggles itself)
  const panel = document.getElementById("holes-panel");
  if (panel.hidden || panel.contains(ev.target) || ev.target.id === "holes") return;
  panel.hidden = true;
});
document.getElementById("np-cancel").addEventListener("click", closePalette);
document.getElementById("np-create").addEventListener("click", createNode);
palette.addEventListener("mousedown", (ev) => {
  if (ev.target === palette) closePalette(); // click the backdrop to dismiss
});

// ---------------------------------------------------------------- edits

function addEdge(source, target) {
  if (view.ir.edges.some((e) => e.source === source && e.target === target)) {
    setStatus("edge already exists", "");
    return;
  }
  const ir = clone(view.ir);
  ir.edges.push({ source, target, kind: "sequential", condition: null, key: null });
  const label = `${prettyName(source)} → ${prettyName(target)}`;
  save(ir, `wired ${label} → code ✓`);
}

function deleteEdge(edge) {
  const ir = clone(view.ir);
  ir.edges = ir.edges.filter((e) => !sameEdge(e, edge));
  view.selectedEdge = null;
  save(ir, "removed edge → code ✓");
}

function sameEdge(a, b) {
  return (
    a.source === b.source &&
    a.target === b.target &&
    a.kind === b.kind &&
    (a.condition || null) === (b.condition || null) &&
    (a.key || null) === (b.key || null)
  );
}

function prettyName(n) {
  return n === START ? "start" : n === END ? "end" : n;
}

function clone(ir) {
  return JSON.parse(JSON.stringify(ir));
}

// ---------------------------------------------------------------- node creation (palette)
//
// Creating a node is the one canvas edit that has no wiring-only expression: the node needs a
// `class X(Node)` to exist. So we add the node (with its typed ports) to the IR and let the bridge
// synthesize an idiomatic stub — a `Hole` body you fill in code. It lands unwired; drag its right
// edge to connect it. Code stays the source of truth: the class is written into the file, not held
// here.

function openPalette() {
  npError("");
  document.getElementById("np-name").value = "";
  document.getElementById("np-in").value = "";
  document.getElementById("np-out").value = "";
  palette.hidden = false;
  document.getElementById("np-name").focus();
}

function closePalette() {
  palette.hidden = true;
}

function createNode() {
  const name = document.getElementById("np-name").value.trim();
  if (!IDENT.test(name)) {
    npError("name must be a valid identifier, e.g. Summarize");
    return;
  }
  if (view.ir.nodes.some((n) => n.name === name) || view.ir.added.includes(name)) {
    npError(`"${name}" already exists`);
    return;
  }
  let inputs, outputs;
  try {
    inputs = parsePorts(document.getElementById("np-in").value);
    outputs = parsePorts(document.getElementById("np-out").value);
  } catch (err) {
    npError(err.message);
    return;
  }
  const ir = clone(view.ir);
  ir.nodes.push({ name, inputs, outputs, has_hole: true });
  ir.added.push(name); // wired into the graph as an (as-yet unconnected) node
  closePalette();
  save(ir, `created ${name} → code ✓`);
}

// "query: str, context: str" -> [{name:"query",type:"str"}, ...]. A bare name defaults to `str`.
function parsePorts(text) {
  const ports = [];
  for (const raw of text.split(",")) {
    const s = raw.trim();
    if (!s) continue;
    const colon = s.indexOf(":");
    const pname = (colon === -1 ? s : s.slice(0, colon)).trim();
    const ptype = (colon === -1 ? "str" : s.slice(colon + 1).trim()) || "str";
    if (!IDENT.test(pname)) throw new Error(`invalid port name: "${pname}"`);
    ports.push({ name: pname, type: ptype });
  }
  return ports;
}

function npError(msg) {
  document.getElementById("np-error").textContent = msg;
}

// ---------------------------------------------------------------- view helpers

function fit() {
  const boxes = [...view.layout.values()];
  if (!boxes.length) {
    view.pan = { x: canvas.clientWidth / 2, y: 60 };
    view.scale = 1;
    return;
  }
  const minX = Math.min(...boxes.map((b) => b.x));
  const maxX = Math.max(...boxes.map((b) => b.x + b.w));
  const minY = Math.min(...boxes.map((b) => b.y));
  const maxY = Math.max(...boxes.map((b) => b.y + b.h));
  const pad = 60;
  const sx = canvas.clientWidth / (maxX - minX + pad * 2);
  const sy = canvas.clientHeight / (maxY - minY + pad * 2);
  view.scale = Math.max(0.35, Math.min(1.4, Math.min(sx, sy)));
  view.pan.x = canvas.clientWidth / 2 - ((minX + maxX) / 2) * view.scale;
  view.pan.y = pad - minY * view.scale;
}

function setStatus(text, cls) {
  const el = document.getElementById("status");
  el.textContent = text;
  el.className = "status " + (cls || "");
}

window.addEventListener("resize", render);
load();
