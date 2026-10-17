// The equity chart (ADR 0019): pure geometry plus a small SVG renderer with a scrubber.
// Drawing is built from DOM nodes only; data never becomes markup.

const SVG = "http://www.w3.org/2000/svg";

// `minSpanRatio` keeps tiny moves tiny: the vertical range is at least that share of the
// average value (0.002 = 0.2%), so a few cents on a large account never look like a cliff.
export function scalePoints(points, width, height, padding = 0, { minSpanRatio = 0 } = {}) {
  const values = points.map((p) => Number(p.equity));
  if (values.length === 0) return { coords: [], min: 0, max: 0 };
  let min = Math.min(...values);
  let max = Math.max(...values);
  const average = values.reduce((a, b) => a + b, 0) / values.length;
  const floor = Math.abs(average) * minSpanRatio;
  if (max - min < floor) {
    const middle = (max + min) / 2;
    min = middle - floor / 2;
    max = middle + floor / 2;
  }
  const span = max - min;
  const innerW = width - 2 * padding;
  const innerH = height - 2 * padding;
  const coords = values.map((v, i) => ({
    x: values.length === 1 ? width / 2 : padding + (innerW * i) / (values.length - 1),
    y: span === 0 ? height / 2 : padding + innerH * (1 - (v - min) / span),
    value: v,
    at: points[i].at,
  }));
  return { coords, min, max };
}

const r = (n) => Math.round(n * 100) / 100;

export function linePath(coords) {
  return coords.map((c, i) => `${i === 0 ? "M" : "L"}${r(c.x)},${r(c.y)}`).join("");
}

export function areaPath(coords, height) {
  if (coords.length === 0) return "";
  const first = coords[0];
  const last = coords[coords.length - 1];
  return `${linePath(coords)}L${r(last.x)},${height}L${r(first.x)},${height}Z`;
}

export function nearestIndex(coords, x) {
  if (coords.length === 0) return -1;
  let best = 0;
  for (let i = 1; i < coords.length; i += 1) {
    if (Math.abs(coords[i].x - x) < Math.abs(coords[best].x - x)) best = i;
  }
  return best;
}

function el(name, attrs) {
  const node = document.createElementNS(SVG, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  return node;
}

// Draw `points` into `svg`; `onScrub(point|null)` follows the pointer (null on leave).
export function renderChart(svg, points, { trend = "flat", baseline = null, onScrub = () => {} } = {}) {
  const width = 800;
  const height = 260;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.replaceChildren();
  const { coords, min, max } = scalePoints(points, width, height, 12, { minSpanRatio: 0.002 });
  if (coords.length === 0) return;

  const defs = el("defs", {});
  const gradient = el("linearGradient", { id: "fill", x1: 0, y1: 0, x2: 0, y2: 1 });
  gradient.append(
    el("stop", { offset: "0%", class: `stop-${trend}`, "stop-opacity": 0.28 }),
    el("stop", { offset: "100%", class: `stop-${trend}`, "stop-opacity": 0 }),
  );
  defs.append(gradient);
  svg.append(defs);

  if (baseline !== null && baseline >= min && baseline <= max && max > min) {
    const y = 12 + (height - 24) * (1 - (baseline - min) / (max - min));
    svg.append(el("line", { x1: 0, x2: width, y1: y, y2: y, class: "chart-baseline" }));
  }
  svg.append(el("path", { d: areaPath(coords, height), class: "chart-area", fill: "url(#fill)" }));
  const line = el("path", { d: linePath(coords), class: `chart-line ${trend}`, pathLength: 1 });
  svg.append(line);

  const cursor = el("line", { x1: 0, x2: 0, y1: 0, y2: height, class: "chart-cursor", visibility: "hidden" });
  const dot = el("circle", { r: 5, class: `chart-dot ${trend}`, visibility: "hidden" });
  svg.append(cursor, dot);

  const move = (event) => {
    const box = svg.getBoundingClientRect();
    const x = ((event.clientX - box.left) / box.width) * width;
    const i = nearestIndex(coords, x);
    if (i < 0) return;
    const c = coords[i];
    for (const node of [cursor, dot]) node.setAttribute("visibility", "visible");
    cursor.setAttribute("x1", c.x);
    cursor.setAttribute("x2", c.x);
    dot.setAttribute("cx", c.x);
    dot.setAttribute("cy", c.y);
    onScrub(points[i]);
  };
  const leave = () => {
    for (const node of [cursor, dot]) node.setAttribute("visibility", "hidden");
    onScrub(null);
  };
  svg.onpointermove = move;
  svg.onpointerdown = move;
  svg.onpointerleave = leave;
}
