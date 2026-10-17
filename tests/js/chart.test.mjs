// Chart geometry (ADR 0019): pure math, so the drawing is predictable.
import test from "node:test";
import assert from "node:assert/strict";
import { scalePoints, linePath, areaPath, nearestIndex } from "../../src/trading_agent/dashboard/static/lib/chart.js";

const points = [
  { at: "2026-10-15T13:30:00Z", equity: "100" },
  { at: "2026-10-15T14:30:00Z", equity: "110" },
  { at: "2026-10-15T15:30:00Z", equity: "105" },
];

test("points fill the box, highest at the top", () => {
  const { coords, min, max } = scalePoints(points, 200, 100, 10);
  assert.equal(min, 100);
  assert.equal(max, 110);
  assert.deepEqual(coords.map((c) => c.x), [10, 100, 190]);
  assert.equal(coords[1].y, 10); // the high is at the top padding
  assert.equal(coords[0].y, 90); // the low is at the bottom padding
});

test("a flat series sits in the middle", () => {
  const flat = scalePoints([{ at: "a", equity: "5" }, { at: "b", equity: "5" }], 100, 50, 0);
  assert.deepEqual(flat.coords.map((c) => c.y), [25, 25]);
});

test("a single point and no points are safe", () => {
  assert.deepEqual(scalePoints([], 100, 50, 0).coords, []);
  const one = scalePoints([{ at: "a", equity: "7" }], 100, 50, 0);
  assert.equal(one.coords.length, 1);
  assert.equal(linePath([]), "");
});

test("paths are valid SVG commands", () => {
  const { coords } = scalePoints(points, 200, 100, 10);
  assert.equal(linePath(coords), "M10,90L100,10L190,50");
  assert.equal(areaPath(coords, 100), "M10,90L100,10L190,50L190,100L10,100Z");
});

test("nearestIndex finds the closest point to the pointer", () => {
  const { coords } = scalePoints(points, 200, 100, 10);
  assert.equal(nearestIndex(coords, 0), 0);
  assert.equal(nearestIndex(coords, 120), 1);
  assert.equal(nearestIndex(coords, 999), 2);
  assert.equal(nearestIndex([], 5), -1);
});

test("tiny moves stay small: a minimum vertical span", () => {
  const cents = [
    { at: "a", equity: "100000" },
    { at: "b", equity: "99999.94" },
  ];
  const { coords } = scalePoints(cents, 200, 100, 0, { minSpanRatio: 0.002 });
  // 6 cents on a $100,000 account is 0.03% of a $200 window: a hair, not a cliff.
  assert.ok(Math.abs(coords[0].y - coords[1].y) < 1);
  // Real moves still use the full height.
  const real = scalePoints(points, 200, 100, 10, { minSpanRatio: 0.002 });
  assert.equal(real.coords[1].y, 10);
});
