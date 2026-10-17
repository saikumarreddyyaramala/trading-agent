// Formatting for the dashboard (ADR 0019): exact, signed, and never "-0".
import test from "node:test";
import assert from "node:assert/strict";
import { money, pct, signClass, relTime, countdown, compact } from "../../src/trading_agent/dashboard/static/lib/format.js";

test("money formats strings and numbers with cents and thousands", () => {
  assert.equal(money("100003.5"), "$100,003.50");
  assert.equal(money(-12.3), "-$12.30");
  assert.equal(money("12.345", { sign: true }), "+$12.35");
  assert.equal(money("-0.004", { sign: true }), "$0.00");
  assert.equal(money(null), "—");
  assert.equal(money("abc"), "—");
});

test("pct is signed and avoids negative zero", () => {
  assert.equal(pct("3.2"), "+3.20%");
  assert.equal(pct("-0.001"), "0.00%");
  assert.equal(pct(-1.5), "-1.50%");
  assert.equal(pct(null), "—");
});

test("signClass picks up, down, or flat", () => {
  assert.equal(signClass("1"), "up");
  assert.equal(signClass(-0.5), "down");
  assert.equal(signClass("0.00"), "flat");
  assert.equal(signClass(null), "flat");
});

test("relTime is short and handles the future", () => {
  const now = Date.parse("2026-10-15T15:00:00Z");
  assert.equal(relTime("2026-10-15T14:59:30Z", now), "now");
  assert.equal(relTime("2026-10-15T14:45:00Z", now), "15m ago");
  assert.equal(relTime("2026-10-15T12:00:00Z", now), "3h ago");
  assert.equal(relTime("2026-10-13T15:00:00Z", now), "2d ago");
  assert.equal(relTime("2026-10-15T15:30:00Z", now), "in 30m");
  assert.equal(relTime(null, now), "");
});

test("countdown reads like a clock", () => {
  assert.equal(countdown(0), "now");
  assert.equal(countdown(65_000), "1m 05s");
  assert.equal(countdown(3_725_000), "1h 02m");
  assert.equal(countdown(90_000_000), "1d 1h");
});

test("compact numbers for token counts", () => {
  assert.equal(compact(950), "950");
  assert.equal(compact(12_345), "12.3K");
  assert.equal(compact(2_500_000), "2.5M");
});

import { dayLabel } from "../../src/trading_agent/dashboard/static/lib/format.js";

test("daily chart points are labeled by their own date, not New York evening", () => {
  // Alpaca stamps a daily point at midnight UTC: that is the 15th's value.
  assert.equal(dayLabel("2026-10-15T00:00:00+00:00"), "Thu, Oct 15");
  assert.equal(dayLabel(null), "");
});
