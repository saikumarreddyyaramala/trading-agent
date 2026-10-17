// The "Today" timeline (ADR 0019): every step of the cycle with its state.
import test from "node:test";
import assert from "node:assert/strict";
import { workflowSteps, STEP_LABELS } from "../../src/trading_agent/dashboard/static/lib/workflow.js";

const overview = {
  cycle: {
    steps: [
      { step: "premarket", status: "ok", detail: "scanned 12", finished_at: "2026-10-15T12:00:00Z" },
      { step: "preflight", status: "ok", detail: "all checks passed", finished_at: "2026-10-15T12:30:00Z" },
      { step: "analysis", status: "proposed", detail: "evidence verified", finished_at: "x" },
      { step: "critique", status: "vetoed", detail: "old news", finished_at: "x" },
      { step: "slot2", status: "skipped", detail: "3 positions open", finished_at: "x" },
    ],
    slots: [],
  },
  next_step: { step: "entry", at: "2026-10-15T13:45:00Z" },
};

test("steps come in cycle order with their state", () => {
  const steps = workflowSteps(overview);
  const keys = steps.map((s) => s.key);
  assert.deepEqual(keys.slice(0, 5), ["premarket", "preflight", "analysis", "critique", "entry"]);
  const byKey = Object.fromEntries(steps.map((s) => [s.key, s]));
  assert.equal(byKey.premarket.state, "done");
  assert.equal(byKey.critique.state, "warn"); // a veto is a stop, not an error
  assert.equal(byKey.entry.state, "next");
  assert.equal(byKey.slot2.state, "skipped");
  assert.equal(byKey.report.state, "pending");
  assert.equal(byKey.preflight.label, STEP_LABELS.preflight);
});

test("errors are errors", () => {
  const steps = workflowSteps({
    cycle: { steps: [{ step: "close", status: "error", detail: "boom", finished_at: "x" }], slots: [] },
    next_step: null,
  });
  assert.equal(steps.find((s) => s.key === "close").state, "error");
});

test("no cycle yet: everything pending, the next step marked", () => {
  const steps = workflowSteps({ cycle: null, next_step: { step: "premarket", at: "x" } });
  assert.equal(steps[0].state, "next");
  assert.ok(steps.slice(1).every((s) => s.state === "pending"));
});
