// The "Today" timeline (ADR 0019): every step of the daily cycle with its state.

export const STEP_LABELS = {
  premarket: "Pre-market scan",
  preflight: "Pre-flight checks",
  analysis: "Research",
  critique: "Critic review",
  entry: "Risk check & entry",
  slot2: "Trade slot 2",
  slot3: "Trade slot 3",
  close: "Force-close",
  reconcile: "Reconcile",
  postmarket: "After-hours scan",
  reflect: "Daily review",
  curate: "Guidebook",
  plan: "Next-day plan",
  report: "Report",
};

const ORDER = Object.keys(STEP_LABELS);
const DONE = new Set(["ok", "proposed", "no_trade", "entered", "approved", "disabled"]);
const WARN = new Set(["vetoed", "rejected", "failed", "not_filled"]);

function stateOf(status) {
  if (status === "error") return "error";
  if (status === "skipped") return "skipped";
  if (WARN.has(status)) return "warn";
  if (DONE.has(status)) return "done";
  return "done";
}

export function workflowSteps(overview) {
  const recorded = new Map();
  for (const step of overview?.cycle?.steps ?? []) recorded.set(step.step, step);
  const next = overview?.next_step?.step ?? null;
  return ORDER.map((key) => {
    const step = recorded.get(key);
    if (step) {
      return {
        key,
        label: STEP_LABELS[key],
        state: stateOf(step.status),
        status: step.status,
        detail: step.detail ?? "",
        at: step.finished_at ?? null,
      };
    }
    return {
      key,
      label: STEP_LABELS[key],
      state: key === next ? "next" : "pending",
      status: key === next ? "next" : "",
      detail: "",
      at: key === next ? overview.next_step.at : null,
    };
  });
}
