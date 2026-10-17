// Formatting for the dashboard (ADR 0019). Inputs are the API's exact decimal strings
// (or numbers); outputs are display text. Nothing here touches the DOM.

const DASH = "—";

function toNumber(value) {
  if (value === null || value === undefined || value === "") return null;
  const n = typeof value === "number" ? value : Number(value);
  return Number.isFinite(n) ? n : null;
}

function round2(n) {
  const r = Math.round(n * 100) / 100;
  return Object.is(r, -0) || r === 0 ? 0 : r; // never "-0.00"
}

const usd = new Intl.NumberFormat("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });

export function money(value, { sign = false } = {}) {
  const n = toNumber(value);
  if (n === null) return DASH;
  const r = round2(n);
  const body = `$${usd.format(Math.abs(r))}`;
  if (r < 0) return `-${body}`;
  return sign && r > 0 ? `+${body}` : body;
}

export function pct(value) {
  const n = toNumber(value);
  if (n === null) return DASH;
  const r = round2(n);
  return `${r > 0 ? "+" : ""}${r.toFixed(2)}%`;
}

export function signClass(value) {
  const n = toNumber(value);
  if (n === null) return "flat";
  const r = round2(n);
  return r > 0 ? "up" : r < 0 ? "down" : "flat";
}

export function relTime(iso, now = Date.now()) {
  if (!iso) return "";
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return "";
  const diff = now - t;
  const future = diff < 0;
  const s = Math.abs(diff) / 1000;
  if (s < 60) return "now";
  const text = s < 3600 ? `${Math.floor(s / 60)}m` : s < 86400 ? `${Math.floor(s / 3600)}h` : `${Math.floor(s / 86400)}d`;
  return future ? `in ${text}` : `${text} ago`;
}

export function countdown(ms) {
  if (!ms || ms <= 0) return "now";
  const s = Math.floor(ms / 1000);
  const d = Math.floor(s / 86400);
  const h = Math.floor((s % 86400) / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  if (d > 0) return `${d}d ${h}h`;
  if (h > 0) return `${h}h ${String(m).padStart(2, "0")}m`;
  return `${m}m ${String(sec).padStart(2, "0")}s`;
}

export function compact(value) {
  const n = toNumber(value) ?? 0;
  if (Math.abs(n) >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (Math.abs(n) >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

const marketTime = new Intl.DateTimeFormat("en-US", {
  timeZone: "America/New_York",
  hour: "numeric",
  minute: "2-digit",
});
const marketDay = new Intl.DateTimeFormat("en-US", {
  timeZone: "America/New_York",
  weekday: "short",
  month: "short",
  day: "numeric",
});

// Market time (New York), e.g. "9:45 AM" or "Mon, Oct 19, 8:00 AM".
export function clockTime(iso, { withDay = false } = {}) {
  if (!iso) return "";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "";
  return withDay ? `${marketDay.format(date)}, ${marketTime.format(date)}` : marketTime.format(date);
}

const utcDay = new Intl.DateTimeFormat("en-US", {
  timeZone: "UTC",
  weekday: "short",
  month: "short",
  day: "numeric",
});

// Daily chart points are stamped at midnight UTC for that day's value.
export function dayLabel(iso) {
  if (!iso) return "";
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? "" : utcDay.format(date);
}
