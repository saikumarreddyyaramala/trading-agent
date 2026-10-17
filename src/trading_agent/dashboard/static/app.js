// Trading agent dashboard (ADR 0019): read-only views over /api, kept live by /api/stream.
// Every value from the API is set with textContent; nothing is parsed as HTML.

import { money, pct, signClass, relTime, countdown, compact, clockTime, dayLabel } from "./lib/format.js";
import { renderChart } from "./lib/chart.js";
import { workflowSteps } from "./lib/workflow.js";
import { renderMarkdown } from "./lib/markdown.js";

const $ = (selector) => document.querySelector(selector);
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const STAR = "★";

const state = {
  overview: null,
  span: "1D",
  history: null,
  news: [],
  newsFilter: "all",
  guidebook: null,
  shownEquity: null,
};

// ---------- DOM helpers ----------

function h(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key === "dataset") Object.assign(node.dataset, value);
    else node.setAttribute(key, value === true ? "" : String(value));
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function safeUrl(url) {
  return typeof url === "string" && /^https?:\/\//i.test(url) ? url : null;
}

function empty(text) {
  return h("div", { class: "empty", text });
}

// Replace a node's children, dropping empty values (replaceChildren would print "null").
function put(node, ...children) {
  node.replaceChildren(...children.flat().filter((c) => c !== null && c !== undefined && c !== false));
}

async function getJSON(path) {
  const response = await fetch(path, { headers: { accept: "application/json" } });
  if (!response.ok) throw new Error(`${path}: ${response.status}`);
  return response.json();
}

let toastTimer = null;
function toast(text) {
  const node = $("#toast");
  node.textContent = text;
  node.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    node.hidden = true;
  }, 4000);
}

// ---------- Portfolio value (animated) ----------

function showEquity(value) {
  const node = $("#equity");
  const target = Number(value);
  if (!Number.isFinite(target)) {
    node.textContent = "—";
    return;
  }
  const from = state.shownEquity;
  state.shownEquity = target;
  if (reduceMotion || from === null || from === target) {
    node.textContent = money(target);
    return;
  }
  const start = performance.now();
  const step = (now) => {
    const t = Math.min((now - start) / 600, 1);
    const eased = 1 - (1 - t) ** 3;
    node.textContent = money(from + (target - from) * eased);
    if (t < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function showChange(amount, percent, label) {
  const node = $("#equity-change");
  node.className = `change ${signClass(percent ?? amount)}`;
  node.textContent =
    amount === null || amount === undefined ? "—" : `${money(amount, { sign: true })} (${pct(percent)}) ${label}`;
}

const SPAN_LABELS = { "1D": "today", "1W": "past week", "1M": "past month", "3M": "past 3 months", "1Y": "past year" };

function showResting() {
  const account = state.overview?.account;
  const history = state.history;
  $("#scrub-time").textContent = "";
  if (state.span === "1D" && account) {
    showEquity(account.equity);
    showChange(account.day_change, account.day_change_pct, "today");
  } else if (history && history.points?.length) {
    showEquity(account?.equity ?? history.points[history.points.length - 1].equity);
    showChange(history.change, history.change_pct, SPAN_LABELS[state.span]);
  } else if (account) {
    showEquity(account.equity);
    showChange(null);
  }
}

// ---------- Chart ----------

async function loadHistory() {
  try {
    state.history = await getJSON(`/api/history?span=${encodeURIComponent(state.span)}`);
  } catch {
    state.history = null;
  }
  drawChart();
  showResting();
}

function drawChart() {
  const svg = $("#chart");
  const points = state.history?.points ?? [];
  $("#chart-empty").hidden = points.length > 1;
  if (points.length < 2) {
    put(svg);
    return;
  }
  const base = Number(state.history.base_value ?? points[0].equity);
  // Color by the rounded percentage, so a few cents on a large account read as flat.
  const trend = base ? signClass(((Number(points[points.length - 1].equity) - base) / base) * 100) : "flat";
  renderChart(svg, points, {
    trend,
    baseline: base,
    onScrub: (point) => {
      if (!point) {
        showResting();
        return;
      }
      state.shownEquity = Number(point.equity);
      $("#equity").textContent = money(point.equity);
      const change = Number(point.equity) - base;
      showChange(change, base ? (change / base) * 100 : null, "");
      $("#scrub-time").textContent =
        state.span === "1D" ? clockTime(point.at) : state.span === "1W" ? clockTime(point.at, { withDay: true }) : dayLabel(point.at);
    },
  });
}

function bindRanges() {
  for (const button of $("#ranges").querySelectorAll("button")) {
    button.addEventListener("click", () => {
      state.span = button.dataset.span;
      for (const other of $("#ranges").querySelectorAll("button")) {
        other.setAttribute("aria-selected", String(other === button));
      }
      loadHistory();
    });
  }
}

// ---------- Overview: header, timeline, positions ----------

function renderOverview(overview) {
  state.overview = overview;
  const env = $("#env-badge");
  env.textContent = overview.paper ? "PAPER" : "LIVE";
  env.className = overview.paper ? "badge" : "badge live";

  const market = overview.market;
  const pill = $("#market-pill");
  if (market) {
    pill.textContent = market.is_open ? "Market open" : `Opens ${clockTime(market.next_open, { withDay: true })}`;
    pill.className = market.is_open ? "pill open" : "pill";
  } else {
    pill.textContent = "Market unknown";
    pill.className = "pill";
  }

  const notices = [];
  if (overview.kill_switch) notices.push("Kill switch is ON: research only, no new orders.");
  notices.push(...(overview.errors ?? []));
  const banner = $("#banner");
  banner.hidden = notices.length === 0;
  banner.textContent = notices.join(" · ");

  if (!document.querySelector("#chart circle[visibility='visible']")) showResting();
  renderTimeline(overview);
  renderPositions(overview.positions ?? []);
  renderSystem(overview);
}

function renderTimeline(overview) {
  const steps = workflowSteps(overview);
  put($("#timeline"), 
    ...steps.map((step) =>
      h(
        "li",
        { class: `step ${step.state}`, title: step.detail || step.label },
        h("span", { class: "marker", "aria-hidden": "true" }),
        h("span", { class: "name", text: step.label }),
        h("span", {
          class: "meta",
          text:
            step.state === "next"
              ? `next · ${clockTime(step.at)}`
              : step.state === "pending"
                ? "later"
                : [step.status, step.detail].filter(Boolean).join(" · "),
        }),
      ),
    ),
  );
}

function tickCountdown() {
  const next = state.overview?.next_step;
  const node = $("#next-step");
  if (!next) {
    node.textContent = "";
    return;
  }
  const label = workflowSteps({ cycle: null, next_step: next }).find((s) => s.key === next.step)?.label ?? next.step;
  node.textContent = `${label} in ${countdown(Date.parse(next.at) - Date.now())}`;
}

function renderPositions(positions) {
  $("#positions-count").textContent = positions.length ? `${positions.length} open` : "";
  if (!positions.length) {
    put($("#positions"), empty("No open positions. Entries happen at the trade slots."));
    return;
  }
  put($("#positions"), 
    ...positions.map((p) => {
      const stop = Number(p.stop_price);
      const target = Number(p.target_price);
      const price = Number(p.price);
      const where = target > stop ? Math.min(Math.max((price - stop) / (target - stop), 0), 1) : 0.5;
      const marker = h("span", { class: "now" });
      marker.style.setProperty("left", `${(where * 100).toFixed(1)}%`);
      return h(
        "div",
        { class: "position" },
        h(
          "div",
          {},
          h("div", { class: "sym", text: p.symbol }),
          h("div", { class: "sub", text: `${p.quantity} sh · avg ${money(p.avg_entry_price)} · now ${money(p.price)}` }),
        ),
        h(
          "div",
          { class: `pnl ${signClass(p.unrealized_pnl)}` },
          h("div", { text: money(p.unrealized_pnl, { sign: true }) }),
          h("div", { class: "sub", text: pct(p.unrealized_pct) }),
        ),
        h("div", { class: "range-bar", title: "Price between the stop-loss and the target" }, marker),
        h(
          "div",
          { class: "range-labels" },
          h("span", { text: `stop ${money(p.stop_price)}` }),
          h("span", { text: `target ${money(p.target_price)}` }),
        ),
      );
    }),
  );
}

// ---------- News ----------

async function loadNews() {
  try {
    state.news = await getJSON("/api/news?limit=30");
  } catch {
    state.news = [];
  }
  renderNews();
}

function renderNews() {
  const cards = state.news.filter((c) =>
    state.newsFilter === "watched" ? c.watched : state.newsFilter === "major" ? c.major : true,
  );
  if (!cards.length) {
    put($("#news"), empty(state.news.length ? "Nothing matches this filter." : "No news in the last 24 hours."));
    return;
  }
  put($("#news"), 
    ...cards.slice(0, 18).map((card) => {
      const url = safeUrl(card.url);
      const chips = [
        ...card.catalysts.map((tag) => h("span", { class: `chip${card.major ? " major" : ""}`, text: tag })),
        card.move_pct !== null
          ? h("span", { class: `chip move ${signClass(card.move_pct)}`, text: `${card.symbols[0]} ${pct(card.move_pct)}` })
          : null,
      ];
      const body = [
        h(
          "div",
          { class: "news-top" },
          h(
            "span",
            { class: "news-symbols" },
            card.watched ? h("span", { class: "star", title: "Held, planned, or traded", text: `${STAR} ` }) : null,
            card.symbols.join(" · ") || "Market",
          ),
          h("span", { text: `${card.source} · ${card.age}` }),
        ),
        h("div", { class: "news-headline", text: card.headline }),
        card.key_fact ? h("div", { class: "news-fact", text: card.key_fact }) : null,
        h("div", { class: "chips" }, ...chips),
      ];
      return url
        ? h("a", { class: `news-card${card.major ? " major" : ""}`, href: url, target: "_blank", rel: "noopener noreferrer" }, ...body)
        : h("div", { class: `news-card${card.major ? " major" : ""}` }, ...body);
    }),
  );
}

function bindNewsFilters() {
  for (const button of $("#news-filters").querySelectorAll("button")) {
    button.addEventListener("click", () => {
      state.newsFilter = button.dataset.filter;
      for (const other of $("#news-filters").querySelectorAll("button")) {
        other.setAttribute("aria-selected", String(other === button));
      }
      renderNews();
    });
  }
}

// ---------- Trades and performance ----------

async function loadTrades() {
  try {
    const [trades, perf] = await Promise.all([getJSON("/api/trades?days=30"), getJSON("/api/performance?days=30")]);
    const stat = (label, value, cls = "") =>
      h("div", { class: "stat" }, h("div", { class: "label", text: label }), h("div", { class: `value ${cls}`, text: value }));
    put($("#perf"), 
      stat("Trades", String(perf.trades ?? 0)),
      stat("Win rate", perf.win_rate_pct === null ? "—" : `${Number(perf.win_rate_pct).toFixed(0)}%`),
      stat("Total P&L", money(perf.total_pnl_usd, { sign: true }), signClass(perf.total_pnl_usd)),
      stat("Avg return", pct(perf.avg_return_pct), signClass(perf.avg_return_pct)),
    );
    put($("#trades"), 
      ...(trades.length
        ? trades.map((t) =>
            h(
              "tr",
              {},
              h("td", { text: t.day }),
              h("td", { text: t.symbol }),
              h("td", { class: "num", text: money(t.entry_price) }),
              h("td", { class: "num", text: t.exit_price === null ? "—" : money(t.exit_price) }),
              h("td", { class: `num ${signClass(t.pnl_usd)}`, text: t.pnl_usd === null ? "open" : money(t.pnl_usd, { sign: true }) }),
              h("td", { class: `num ${signClass(t.return_pct)}`, text: pct(t.return_pct) }),
              h("td", { text: t.exit_via }),
            ),
          )
        : [
            h(
              "tr",
              {},
              h("td", { colspan: "7", class: "empty", text: "No trades yet. The first paper trades come from the next trading day's cycle." }),
            ),
          ]),
    );
  } catch {
    put($("#perf"), empty("Trades are unavailable right now."));
  }
}

// ---------- Activity ----------

function activityItem(item) {
  return h(
    "li",
    {},
    h("span", { class: `kind ${item.kind}`, "aria-hidden": "true" }),
    h("div", {}, h("div", { class: "title", text: item.title }), item.detail ? h("div", { class: "detail", text: item.detail }) : null),
    h("span", { class: "when", text: relTime(item.at), title: clockTime(item.at, { withDay: true }) }),
  );
}

async function loadActivity() {
  try {
    const items = await getJSON("/api/activity?limit=40");
    put($("#activity"), 
      ...(items.length ? items.map(activityItem) : [empty("Nothing yet. Steps, orders, and lessons appear here live.")]),
    );
  } catch {
    put($("#activity"), empty("Activity is unavailable right now."));
  }
}

function addActivity(item) {
  const list = $("#activity");
  list.querySelector(".empty")?.remove();
  list.prepend(activityItem(item));
  while (list.children.length > 60) list.lastElementChild.remove();
  if (item.kind === "order" || item.kind === "position") toast(item.title);
}

// ---------- Plan, watch, learning ----------

function priorityTag(item) {
  return h("span", { class: `tag p${item.priority}`, text: item.kind });
}

async function loadPlanAndWatch() {
  try {
    const [plan, watch] = await Promise.all([getJSON("/api/plan"), getJSON("/api/watch")]);
    const sections = [];
    for (const [title, found] of [
      ["For today", plan.today],
      [`For ${plan.next?.for_day ?? "next session"}`, plan.next],
    ]) {
      if (!found) continue;
      sections.push(h("div", { class: "sub-title", text: title }));
      sections.push(
        found.items.length
          ? h("ul", { class: "list" }, ...found.items.map((i) => h("li", {}, priorityTag(i), i.text)))
          : empty(found.summary),
      );
    }
    put($("#plan"), ...(sections.length ? sections : [empty("No plan yet. One is written after each session.")]));

    const watchParts = [];
    for (const [title, report] of [
      ["Pre-market", watch.pre],
      ["After hours", watch.post],
    ]) {
      if (!report) continue;
      watchParts.push(h("div", { class: "sub-title", text: `${title} · ${report.items.length} symbols` }));
      watchParts.push(
        report.alerts.length ? h("ul", { class: "list" }, ...report.alerts.map((a) => h("li", { text: a }))) : empty("No alerts."),
      );
    }
    put($("#watch"), 
      ...(watchParts.length ? watchParts : [empty("Scans run at 8:00 AM and 4:45 PM ET on trading days.")]),
    );
  } catch {
    put($("#plan"), empty("The plan is unavailable right now."));
  }
}

async function loadLearning() {
  try {
    const data = await getJSON("/api/learning");
    state.guidebook = data.guidebook;
    const parts = [];
    parts.push(h("div", { class: "sub-title", text: `Principles · ${data.principles.length}` }));
    parts.push(
      data.principles.length
        ? h("ul", { class: "list" }, ...data.principles.map((l) => h("li", {}, h("span", { class: "tag", text: l.category }), l.text)))
        : empty("None yet: lessons become principles once trades prove them."),
    );
    parts.push(h("div", { class: "sub-title", text: `Recent lessons · ${data.lessons.length}` }));
    parts.push(
      data.lessons.length
        ? h("ul", { class: "list" }, ...data.lessons.slice(0, 6).map((l) => h("li", {}, h("span", { class: "tag", text: l.category }), l.text)))
        : empty("None yet: the evening review writes them."),
    );
    const live = data.playbooks.filter((p) => p.status !== "retired");
    parts.push(h("div", { class: "sub-title", text: `Playbooks · ${live.length}` }));
    parts.push(
      live.length
        ? h(
            "ul",
            { class: "list" },
            ...live.map((p) => h("li", { title: p.when_to_use }, p.title, h("span", { class: `status-pill ${p.status}`, text: p.status }))),
          )
        : empty("None yet."),
    );
    put($("#learning"), ...parts);
    $("#open-guidebook").hidden = !data.guidebook;
  } catch {
    put($("#learning"), empty("Learning is unavailable right now."));
  }
}

function openGuidebook() {
  const book = state.guidebook;
  if (!book) return;
  $("#guidebook-meta").textContent = `Version ${book.version} · ${clockTime(book.created_at, { withDay: true })} · ${book.changes.join("; ")}`;
  renderMarkdown($("#guidebook-body"), book.content);
  $("#guidebook").hidden = false;
  $("#close-guidebook").focus();
}

// ---------- Settings: costs and system ----------

async function loadCosts() {
  try {
    const report = await getJSON("/api/costs?days=30");
    const today = report.days[report.days.length - 1];
    const budget = report.daily_budget_usd === null ? null : Number(report.daily_budget_usd);
    const spent = Number(today?.known_llm_usd ?? 0);
    const used = budget ? Math.min(spent / budget, 1) : 0;
    const fill = h("span", { class: used >= 1 ? "over" : "" });
    fill.style.setProperty("width", `${(used * 100).toFixed(1)}%`);
    const dayTotal = (d) => Number(d.known_llm_usd) + Number(d.services_usd);
    const peak = Math.max(...report.days.map(dayTotal), 0.0001);
    const bars = report.days.map((d) => {
      const bar = h("span", { title: `${d.day}: ${money(dayTotal(d))}` });
      bar.style.setProperty("height", `${Math.max((dayTotal(d) / peak) * 100, 2).toFixed(1)}%`);
      return bar;
    });
    const agents = Object.entries(today?.by_agent ?? {});
    const total = report.total;
    put($("#costs"), 
      h(
        "div",
        { class: "budget" },
        h(
          "dl",
          { class: "kv" },
          h("dt", { text: "Today, LLM" }),
          h("dd", { text: `${money(spent)}${today?.unpriced_calls ? " + unpriced" : ""}${budget ? ` of ${money(budget)}` : ""}` }),
        ),
        budget
          ? h(
              "div",
              { class: "budget-bar", role: "progressbar", "aria-valuemin": "0", "aria-valuemax": "100", "aria-valuenow": String(Math.round(used * 100)) },
              fill,
            )
          : null,
      ),
      agents.length
        ? h(
            "dl",
            { class: "kv" },
            ...agents.flatMap(([name, cost]) => [
              h("dt", { text: name }),
              h("dd", {
                text: `${cost.usd === null ? `${money(cost.known_usd)} + unpriced` : money(cost.usd)} · ${cost.calls} calls · ${compact(cost.input_tokens + cost.output_tokens)} tokens`,
              }),
            ]),
          )
        : empty("No model calls today."),
      h("div", { class: "sub-title", text: "Last 30 days" }),
      h("div", { class: "bars", "aria-hidden": "true" }, ...bars),
      h(
        "dl",
        { class: "kv" },
        h("dt", { text: "LLM" }),
        h("dd", { text: total.llm_usd === null ? `${money(total.known_llm_usd)} + unpriced` : money(total.llm_usd) }),
        h("dt", { text: "Services" }),
        h("dd", { text: money(total.services_usd) }),
        h("dt", { text: "Total" }),
        h("dd", {
          text: total.total_usd === null ? `${money(Number(total.known_llm_usd) + Number(total.services_usd))} + unpriced` : money(total.total_usd),
        }),
        h("dt", { text: "Calls" }),
        h("dd", { text: `${total.calls} (${total.failed_calls} failed)` }),
      ),
      h("div", { class: "sub-title", text: "Paid services" }),
      h("dl", { class: "kv" }, ...report.services.flatMap((s) => [h("dt", { text: s.name }), h("dd", { text: `${money(s.monthly_usd)}/month` })])),
      report.unpriced_models.length
        ? h("p", {
            class: "note",
            text: `No price set for ${report.unpriced_models.join(", ")}. Add it under costs.llm_prices in config/base.yaml to see real dollars and let the daily budget count it.`,
          })
        : null,
    );
  } catch {
    put($("#costs"), empty("Costs are unavailable right now."));
  }
}

function renderSystem(overview) {
  const rows = [
    ["Environment", overview.paper ? "Paper (simulated money)" : "LIVE (real money)"],
    ["Account", overview.account?.account ?? "—"],
    ["Cash", money(overview.account?.cash)],
    ["Buying power", money(overview.account?.buying_power)],
    ["Kill switch", overview.kill_switch ? "ON" : "off"],
    ["Updated", clockTime(overview.generated_at, { withDay: true })],
  ];
  put($("#system"), ...rows.flatMap(([k, v]) => [h("dt", { text: k }), h("dd", { text: v })]));
}

function openDrawer() {
  $("#drawer").hidden = false;
  $("#close-settings").focus();
  loadCosts();
}

function closeOverlays() {
  $("#drawer").hidden = true;
  $("#guidebook").hidden = true;
}

function bindOverlays() {
  $("#open-settings").addEventListener("click", openDrawer);
  $("#close-settings").addEventListener("click", closeOverlays);
  $("#open-guidebook").addEventListener("click", openGuidebook);
  $("#close-guidebook").addEventListener("click", closeOverlays);
  for (const overlay of [$("#drawer"), $("#guidebook")]) {
    overlay.addEventListener("click", (event) => {
      if (event.target === overlay) closeOverlays();
    });
  }
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeOverlays();
  });
}

// ---------- Theme ----------

function bindTheme() {
  let saved = null;
  try {
    saved = localStorage.getItem("theme");
  } catch {
    saved = null;
  }
  if (saved === "light" || saved === "dark") document.documentElement.dataset.theme = saved;
  const toggle = $("#theme-toggle");
  toggle.checked = document.documentElement.dataset.theme === "light";
  toggle.addEventListener("change", () => {
    const theme = toggle.checked ? "light" : "dark";
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem("theme", theme);
    } catch {
      /* private mode: the choice lasts for this visit */
    }
    drawChart();
  });
}

// ---------- Live updates ----------

function connect() {
  const conn = $("#conn");
  const text = $("#conn-text");
  const source = new EventSource("/api/stream");
  source.addEventListener("open", () => {
    conn.className = "conn live";
    text.textContent = "Live";
  });
  source.addEventListener("overview", (event) => renderOverview(JSON.parse(event.data)));
  source.addEventListener("activity", (event) => addActivity(JSON.parse(event.data)));
  source.addEventListener("error", () => {
    conn.className = "conn lost";
    text.textContent = "Reconnecting";
  });
}

async function start() {
  bindTheme();
  bindRanges();
  bindNewsFilters();
  bindOverlays();
  try {
    renderOverview(await getJSON("/api/overview"));
  } catch {
    $("#banner").hidden = false;
    $("#banner").textContent = "The dashboard API is not answering. Is `trading-agent dashboard` running?";
  }
  await Promise.all([loadHistory(), loadNews(), loadTrades(), loadActivity(), loadPlanAndWatch(), loadLearning()]);
  connect();
  setInterval(tickCountdown, 1000);
  tickCountdown();
  setInterval(loadHistory, 60_000);
  setInterval(loadTrades, 60_000);
  setInterval(loadNews, 120_000);
  setInterval(loadPlanAndWatch, 120_000);
  setInterval(loadLearning, 300_000);
}

start();
