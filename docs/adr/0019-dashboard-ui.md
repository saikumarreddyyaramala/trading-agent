# ADR 0019: The dashboard interface: self-contained, safe, and live

- Status: Accepted
- Date: 2026-10-17

## Context

The operator asked for a lightweight dashboard: live graphs, animation, and the clarity of
Robinhood without copying it. It should keep them aware of the workflow and the portfolio,
show the news as key facts, and keep costs under Settings. It reads the API from ADR 0018.

## Decisions

### 1. No build step, no third parties
The interface is plain HTML, CSS, and native JavaScript modules in
`dashboard/static/`. The dashboard server serves them. There is no framework, bundler,
`node_modules`, CDN, web font, or analytics. It works offline and starts instantly, and
nothing outside this repository runs in the page.

### 2. Safe by construction
- A strict Content Security Policy: `default-src 'self'`, scripts and styles only from the
  page's own files, `connect-src 'self'`, and `frame-ancestors 'none'`. There are no inline
  scripts or event attributes.
- Every value from the API, including news headlines, which are third-party text, is
  inserted with `textContent`, never parsed as HTML. Tests forbid `innerHTML`,
  `insertAdjacentHTML`, `document.write`, `eval`, and `new Function` in every script.
- The guidebook viewer renders a small Markdown subset (headings, lists, paragraphs) as
  DOM nodes, so markup in the data stays plain text.
- External news links must be `http(s)` and open with `rel="noopener noreferrer"`.
- Children are replaced only through one helper that drops empty values, so a missing
  section never prints "null". A test enforces this.

### 3. What it shows
- **Portfolio**: the value, which counts up when it changes, and the day's change in green
  or red. Below it, an animated equity chart with 1D, 1W, 1M, 3M, and 1Y ranges. Dragging
  or hovering scrubs the value and the time, Robinhood-style.
  - The vertical range is at least 0.2% of the account, so cents never look like a cliff.
  - The color follows the rounded percentage, so a flat day reads flat.
  - Days before the account was funded are left out.
  - Daily points are labeled by their own date.
- **Today**: every step of the cycle as a timeline: done, stopped by a gate, failed,
  skipped, next with a live countdown, or later.
- **Positions**: P&L, and where the price sits between the stop-loss and the target.
- **News**: the cards from ADR 0018, with filters for All, Watched, and Major.
- **Trades**: the last 30 days, with the win rate, total P&L, and average return.
- **Activity**: steps, orders, exits, lessons, and guidebook editions, streamed live.
  Orders and exits also show a toast.
- **Plan and Market watch**: the next-day plan and the scan alerts.
- **Learning**: principles, recent lessons, playbooks, and a guidebook viewer.
- **Settings** (the gear):
  - costs: today against the budget, by agent with calls and tokens, a 30-day bar chart,
    totals, paid services, and a warning naming any model without a price;
  - a light or dark theme;
  - system details.

### 4. Live, and honest about it
The page connects to `/api/stream`, which reconnects on its own. A dot in the top bar shows
Live or Reconnecting. Slower data refreshes on timers: the chart and trades every minute,
news and the plan every two minutes. Source failures show in a banner, never as silent
blanks.

### 5. Accessible and calm
- Semantic landmarks, labeled controls, `aria-live` on the timeline, activity, and toasts,
  and visible focus rings.
- Escape closes the drawer and the guidebook.
- `prefers-reduced-motion` turns off every animation.
- The layout goes from two columns to one under 960px and stays usable at 375px wide, with
  no sideways scrolling.
- Dark theme first, one accent per direction.

### 6. Tested where it matters
- The UI logic (number and time formatting, chart geometry, the timeline, the Markdown
  subset) is pure JavaScript, unit-tested with Node's built-in test runner (`make test-ui`,
  and in CI). No npm packages are needed.
- Python tests check:
  - how the page is served, with its headers;
  - that nothing loads from another origin;
  - the unsafe-API bans;
  - that every element the script looks up exists in the page.
- The page was also checked in a browser against the real paper account. That found and
  fixed: the cliff on a 6-cent day, $0 days before funding, daily points labeled as the
  previous evening, a stray "null", and news sentences glued together without a space.

## Consequences

- The interface is about 60 KB of readable source that can be changed without a toolchain.
- Hand-written SVG charts are limited to what the dashboard needs (one equity line). A
  richer chart would need a vendored library, reviewed and served locally, never a CDN.
- `make check` and CI now also need Node 22 or newer.
