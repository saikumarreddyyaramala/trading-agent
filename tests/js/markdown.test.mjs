// The guidebook viewer (ADR 0019): a tiny, safe Markdown subset, never HTML.
import test from "node:test";
import assert from "node:assert/strict";
import { parseMarkdown } from "../../src/trading_agent/dashboard/static/lib/markdown.js";

test("headings, lists, numbered steps, and paragraphs", () => {
  const blocks = parseMarkdown("# Guidebook\n\n## Part 1\n\n1. Long only.\n2. Stops.\n- [entry] Check volume.\n### gap-check: Title (candidate)\nWhen to use: gaps.\n");
  assert.deepEqual(blocks, [
    { type: "h1", text: "Guidebook" },
    { type: "h2", text: "Part 1" },
    { type: "ol", items: ["Long only.", "Stops."] },
    { type: "ul", items: ["[entry] Check volume."] },
    { type: "h3", text: "gap-check: Title (candidate)" },
    { type: "p", text: "When to use: gaps." },
  ]);
});

test("HTML stays text", () => {
  const [block] = parseMarkdown("<script>alert(1)</script>");
  assert.deepEqual(block, { type: "p", text: "<script>alert(1)</script>" });
});

test("empty input", () => {
  assert.deepEqual(parseMarkdown(""), []);
  assert.deepEqual(parseMarkdown(null), []);
});
