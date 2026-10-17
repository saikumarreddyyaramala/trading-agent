// The guidebook viewer (ADR 0019): headings, lists, and paragraphs only. Text is always
// text: `renderMarkdown` builds nodes with textContent, so markup in the data is inert.

export function parseMarkdown(source) {
  const blocks = [];
  let list = null;
  const flush = () => {
    if (list) blocks.push(list);
    list = null;
  };
  for (const raw of String(source ?? "").split(/\r?\n/)) {
    const line = raw.trim();
    if (!line) {
      flush();
      continue;
    }
    const heading = /^(#{1,3})\s+(.*)$/.exec(line);
    const numbered = /^\d+\.\s+(.*)$/.exec(line);
    const bullet = /^[-*]\s+(.*)$/.exec(line);
    if (heading) {
      flush();
      blocks.push({ type: `h${heading[1].length}`, text: heading[2] });
    } else if (numbered) {
      if (!list || list.type !== "ol") flush();
      list = list ?? { type: "ol", items: [] };
      list.items.push(numbered[1]);
    } else if (bullet) {
      if (!list || list.type !== "ul") flush();
      list = list ?? { type: "ul", items: [] };
      list.items.push(bullet[1]);
    } else {
      flush();
      blocks.push({ type: "p", text: line });
    }
  }
  flush();
  return blocks;
}

export function renderMarkdown(container, source) {
  container.replaceChildren();
  for (const block of parseMarkdown(source)) {
    if (block.type === "ol" || block.type === "ul") {
      const list = document.createElement(block.type);
      for (const item of block.items) {
        const li = document.createElement("li");
        li.textContent = item;
        list.append(li);
      }
      container.append(list);
    } else {
      const node = document.createElement(block.type);
      node.textContent = block.text;
      container.append(node);
    }
  }
}
