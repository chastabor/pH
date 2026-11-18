// The drill-down layer `build.py` adds to each page. Clicking a node or an arrow
// that has details opens a panel with them; the launcher lists everything the
// page can open. `?drill=node:<id>`, `edge:<id>`, `topic:<id>` or `index` opens
// one on load, which is also how one page links into another.
(() => {
  const source = document.getElementById("ph-drill-data");
  if (!source) return;
  const data = JSON.parse(source.textContent);
  const svg = document.querySelector("svg[data-preset]") || document.querySelector("svg");

  const MODES = {
    emit: "a fact: every listener is told, return values are ignored, a listener that raises is logged",
    serial: "listeners are awaited one at a time, in registration order, until one bails",
    parallel: "every listener runs at once and all are awaited; failures are collected",
    waterfall: "around-middleware: each listener gets next() and may change the value or refuse by not calling it",
  };
  const TRANSPORTS = {
    bus: "bus event",
    service: "service call",
    frame: "fd 3 frame",
    socket: "unix socket",
    http: "HTTP",
    process: "subprocess",
    log: "log record",
    file: "file",
    call: "function call",
  };

  const el = (tag, attrs = {}, ...kids) => {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs)) {
      if (value == null || value === false) continue;
      if (key === "html") node.innerHTML = value;
      else if (key === "text") node.textContent = value;
      else if (key === "class") node.className = value;
      else node.setAttribute(key, value);
    }
    for (const kid of kids.flat()) if (kid) node.append(kid);
    return node;
  };
  const chip = (label, value, tone) =>
    el("span", { class: "ph-chip", "data-tone": tone }, label ? `${label} ` : "", el("b", { text: value }));
  const code = (text) => el("code", { text });
  // A block of code scrolls sideways; its button wraps long lines instead.
  const pre = (text) => {
    const block = el("pre", {}, el("code", { text }));
    if (!text.split("\n").some((line) => line.length > 74)) return block;
    const toggle = el("button", { class: "ph-wrap", type: "button", "aria-pressed": "false", text: "wrap" });
    toggle.addEventListener("click", () => {
      const on = block.classList.toggle("is-wrapped");
      toggle.setAttribute("aria-pressed", String(on));
      toggle.textContent = on ? "unwrap" : "wrap";
    });
    return el("div", { class: "ph-pre" }, toggle, block);
  };
  // A long field table starts folded, so a panel of several structures can be scanned.
  const folded = (count, noun, content) =>
    count <= 8 ? content : el("details", { class: "ph-fold" }, el("summary", { text: `${count} ${noun}` }), content);
  const table = (head, rows) =>
    el(
      "div",
      { class: "ph-table-wrap" },
      el(
        "table",
        {},
        el("thead", {}, el("tr", {}, head.map((cell) => el("th", { html: cell })))),
        el("tbody", {}, rows.map((row) => el("tr", {}, row.map((cell) => (cell instanceof Node ? el("td", {}, cell) : el("td", { html: cell })))))),
      ),
    );
  const fieldCell = (field) =>
    el(
      "span",
      {},
      el("span", { class: "ph-field ph-mono", text: field.name }),
      field.required ? el("span", { class: "ph-req", text: " required" }) : null,
      el("span", { class: "ph-type ph-mono", text: field.type || "" }),
    );
  const fieldsTable = (fields, withDefault) =>
    table(
      withDefault ? ["field", "default", "what it is"] : ["field", "what it is"],
      fields.map((field) =>
        withDefault
          ? [fieldCell(field), field.required || field.default == null || field.default === "" ? "" : code(String(field.default)), field.doc || ""]
          : [fieldCell(field), field.doc || ""],
      ),
    );

  const renderers = {
    text: (block) => [el("div", { html: block.html })],
    code: (block) => [pre(block.text)],
    list: (block) => [el("div", { class: "ph-callout", "data-tone": block.tone }, el("ul", {}, block.items.map((item) => el("li", { html: item }))))],
    table: (block) => [table(block.columns, block.rows)],
    links: (block) => [el("ul", {}, block.items.map((item) => el("li", {}, el("a", { href: item.href, text: item.label }))))],
    struct: (block) => [
      el("p", { class: "ph-block-lead" }, code(block.name), " ", el("span", { class: "ph-note", text: block.module })),
      block.doc ? el("p", { html: block.doc }) : null,
      block.fields.length ? folded(block.fields.length, "fields", fieldsTable(block.fields, true)) : null,
      block.methods && block.methods.length
        ? folded(
            block.methods.length,
            "methods",
            table(
              ["method", "what it does"],
              block.methods.map((one) => [el("span", { class: "ph-mono ph-method", text: one.name }), one.doc || ""]),
            ),
          )
        : null,
      block.partial ? el("p", { class: "ph-note", text: "Only the parts that matter here; the class has more." }) : null,
    ],
    row: (block) => {
      const out = [
        el("p", { class: "ph-block-lead" }, code(block.name)),
        el(
          "div",
          { class: "ph-chips" },
          chip("module", block.module),
          block.affects ? chip("affects", block.affects) : null,
          block.injects.map((key) => chip("injects", key)),
        ),
      ];
      for (const shipped of block.shipped) {
        const lines = [`- id: ${shipped.id}`, `  name: ${block.name}`];
        if (shipped.disabled) lines.push("  disabled: true");
        if (shipped.config) lines.push("  config:", ...shipped.config.split("\n").map((line) => `    ${line}`));
        out.push(
          el("p", {}, "Shipped in ", code(shipped.file), " ", chip("", shipped.disabled ? "off until a profile arms it" : "on", shipped.disabled ? "off" : "on")),
          pre(lines.join("\n")),
        );
      }
      if (!block.shipped.length) out.push(el("p", { class: "ph-note", text: "No shipped profile mounts this row; a profile inserts it." }));
      const off = block.shipped.find((shipped) => shipped.disabled);
      if (off) {
        out.push(
          el("p", {}, "To switch it on, in a profile's ", code("rows:"), " or for one run:"),
          pre(`- id: ${off.id}\n  disabled: false\n\nphern --patch '{id: ${off.id}, disabled: false}' ...`),
        );
      }
      out.push(
        block.options.length
          ? folded(block.options.length, "options", fieldsTable(block.options, true))
          : el("p", { class: "ph-note", text: "This row takes no configuration." }),
      );
      return out;
    },
    event: (block) => [
      el("p", { class: "ph-block-lead" }, code(block.name)),
      el("div", { class: "ph-chips" }, chip("dispatch", block.mode, block.mode), chip("declared by", block.producer || "?")),
      el("p", { class: "ph-note", text: `${block.mode}: ${MODES[block.mode] || ""}.` }),
      block.doc ? el("p", { html: block.doc }) : null,
      block.consumers.length
        ? table(
            ["listener", "in profiles"],
            block.consumers.map((one) => [code(one.module), one.profiles.join(", ")]),
          )
        : el("p", { class: "ph-note", text: "No row in the shipped profiles listens to it." }),
      block.payload
        ? el(
            "div",
            {},
            el("p", { class: "ph-block-lead" }, "Payload ", code(block.payload.name), " ", el("span", { class: "ph-note", text: block.payload.module })),
            block.payload.doc ? el("p", { html: block.payload.doc }) : null,
            folded(block.payload.fields.length, "fields", fieldsTable(block.payload.fields, true)),
          )
        : block.args
          ? null
          : el("p", { class: "ph-note", text: "Declared without a structured payload type." }),
      block.args
        ? el("div", {}, el("p", { class: "ph-block-lead", text: "Arguments, as the dispatch site passes them" }), table(["argument", "type", "what it is"], block.args))
        : null,
    ],
    frame: (block) => [
      el("p", { class: "ph-block-lead" }, code(block.name)),
      el(
        "div",
        { class: "ph-chips" },
        chip("sent by", block.from),
        chip("", "one line of JSON on fd 3"),
        block.unproduced ? chip("", "defined, nothing sends it yet", "off") : null,
      ),
      table(
        ["field", "required"],
        [...block.required.map((name) => [code(name), "yes"]), ...block.optional.map((name) => [code(name), "no"])],
      ),
      block.example ? pre(block.example) : null,
    ],
    record: (block) => [
      el("p", { class: "ph-block-lead" }, code(block.name)),
      el(
        "div",
        { class: "ph-chips" },
        block.writer.filter(Boolean).map((module) => chip("written by", module)),
        chip("", block.ignorable ? "ignorable by an older reader" : "required to read the log", block.ignorable ? "on" : "off"),
        block.auditOnly ? chip("", "auditor's view only") : null,
      ),
      block.exported.length ? el("p", {}, "Telemetry ships: ", block.exported.flatMap((name, index) => [index ? ", " : "", code(name)])) : null,
      block.example ? pre(block.example) : null,
    ],
  };
  const HEADINGS = { row: "Row", event: "Bus event", struct: "Structure", frame: "Frame", record: "Log record" };

  const renderBlock = (block) => {
    const title = block.title || HEADINGS[block.kind];
    return el(
      "section",
      { class: "ph-block" },
      title ? el("h3", { html: title }) : null,
      (renderers[block.kind] || (() => []))(block),
      block.note ? el("div", { class: "ph-note", html: block.note }) : null,
    );
  };

  // --- the panel -------------------------------------------------------------
  const kicker = el("span", { class: "ph-drill-kicker" });
  const heading = el("span");
  const body = el("div", { class: "ph-drill-body" });
  const back = el("button", { class: "ph-drill-btn", type: "button", "aria-label": "Everything this page can open", text: "All" });
  const close = el("button", { class: "ph-drill-btn", type: "button", "aria-label": "Close details", text: "✕" });
  const panel = el(
    "aside",
    { class: "ph-drill", "aria-label": "Details", tabindex: "-1" },
    el("div", { class: "ph-drill-head" }, el("h2", {}, kicker, heading), back, close),
    body,
  );
  const launch = el("button", { class: "ph-launch", type: "button" }, "Details");
  document.body.append(panel, launch);

  const nodeLabel = (id) => (data.nodes[id] && (data.nodes[id].title || data.nodes[id].label)) || svg?.querySelector(`[data-node-id="${CSS.escape(id)}"]`)?.getAttribute("data-node-label") || id;
  const edgeEnds = (id) => {
    const path = svg?.querySelector(`[data-edge-id="${CSS.escape(id)}"]`);
    if (!path) return "";
    const from = path.getAttribute("data-edge-from");
    const to = path.getAttribute("data-edge-to");
    return from && to ? `${nodeLabel(from)} → ${nodeLabel(to)}` : "";
  };

  let marked = null;
  const mark = (target) => {
    marked?.classList.remove("ph-is-open-target");
    marked = target || null;
    marked?.classList.add("ph-is-open-target");
  };
  const remember = (drill) => {
    try {
      const url = new URL(location.href);
      if (drill) url.searchParams.set("drill", drill);
      else url.searchParams.delete("drill");
      history.replaceState(null, "", url);
    } catch {
      // A file: page may refuse; the panel works without the address.
    }
  };

  const show = (kick, title, parts, drill, target) => {
    kicker.textContent = kick;
    heading.textContent = title;
    body.replaceChildren(...parts.filter(Boolean));
    body.scrollTop = 0;
    panel.classList.add("is-open");
    back.hidden = drill === "index";
    mark(target);
    remember(drill);
  };

  const entryParts = (entry, extraChips = []) => [
    el(
      "div",
      { class: "ph-chips" },
      extraChips,
      entry.transport ? chip("", TRANSPORTS[entry.transport] || entry.transport) : null,
      entry.mode ? chip("dispatch", entry.mode, entry.mode) : null,
    ),
    entry.summary ? el("div", { html: entry.summary }) : null,
    ...entry.blocks.map(renderBlock),
  ];

  const item = (drill, title, sub) => {
    const button = el("button", { class: "ph-drill-item", type: "button" }, el("strong", { text: title }), sub ? el("span", { text: sub }) : null);
    button.addEventListener("click", () => open(drill));
    return el("li", {}, button);
  };
  const group = (title, items) => (items.length ? el("section", { class: "ph-block" }, el("h3", { text: title }), el("ul", { class: "ph-drill-list" }, items)) : null);
  const plain = (htmlText) => el("div", { html: htmlText || "" }).textContent.trim();
  const brief = (entry) => {
    const text = plain(entry.summary);
    return text.length > 110 ? `${text.slice(0, 108)}…` : text;
  };

  const openIndex = () =>
    show(
      "This page",
      data.title,
      [
        data.intro ? el("div", { html: data.intro }) : null,
        el("p", { class: "ph-note", text: "Click a box or an arrow in the diagram, or pick one here." }),
        group("Topics", data.topics.map((topic) => item(`topic:${topic.id}`, topic.title || topic.id, brief(topic)))),
        group("Boxes", Object.entries(data.nodes).map(([id, entry]) => item(`node:${id}`, entry.title || entry.label, brief(entry)))),
        group(
          "Arrows",
          Object.entries(data.edges).map(([id, entry]) =>
            item(`edge:${id}`, entry.title || entry.label, [edgeEnds(id), entry.mode ? `${entry.mode}` : TRANSPORTS[entry.transport] || ""].filter(Boolean).join(" · ")),
          ),
        ),
        el(
          "section",
          { class: "ph-block" },
          el("h3", { text: "Other pages" }),
          el("ul", {}, data.pages.filter((page) => page.page !== data.page).map((page) => el("li", {}, el("a", { href: `${page.page}.html?drill=index`, text: page.title })))),
        ),
      ],
      "index",
    );

  function open(drill) {
    const [kind, ...rest] = String(drill).split(":");
    const id = rest.join(":");
    if (kind === "node" && data.nodes[id]) {
      const entry = data.nodes[id];
      return show("Box", entry.title || entry.label, entryParts(entry), drill, svg?.querySelector(`[data-node-id="${CSS.escape(id)}"]`));
    }
    if (kind === "edge" && data.edges[id]) {
      const entry = data.edges[id];
      const ends = edgeEnds(id);
      return show("Arrow", entry.title || entry.label, entryParts(entry, ends ? [chip("", ends)] : []), drill, svg?.querySelector(`[data-edge-id="${CSS.escape(id)}"]`));
    }
    const topic = kind === "topic" && data.topics.find((one) => one.id === id);
    if (topic) return show("Topic", topic.title || topic.id, entryParts(topic), drill);
    return openIndex();
  }
  const shut = () => {
    panel.classList.remove("is-open");
    mark(null);
    remember("");
  };

  launch.addEventListener("click", () => (panel.classList.contains("is-open") ? shut() : openIndex()));
  back.addEventListener("click", openIndex);
  close.addEventListener("click", shut);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && panel.classList.contains("is-open")) shut();
  });
  // A link to something on this page opens it here rather than loading the page again.
  panel.addEventListener("click", (event) => {
    const link = event.target.closest("a[href]");
    if (!link) return;
    const href = link.getAttribute("href");
    const local = href.startsWith("?drill=") ? href : href.startsWith(`${data.page}.html?drill=`) ? href.slice(`${data.page}.html`.length) : "";
    if (!local) return;
    event.preventDefault();
    open(new URLSearchParams(local).get("drill"));
  });

  // --- the diagram -----------------------------------------------------------
  if (svg) {
    const NS = "http://www.w3.org/2000/svg";
    for (const id of Object.keys(data.nodes)) {
      const node = svg.querySelector(`[data-node-id="${CSS.escape(id)}"]`);
      if (!node) continue;
      node.classList.add("ph-has-detail");
      const shape = node.querySelector("rect, circle, ellipse, polygon, path") || node;
      try {
        const box = shape.getBBox();
        const dot = document.createElementNS(NS, "circle");
        dot.setAttribute("class", "ph-mark");
        dot.setAttribute("cx", box.x + box.width);
        dot.setAttribute("cy", box.y);
        dot.setAttribute("r", 4);
        node.append(dot);
      } catch {
        // Not laid out (a hidden view): the cursor still says it opens.
      }
    }
    // The viewer lays its own hit target over every arrow once it starts
    // (`data-relationship-id`), so an arrow needs nothing added here: a press on
    // that target, or on the arrow or its label, opens the arrow's details.
    const edgeOf = (target) => {
      const hit = target.closest("[data-relationship-id]");
      if (hit) return hit.getAttribute("data-relationship-id");
      const drawn = target.closest("[data-edge-id]");
      if (drawn) return drawn.getAttribute("data-edge-id");
      const label = target.closest("[data-edge-key]");
      const of = label && svg.querySelector(`[data-edge-id][data-edge-key="${CSS.escape(label.getAttribute("data-edge-key"))}"]`);
      return of ? of.getAttribute("data-edge-id") : null;
    };
    const openAt = (target) => {
      const edge = edgeOf(target);
      if (edge && data.edges[edge]) return open(`edge:${edge}`), true;
      const node = target.closest("[data-node-id]");
      if (node && data.nodes[node.getAttribute("data-node-id")]) return open(`node:${node.getAttribute("data-node-id")}`), true;
      return false;
    };
    // A drag pans the diagram; only a press that stayed put opens something.
    let down = null;
    svg.addEventListener("pointerdown", (event) => (down = [event.clientX, event.clientY]), true);
    svg.addEventListener(
      "click",
      (event) => {
        if (down && Math.hypot(event.clientX - down[0], event.clientY - down[1]) > 5) return;
        openAt(event.target);
      },
      true,
    );
    svg.addEventListener(
      "keydown",
      (event) => {
        if (event.key === "Enter" || event.key === " ") openAt(event.target);
      },
      true,
    );
  }

  const count = Object.keys(data.nodes).length + Object.keys(data.edges).length + data.topics.length;
  if (count) launch.append(el("small", { text: String(count) }));
  const asked = new URLSearchParams(location.search).get("drill");
  if (asked) {
    // Asked for by the address: already open when the page is first seen.
    panel.classList.add("is-instant");
    open(asked);
    requestAnimationFrame(() => requestAnimationFrame(() => panel.classList.remove("is-instant")));
  }
})();
