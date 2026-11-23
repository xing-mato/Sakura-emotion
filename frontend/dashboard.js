// 情绪仪表盘 · Sakura 插件设置模块
//
// 契约（源码见 desktop/frontend/settings/plugin-module.js）：
//   const module = await import(blob(source))
//   const mounted = await module.mount({ ...context, signal })
//   mounted 必须返回 { element, dispose() }，可选 update()
//
// 面板分两层：
//   1. 在场的那两三根（主 / 次 / 辅）——她此刻的全部；
//   2. 当前不在场的情绪收进抽屉，点箭头才展开，只是让人知道池子里有哪些词。
// 不解释、不推荐、不写"她此刻会怎样"。
//
// 三个踩过的坑，写在这里免得再踩：
//   1. 页面 CSP 是 style-src 'self'，没有 unsafe-inline。
//      往 DOM 里塞 <style> 文本会被静默拦掉（样式全丢，页面变成一串裸文字），
//      必须用构造式样式表 + document.adoptedStyleSheets。
//   2. CSSOM（element.style.width = ...）不受这条 CSP 限制，宽度可以照常用。
//   3. action() 只是把结果返回给我们，宿主不会替我们把 result.values 写回 read()。
//      所以刷新之后必须自己用返回值覆盖本地 payload，否则面板看起来"没反应"。
//
// 数据来自 section 的只读 data 字段 payload，用 read("payload") 取；
// 刷新按钮调用插件的 refresh 动作，由插件去问模型。

const ROLES = ["主", "次", "辅"];
const EMPTY = "—";
const FLASH_MS = 6000;
// 颜色只跟情绪的分组走：暖 / 钝 / 冷 / 混
const TONE = {
  喜悦: "pos", 爱慕: "pos", 温柔: "pos", 好奇: "pos", 自信: "pos", 害羞: "pos",
  欣慰: "pos", 期待: "pos", 撒娇: "pos",
  无奈: "neu", 无语: "neu", 疑惑: "neu", 尴尬: "neu", 惊讶: "neu",
  委屈: "neg", 失望: "neg", 悲伤: "neg", 生气: "neg", 害怕: "neg", 厌恶: "neg",
  担忧: "neg", 孤独: "neg",
  吃醋: "mix", 傲娇: "mix",
};

const CSS = [
  ".epd{display:grid;gap:12px;font-size:13px;color:var(--sakura-text,inherit)}",
  ".epd-base{font-size:11.5px;opacity:.45;letter-spacing:.02em}",
  ".epd-trackline{font-size:11.5px;opacity:.5;letter-spacing:.02em;margin-top:-2px}",
  ".epd-list{display:grid;gap:10px;margin:0;padding:0;list-style:none}",
  ".epd-row{display:grid;grid-template-columns:1.9em 3.2em minmax(0,1fr) 3.2em;align-items:center;gap:12px;transition:opacity .25s}",
  ".epd-row[data-empty=true]{opacity:.4}",
  ".epd-role{font-size:11px;opacity:.5;text-align:center;border:1px solid var(--sakura-border,rgba(127,127,127,.28));border-radius:999px;padding:1px 0}",
  ".epd-row.is-lead .epd-role{opacity:.8;border-color:var(--sakura-accent,rgba(127,127,127,.45))}",
  ".epd-name{opacity:.7;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}",
  ".epd-track{position:relative;height:8px;border-radius:999px;background:var(--hairline,rgba(127,127,127,.16));overflow:hidden;box-shadow:inset 0 0 0 1px rgba(127,127,127,.05)}",
  ".epd-fill{display:block;height:100%;width:0;border-radius:999px;transition:width .55s cubic-bezier(.22,.61,.36,1)}",
  ".epd-row[data-tone=pos] .epd-fill,.epd-other[data-tone=pos] .epd-fill{background:linear-gradient(90deg,rgba(217,115,154,.5),rgba(217,115,154,.95))}",
  ".epd-row[data-tone=neu] .epd-fill,.epd-other[data-tone=neu] .epd-fill{background:linear-gradient(90deg,rgba(150,154,140,.42),rgba(150,154,140,.88))}",
  ".epd-row[data-tone=neg] .epd-fill,.epd-other[data-tone=neg] .epd-fill{background:linear-gradient(90deg,rgba(125,138,166,.45),rgba(125,138,166,.92))}",
  ".epd-row[data-tone=mix] .epd-fill,.epd-other[data-tone=mix] .epd-fill{background:linear-gradient(90deg,rgba(154,127,176,.45),rgba(154,127,176,.92))}",
  ".epd-val{text-align:right;font-variant-numeric:tabular-nums;opacity:.75}",
  ".epd-row.is-lead .epd-name{opacity:1;font-weight:600}",
  ".epd-row.is-lead .epd-val{opacity:1}",
  ".epd-more{display:grid;gap:8px}",
  ".epd-toggle{display:inline-flex;align-items:center;gap:6px;justify-self:start;border:0;background:transparent;color:inherit;font:inherit;font-size:11.5px;opacity:.5;padding:2px 0;cursor:pointer;transition:opacity .2s}",
  ".epd-toggle:hover{opacity:.9}",
  ".epd-toggle:focus-visible{outline:2px solid var(--sakura-accent,rgba(127,127,127,.5));outline-offset:2px;border-radius:6px}",
  ".epd-arrow{display:inline-block;font-size:10px;transition:transform .2s}",
  ".epd-arrow.is-open{transform:rotate(90deg)}",
  ".epd-others{display:grid;gap:6px;margin:0;padding:0;list-style:none}",
  ".epd-other{display:grid;grid-template-columns:3.2em minmax(0,1fr) 3.2em;align-items:center;gap:12px;opacity:.38}",
  ".epd-other .epd-track{height:6px;background:transparent;box-shadow:inset 0 0 0 1px var(--hairline,rgba(127,127,127,.2))}",
  ".epd-other .epd-name{font-size:12px}",
  ".epd-other[data-active=true]{opacity:.82}",
  ".epd-foot{display:flex;align-items:center;gap:12px;justify-content:space-between;border-top:1px solid var(--hairline,rgba(127,127,127,.14));padding-top:10px}",
  ".epd-tools{display:inline-flex;align-items:center;gap:10px}",
  ".epd-note{font-size:11.5px;opacity:.55;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}",
  ".epd-note.is-flash{opacity:.85}",
  ".epd-refresh{border:1px solid var(--sakura-border,rgba(127,127,127,.3));background:transparent;color:inherit;border-radius:999px;padding:4px 14px;font-size:12px;cursor:pointer;transition:background .2s,border-color .2s,opacity .2s}",
  ".epd-refresh:hover:not(:disabled){background:var(--sakura-panel-bg,rgba(127,127,127,.08))}",
  ".epd-refresh:disabled{opacity:.5;cursor:default}",
  ".epd-ghost{border:0;background:transparent;color:inherit;font:inherit;font-size:11.5px;opacity:.5;padding:4px 2px;cursor:pointer}",
  ".epd-ghost:hover{opacity:.9}",
  ".epd-mask{position:fixed;inset:0;background:rgba(0,0,0,.30);display:flex;align-items:center;justify-content:center;z-index:9999;padding:16px}",
  ".epd-dialog{background:var(--sakura-panel-bg,#fff);border:1px solid var(--sakura-border,rgba(127,127,127,.3));border-radius:14px;padding:18px 20px;width:min(340px,100%);box-shadow:0 16px 48px rgba(0,0,0,.22);display:grid;gap:10px;font-size:13px;color:var(--sakura-text,inherit)}",
  ".epd-dialog-title{font-size:14px;font-weight:600;line-height:1.3}",
  ".epd-dialog-body{opacity:.72;line-height:1.55;margin:0}",
  ".epd-dialog-actions{display:flex;justify-content:flex-end;gap:10px;margin-top:2px}",
  ".epd-btn{border:1px solid var(--sakura-border,rgba(127,127,127,.32));background:transparent;color:inherit;border-radius:999px;padding:5px 16px;font-size:12px;cursor:pointer}",
  ".epd-btn:hover{background:var(--sakura-panel-bg-light,rgba(127,127,127,.08))}",
  ".epd-btn.is-danger{background:#c9556f;border-color:#c9556f;color:#fff}",
  "@media (max-width:520px){.epd-row{grid-template-columns:1.7em 2.8em minmax(0,1fr) 2.9em;gap:9px}}",
].join("\n");

export function mount(context) {
  const document = context.document || globalThis.document;
  const read = typeof context.read === "function" ? context.read : () => null;
  const action = typeof context.action === "function" ? context.action : null;
  const onError = typeof context.onError === "function" ? context.onError : null;

  const root = document.createElement("section");
  root.className = "epd";
  root.setAttribute("aria-label", "情绪");

  let sheet = null;
  try {
    if (typeof CSSStyleSheet === "function" && "adoptedStyleSheets" in document) {
      sheet = new CSSStyleSheet();
      sheet.replaceSync(CSS);
      document.adoptedStyleSheets = [...document.adoptedStyleSheets, sheet];
    }
  } catch (error) {
    sheet = null;
  }

  const baselineLine = document.createElement("div");
  baselineLine.className = "epd-base";
  baselineLine.hidden = true;

  const list = document.createElement("ol");
  list.className = "epd-list";

  const rows = [];
  ROLES.forEach((role, index) => {
    const li = document.createElement("li");
    li.className = "epd-row";
    li.dataset.tone = "neu";
    li.dataset.empty = "true";
    if (index === 0) li.classList.add("is-lead");
    const roleEl = document.createElement("span");
    roleEl.className = "epd-role";
    roleEl.textContent = role;
    const name = document.createElement("span");
    name.className = "epd-name";
    name.textContent = EMPTY;
    const track = document.createElement("span");
    track.className = "epd-track";
    const fill = document.createElement("span");
    fill.className = "epd-fill";
    track.append(fill);
    const value = document.createElement("span");
    value.className = "epd-val";
    value.textContent = EMPTY;
    li.append(roleEl, name, track, value);
    rows.push({ li, name, fill, value });
    list.append(li);
  });

  const more = document.createElement("div");
  more.className = "epd-more";
  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.className = "epd-toggle";
  toggle.setAttribute("aria-expanded", "false");
  const arrow = document.createElement("span");
  arrow.className = "epd-arrow";
  arrow.textContent = "\u25B8";
  const toggleLabel = document.createElement("span");
  toggleLabel.textContent = "其余情绪";
  toggle.append(arrow, toggleLabel);
  const others = document.createElement("ol");
  others.className = "epd-others";
  others.hidden = true;
  more.append(toggle, others);

  const foot = document.createElement("div");
  foot.className = "epd-foot";
  const note = document.createElement("span");
  note.className = "epd-note";
  const refreshButton = document.createElement("button");
  refreshButton.type = "button";
  refreshButton.className = "epd-refresh";
  refreshButton.textContent = "刷新";
  const exportButton = document.createElement("button");
  exportButton.type = "button";
  exportButton.className = "epd-ghost";
  exportButton.textContent = "导出日志";
  const resetButton = document.createElement("button");
  resetButton.type = "button";
  resetButton.className = "epd-ghost";
  resetButton.textContent = "清零";
  const tools = document.createElement("span");
  tools.className = "epd-tools";
  tools.append(exportButton, resetButton);
  const right = document.createElement("span");
  right.className = "epd-tools";
  right.append(tools, refreshButton);
  foot.append(note, right);

  const trackLine = document.createElement("div");
  trackLine.className = "epd-trackline";
  trackLine.hidden = true;
  root.append(baselineLine, list, trackLine, more, foot);

  let busy = false;
  let busySince = 0;
  // 宿主偶尔不返回结果：busy 一旦卡住，轮询和所有按钮都会跟着停。
  // 超过 70 秒就当那次操作已经结束。
  function isBusy() {
    if (!busy) return false;
    if (Date.now() - busySince > 70000) {
      busy = false;
      return false;
    }
    return true;
  }
  function setBusy(value) {
    busy = value;
    busySince = value ? Date.now() : 0;
  }
  let disposed = false;
  let expanded = false;
  let pollTimer = null;
  const POLL_MS = 8000;
  let othersSignature = "";
  let otherNodes = [];
  // 刷新拿回来的新值：宿主不会写回 read()，所以本地留一份盖在上面
  let localPayload = null;
  let flashTimer = null;

  function sectionPayload() {
    const fallback = context.section && context.section.values && context.section.values.payload;
    return fallback && typeof fallback === "object" ? fallback : null;
  }

  function currentPayload() {
    const fromRead = read("payload");
    if (localPayload) {
      if (fromRead && JSON.stringify(fromRead) === JSON.stringify(localPayload)) {
        localPayload = null;
        return fromRead;
      }
      return localPayload;
    }
    if (fromRead && typeof fromRead === "object") return fromRead;
    return sectionPayload();
  }

  function baseNoteText() {
    const payload = currentPayload() || {};
    return payload.updated || "还没有判断过";
  }

  function renderOthers(items) {
    const signature = items.map(item => item.key + ":" + item.value).join("|");
    if (signature !== othersSignature) {
      othersSignature = signature;
      for (const node of otherNodes) node.remove();
      otherNodes = [];
      for (const item of items) {
        const li = document.createElement("li");
        li.className = "epd-other";
        const name = document.createElement("span");
        name.className = "epd-name";
        name.textContent = item.key;
        const track = document.createElement("span");
        track.className = "epd-track";
        const fill = document.createElement("span");
        fill.className = "epd-fill";
        const percent = Math.max(0, Math.min(100, Math.round(Number(item.value) || 0)));
        fill.style.width = percent + "%";
        track.append(fill);
        const value = document.createElement("span");
        value.className = "epd-val";
        value.textContent = percent + "%";
        li.dataset.tone = TONE[item.key] || "neg";
        // 余韵是真实值：点亮对应颜色；0 的仍然是淡的（只是目录）
        li.dataset.active = percent > 0 ? "true" : "false";
        li.append(name, track, value);
        others.append(li);
        otherNodes.push(li);
      }
    }
    toggleLabel.textContent = items.length > 0 ? "其余 " + items.length + " 种" : "没有其它情绪";
    toggle.hidden = items.length === 0;
  }

  function render() {
    const payload = currentPayload() || {};
    const slots = Array.isArray(payload.slots) ? payload.slots : [];
    rows.forEach((row, index) => {
      const slot = slots[index];
      const key = slot && typeof slot.key === "string" ? slot.key : "";
      const value = slot ? Math.max(0, Math.min(100, Math.round(Number(slot.value) || 0))) : 0;
      row.name.textContent = key || EMPTY;
      row.fill.style.width = value + "%";
      row.value.textContent = key ? value + "%" : EMPTY;
      row.li.dataset.tone = TONE[key] || "neu";
      row.li.dataset.empty = key ? "false" : "true";
      row.li.title = key && slot.role ? slot.role + " · " + key : "";
    });
    const baseline = typeof payload.baseline === "string" ? payload.baseline.trim() : "";
    baselineLine.textContent = baseline ? "底色：" + baseline : "";
    baselineLine.hidden = !baseline;
    renderOthers(Array.isArray(payload.others) ? payload.others : []);
    arrow.classList.toggle("is-open", expanded);
    toggle.setAttribute("aria-expanded", expanded ? "true" : "false");
    others.hidden = !expanded;
    note.textContent = baseNoteText();
    const turns = Number(payload.turns) || 0;
    // 她这一轮实际用了哪几种语气（先判之后没有，她说完校正完才有）
    const track = Array.isArray(payload.track)
      ? payload.track.filter((item) => typeof item === "string" && item)
      : [];
    if (track.length >= 2) {
      trackLine.textContent = "本轮：" + track.join(" → ");
      trackLine.hidden = false;
    } else if (payload.recheck_failed === true && turns > 0) {
      trackLine.textContent = "本轮：（未校正）";
      trackLine.hidden = false;
    } else if (turns > 0 && payload.rechecked !== true) {
      trackLine.textContent = "本轮：等她说完再校正";
      trackLine.hidden = false;
    } else {
      trackLine.textContent = "";
      trackLine.hidden = true;
    }
    note.title = note.textContent + (turns > 0 ? " · 已判断 " + turns + " 轮" : " · 还没按轮判断过");
    note.classList.remove("is-flash");
    refreshButton.hidden = !action;
    // 重判会把她现在的情绪整个换掉：只在面板空着的时候给按。
    const hasMood = slots.length > 0;
    refreshButton.disabled = hasMood;
    refreshButton.title = hasMood ? "此刻她有情绪，不重算。等面板空了或清零后再用。" : "";
  }

  function onToggle() {
    expanded = !expanded;
    render();
  }

  let mask = null;

  function closeDialog() {
    if (mask && mask.parentNode) mask.parentNode.removeChild(mask);
    mask = null;
    if (typeof document.removeEventListener === "function") {
      document.removeEventListener("keydown", onDialogKey);
    }
  }

  function onDialogKey(event) {
    if (event && event.key === "Escape") closeDialog();
  }

  function confirmDialog(options) {
    closeDialog();
    const backdrop = document.createElement("div");
    backdrop.className = "epd-mask";
    const panel = document.createElement("div");
    panel.className = "epd-dialog";
    panel.setAttribute("role", "dialog");
    panel.setAttribute("aria-modal", "true");
    const title = document.createElement("div");
    title.className = "epd-dialog-title";
    title.textContent = options.title;
    const body = document.createElement("p");
    body.className = "epd-dialog-body";
    body.textContent = options.body;
    const actions = document.createElement("div");
    actions.className = "epd-dialog-actions";
    const cancel = document.createElement("button");
    cancel.type = "button";
    cancel.className = "epd-btn";
    cancel.textContent = "取消";
    const ok = document.createElement("button");
    ok.type = "button";
    ok.className = "epd-btn" + (options.danger ? " is-danger" : "");
    ok.textContent = options.confirmLabel || "确认";
    actions.append(cancel, ok);
    panel.append(title, body, actions);
    backdrop.append(panel);
    (document.body || root).append(backdrop);
    mask = backdrop;
    cancel.addEventListener("click", closeDialog);
    backdrop.addEventListener("click", (event) => { if (event.target === backdrop) closeDialog(); });
    ok.addEventListener("click", () => {
      closeDialog();
      options.onConfirm();
    });
    if (typeof document.addEventListener === "function") {
      document.addEventListener("keydown", onDialogKey);
    }
    return panel;
  }

  async function runAction(actionId, options) {
    if (!action || isBusy()) return;
    setBusy(true);
    flash(options.waiting || "正在处理…");
    try {
      const result = await action(actionId, {});
      // 宿主把新数据一起返回时直接盖上去：不然要等下一次轮询（8 秒），看起来像没生效
      const patched = result && result.values && result.values.payload;
      if (patched && typeof patched === "object") localPayload = patched;
      const message = result && typeof result.message === "string" ? result.message : "";
      flash(message || options.done || "已完成");
      render();
    } catch (error) {
      flash("操作失败：" + (error && error.message ? String(error.message).slice(0, 100) : "未知错误"));
      if (onError) onError(error);
    } finally {
      setBusy(false);
    }
  }

  function askExport() {
    confirmDialog({
      title: "导出运行日志？",
      body: "文件会包含当前情绪值、逐轮判定记录和聊天原文。聊天记录可能含有隐私内容，导出后请自行保管。\n" +
            "确认后会弹出保存对话框：扩展名选 .txt 得到可读文本，选 .jsonl 得到可直接分析的数据。",
      confirmLabel: "选择位置并导出",
      onConfirm: () => saveExport(),
    });
  }

  function exportStamp() {
    const now = new Date();
    const pad = (value) => String(value).padStart(2, "0");
    return String(now.getFullYear()) + pad(now.getMonth() + 1) + pad(now.getDate()) +
      "-" + pad(now.getHours()) + pad(now.getMinutes()) + pad(now.getSeconds());
  }

  async function saveExport() {
    if (!action || isBusy()) return;
    setBusy(true);
    flash("正在导出…");
    try {
      const picker = typeof window.showSaveFilePicker === "function" ? window.showSaveFilePicker : null;
      if (!picker) {
        // 宿主不支持保存对话框：退回插件自己写进数据目录再打开。
        const result = await action("export_log", {});
        flash((result && result.message) || "已导出");
        return;
      }
      // 只有一种格式：JSONL。先把内容生成好，文件名由插件给（带角色和时间）
      const result = await action("export_json", {});
      const payload = result && result.values && result.values.export;
      if (!payload || payload.ok !== true) {
        flash((result && result.message) || "导出失败");
        return;
      }
      let handle = null;
      try {
        handle = await picker.call(window, {
          suggestedName: String(payload.filename || "emotion-pool.jsonl"),
          types: [
            { description: "JSONL 数据", accept: { "application/jsonl": [".jsonl"] } },
          ],
        });
      } catch (error) {
        if (error && (error.name === "AbortError" || error.code === 20)) return;   // 用户取消
        handle = null;
      }
      if (!handle) {
        const fallback = await action("export_log", {});
        flash((fallback && fallback.message) || "已导出");
        return;
      }
      const writable = await handle.createWritable();
      await writable.write(String(payload.text || ""));
      await writable.close();
      flash("已保存：" + handle.name);
    } catch (error) {
      flash("保存失败：" + (error && error.message ? String(error.message).slice(0, 90) : "未知错误"));
      if (onError) onError(error);
    } finally {
      setBusy(false);
    }
  }

  function askReset() {
    confirmDialog({
      title: "清空当前情绪？",
      body: "不推荐这么做，但这由你。清零会把此刻的情绪值全部归零；涟漪记录和聊天记录都还在。",
      confirmLabel: "确认清零",
      danger: true,
      onConfirm: () => runAction("reset", { waiting: "正在清零…", done: "已清零" }),
    });
  }

  function flash(text) {
    if (flashTimer) {
      clearTimeout(flashTimer);
      flashTimer = null;
    }
    note.textContent = text;
    note.title = text;
    note.classList.add("is-flash");
    flashTimer = setTimeout(() => {
      flashTimer = null;
      if (disposed) return;
      note.textContent = baseNoteText();
      note.title = note.textContent;
      note.classList.remove("is-flash");
    }, FLASH_MS);
  }

  async function refresh() {
    if (!action || isBusy()) return;
    setBusy(true);
    refreshButton.disabled = true;
    refreshButton.textContent = "想一想…";
    flash("正在请她重新想一想…");
    try {
      const result = await action("refresh", {});
      const patched = result && result.values && result.values.payload;
      if (patched && typeof patched === "object") localPayload = patched;
      render();
      const message = result && typeof result.message === "string" ? result.message : "";
      flash(message || "已刷新");
    } catch (error) {
      render();
      flash("刷新失败：" + (error && error.message ? String(error.message).slice(0, 120) : "未知错误"));
      if (onError) onError(error);
    } finally {
      setBusy(false);
      refreshButton.disabled = false;
      refreshButton.textContent = "刷新";
    }
  }

  async function peek() {
    if (!action || isBusy() || disposed) return;
    try {
      const result = await action("peek", {});
      const patched = result && result.values && result.values.payload;
      if (!patched || typeof patched !== "object") return;
      if (JSON.stringify(patched) !== JSON.stringify(currentPayload())) {
        localPayload = patched;
      }
      // 即使值没变也走一遍 render：宿主可能会自己把 read() 更新成新值，
      // 这时 localPayload 该让位，但 DOM 仍然要按最新的画。
      if (!flashTimer) render();
    } catch (error) {
      /* 轮询失败不打扰用户，下一次再试 */
    }
  }

  function startPolling() {
    stopPolling();
    if (!action) return;
    pollTimer = setInterval(() => {
      const state = document.visibilityState;
      if (state && state !== "visible") return;
      peek();
    }, POLL_MS);
  }

  function stopPolling() {
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }

  function onVisibility() {
    if (!document.visibilityState || document.visibilityState === "visible") peek();
  }

  toggle.addEventListener("click", onToggle);
  refreshButton.addEventListener("click", refresh);
  exportButton.addEventListener("click", askExport);
  resetButton.addEventListener("click", askReset);
  if (typeof document.addEventListener === "function") {
    document.addEventListener("visibilitychange", onVisibility);
  }
  render();
  startPolling();

  return {
    element: root,
    update() { render(); },
    dispose() {
      disposed = true;
      stopPolling();
      if (flashTimer) {
        clearTimeout(flashTimer);
        flashTimer = null;
      }
      if (typeof document.removeEventListener === "function") {
        document.removeEventListener("visibilitychange", onVisibility);
      }
      closeDialog();
      toggle.removeEventListener("click", onToggle);
      refreshButton.removeEventListener("click", refresh);
      exportButton.removeEventListener("click", askExport);
      resetButton.removeEventListener("click", askReset);
      if (sheet && "adoptedStyleSheets" in document) {
        document.adoptedStyleSheets = document.adoptedStyleSheets.filter(item => item !== sheet);
      }
      root.remove();
    },
  };
}
