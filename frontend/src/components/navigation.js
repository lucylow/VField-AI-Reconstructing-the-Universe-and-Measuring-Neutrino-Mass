import { $, $$, setText } from "../core/dom.js";

const titles = {
  overview: "Overview",
  expansion: "Cosmic expansion",
  field: "Cosmic web",
  cmb: "CMB sky",
  lensing: "Gravitational lensing",
  spectra: "Power spectrum",
  parameters: "Parameter space",
  diagnostics: "Run diagnostics",
};

export function initNavigation(onNavigate) {
  $$("[data-view]").forEach((button) => {
    button.addEventListener("click", () => onNavigate(button.dataset.view));
  });
  $$("[data-navigate]").forEach((button) => {
    button.addEventListener("click", () => onNavigate(button.dataset.navigate));
  });
  const sidebar = $("#sidebar");
  const menuButton = $("#mobile-menu");
  const menuScrim = $("#mobile-nav-scrim");
  const mobileViewport = window.matchMedia?.("(max-width: 760px)");
  const setMenuOpen = (opened) => {
    sidebar?.classList.toggle("sidebar-open", opened);
    const hiddenFromKeyboard = Boolean(mobileViewport?.matches && !opened);
    if (sidebar) {
      sidebar.inert = hiddenFromKeyboard;
      sidebar.setAttribute("aria-hidden", String(hiddenFromKeyboard));
    }
    if (menuScrim) menuScrim.hidden = !opened;
    menuButton?.setAttribute("aria-expanded", String(opened));
    menuButton?.setAttribute("aria-label", opened ? "Close navigation" : "Open navigation");
  };
  setMenuOpen(false);
  menuButton?.setAttribute("aria-controls", "sidebar");
  menuButton?.addEventListener("click", () => {
    const opened = !sidebar?.classList.contains("sidebar-open");
    setMenuOpen(opened);
    if (opened) sidebar?.querySelector("[data-view]")?.focus();
  });
  mobileViewport?.addEventListener?.("change", () => setMenuOpen(false));
  menuScrim?.addEventListener("click", () => {
    setMenuOpen(false);
    menuButton?.focus();
  });
  window.addEventListener("click", (event) => {
    if (sidebar?.classList.contains("sidebar-open")
      && !sidebar.contains(event.target)
      && !menuButton?.contains(event.target)) setMenuOpen(false);
  });

  window.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && sidebar?.classList.contains("sidebar-open")) {
      setMenuOpen(false);
      menuButton?.focus();
      return;
    }
    if (event.altKey || event.ctrlKey || event.metaKey || isTypingTarget(event.target)) return;
    const views = ["overview", "expansion", "field", "cmb", "lensing", "spectra", "parameters", "diagnostics"];
    const index = Number(event.key) - 1;
    if (index >= 0 && index < views.length) onNavigate(views[index]);
  });

  const initialView = viewFromHash();
  if (initialView) onNavigate(initialView, { updateHistory: false });
  window.addEventListener("popstate", () => {
    const view = viewFromHash();
    if (view) onNavigate(view, { updateHistory: false });
  });
}

export function navigateTo(view, { updateHistory = true } = {}) {
  const target = $("#view-" + view);
  if (!target || !titles[view]) return;
  const route = `#view=${view}`;
  if (updateHistory && window.location.hash !== route) {
    try { window.history.pushState({ view }, "", route); } catch { /* navigation still works without URL history */ }
  }
  $$(".view").forEach((node) => node.classList.toggle("active", node === target));
  $$(".nav-item[data-view]").forEach((node) => {
    const active = node.dataset.view === view;
    node.classList.toggle("active", active);
    if (active) node.setAttribute("aria-current", "page");
    else node.removeAttribute("aria-current");
  });
  setText("#current-breadcrumb", titles[view] ?? "Workspace");
  const sidebar = $("#sidebar");
  sidebar?.classList.remove("sidebar-open");
  const sidebarHidden = Boolean(window.matchMedia?.("(max-width: 760px)").matches);
  if (sidebar) {
    sidebar.inert = sidebarHidden;
    sidebar.setAttribute("aria-hidden", String(sidebarHidden));
  }
  $("#mobile-menu")?.setAttribute("aria-expanded", "false");
  $("#mobile-menu")?.setAttribute("aria-label", "Open navigation");
  const menuScrim = $("#mobile-nav-scrim");
  if (menuScrim) menuScrim.hidden = true;
  target.querySelector("h1")?.setAttribute("tabindex", "-1");
  target.querySelector("h1")?.focus({ preventScroll: true });
  window.scrollTo({ top: 0, behavior: "auto" });
  window.dispatchEvent(new CustomEvent("nufield:viewchange", { detail: { view } }));
}

function viewFromHash() {
  const match = window.location.hash.match(/^#view=([a-z-]+)$/);
  return match && titles[match[1]] ? match[1] : null;
}

function isTypingTarget(target) {
  return target instanceof HTMLElement && (target.matches("input, textarea, select, [contenteditable='true']") || target.isContentEditable);
}
