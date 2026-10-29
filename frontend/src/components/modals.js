import { $, $$, setHidden } from "../core/dom.js";

let returnFocusTo = null;

export function initModals() {
  $("#open-import")?.addEventListener("click", openImportModal);
  $("#overview-import-button")?.addEventListener("click", openImportModal);
  $("#table-import-button")?.addEventListener("click", openImportModal);
  $("#diagnostics-import-button")?.addEventListener("click", openImportModal);
  $$("[data-open-import]").forEach((button) => button.addEventListener("click", openImportModal));
  $("#open-data-info")?.addEventListener("click", () => openModal("#info-modal"));
  $("#about-button")?.addEventListener("click", () => openModal("#info-modal"));
  $$("[data-close-modal]").forEach((button) => button.addEventListener("click", () => closeTopModal()));
  $$(".modal-backdrop").forEach((backdrop) => {
    backdrop.addEventListener("click", (event) => {
      if (event.target === backdrop) closeModal(backdrop);
    });
  });
  window.addEventListener("keydown", (event) => {
    const opened = $$(".modal-backdrop:not([hidden])").at(-1);
    if (!opened) return;
    if (event.key === "Escape") {
      event.preventDefault();
      closeModal(opened);
      return;
    }
    if (event.key !== "Tab") return;
    const dialog = opened.querySelector('[role="dialog"]');
    if (!dialog) return;
    const focusable = [...dialog.querySelectorAll(
      "a[href], button:not([disabled]), input:not([disabled]):not([type='hidden']), " +
      "select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex='-1'])"
    )].filter((element) => !element.hidden && element.getAttribute("aria-hidden") !== "true");
    if (!focusable.length) {
      event.preventDefault();
      dialog.setAttribute("tabindex", "-1");
      dialog.focus();
      return;
    }
    const first = focusable[0];
    const last = focusable.at(-1);
    if (!dialog.contains(document.activeElement)) {
      event.preventDefault();
      (event.shiftKey ? last : first).focus();
    } else if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  });
}

export function openImportModal() {
  openModal("#import-modal");
}

export function openModal(selector) {
  const modal = $(selector);
  if (!modal) return;
  returnFocusTo = document.activeElement instanceof HTMLElement ? document.activeElement : null;
  modal.hidden = false;
  document.body.classList.add("modal-open");
  modal.querySelector("button, input, [tabindex='0']")?.focus();
}

export function closeTopModal() {
  const opened = $$(".modal-backdrop:not([hidden])").at(-1);
  if (opened) closeModal(opened);
}

function closeModal(modal) {
  setHidden(modal, true);
  if ($$(".modal-backdrop:not([hidden])").length === 0) {
    document.body.classList.remove("modal-open");
    if (returnFocusTo?.isConnected) returnFocusTo.focus();
    returnFocusTo = null;
  }
}
