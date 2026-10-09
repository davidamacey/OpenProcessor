/**
 * Trap Tab/Shift+Tab focus inside a modal and, optionally, close it on
 * Escape — pairs with `focusOnMount`, which only places the *initial*
 * focus (m12, 2026-09-24 interactive pass: Tab was escaping the Add Class
 * modal into the page body and nav links behind it).
 *
 * Usage: `<div use:trapFocus={{ onEscape: onclose }} ...>`. Restores focus
 * to whatever was focused before the modal opened, on destroy — otherwise
 * focus is left on `<body>` and keyboard users lose their place.
 */

const FOCUSABLE_SELECTOR = [
  'a[href]',
  'button:not([disabled])',
  'input:not([disabled])',
  'select:not([disabled])',
  'textarea:not([disabled])',
  '[tabindex]:not([tabindex="-1"])',
].join(',');

function focusable(node: HTMLElement): HTMLElement[] {
  // Layout-based visibility (offsetParent) isn't available under jsdom, so
  // this checks the DOM attributes a hidden control would actually carry
  // instead — good enough for the modals this action targets, which don't
  // nest a hidden panel inside the trap.
  return Array.from(node.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR)).filter(
    (el) =>
      !el.hidden &&
      el.style.display !== 'none' &&
      el.getAttribute('aria-hidden') !== 'true',
  );
}

export interface TrapFocusOptions {
  onEscape?: () => void;
}

interface OpenTrap {
  node: HTMLElement;
  handle: (e: KeyboardEvent) => void;
}

// Open traps, innermost last. A focused button that turns `disabled` (a
// dialog's busy Confirm) drops focus to <body>, and keys then never reach
// the dialog's own listener, so a document listener hands any key that
// landed outside every open trap to the topmost one.
const openTraps: OpenTrap[] = [];

function onDocumentKeydown(e: KeyboardEvent): void {
  const top = openTraps[openTraps.length - 1];
  if (!top) return;
  const target = e.target instanceof Node ? e.target : null;
  if (target && openTraps.some((t) => t.node.contains(target))) return;
  top.handle(e);
}

export function trapFocus(node: HTMLElement, opts: TrapFocusOptions = {}) {
  let options = opts;
  const previouslyFocused =
    document.activeElement instanceof HTMLElement ? document.activeElement : null;

  function onKeydown(e: KeyboardEvent): void {
    if (e.key === 'Escape') {
      if (options.onEscape) {
        e.preventDefault();
        options.onEscape();
      }
      return;
    }
    if (e.key !== 'Tab') return;
    const els = focusable(node);
    if (els.length === 0) {
      e.preventDefault();
      node.focus();
      return;
    }
    const first = els[0];
    const last = els[els.length - 1];
    const active = document.activeElement;
    // idx is -1 both when nothing inside the dialog is focused yet (the
    // wrapper itself, from focusOnMount) and when focus has somehow left
    // the dialog — either way, re-enter at an edge instead of leaking Tab
    // out to the page behind it.
    const idx = active instanceof HTMLElement ? els.indexOf(active) : -1;
    if (e.shiftKey) {
      if (idx === 0 || idx === -1) {
        e.preventDefault();
        last.focus();
      }
    } else {
      if (idx === els.length - 1 || idx === -1) {
        e.preventDefault();
        first.focus();
      }
    }
  }

  node.addEventListener('keydown', onKeydown);
  const entry: OpenTrap = { node, handle: onKeydown };
  if (openTraps.length === 0) document.addEventListener('keydown', onDocumentKeydown);
  openTraps.push(entry);

  return {
    update(next: TrapFocusOptions = {}): void {
      options = next;
    },
    destroy(): void {
      node.removeEventListener('keydown', onKeydown);
      const i = openTraps.indexOf(entry);
      if (i !== -1) openTraps.splice(i, 1);
      if (openTraps.length === 0)
        document.removeEventListener('keydown', onDocumentKeydown);
      if (previouslyFocused && document.contains(previouslyFocused)) {
        previouslyFocused.focus();
      }
    },
  };
}
