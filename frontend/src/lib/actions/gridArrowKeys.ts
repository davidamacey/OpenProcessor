/**
 * Svelte action: arrow keys move the focus between the cards of a grid (roving tabindex).
 * Cards are the descendants marked `data-grid-item`; only the focused one is a tab stop, so
 * Tab enters the grid once and leaves it. Left and Right step one card, Up and Down one row
 * (the row length is read from the layout), Home and End jump to the ends. A card that is a
 * button opens with Enter as it always did.
 */
import type { Action } from 'svelte/action';

const KEYS = new Set(['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End']);

function cards(node: HTMLElement): HTMLElement[] {
  return [...node.querySelectorAll<HTMLElement>('[data-grid-item]')];
}

function rowLength(items: HTMLElement[]): number {
  const top = items[0]?.offsetTop;
  const n = items.findIndex((el) => el.offsetTop !== top);
  return n < 0 ? items.length : n;
}

function target(key: string, at: number, items: HTMLElement[]): number {
  if (key === 'Home') return 0;
  if (key === 'End') return items.length - 1;
  const step = key === 'ArrowLeft' || key === 'ArrowRight' ? 1 : rowLength(items);
  const next = key === 'ArrowLeft' || key === 'ArrowUp' ? at - step : at + step;
  return Math.min(items.length - 1, Math.max(0, next));
}

export const gridArrowKeys: Action<HTMLElement> = (node) => {
  function onfocusin(e: FocusEvent): void {
    const hit = (e.target as HTMLElement).closest<HTMLElement>('[data-grid-item]');
    if (!hit) return;
    for (const el of cards(node)) el.tabIndex = el === hit ? 0 : -1;
  }

  function onkeydown(e: KeyboardEvent): void {
    if (!KEYS.has(e.key) || e.ctrlKey || e.metaKey || e.altKey || e.shiftKey) return;
    const items = cards(node);
    const at = items.findIndex((el) => el === document.activeElement);
    if (at < 0) return;
    e.preventDefault();
    e.stopPropagation();
    items[target(e.key, at, items)]?.focus();
  }

  node.addEventListener('focusin', onfocusin);
  node.addEventListener('keydown', onkeydown);
  return {
    destroy() {
      node.removeEventListener('focusin', onfocusin);
      node.removeEventListener('keydown', onkeydown);
    },
  };
};
