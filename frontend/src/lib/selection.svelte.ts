/**
 * Multi-select behavior shared by the crop grid and the region gallery.
 *
 * Both grids implement the same three-branch click contract — shift-range,
 * ctrl/cmd-toggle, plain — and differ only in what a plain click does:
 *
 * - `replace` (crop grid): plain click selects exactly that card.
 * - `toggle` (region gallery): plain click adds/removes it, accumulating.
 *
 * The range branch unions with the current selection so shift-after-ctrl
 * extends rather than replaces, and falls through to the plain branch when
 * the anchor has scrolled out of the ordered list.
 */

import { SvelteSet } from 'svelte/reactivity';

export type PlainClickMode = 'replace' | 'toggle';

export interface Selection {
  /** The live selection. Assign a new Set to replace it wholesale. */
  ids: Set<string>;
  /** Last card clicked without shift — the origin of a range select. */
  anchorId: string | null;
  readonly size: number;
  has(id: string): boolean;
  /** @param orderedIds ids in the order currently displayed. */
  click(id: string, e: MouseEvent | undefined, orderedIds: string[]): void;
  selectAll(ids: string[]): void;
  clear(): void;
}

export function createSelection(
  opts: { plainClick: PlainClickMode } = { plainClick: 'replace' },
): Selection {
  // A single long-lived SvelteSet, always mutated in place (never
  // reassigned) — SvelteSet is already reactive on `.add`/`.delete`/
  // `.clear`, so wrapping it in `$state` too would be redundant
  // (svelte/no-unnecessary-state-wrap) as well as pointless churn on
  // every click.
  const selected = new SvelteSet<string>();
  let anchorId = $state<string | null>(null);

  function replaceWith(ids: Iterable<string>): void {
    selected.clear();
    for (const id of ids) selected.add(id);
  }

  return {
    get ids() {
      return selected;
    },
    set ids(next: Set<string>) {
      replaceWith(next);
    },
    get anchorId() {
      return anchorId;
    },
    set anchorId(next: string | null) {
      anchorId = next;
    },
    get size() {
      return selected.size;
    },
    has(id: string): boolean {
      return selected.has(id);
    },
    click(id: string, e: MouseEvent | undefined, orderedIds: string[]): void {
      const isToggle = !!(e && (e.ctrlKey || e.metaKey));
      const isRange = !!(e && e.shiftKey);

      if (isRange && anchorId) {
        const a = orderedIds.indexOf(anchorId);
        const b = orderedIds.indexOf(id);
        if (a !== -1 && b !== -1) {
          const [lo, hi] = a <= b ? [a, b] : [b, a];
          for (let i = lo; i <= hi; i++) selected.add(orderedIds[i]!);
          return;
        }
        // Anchor no longer visible — fall through.
      }

      if (isToggle || opts.plainClick === 'toggle') {
        if (selected.has(id)) selected.delete(id);
        else selected.add(id);
        anchorId = id;
        return;
      }

      replaceWith([id]);
      anchorId = id;
    },
    selectAll(ids: string[]): void {
      replaceWith(ids);
    },
    clear(): void {
      selected.clear();
      anchorId = null;
    },
  };
}
