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
  let selected = $state<Set<string>>(new Set());
  let anchorId = $state<string | null>(null);

  return {
    get ids() {
      return selected;
    },
    set ids(next: Set<string>) {
      selected = next;
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
          const next = new Set(selected);
          for (let i = lo; i <= hi; i++) next.add(orderedIds[i]!);
          selected = next;
          return;
        }
        // Anchor no longer visible — fall through.
      }

      if (isToggle || opts.plainClick === 'toggle') {
        const next = new Set(selected);
        if (next.has(id)) next.delete(id);
        else next.add(id);
        selected = next;
        anchorId = id;
        return;
      }

      selected = new Set([id]);
      anchorId = id;
    },
    selectAll(ids: string[]): void {
      selected = new Set(ids);
    },
    clear(): void {
      selected = new Set();
      anchorId = null;
    },
  };
}
