/**
 * Pure keymap-table builder for the Plates tab, extracted from the
 * inline `$effect` in `review/+page.svelte` (Phase 0 seam ahead of
 * P2.8's `reviewTabs.ts` data-driving,
 * docs/genericization-plan-2026-09-13.md §3.6/§5a).
 *
 * This returns DATA (combo -> handler + description), not the
 * `keyboardStore.register` side effect itself — the page still owns
 * registration/cleanup, this module owns "what the table looks like
 * for a given mode." That split is what P2.8 needs: a slot's
 * `QueueCapability.keymap` is exactly this shape, and
 * `reservedHotkeyLetters()` (classHotkey.ts, not yet updated — see
 * Finding C.2) will eventually derive from tables built this way
 * instead of a hand-maintained literal list.
 */

export interface PlateKeymapHandlers {
  confirmPlate: () => void | Promise<void>;
  rejectPlate: () => void | Promise<void>;
  markFalsePositive: () => void | Promise<void>;
  toggleEdit: () => void;
  plateBack: () => void | Promise<void>;
  advance: () => void;
  saveBboxAndExit: () => void | Promise<void>;
}

export interface KeymapEntry {
  combo: string;
  fn: () => void | Promise<void>;
  description: string;
}

/**
 * The Plates tab's keymap for the given mode. Scan mode (default):
 * Enter/D/F/E/arrowleft/B/arrowright. Edit mode: Enter/Escape only —
 * the bbox canvas owns arrow/[ / ]/Backspace directly (see the
 * `canvasKey` window listener the page still wires separately; that
 * one isn't a `keyboardStore` combo and isn't part of this table).
 */
export function buildPlateKeymap(
  editMode: boolean,
  h: PlateKeymapHandlers,
): KeymapEntry[] {
  if (editMode) {
    return [
      { combo: 'enter', fn: h.saveBboxAndExit, description: 'Save bbox & exit edit' },
      { combo: 'escape', fn: h.toggleEdit, description: 'Cancel edit' },
    ];
  }
  return [
    { combo: 'enter', fn: h.confirmPlate, description: 'Confirm plate & advance' },
    { combo: 'd', fn: h.rejectPlate, description: 'Reject (no plate visible)' },
    { combo: 'f', fn: h.markFalsePositive, description: 'False positive (keep box)' },
    { combo: 'e', fn: h.toggleEdit, description: 'Edit bbox' },
    // Back: re-insert the most-recently-confirmed plate so the operator
    // can correct mistakes without scrolling back through the queue.
    { combo: 'arrowleft', fn: h.plateBack, description: 'Back to last confirmed plate' },
    { combo: 'b', fn: h.plateBack, description: 'Back (alias)' },
    { combo: 'arrowright', fn: h.advance, description: 'Next item' },
  ];
}

/** Single-character combos in a keymap table — what
 *  `reservedHotkeyLetters()` needs to derive the reserved set (Finding
 *  C.2). Multi-char combos (`arrowleft`, `enter`, `escape`) are excluded
 *  since a class hotkey is always a single character. */
export function singleCharCombos(entries: KeymapEntry[]): string[] {
  return entries.map((e) => e.combo).filter((c) => c.length === 1);
}
