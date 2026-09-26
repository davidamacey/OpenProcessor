/**
 * Pure keymap-table builder for a slot's review queue tab
 * (docs/genericization-plan-2026-09-13.md §9.5/P2.8c). It reads the spec's
 * `queue.keymap` directly, so a slot's combos are declared exactly once
 * (no second hand-maintained copy to drift, Finding C.2).
 *
 * Returns DATA (action id + combo -> handler + description), not the
 * `keyboardStore.registerAction` side effect itself — the page still owns
 * registration/cleanup.
 *
 * Every entry names its keymap action id (docs/design/configurable-
 * keyboard-shortcuts-plan-2026-09-26.md §5.3). The combos for the slot's
 * own verbs (reject / false positive / edit / back) come from the slot's
 * `queue.keymap` — the served region slot derives that from the keymap
 * store (`servedRegionSlot.ts`), a tier-2 slot declares its own. The
 * slot-independent ones (confirm and next in scan mode, save and cancel
 * in edit mode) come straight from the keymap store, and every
 * description is the store's label for the id.
 */

import type { SlotSpec } from '../annotations/types';
import { formatShortcutKey } from '$lib/keyboardDisplay';
import { keymapStore } from '$stores/keymap.svelte';

export interface SlotKeymapHandlers {
  confirm: () => void | Promise<void>;
  reject: () => void | Promise<void>;
  markFalsePositive?: () => void | Promise<void>;
  toggleEdit: () => void;
  back?: () => void | Promise<void>;
  advance: () => void;
  saveAndExit: () => void | Promise<void>;
}

export interface KeymapEntry {
  actionId: string;
  combo: string;
  fn: () => void | Promise<void>;
  description: string;
}

/**
 * A slot's queue-tab keymap for the given mode. Scan mode (default)
 * emits, in order: `review.region.confirm`, each `reject` combo, each
 * `markFalsePositive` combo (only if the slot declares one AND the
 * handler is supplied), each `editBox` combo, each `back` combo, then
 * `review.region.next`. Edit mode: `box_edit.save` / `box_edit.cancel`
 * only — the bbox canvas owns the nudge/edge/clear keys directly.
 *
 * For a slot with the standard region keymap this produces the
 * enter/d/f/e/arrowleft/b/arrowright sequence — pinned in
 * `slotKeymap.test.ts`.
 */
export function buildSlotKeymap(
  spec: SlotSpec,
  editMode: boolean,
  h: SlotKeymapHandlers,
): KeymapEntry[] {
  const vars = { region: spec.label.singular };
  const entries: KeymapEntry[] = [];
  const add = (
    actionId: string,
    combos: string[],
    fn: () => void | Promise<void>,
  ): void => {
    const description = keymapStore.label(actionId, vars);
    for (const combo of combos) entries.push({ actionId, combo, fn, description });
  };
  const stored = (id: string) => keymapStore.keysFor(id);
  if (editMode) {
    add('box_edit.save', stored('box_edit.save'), h.saveAndExit);
    add('box_edit.cancel', stored('box_edit.cancel'), h.toggleEdit);
    return entries;
  }
  const km = spec.capabilities.queue?.keymap ?? {};
  add('review.region.confirm', stored('review.region.confirm'), h.confirm);
  add('review.region.reject', km.reject ?? [], h.reject);
  if (h.markFalsePositive) {
    add('review.region.false_positive', km.markFalsePositive ?? [], h.markFalsePositive);
  }
  add('review.region.edit_box', km.editBox ?? [], h.toggleEdit);
  if (h.back) add('review.region.back', km.back ?? [], h.back);
  add('review.region.next', stored('review.region.next'), h.advance);
  return entries;
}

/** Single-character combos in a keymap table — what
 *  `reservedHotkeyLetters()` (`../classHotkey.ts`) uses to derive the
 *  reserved set (Finding C.2, closed structurally by P2.8c). Multi-char
 *  combos (`arrowleft`, `enter`, `escape`) are excluded since a class
 *  hotkey is always a single character. */
export function singleCharCombos(entries: KeymapEntry[]): string[] {
  return entries.map((e) => e.combo).filter((c) => c.length === 1);
}

/**
 * The on-screen glyph for a slot's reject/"no {label} visible" action —
 * the review-tab hint strip used to hardcode this to the literal `"D"`
 * regardless of what the active slot's own keymap actually binds, which
 * silently went wrong for any slot whose `queue.keymap.reject` isn't
 * `['d']` (dispatch itself was never wrong — `buildSlotKeymap` above
 * already reads `queue.keymap` correctly; only the hint text had a
 * second, hardcoded copy). Reads the exact same `keymap.reject` lookup
 * `buildSlotKeymap` uses — first bound combo, formatted for display —
 * so there is only ever one place a slot's reject key is declared. A
 * slot with no reject combo prints the keymap's own reject key.
 */
export function rejectKeyGlyph(spec: SlotSpec): string {
  const combo = spec.capabilities.queue?.keymap.reject?.[0];
  return combo ? formatShortcutKey(combo) : keymapStore.glyph('review.region.reject');
}
