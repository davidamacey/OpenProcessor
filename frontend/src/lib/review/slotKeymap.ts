/**
 * Pure keymap-table builder for a slot's review queue tab, generalized
 * (docs/genericization-plan-2026-09-13.md §9.5/P2.8c) from the
 * plate-only `buildPlateKeymap` that used to hand-maintain a SECOND copy
 * of `licensePlateSlot.capabilities.queue.keymap` — same drift hazard as
 * Finding C.2, one layer down. This reads the spec's `queue.keymap`
 * directly, so a slot's combos are declared exactly once.
 *
 * Returns DATA (combo -> handler + description), not the
 * `keyboardStore.register` side effect itself — the page still owns
 * registration/cleanup.
 *
 * `enter`/`escape` in edit mode and `arrowright` (advance) in scan mode
 * stay unconditional, matching every capable slot's actual behavior
 * today (edit save/cancel and "next item" apply regardless of what a
 * profile's `queue.keymap` declares) rather than being second capability
 * fields with no current variation to justify them.
 */

import type { SlotSpec } from '../annotations/types';

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
  combo: string;
  fn: () => void | Promise<void>;
  description: string;
}

/**
 * A slot's queue-tab keymap for the given mode. Scan mode (default)
 * emits, in order: `enter` (confirm), each `reject` combo, each
 * `markFalsePositive` combo (only if the slot declares one AND the
 * handler is supplied), each `editBox` combo, each `back` combo, then
 * `arrowright` (advance). Edit mode: `enter` (save+exit) / `escape`
 * (cancel) only — the bbox canvas owns arrow/[ / ]/Backspace directly.
 *
 * For `licensePlateSlot` this produces the exact same
 * enter/d/f/e/arrowleft/b/arrowright sequence the old hand-written
 * `buildPlateKeymap` did — see `slotKeymap.test.ts`'s
 * `licensePlateSlot` case, the no-regression proof.
 */
export function buildSlotKeymap(
  spec: SlotSpec,
  editMode: boolean,
  h: SlotKeymapHandlers,
): KeymapEntry[] {
  const label = spec.label.singular;
  if (editMode) {
    return [
      { combo: 'enter', fn: h.saveAndExit, description: `Save ${label} & exit edit` },
      { combo: 'escape', fn: h.toggleEdit, description: 'Cancel edit' },
    ];
  }
  const km = spec.capabilities.queue?.keymap ?? {};
  const entries: KeymapEntry[] = [
    { combo: 'enter', fn: h.confirm, description: `Confirm ${label} & advance` },
  ];
  for (const c of km.reject ?? []) {
    entries.push({ combo: c, fn: h.reject, description: `Reject (no ${label} visible)` });
  }
  if (h.markFalsePositive) {
    for (const c of km.markFalsePositive ?? []) {
      entries.push({
        combo: c,
        fn: h.markFalsePositive,
        description: 'False positive (keep box)',
      });
    }
  }
  for (const c of km.editBox ?? []) {
    entries.push({ combo: c, fn: h.toggleEdit, description: `Edit ${label} box` });
  }
  if (h.back) {
    for (const c of km.back ?? []) {
      entries.push({
        combo: c,
        fn: h.back,
        description: `Back to last confirmed ${label}`,
      });
    }
  }
  entries.push({ combo: 'arrowright', fn: h.advance, description: 'Next item' });
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
 * `buildSlotKeymap` uses — first bound combo, uppercased for display —
 * so there is only ever one place a slot's reject key is declared.
 */
export function rejectKeyGlyph(spec: SlotSpec): string {
  const combo = spec.capabilities.queue?.keymap.reject?.[0];
  return (combo ?? 'd').toUpperCase();
}
