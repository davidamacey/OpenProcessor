/**
 * Shared "bind a letter to a class" flow.
 *
 * Used by /classes and the ~ shortcut overlay, which previously carried
 * verbatim copies of this validation.
 */

import { ApiError, renameClass } from '$lib/api';
import { isPickerHiddenClass } from '$lib/classVisibility';
import { slotRegistry } from '$lib/annotations/registeredSlots';
import type { SlotRegistry } from '$lib/annotations/registry';
import { classesStore } from '$stores/classes.svelte';
import { toastStore } from '$stores/toast.svelte';
import type { RegistryClass } from '$lib/types';

/**
 * Single-char keys reserved for a labeling action — never bindable to a
 * class. Two window keydown listeners run for every keypress — the
 * keyboardStore dispatcher and the layout's class-letter listener — and
 * preventDefault in one does not stop the other, so a class bound to a
 * reserved letter would fire both actions on the same keypress.
 *
 * `classesStore.reservedHotkeys` is `GET {API_PREFIX}/classes`'s own
 * `reserved_hotkeys` field — the base action keys (`g n d z x u a m /`)
 * union every *server-known* slot's keymap letters. It is never
 * recomputed client-side; this function only adds one thing on top: any
 * registered slot's keymap letters the server doesn't know about yet — a
 * tier-2 deployment profile registered purely client-side (no backend
 * checkout) has no way to tell the backend about its own keymap. That
 * union is redundant for a slot the backend already knows (the server's
 * reserved set already includes that slot's keymap letters) but is what
 * keeps a *second*, backend-unaware slot safe without a human
 * re-auditing every class hotkey.
 */
export function reservedHotkeyLetters(
  registry: SlotRegistry = slotRegistry,
): Set<string> {
  const out = new Set(classesStore.reservedHotkeys);
  for (const spec of registry.queues) {
    for (const combos of Object.values(spec.capabilities.queue?.keymap ?? {})) {
      for (const combo of combos ?? []) {
        if (combo.length === 1) out.add(combo);
      }
    }
  }
  return out;
}

/**
 * Validate and persist a class's hotkey letter. Empty string clears it.
 *
 * Toasts on both success and rejection; never throws. Callers own their own
 * busy/pending flag.
 */
export async function setClassHotkey(cls: RegistryClass, raw: string): Promise<void> {
  const next = raw.trim().toLowerCase();
  const current = (cls.hotkey_letter ?? '').toLowerCase();
  if (next === current) return;
  if (next.length > 1) {
    toastStore.error('Hotkey must be a single character.');
    return;
  }
  if (next) {
    if (reservedHotkeyLetters().has(next)) {
      toastStore.error(
        `'${next}' is reserved for a labeling action — pick another letter.`,
      );
      return;
    }
    if (isPickerHiddenClass(cls.name)) {
      toastStore.error(
        `${cls.name} is hidden from class-assignment surfaces and can't be bound to a hotkey.`,
      );
      return;
    }
    // Reject duplicates against other classes' already-bound letters. The
    // backend enforces this too (PUT {API_PREFIX}/classes/{id} returns 409), but
    // catching it client-side gives a clearer message with no round-trip.
    const owner = classesStore.classes.find(
      (c) =>
        c.id !== cls.id &&
        !c.deprecated &&
        (c.hotkey_letter ?? '').toLowerCase() === next,
    );
    if (owner) {
      toastStore.error(`'${next}' is already assigned to ${owner.name}.`);
      return;
    }
  }
  try {
    // PUT {API_PREFIX}/classes/{id} treats '' as "clear binding".
    await renameClass(cls.id, { hotkey_letter: next });
    toastStore.success(
      next ? `${cls.name} → hotkey '${next}'` : `${cls.name} → hotkey cleared`,
    );
    await classesStore.clearAndRefetch();
  } catch (e) {
    // Show the server's 400/409/422 detail verbatim when it sent one — the
    // client-side checks above cover the common cases, but a race (another
    // operator bound the same letter a moment ago) or a rule the client
    // doesn't know about yet still needs the server's own words.
    const detail = e instanceof ApiError ? (e.detail ?? e.message) : (e as Error).message;
    toastStore.error(`Hotkey set failed: ${detail}`);
  }
}
