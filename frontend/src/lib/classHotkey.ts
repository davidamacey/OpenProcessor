/**
 * Shared "bind a letter to a class" flow.
 *
 * Used by /classes and the ~ shortcut overlay, which previously carried
 * verbatim copies of this validation.
 */

import { renameClass } from '$lib/api';
import { isPickerHiddenClass } from '$lib/classVisibility';
import { slotRegistry } from '$lib/annotations/registeredSlots';
import type { SlotRegistry } from '$lib/annotations/registry';
import { classesStore } from '$stores/classes.svelte';
import { toastStore } from '$stores/toast.svelte';
import type { RegistryClass } from '$lib/types';

/**
 * Single-char keys the labeling pages bind to actions.
 *
 * Two window keydown listeners run for every keypress — the keyboardStore
 * dispatcher and the layout's class-letter listener — and preventDefault in
 * one does not stop the other. So a class bound to 'd' would discard the
 * selection AND label it in the same keypress. (b / e / f are also bound on
 * the review plates tab, but class letters are inert there: the review page
 * registers no drop handler on that tab.)
 *
 * '/' is reserved too (Phase 7, audit remediation plan P1-4): it opens the
 * `/review` fuzzy-search class picker via keyboardStore. Same collision
 * shape as the letters above — a class bound to '/' would both open the
 * picker and assign itself on the same keypress.
 */
export const RESERVED_HOTKEY_LETTERS = new Set([
  'g',
  'n',
  'd',
  'z',
  'x',
  'u',
  'a',
  'm',
  '/',
]);

/**
 * `RESERVED_HOTKEY_LETTERS` above ∪ every single-character combo any
 * queue-capable slot's `QueueCapability.keymap` declares — closes Finding
 * C.2 structurally (docs/genericization-plan-2026-09-13.md §9.5/P2.8c):
 * a class can no longer be bound to a letter a slot's review-tab keymap
 * owns, without anyone having to remember to extend a hand-maintained
 * list when a new slot ships. For `license_plate` today this adds
 * `d`/`f`/`e`/`b` on top of the base set (`d` was already reserved).
 * Class letters are already inert while a slot tab is active
 * (`isSlotSuppressedTab`) so this is defense in depth, not a fix for a
 * live collision — but it is what makes a *second* capable slot safe
 * without a human re-auditing every letter it uses.
 */
export function reservedHotkeyLetters(
  registry: SlotRegistry = slotRegistry,
): Set<string> {
  const out = new Set(RESERVED_HOTKEY_LETTERS);
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
    // backend enforces this too (PUT {API_PREFIX}/classes/{id} returns 400), but
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
    toastStore.error(`Hotkey set failed: ${(e as Error).message}`);
  }
}
