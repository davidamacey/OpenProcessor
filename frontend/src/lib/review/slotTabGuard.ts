/**
 * The predicate behind Finding C.2's "class letters are inert on the
 * Plates tab" invariant (docs/genericization-plan-2026-09-13.md), pulled
 * out of the inline `if (tab === 'plates') return;` guard in
 * `review/+page.svelte`'s class-drop `$effect` so it has one, testable
 * definition instead of being re-derived by reading a `return` statement.
 *
 * Today there is exactly one slot tab (`'plates'`). When P2.8 makes
 * `ReviewTab` a `CoreReviewTab | SlotReviewTab` union (`slot:${string}`),
 * this becomes `tab.startsWith('slot:')` and every slot tab gets the
 * same suppression for free, by construction — not by remembering to
 * add another literal to a growing list.
 */

/** Tab ids that suppress class-drop hotkey registration today. */
const SLOT_TABS = new Set(['plates']);

export function isSlotSuppressedTab(tab: string): boolean {
  return SLOT_TABS.has(tab);
}
