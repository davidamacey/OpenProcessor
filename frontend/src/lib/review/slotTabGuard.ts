/**
 * The predicate behind Finding C.2's "class letters are inert on a slot
 * tab" invariant (docs/genericization-plan-2026-09-13.md), pulled out of
 * the inline `if (tab === 'plates') return;` guard in
 * `review/+page.svelte`'s class-drop `$effect` so it has one, testable
 * definition instead of being re-derived by reading a `return` statement.
 *
 * P2.8b (the §9.5 addendum) closed the "when P2.8 lands" TODO this file
 * used to carry: `ReviewTab` is now `CoreReviewTab | SlotReviewTab`
 * (`slot:${string}`), so this is a structural check via `isSlotTab` —
 * every slot tab is suppressed for free, by construction, not by
 * remembering to add another literal to a growing list.
 */

/** Loosely typed to `string` (not `ReviewTab`) so a stray/unknown tab id
 *  fails safe (not suppressed) rather than a type error at the call
 *  site — mirrors the old Set-based implementation's tolerance. */
export function isSlotSuppressedTab(tab: string): boolean {
  return tab.startsWith('slot:');
}
