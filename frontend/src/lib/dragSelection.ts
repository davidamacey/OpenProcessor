import type { Selection } from '$lib/selection.svelte';

/**
 * The drag set a grabbed card stands for (Finder pattern): grabbing any selected card drags the
 * whole selection; grabbing an unselected card replaces the selection with just that one.
 * `consider` fires repeatedly during a drag, so a card already in `dragIds` changes nothing.
 * Returns the new drag ids, or `null` when nothing changed.
 */
export function captureDragSelection(
  draggedId: string | undefined,
  dragIds: readonly string[],
  sel: Selection,
): string[] | null {
  if (!draggedId || dragIds.includes(draggedId)) return null;
  if (sel.has(draggedId) && sel.size > 1) return [...sel.ids];
  // eslint-disable-next-line svelte/prefer-svelte-reactivity -- copied into the selection's own SvelteSet by its setter
  sel.ids = new Set([draggedId]);
  return [draggedId];
}
