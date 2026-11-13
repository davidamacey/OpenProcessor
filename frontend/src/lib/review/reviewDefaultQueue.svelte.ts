/**
 * Review without a queue opens the served default queue: the first, in served order, that holds
 * work. A grid's list (an item the user came from) is not a queue and keeps walking the
 * Detections order. While the vocabulary loads nothing is read; when every queue is empty there is
 * nothing to review.
 */
import type { WorkspaceController } from '$lib/workspace/workspaceController.svelte';
import { reviewTabsVocabularyStore } from '$stores/reviewTabsVocabulary.svelte';

/** `queue` is read lazily (it follows the URL); `walksDetections` is fixed at mount, because
 *  locating the item clears the pending item and the list must not then jump queue. */
export function createReviewDefaultQueue(
  ws: WorkspaceController,
  queue: () => string | null,
  walksDetections: boolean,
) {
  const vocabulary = reviewTabsVocabularyStore;
  const open = $derived(!queue() && !walksDetections);
  const defaultPending = $derived(
    open && (!vocabulary.loaded || vocabulary.defaultTab != null),
  );
  const nothingToReview = $derived(
    open && vocabulary.loaded && vocabulary.defaultTab === null,
  );

  $effect(() => {
    const served = vocabulary.defaultTab;
    if (defaultPending && vocabulary.loaded && served)
      ws.setQueue(served, { replace: true });
  });

  return {
    get defaultPending() {
      return defaultPending;
    },
    get nothingToReview() {
      return nothingToReview;
    },
  };
}
