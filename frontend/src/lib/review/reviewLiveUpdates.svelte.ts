/**
 * `/review` live "N new" pill: counts crop events from the curation SSE
 * stream so the operator decides when to pull the new batch into the queue.
 *
 * Moved verbatim out of `review/+page.svelte`.
 */

import { onMount } from 'svelte';
import { subscribeCurationEvents, type CurationEventSubscription } from '$lib/sse';

export function createReviewLiveUpdates(reload: () => void) {
  // -- SSE: live updates as ingest + workers classify new crops -------
  // We don't fetch each new crop individually (no single-crop endpoint
  // is exposed); instead we count incoming crop.* events and surface a
  // "X new crops" pill the user can click to refresh the queue. Auto-
  // refreshing while the operator is mid-keystroke would be jarring —
  // they decide when to pull in the new batch.
  let liveNewCount = $state<number>(0);
  let liveSub: CurationEventSubscription | null = null;
  onMount(() => {
    liveSub = subscribeCurationEvents({
      // Both classification + any slot-verify changes are interesting on
      // the review page — the operator may be on any tab. `_verified` is
      // structural (matches every crop.<slot.key>_verified event, plus
      // the generic crop.region_verified — see
      // sse.ts's slotVerifiedEventTypes()), not a single hardcoded slot.
      onEvent: (ev) => {
        if (
          ev.type === 'crop.classified' ||
          ev.type === 'crop.created' ||
          ev.type.endsWith('_verified')
        ) {
          liveNewCount += 1;
        }
      },
    });
    return () => {
      liveSub?.close();
      liveSub = null;
    };
  });

  function refreshFromLive(): void {
    liveNewCount = 0;
    reload();
  }

  return {
    get liveNewCount() {
      return liveNewCount;
    },
    refreshFromLive,
  };
}
