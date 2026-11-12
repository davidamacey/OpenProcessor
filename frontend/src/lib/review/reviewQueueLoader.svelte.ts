/**
 * The Review view's queue: the pager, the cursor, the handled-ids dedup set, the locate jump to
 * the focused item and the refetch effects that load it.
 *
 * The items come from `reviewSource`: a served queue reads `GET /review/{queue}` in its own order;
 * with no queue the Workspace query is walked on `GET /crops`. `queue`, `cursor` and `handledIds`
 * live here, once, and every other review controller reads and writes them through this object
 * (by reference), never through a copy. The immediate effect loads page 1 and the locate effect
 * waits for that first page.
 */

import { untrack } from 'svelte';
import { apiErrorText } from '$lib/api_errors';
import { withoutOpenVocab } from '$lib/itemFilter/itemFilterState.svelte';
import { createPager } from '$lib/pager.svelte';
import { createReviewQueueController } from '$lib/review/reviewController.svelte';
import {
  emptyQueueMessage,
  locateMissMessage,
  queuePosition,
} from '$lib/review/reviewCopy';
import { reviewSource, type ReviewSourceQuery } from '$lib/review/reviewSource.svelte';
import type { ReviewTabsController } from '$lib/review/reviewTabsController.svelte';
import { isSemanticSearchAvailable } from '$lib/strategies';
import type { CropFilter, HiddenCounts } from '$lib/types';
import type { SortOption } from '$lib/types_cursor';
import type { ReviewItem } from '$lib/types_review';
import type { WorkspaceController } from '$lib/workspace/workspaceController.svelte';
import { reviewTabsVocabularyStore } from '$stores/reviewTabsVocabulary.svelte';
import { strategiesStore } from '$stores/strategies.svelte';
import { toastStore } from '$stores/toast.svelte';

/** Served page size of the review queue. */
export const REVIEW_PAGE_SIZE = 30;

export interface ReviewQueueLoaderOptions {
  ws: WorkspaceController;
  tabs: ReviewTabsController;
}

export function createReviewQueueLoader(opts: ReviewQueueLoaderOptions) {
  const { ws, tabs } = opts;

  // The one place the Workspace query, the queue and its queue-only filters become a source.
  const sourceQuery = $derived.by<ReviewSourceQuery>(() => {
    const queue = tabs.queue;
    const filter: Record<string, unknown> = {
      ...ws.itemQuery(withoutOpenVocab),
      ...tabs.queueFilterValues,
    };
    // A queue is read from its own route; as a filter it would only repeat itself (and the
    // region queue is not a filter value at all).
    if (queue) delete filter.review_tab;
    return {
      queue,
      filter: filter as CropFilter,
      sort: ws.url.sort[queue ? 'review' : 'detections'] ?? null,
    };
  });
  const source = $derived(reviewSource(sourceQuery));
  const sourceKey = $derived(JSON.stringify(sourceQuery));

  let cursor = $state<number>(0); // index within accumulated items

  // Set from the response whenever the requested sort couldn't be honored server-side (its field
  // isn't backfilled yet). Rendered as a small inline note, never a toast.
  let sortFallbackReason = $state<string | null>(null);
  // The server's own explanation for an empty queue, distinct from sortFallbackReason.
  let emptyReason = $state<string | null>(null);
  // What the queue hides (test holdout, ignored items), as the last page served it.
  let hidden = $state<HiddenCounts | null>(null);
  let sortApplied = $state<string | null>(null);
  // The orders the source serves (`sort_options`): a queue's registry sorts, or the crop sorts.
  let sortOptions = $state<SortOption[]>([]);

  // One fetchPage closure means the queue + filter set can't drift between page 1 and the pages
  // the cursor pulls in behind it.
  const queue = createPager<ReviewItem>({
    fetchPage: async (cursor) => {
      const res = await source.page(cursor, REVIEW_PAGE_SIZE);
      sortFallbackReason = res.sort_fallback_reason ?? null;
      sortApplied = res.sort_applied ?? null;
      sortOptions = res.sort_options ?? [];
      emptyReason = res.empty_reason ?? null;
      hidden = res.hidden ?? null;
      return res;
    },
    keyOf: (i) => i.id,
    onReset: () => {
      cursor = 0;
      handledIds.clear();
    },
    accept: (i) => !handledIds.has(i.id),
  });

  // Eagerly prefetch the next page when the cursor is within this many items of the end of the
  // loaded buffer, so confirming the last loaded item never shows "no more" for a frame.
  const PREFETCH_AHEAD = 5;
  function maybePrefetch(): void {
    if (queue.loadingMore || !queue.hasMore) return;
    if (queue.items.length - cursor <= PREFETCH_AHEAD) {
      void loadMore();
    }
  }

  // "More like this" needs the semantic-search overlay; hidden on a backend without it.
  const semanticSearchAvailable = $derived(
    isSemanticSearchAvailable(strategiesStore.methods.overlays),
  );

  // Crop ids this session has already assigned / discarded / triaged. The server collection
  // SHRINKS as you label, so page N+1 can contain a crop that page N would have held before the
  // shift; loadMore's dedup only compares against the buffer, which no longer holds a handled
  // crop. Entries come back out on rollback and on undo-restore. Not $state: only loadMore reads
  // it, inside an async callback.
  // eslint-disable-next-line svelte/prefer-svelte-reactivity -- deliberately not reactive, see the comment above
  const handledIds = new Set<string>();

  // Queue action controller (assign / discard / skip / undo); `queue`, `cursor` and `handledIds`
  // stay owned here and are handed in by reference/accessor.
  const queueController = createReviewQueueController({
    queue,
    handledIds,
    getCursor: () => cursor,
    setCursor: (v) => {
      cursor = v;
    },
    maybePrefetch: () => maybePrefetch(),
  });

  // The served queue position, not the index within the loaded buffer (a deep link loads only the
  // located page).
  const currentPosition = $derived(queuePosition(queue.firstIndex, cursor));

  const loadFirst = () => queue.loadFirst();
  const loadMore = () => queue.loadMore();

  // The focused item: ask the server exactly where it sits (one request, `/review/{queue}/locate`
  // or `/crops/locate`) and load only the page that starts there.
  let jumpingToCrop = $state(false);
  async function jumpToPendingCrop(): Promise<void> {
    const cropId = tabs.pendingCropId;
    if (cropId == null) return;
    jumpingToCrop = true;
    try {
      const loc = await source.locate(cropId);
      if (!loc.found || loc.index == null) {
        toastStore.info(loc.reasonMessage ?? locateMissMessage(loc.reason));
        return;
      }
      await queue.loadAt(loc.cursorBefore, loc.index);
      const idx = queue.items.findIndex((i) => i.id === cropId);
      cursor = Math.max(0, idx);
      // The located page may end soon after the item; fetch the rows after it now.
      maybePrefetch();
    } catch (e) {
      toastStore.error(`Locate failed: ${apiErrorText(e)}`);
    } finally {
      jumpingToCrop = false;
      tabs.pendingCropId = null;
      tabs.awaitingDeepLink = false;
    }
  }

  $effect(() => {
    if (
      tabs.pendingCropId == null ||
      queue.loading ||
      queue.loadingMore ||
      jumpingToCrop
    ) {
      return;
    }
    if (!queue.loaded) return; // first page not in yet
    void jumpToPendingCrop();
  });

  // A change of queue, filter or sort reloads page 1. Guarded by a same-key check: the effect body
  // can run an extra time for the same values, and a reload resets the cursor.
  let lastKey: string | null = null;
  $effect(() => {
    const key = sourceKey;
    // A queue's own filters (combine_conflict, region_status, ...) are the served ones: read the
    // first page only once the vocabulary says which the URL's values are, so a link's filter is
    // never missing from the first request.
    if (tabs.queue && !reviewTabsVocabularyStore.loaded) return;
    if (key === lastKey) return;
    lastKey = key;
    untrack(() => void loadFirst());
  });

  const current = $derived<ReviewItem | null>(queue.items[cursor] ?? null);

  const filtersActive = $derived(
    !ws.filter.isEmpty || Object.keys(tabs.queueFilterValues).length > 0,
  );
  const emptyMessage = $derived(
    emptyQueueMessage({
      label: tabs.activeQueueLabel,
      description:
        reviewTabsVocabularyStore.descriptionFor(tabs.activeTabEndpointId) ?? null,
      sortFallbackReason,
      emptyReason,
      emptyState: reviewTabsVocabularyStore.emptyState,
      filtersActive,
    }),
  );
  return {
    queue,
    handledIds,
    queueController,
    get cursor() {
      return cursor;
    },
    set cursor(v: number) {
      cursor = v;
    },
    get current() {
      return current;
    },
    get sortFallbackReason() {
      return sortFallbackReason;
    },
    get emptyReason() {
      return emptyReason;
    },
    get hidden() {
      return hidden;
    },
    get sortApplied() {
      return sortApplied;
    },
    get sortOptions() {
      return sortOptions;
    },
    maybePrefetch,
    get semanticSearchAvailable() {
      return semanticSearchAvailable;
    },
    get currentPosition() {
      return currentPosition;
    },
    loadFirst,
    get emptyMessage() {
      return emptyMessage;
    },
  };
}

export type ReviewQueueLoader = ReturnType<typeof createReviewQueueLoader>;
