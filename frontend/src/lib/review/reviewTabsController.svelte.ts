/**
 * The Review view's queue state. The queue is `ws.url.queue` (a served queue id, or none to walk
 * the Workspace query on `/crops`) and the item to land on is `ws.url.focus`; this writes neither.
 * It maps the queue to a slot tab when a slot reads it (the region queue), and reads the served
 * per-queue vocabulary: label, honoured filters and the queue-only filter values the URL carries.
 */
import { ITEM_FILTER_PARAMS } from '$lib/itemFilter/itemFilterControls';
import { createReviewDefaultQueue } from '$lib/review/reviewDefaultQueue.svelte';
import { REVIEW_TABS, tabForQueue } from '$lib/reviewTabs';
import type { WorkspaceController } from '$lib/workspace/workspaceController.svelte';
import { reviewTabsVocabularyStore } from '$stores/reviewTabsVocabulary.svelte';
import type { ReviewTab } from '$lib/types_review';

/** The URL value of a queue-only filter: one value, or several for a repeatable one. */
export type ServedFilterValue = string | string[];

export function createReviewTabsController(ws: WorkspaceController) {
  const queue = $derived(ws.url.queue);
  const tab = $derived<ReviewTab>(tabForQueue(queue));
  // The slot backing the current queue, if any: the one value every slot-queue call site hangs off.
  const activeSlot = $derived(REVIEW_TABS.find((t) => t.id === tab)?.slot ?? null);
  // The vocabulary is keyed by the served queue id (a slot reads the queue named by its endpointId).
  const activeTabEndpointId = $derived(queue ?? '');

  // The item a deep link asks for, read once at mount: Review rewrites `focus` as it advances.
  // svelte-ignore state_referenced_locally
  let pendingCropId = $state<string | null>(
    ws.url.focus?.kind === 'item' ? ws.url.focus.id : null,
  );
  // True from mount until that item is located (or the attempt gives up). While true the keys
  // register nothing, so a keypress cannot act on whatever item #1 of the first page is.
  let awaitingDeepLink = $state<boolean>(pendingCropId != null);

  // svelte-ignore state_referenced_locally
  const walksDetections = !queue && pendingCropId != null;
  const defaultQueue = createReviewDefaultQueue(ws, () => queue, walksDetections);

  /** The queue-only filters this queue serves (region status, text, ...): every served spec the
   *  shared item filter does not own. They live in the URL as plain parameters. */
  const queueFilterSpecs = $derived.by(() => {
    // The slot names its text filter in its own words (a deployment profile's "SSCC").
    const text = activeSlot?.capabilities.queue?.textFilter;
    return reviewTabsVocabularyStore
      .filterSpecsFor(activeTabEndpointId)
      .filter((s) => !ITEM_FILTER_PARAMS.has(s.param))
      .map((s) => (text && s.param === text.param ? { ...s, label: text.label } : s));
  });
  const queueFilterValues = $derived.by<Record<string, ServedFilterValue>>(() => {
    const out: Record<string, ServedFilterValue> = {};
    for (const spec of queueFilterSpecs) {
      const all = ws.searchParams.getAll(spec.param);
      if (all.length > 0) out[spec.param] = all.length > 1 ? all : all[0]!;
    }
    return out;
  });

  const activeQueueLabel = $derived(
    reviewTabsVocabularyStore.labelFor(
      activeTabEndpointId,
      activeSlot?.capabilities.queue?.tabLabel ?? (queue ? queue : 'Detections'),
    ),
  );

  return {
    get queue() {
      return queue;
    },
    get tab() {
      return tab;
    },
    get activeSlot() {
      return activeSlot;
    },
    get activeTabEndpointId() {
      return activeTabEndpointId;
    },
    get pendingCropId() {
      return pendingCropId;
    },
    set pendingCropId(v: string | null) {
      pendingCropId = v;
    },
    get awaitingDeepLink() {
      return awaitingDeepLink;
    },
    set awaitingDeepLink(v: boolean) {
      awaitingDeepLink = v;
    },
    get queueFilterSpecs() {
      return queueFilterSpecs;
    },
    get queueFilterValues() {
      return queueFilterValues;
    },
    get defaultPending() {
      return defaultQueue.defaultPending;
    },
    get nothingToReview() {
      return defaultQueue.nothingToReview;
    },
    get activeQueueLabel() {
      return activeQueueLabel;
    },
  };
}

export type ReviewTabsController = ReturnType<typeof createReviewTabsController>;
