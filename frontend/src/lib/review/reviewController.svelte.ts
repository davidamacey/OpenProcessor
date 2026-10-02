/**
 * Review-queue action controller — assign / discard / skip / undo,
 * extracted out of `review/+page.svelte` (docs/design/test-audit-2026-09-24.md
 * P1-4) so the queue-mutation logic (optimistic remove + rollback-on-failure,
 * cursor advance, undo re-insert) is unit-testable without mounting the
 * page. Three of these survived mutation under only the source-scan test
 * suite before this extraction: a failed `assign()` no longer restoring the
 * item, a failed `discard()` no longer restoring the item, and `skip()` no
 * longer advancing the cursor (see the audit's §2.2 table).
 *
 * Follows the existing `slotGalleryController.svelte.ts` factory-function
 * convention (state via closures, not a class) but stays deliberately thin:
 * the page still owns `queue` / `cursor` / `handledIds` (shared with
 * slot-tab actions — confirmSlot/rejectSlot/markFalsePositive — and with
 * arrow-key queue navigation that this controller doesn't touch), and hands
 * them in by reference/accessor so there is exactly one copy of each, never
 * a page-level and a controller-level copy that can drift apart.
 */

import { putCropLabel, reviewDismissCrop } from '$lib/api';
import type { Pager } from '$lib/pager.svelte';
import type { ReviewItem } from '$lib/types';
import { classesStore } from '$stores/classes.svelte';
import { toastStore } from '$stores/toast.svelte';
import { undoStore } from '$stores/undo.svelte';

export interface ReviewQueueControllerOptions {
  /** The page's queue pager — items/total are mutated in place. */
  queue: Pager<ReviewItem>;
  /** The page's handled-ids set (dedup guard for the pager's `accept`). */
  handledIds: Set<string>;
  getCursor: () => number;
  setCursor: (next: number) => void;
  /** Eagerly pull the next page when the cursor nears the loaded end. */
  maybePrefetch: () => void;
}

export function createReviewQueueController(opts: ReviewQueueControllerOptions) {
  const { queue, handledIds, getCursor, setCursor, maybePrefetch } = opts;

  // m2 (2026-09-24 interactive pass): undoLast() re-inserts a Crop (from
  // undoStore, which only knows label fields) into a ReviewItem[] queue.
  // ReviewItem carries a served `reason` Crop doesn't have, so this
  // caches the original item's reason at removal time and reuses it on
  // undo — instead of inventing a "restored by undo" string that was
  // never served by the backend.
  // eslint-disable-next-line svelte/prefer-svelte-reactivity -- internal bookkeeping map, never read reactively by a template/derived
  const removedItemReasons = new Map<string, string | null>();

  /**
   * Optimistically drop an item from the queue and advance.
   *
   * Returns the undo closure that puts it back at the same index with the
   * same cursor. EVERY caller must invoke it when the API call fails —
   * otherwise the item vanishes from the operator's queue while the server
   * still holds it unchanged, and it is never seen again this session.
   */
  function removeFromQueue(item: ReviewItem): () => void {
    removedItemReasons.set(item.id, item.reason ?? null);
    const found = queue.items.findIndex((x) => x.id === item.id);
    const removedIdx = found >= 0 ? found : getCursor();
    const priorCursor = getCursor();
    queue.items = queue.items.filter((x) => x.id !== item.id);
    queue.total = Math.max(0, queue.total - 1);
    setCursor(Math.min(getCursor(), Math.max(0, queue.items.length - 1)));
    handledIds.add(item.id);
    maybePrefetch();
    return () => {
      handledIds.delete(item.id);
      const at = Math.min(removedIdx, queue.items.length);
      queue.items = [...queue.items.slice(0, at), item, ...queue.items.slice(at)];
      queue.total += 1;
      setCursor(priorCursor);
    };
  }

  async function assign(item: ReviewItem, classId: number): Promise<void> {
    const cls = classesStore.byId(classId);
    // Optimistic: drop from list and advance.
    const restore = removeFromQueue(item);
    try {
      await putCropLabel(item.id, classId);
      undoStore.recordWrites([item.id]);
      toastStore.success(`Labeled "${cls?.name ?? classId}".`);
    } catch (e) {
      restore();
      toastStore.error(`Label failed: ${(e as Error).message}`);
    }
  }

  function skip(): void {
    setCursor(Math.min(queue.items.length - 1, getCursor() + 1));
    maybePrefetch();
  }

  async function discard(item: ReviewItem): Promise<void> {
    // Discard = "permanently dismiss this crop from every review queue."
    // Deliberately does NOT push an undoStore entry — Z undoes a *label*
    // write, a different action from a dismiss (see the page's "Dismissed"
    // panel / reviewUndismissCrop for reversing a dismiss instead).
    const restore = removeFromQueue(item);
    try {
      await reviewDismissCrop(item.id);
      toastStore.success('Dismissed from review (permanent).');
    } catch (e) {
      restore();
      toastStore.error(`Discard failed: ${(e as Error).message}`);
    }
  }

  async function undoLast(): Promise<void> {
    const crops = await undoStore.undoLast();
    if (crops.length === 0) return;
    // The item(s) were removed from the queue by assign/discard, so
    // re-insert each restored item at the cursor so the operator can see
    // (and re-verify) what the undo brought back.
    for (const crop of crops) {
      handledIds.delete(crop.id);
      const restored: ReviewItem = {
        ...crop,
        reason: removedItemReasons.get(crop.id) ?? null,
      } as ReviewItem;
      removedItemReasons.delete(crop.id);
      const without = queue.items.filter((it) => it.id !== crop.id);
      const at = Math.min(getCursor(), without.length);
      queue.total += without.length === queue.items.length ? 1 : 0;
      queue.items = [...without.slice(0, at), restored, ...without.slice(at)];
      setCursor(at);
    }
  }

  return { removeFromQueue, assign, skip, discard, undoLast };
}

export type ReviewQueueController = ReturnType<typeof createReviewQueueController>;
