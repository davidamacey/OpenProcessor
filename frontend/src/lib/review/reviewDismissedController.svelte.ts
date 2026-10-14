/**
 * `/review` dismissed-crops panel (un-dismiss): lists the crops `discard()`
 * dismissed from every review queue, each with a Restore action.
 *
 * Moved verbatim out of `review/+page.svelte`.
 */

import { apiErrorText, getCrops, reviewUndismissCrop } from '$lib/api';
import type { Crop } from '$lib/types';
import { toastStore } from '$stores/toast.svelte';

export function createReviewDismissedController() {
  // -- dismissed-crops panel (un-dismiss) --------------------------------
  // Minimal reachable-from-/review surface for reviewUndismissCrop: a
  // toggleable panel listing crops discard() has dismissed, each with a
  // Restore action. Loaded lazily (only when opened), not kept in sync
  // with the live queue.
  let dismissedPanelOpen = $state<boolean>(false);
  let dismissedItems = $state<Crop[]>([]);
  let dismissedLoading = $state<boolean>(false);
  let dismissedError = $state<string | null>(null);

  async function toggleDismissedPanel(): Promise<void> {
    dismissedPanelOpen = !dismissedPanelOpen;
    if (!dismissedPanelOpen) return;
    dismissedLoading = true;
    dismissedError = null;
    try {
      // M1 (2026-09-24 interactive pass): GET {API_PREFIX}/crops's `sort`
      // is a '<field>[:asc|desc]' pair against a closed field list
      // (contracts/openapi/curation.json) — 'recent' isn't
      // one of them and 400s every time. 'updated_at:desc' is the
      // server's own documented default and matches "most recently
      // dismissed first".
      const res = await getCrops({
        review_dismissed: true,
        limit: 60,
        sort: 'updated_at:desc',
      });
      dismissedItems = res.items;
    } catch (e) {
      // Show the failure inline instead of falling through to "No
      // dismissed crops" — that empty state used to render even when the
      // request itself failed, making a real 400 look like there was
      // simply nothing to restore.
      dismissedError = apiErrorText(e);
      dismissedItems = [];
    } finally {
      dismissedLoading = false;
    }
  }

  async function undismiss(crop: Crop): Promise<void> {
    try {
      await reviewUndismissCrop(crop.id);
      dismissedItems = dismissedItems.filter((c) => c.id !== crop.id);
      toastStore.success('Restored to review.');
    } catch (e) {
      toastStore.error(`Restore failed: ${apiErrorText(e)}`);
    }
  }

  return {
    get dismissedPanelOpen() {
      return dismissedPanelOpen;
    },
    set dismissedPanelOpen(v: boolean) {
      dismissedPanelOpen = v;
    },
    get dismissedItems() {
      return dismissedItems;
    },
    get dismissedLoading() {
      return dismissedLoading;
    },
    get dismissedError() {
      return dismissedError;
    },
    toggleDismissedPanel,
    undismiss,
  };
}

export type ReviewDismissedController = ReturnType<
  typeof createReviewDismissedController
>;
