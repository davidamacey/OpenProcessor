/**
 * Live refresh for /clusters/[id]: the class-scoped event stream counts new
 * crops, refreshes in place while the operator is at the top of the grid,
 * and otherwise offers a "new crops" pill.
 */
import { subscribeCurationEvents, type CurationEventSubscription } from '$lib/sse';
import type { ClusterDetailController } from '$lib/clusters/clusterDetailController.svelte';
import { toastStore } from '$stores/toast.svelte';

export function createClusterDetailLive(view: ClusterDetailController) {
  const { cropPager, loadFirst } = view;

  // ---------------- SSE: live updates for this cluster's class ----
  // cluster_id == class_id holds for class-kind clusters (see
  // `ClusterKind` in src/lib/types.ts and the scoping note on getCluster
  // in src/lib/api.ts), so we subscribe with `class_id=clusterId`.
  // Crop.created without a class is hidden from per-class pages by
  // event_hub's filter. Candidate clusters have no class, and event_hub
  // filters on exact class_id equality, so a subscription there could
  // never deliver anything — skip it rather than hold an idle stream open.
  let liveNewCount = $state<number>(0);
  let scrolledPastFirst20 = $state<boolean>(false);
  let liveSub: CurationEventSubscription | null = null;
  let scrollEl = $state<HTMLDivElement | null>(null);
  const liveClassId = $derived(view.clsForCluster?.id ?? null);

  $effect(() => {
    if (liveClassId == null) return;
    liveSub = subscribeCurationEvents({
      class_id: liveClassId,
      onEvent: (ev) => {
        if (ev.type === 'crop.classified' || ev.type === 'crop.created') {
          liveNewCount += 1;
        }
      },
    });
    return () => {
      liveSub?.close();
      liveSub = null;
    };
  });

  // Scroll-aware: only show the pill once the user has scrolled past
  // the first ~20 cards (otherwise just refresh in-place silently).
  function onScroll(): void {
    if (!scrollEl) return;
    scrolledPastFirst20 = scrollEl.scrollTop > 480; // ~3 rows at 8-col grid
  }

  $effect(() => {
    // Auto-refresh while user is at the top — they're not actively
    // labeling far down the list, so prepending new cards is safe.
    if (
      liveNewCount > 0 &&
      !scrolledPastFirst20 &&
      !cropPager.loading &&
      !cropPager.loadingMore
    ) {
      const n = liveNewCount;
      liveNewCount = 0;
      void loadFirst().then(() => {
        // Surface a small toast so the user knows something refreshed.
        toastStore.info(`${n} new crop${n === 1 ? '' : 's'} loaded.`);
      });
    }
  });

  function refreshFromLive(): void {
    liveNewCount = 0;
    if (scrollEl) scrollEl.scrollTop = 0;
    void loadFirst();
  }

  return {
    get liveNewCount() {
      return liveNewCount;
    },
    get scrolledPastFirst20() {
      return scrolledPastFirst20;
    },
    get scrollEl() {
      return scrollEl;
    },
    set scrollEl(v: HTMLDivElement | null) {
      scrollEl = v;
    },
    onScroll,
    refreshFromLive,
  };
}

export type ClusterDetailLive = ReturnType<typeof createClusterDetailLive>;
