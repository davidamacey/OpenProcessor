<!--
  The /review tab strip with the queue counter, the shortcuts button, the
  Dismissed panel toggle and the live "N new" pill.
-->
<script lang="ts">
  import ScrollStrip from '$lib/components/ScrollStrip.svelte';
  import HiddenItemsNote from '$lib/components/itemFilter/HiddenItemsNote.svelte';
  import ShortcutsButton from '$lib/components/ShortcutsButton.svelte';
  import type { ReviewQueueLoader } from '$lib/review/reviewQueueLoader.svelte';
  import type { ReviewTabsController } from '$lib/review/reviewTabsController.svelte';
  import type { ReviewTabDef } from '$lib/reviewTabs';
  import { regionProfileStore } from '$stores/regionProfile.svelte';
  import { reviewTabsVocabularyStore } from '$stores/reviewTabsVocabulary.svelte';

  interface Props {
    tabs: ReviewTabsController;
    loader: ReviewQueueLoader;
    liveNewCount: number;
    refreshFromLive: () => void;
    toggleDismissedPanel: () => void;
  }

  let { tabs, loader, liveNewCount, refreshFromLive, toggleDismissedPanel }: Props =
    $props();

  const tab = $derived(tabs.tab);
  const visibleTabs = $derived(tabs.visibleTabs);
  const queue = $derived(loader.queue);
  const currentPosition = $derived(loader.currentPosition);
  const selectTab = (t: ReviewTabDef) => tabs.selectTab(t);
</script>

<!-- Tabs — horizontally scrollable on narrow viewports so all tabs stay reachable
     without colliding with the loaded-count chip on the right. -->
<div
  class="flex flex-wrap items-center gap-1 border-b border-zinc-800 px-4 md:flex-nowrap"
>
  <!-- R2 (visual audit 2026-09-24): at 800px the plain overflow row
       clipped "New class proposals"/"Regions" with no hint, and an
       active tab past the edge was invisible — ScrollStrip shows a
       chevron where tabs are hidden and scrolls the active one in. -->
  <ScrollStrip
    class="gap-1"
    navClass="basis-full md:basis-auto"
    activeKey={tab}
    testId="review-tabs"
  >
    {#each visibleTabs as t (t.id)}
      {@const count = reviewTabsVocabularyStore.countFor(t.endpointId)}
      {@const unavailable = reviewTabsVocabularyStore.unavailableReasonFor(t.endpointId)}
      {@const dim = unavailable !== null || count === 0}
      <!-- The count sits beside the button, not in it, so the button's name stays the tab's label. -->
      <span
        class="flex shrink-0 items-baseline border-b-2 {tab === t.id
          ? 'border-blue-500 text-white'
          : dim
            ? 'border-transparent text-zinc-600 hover:text-zinc-300'
            : 'border-transparent text-zinc-400 hover:text-zinc-200'}"
      >
        <button
          type="button"
          data-active={tab === t.id ? 'true' : undefined}
          class="py-2.5 pr-1 pl-3 text-sm"
          title={[reviewTabsVocabularyStore.descriptionFor(t.endpointId), unavailable]
            .filter(Boolean)
            .join(' — ') || undefined}
          onclick={() => selectTab(t)}
        >
          {reviewTabsVocabularyStore.labelFor(t.endpointId, t.label)}
        </button>
        {#if unavailable !== null}
          <span class="pr-3 text-[10px] text-zinc-500" data-testid="tab-unavailable"
            >needs setup</span
          >
        {:else if count !== null}
          <span class="pr-3 font-mono text-[10px] text-zinc-500" data-testid="tab-count"
            >{count.toLocaleString()}</span
          >
        {/if}
      </span>
    {/each}
  </ScrollStrip>
  {#if regionProfileStore.unknown}
    <!-- Boot couldn't read the region profile yet; the region tab
         appears once a /health poll answers (the layout re-mounts). -->
    <span
      class="shrink-0 pl-2 text-[11px] text-zinc-500"
      data-testid="region-profile-loading">loading region profile…</span
    >
  {/if}
  <span
    data-testid="queue-counter"
    class="shrink-0 py-1.5 pl-0 font-mono text-xs text-zinc-500 md:py-0 md:pl-2"
  >
    {queue.items.length > 0 ? `#${currentPosition}` : '—'} · {queue.items.length} loaded ·
    {queue.total}
    total
  </span>
  <HiddenItemsNote hidden={loader.hidden} />
  <span class="shrink-0 pl-2"><ShortcutsButton /></span>
  <button
    type="button"
    class="ml-2 shrink-0 rounded border border-zinc-700 px-2 py-1 text-[11px] text-zinc-400 hover:border-zinc-500 hover:text-zinc-200"
    onclick={toggleDismissedPanel}
    title="Crops permanently dismissed from review — restore one back into the queue"
  >
    Dismissed
  </button>
  {#if liveNewCount > 0}
    <button
      type="button"
      class="ml-2 shrink-0 animate-pulse rounded-full border border-blue-500/60 bg-blue-500/15 px-2.5 py-1 text-[11px] text-blue-200 hover:bg-blue-500/25"
      onclick={refreshFromLive}
      title="Reload the queue with the latest crops"
    >
      {liveNewCount} new · refresh
    </button>
  {/if}
</div>
