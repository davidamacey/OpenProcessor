<!--
  The Review view's header: the queue it reads, where the focus sits, what the queue hides, the
  live "N new" pill, the Dismissed panel toggle and the key hints for the active kind of queue.
-->
<script lang="ts">
  import HiddenItemsNote from '$lib/components/itemFilter/HiddenItemsNote.svelte';
  import ShortcutsButton from '$lib/components/ShortcutsButton.svelte';
  import { kg } from '$lib/review/reviewKeys';
  import type { ReviewQueueLoader } from '$lib/review/reviewQueueLoader.svelte';
  import { rejectKeyGlyph } from '$lib/review/slotKeymap';
  import type { ReviewTabsController } from '$lib/review/reviewTabsController.svelte';
  import { regionProfileStore } from '$stores/regionProfile.svelte';

  interface Props {
    tabs: ReviewTabsController;
    loader: ReviewQueueLoader;
    liveNewCount: number;
    refreshFromLive: () => void;
    toggleDismissedPanel: () => void;
    isMultiBoxSlot: boolean;
    editMode: boolean;
  }

  let {
    tabs,
    loader,
    liveNewCount,
    refreshFromLive,
    toggleDismissedPanel,
    isMultiBoxSlot,
    editMode,
  }: Props = $props();

  const queue = $derived(loader.queue);
  const activeSlot = $derived(tabs.activeSlot);
</script>

<div
  class="flex min-w-0 flex-wrap items-center gap-x-3 gap-y-1 border-b border-zinc-800 px-4 py-2"
  data-testid="review-header"
>
  <h1 class="text-sm font-medium text-zinc-100" data-testid="review-queue-label">
    {tabs.activeQueueLabel}
  </h1>
  {#if regionProfileStore.unknown}
    <span class="text-[11px] text-zinc-500" data-testid="region-profile-loading"
      >loading region profile…</span
    >
  {/if}
  <span data-testid="queue-counter" class="font-mono text-xs text-zinc-500">
    {queue.items.length > 0 ? `#${loader.currentPosition}` : '—'} · {queue.items.length} loaded
    · {queue.total} total
  </span>
  <HiddenItemsNote hidden={loader.hidden} />
  <span class="grow"></span>
  <span class="hidden text-[11px] text-zinc-500 lg:inline">
    {#if isMultiBoxSlot && activeSlot}
      <kbd>{kg('review.region.confirm')}</kbd> confirm proposed ·
      <kbd>{kg('review.region.accept_box')}</kbd> accept box ·
      <kbd>{kg('review.region.reject_box')}</kbd> reject box ·
      <kbd>{kg('box_edit.next_box')}</kbd> next box
      {#if editMode}
        · <kbd>{kg('box_edit.delete_box')}</kbd> delete box · drag empty area to add
      {:else}
        · <kbd>{kg('review.region.edit_box')}</kbd> edit
      {/if}
      · <kbd>{kg('review.skip')}</kbd> skip
    {:else if activeSlot}
      <kbd>{kg('review.region.confirm')}</kbd> confirm ·
      <kbd>{rejectKeyGlyph(activeSlot)}</kbd> reject
      {#if activeSlot.capabilities.lifecycle?.falsePositiveState}
        · <kbd>{kg('review.region.false_positive')}</kbd> false-pos
      {/if}
      · <kbd>{kg('review.skip')}</kbd> skip · <kbd>{kg('review.region.back')}</kbd> back
    {:else}
      per-class letter assigns · <kbd>{kg('review.queue.class_picker')}</kbd> search all
      classes · <kbd>{kg('review.queue.confirm')}</kbd> confirm ·
      <kbd>{kg('review.skip')}</kbd> skip · <kbd>{kg('review.queue.discard')}</kbd>
      discard · <kbd>{kg('review.undo')}</kbd> undo
    {/if}
    · <kbd>{kg('workspace.escape')}</kbd> back to the grid
  </span>
  <ShortcutsButton />
  <button
    type="button"
    class="shrink-0 rounded border border-zinc-700 px-2 py-1 text-[11px] text-zinc-400 hover:border-zinc-500 hover:text-zinc-200"
    onclick={toggleDismissedPanel}
    title="Crops permanently dismissed from review — restore one back into the queue"
  >
    Dismissed
  </button>
  {#if liveNewCount > 0}
    <button
      type="button"
      class="shrink-0 animate-pulse rounded-full border border-blue-500/60 bg-blue-500/15 px-2.5 py-1 text-[11px] text-blue-200 hover:bg-blue-500/25"
      onclick={refreshFromLive}
      title="Reload the queue with the latest crops"
    >
      {liveNewCount} new · refresh
    </button>
  {/if}
</div>
