<script lang="ts">
  import DismissedPanel from '$lib/components/review/DismissedPanel.svelte';
  import ClassPickerModal from '$lib/components/review/ClassPickerModal.svelte';
  import RejectReasonPrompt from '$lib/components/review/RejectReasonPrompt.svelte';
  import ReviewCropPane from '$lib/components/review/ReviewCropPane.svelte';
  import ReviewEmptyQueue from '$lib/components/review/ReviewEmptyQueue.svelte';
  import ReviewFilterBar from '$lib/components/review/ReviewFilterBar.svelte';
  import ReviewStatusBar from '$lib/components/review/ReviewStatusBar.svelte';
  import ReviewStrategyRow from '$lib/components/review/ReviewStrategyRow.svelte';
  import ReviewTabsBar from '$lib/components/review/ReviewTabsBar.svelte';
  import { createReviewClassPicker } from '$lib/review/reviewClassPicker.svelte';
  import { createReviewDismissedController } from '$lib/review/reviewDismissedController.svelte';
  import { createReviewDiverseController } from '$lib/review/reviewDiverseController.svelte';
  import { createReviewFilterController } from '$lib/review/reviewFilterController.svelte';
  import { createReviewItemActions } from '$lib/review/reviewItemActions.svelte';
  import { installReviewKeyboard } from '$lib/review/reviewKeyboard.svelte';
  import { createReviewLiveUpdates } from '$lib/review/reviewLiveUpdates.svelte';
  import { createReviewQueueLoader } from '$lib/review/reviewQueueLoader.svelte';
  import { createReviewSlotActions } from '$lib/review/reviewSlotActions.svelte';
  import { createReviewSlotEditController } from '$lib/review/reviewSlotEditController.svelte';
  import { createReviewTabsController } from '$lib/review/reviewTabsController.svelte';

  // The review page is wiring only: each concern lives in a controller under
  // $lib/review and renders through a component under
  // $lib/components/review. The filter reset and the picker close are read
  // lazily because their controllers are built after the tabs one; every
  // other dependency is passed once it exists.
  const tabs = createReviewTabsController({
    resetTabFilters: (url) => filters.resetForTab(url),
    closePicker: () => picker.closePicker(),
  });
  const diverse = createReviewDiverseController({
    tabs,
    scope: () => ({
      sourceFilter: filters.sourceFilter,
      classNames: filters.itemFilter.classNames,
    }),
  });
  const filters = createReviewFilterController({ tabs, diverse });
  // `queue`, `cursor` and `handledIds` live once, in the loader, and every
  // controller below reads and writes them through it by reference.
  const loader = createReviewQueueLoader({ tabs, diverse, filters });
  const live = createReviewLiveUpdates(() => void loader.loadFirst());
  const actions = createReviewItemActions(loader);
  const picker = createReviewClassPicker({ tabs, loader, actions });
  const dismissed = createReviewDismissedController();
  const edit = createReviewSlotEditController({ tabs, loader });
  const slot = createReviewSlotActions({ tabs, loader, actions, edit });
  installReviewKeyboard({ tabs, loader, actions, picker, edit, slot });

  const queue = $derived(loader.queue);
  const current = $derived(loader.current);
  const awaitingDeepLink = $derived(tabs.awaitingDeepLink);

  // -- item-detail "Details" disclosure (G7/G9/G8) -----------------------
  // Collapsed by default; CropMetaPanel only mounts (and fetches
  // history/image) once the operator opens it.
  let detailsOpen = $state<boolean>(false);
</script>

<div class="flex h-full flex-col">
  <h1 class="sr-only">Review</h1>
  <ReviewTabsBar
    {tabs}
    {loader}
    liveNewCount={live.liveNewCount}
    refreshFromLive={live.refreshFromLive}
    toggleDismissedPanel={dismissed.toggleDismissedPanel}
  />

  <ReviewStrategyRow {tabs} {diverse} {filters} {loader} />

  <ReviewFilterBar
    {tabs}
    {diverse}
    {filters}
    isMultiBoxSlot={edit.isMultiBoxSlot}
    editMode={edit.editMode}
  />

  {#if tabs.unavailableTabNotice}
    <div
      class="mx-4 mt-3 flex items-center gap-3 rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-200"
      data-testid="tab-unavailable"
      role="status"
    >
      <span class="grow">{tabs.unavailableTabNotice}</span>
      <button
        type="button"
        class="btn-sm"
        onclick={() => (tabs.unavailableTabNotice = null)}>Dismiss</button
      >
    </div>
  {/if}

  <!-- Body -->
  <!-- F8 D4: below lg the two panels stack; the body scrolls as a whole
       there instead of squeezing each panel into half the height (the
       metadata pane was ~79px tall at 800px). -->
  <div
    class="grid min-h-0 flex-1 grid-cols-1 content-start gap-4 overflow-y-auto p-4 lg:grid-cols-2 lg:content-normal lg:overflow-hidden"
    data-testid="review-body"
  >
    {#if queue.loading && queue.items.length === 0}
      <p class="col-span-full text-sm text-zinc-500">Loading...</p>
    {:else if awaitingDeepLink}
      <!-- DQ-M7: page 1 (item #1) may already be loaded underneath this —
           don't render it, or the operator briefly sees and could act on
           the wrong crop while the target page is still being located. -->
      <p class="col-span-full text-sm text-zinc-500">Locating crop…</p>
    {:else if queue.error}
      <p class="col-span-full text-sm text-red-300">API unavailable: {queue.error}</p>
    {:else if !current}
      <ReviewEmptyQueue
        emptyMessage={loader.emptyMessage}
        emptyReason={loader.emptyReason}
        sortFallbackReason={loader.sortFallbackReason}
      />
    {:else}
      <ReviewCropPane
        {current}
        {tabs}
        {loader}
        {actions}
        {picker}
        {edit}
        {slot}
        bind:detailsOpen
      />
    {/if}
  </div>

  <ReviewStatusBar {queue} currentPosition={loader.currentPosition} />
</div>

<DismissedPanel {dismissed} />

<RejectReasonPrompt
  activeSlot={tabs.activeSlot}
  rejectReasonPromptOpen={slot.rejectReasonPromptOpen}
  bind:rejectReasonPromptValue={slot.rejectReasonPromptValue}
  submitRejectReasonPrompt={slot.submitRejectReasonPrompt}
  cancelRejectReasonPrompt={slot.cancelRejectReasonPrompt}
/>

<ClassPickerModal {picker} />
