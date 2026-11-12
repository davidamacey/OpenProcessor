<script lang="ts">
  /**
   * Review: the two panes of the old Review page over the Workspace's own query and order. The
   * items come from the served queue (or the Detections order when no queue is set), the item to
   * land on is `focus`, and the filter is the Workspace filter; this view is wiring only, each
   * concern lives in a controller under `$lib/review` and renders through a component under
   * `$lib/components/review`. Enter on a grid card opens it here; Esc returns to the same card.
   */
  import ClassPicker from '$components/ClassPicker.svelte';
  import DismissedPanel from '$lib/components/review/DismissedPanel.svelte';
  import Filmstrip from '$lib/components/review/Filmstrip.svelte';
  import QueueFilters from '$lib/components/review/QueueFilters.svelte';
  import RejectReasonPrompt from '$lib/components/review/RejectReasonPrompt.svelte';
  import ReviewCropPane from '$lib/components/review/ReviewCropPane.svelte';
  import ReviewEmptyQueue from '$lib/components/review/ReviewEmptyQueue.svelte';
  import ReviewHeader from '$lib/components/review/ReviewHeader.svelte';
  import { createReviewDismissedController } from '$lib/review/reviewDismissedController.svelte';
  import { createReviewItemActions } from '$lib/review/reviewItemActions.svelte';
  import { installReviewKeyboard } from '$lib/review/reviewKeyboard.svelte';
  import { createReviewLiveUpdates } from '$lib/review/reviewLiveUpdates.svelte';
  import { createReviewQueueLoader } from '$lib/review/reviewQueueLoader.svelte';
  import { createReviewSlotActions } from '$lib/review/reviewSlotActions.svelte';
  import { createReviewSlotEditController } from '$lib/review/reviewSlotEditController.svelte';
  import { createReviewTabsController } from '$lib/review/reviewTabsController.svelte';
  import { isSlotTab } from '$lib/reviewTabs';
  import type { WorkspaceController } from '$lib/workspace/workspaceController.svelte';
  import { classesStore } from '$stores/classes.svelte';

  let { ws }: { ws: WorkspaceController } = $props();

  // `ws` is fixed for the life of the view, so reading it once is right.
  // svelte-ignore state_referenced_locally
  const tabs = createReviewTabsController(ws);
  // svelte-ignore state_referenced_locally
  const loader = createReviewQueueLoader({ ws, tabs });
  const live = createReviewLiveUpdates(() => void loader.loadFirst());
  const actions = createReviewItemActions(loader);
  const dismissed = createReviewDismissedController();
  const edit = createReviewSlotEditController({ tabs, loader });
  const slot = createReviewSlotActions({ tabs, loader, actions, edit });

  // The shared class picker, opened by the class-picker key or the "search all classes" button.
  let pickerOpen = $state(false);
  const picker = {
    openPicker(): void {
      if (isSlotTab(tabs.tab) || !loader.current) return;
      pickerOpen = true;
    },
  };
  const recent = $derived(
    actions.currentHintIds
      .map((id) => classesStore.byId(id)?.name)
      .filter((n): n is string => !!n),
  );
  installReviewKeyboard({ tabs, loader, actions, picker, edit, slot });

  const queue = $derived(loader.queue);
  const current = $derived(loader.current);

  // The focus follows the item Review stands on, so Esc (and a reload) land on it.
  $effect(() => {
    if (tabs.awaitingDeepLink || !current) return;
    if (ws.url.focus?.kind === 'item' && ws.url.focus.id === current.id) return;
    ws.setReviewFocus(current.id);
  });

  // The served page around the focus: the loaded items near the cursor.
  const strip = $derived(
    queue.items
      .slice(Math.max(0, loader.cursor - 6), loader.cursor + 10)
      .map((c) => ({ id: c.id, displayName: c.display_name })),
  );

  // "Details" disclosure: collapsed by default; CropMetaPanel only mounts (and fetches
  // history/image) once the operator opens it.
  let detailsOpen = $state<boolean>(false);
</script>

<div class="flex h-full flex-col" data-testid="review-view">
  <ReviewHeader
    {tabs}
    {loader}
    liveNewCount={live.liveNewCount}
    refreshFromLive={live.refreshFromLive}
    toggleDismissedPanel={dismissed.toggleDismissedPanel}
    isMultiBoxSlot={edit.isMultiBoxSlot}
    editMode={edit.editMode}
  />
  <QueueFilters {ws} {tabs} />

  <!-- Below lg the two panels stack; the body scrolls as a whole there instead of squeezing each
       panel into half the height. -->
  <div
    class="grid min-h-0 flex-1 grid-cols-1 content-start gap-4 overflow-y-auto p-4 lg:grid-cols-2 lg:content-normal lg:overflow-hidden"
    data-testid="review-body"
  >
    {#if queue.loading && queue.items.length === 0}
      <p class="col-span-full text-sm text-zinc-500">Loading...</p>
    {:else if tabs.awaitingDeepLink}
      <!-- Page 1 (item #1) may already be loaded underneath this: don't render it, or the
           operator briefly sees and could act on the wrong item while the target is located. -->
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

  {#if current}
    <Filmstrip
      items={strip}
      focusedId={current.id}
      index={loader.currentPosition - 1}
      total={queue.total}
      onSelect={(id) => {
        const at = queue.items.findIndex((c) => c.id === id);
        if (at >= 0) loader.cursor = at;
      }}
    />
  {/if}
</div>

<DismissedPanel {dismissed} />

<RejectReasonPrompt
  activeSlot={tabs.activeSlot}
  rejectReasonPromptOpen={slot.rejectReasonPromptOpen}
  bind:rejectReasonPromptValue={slot.rejectReasonPromptValue}
  submitRejectReasonPrompt={slot.submitRejectReasonPrompt}
  cancelRejectReasonPrompt={slot.cancelRejectReasonPrompt}
/>

<ClassPicker
  open={pickerOpen}
  mode="assign"
  {recent}
  onpick={(cls) => {
    pickerOpen = false;
    void actions.assign(cls.id);
  }}
  onclose={() => (pickerOpen = false)}
/>
