<!--
  The /review dismissed-crops dialog: crops discard() dismissed from every
  review queue, each restorable.
-->
<script lang="ts">
  import Modal from '$components/Modal.svelte';
  import type { ReviewDismissedController } from '$lib/review/reviewDismissedController.svelte';

  let { dismissed }: { dismissed: ReviewDismissedController } = $props();

  const dismissedPanelOpen = $derived(dismissed.dismissedPanelOpen);
  const dismissedItems = $derived(dismissed.dismissedItems);
  const dismissedLoading = $derived(dismissed.dismissedLoading);
  const dismissedError = $derived(dismissed.dismissedError);
  const undismiss = $derived(dismissed.undismiss);
</script>

{#if dismissedPanelOpen}
  <!-- Minimal un-dismiss surface: crops discard() has permanently
       dismissed from every review queue, each restorable via
       reviewUndismissCrop. Backdrop click and Esc both close. -->
  <Modal
    label="Dismissed crops"
    onclose={() => (dismissed.dismissedPanelOpen = false)}
    layer="top"
    align="top"
    scrim="bg-black/50"
    panelClass="w-full max-w-lg overflow-hidden rounded-lg border border-zinc-700 bg-zinc-900 shadow-xl"
  >
    <div class="flex items-center justify-between border-b border-zinc-800 px-3 py-2">
      <span class="text-sm font-medium text-zinc-200">Dismissed crops</span>
      <button
        type="button"
        class="text-xs text-zinc-500 hover:text-zinc-300"
        onclick={() => (dismissed.dismissedPanelOpen = false)}
      >
        close
      </button>
    </div>
    <ul class="max-h-96 overflow-y-auto py-1 text-sm">
      {#if dismissedLoading}
        <li class="px-3 py-2 text-zinc-500">Loading…</li>
      {:else if dismissedError}
        <li class="px-3 py-2 text-red-300">
          Failed to load dismissed crops: {dismissedError}
        </li>
      {:else if dismissedItems.length === 0}
        <li class="px-3 py-2 text-zinc-500">No dismissed crops.</li>
      {/if}
      {#each dismissedItems as crop (crop.id)}
        <li class="flex items-center justify-between gap-2 px-3 py-1.5">
          <span class="truncate text-zinc-300">{crop.class_name ?? crop.id}</span>
          <button
            type="button"
            class="shrink-0 rounded border border-zinc-700 px-2 py-0.5 text-xs text-zinc-300 hover:border-blue-500 hover:text-blue-300"
            onclick={() => undismiss(crop)}
          >
            Restore
          </button>
        </li>
      {/each}
    </ul>
  </Modal>
{/if}
