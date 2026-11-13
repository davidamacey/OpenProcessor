<script lang="ts">
  /**
   * "Diverse sample…": picks k items that cover the Workspace filter as widely as possible and
   * makes them the selection. The server answers at once for a small pool and runs a job for a
   * large one; this shows which, with a Cancel while it runs.
   */
  import { onDestroy } from 'svelte';
  import Modal from '$components/Modal.svelte';
  import { createDiverseSample } from '$lib/workspace/diverseSample.svelte';
  import type { WorkspaceController } from '$lib/workspace/workspaceController.svelte';

  let { ws, open = $bindable(false) }: { ws: WorkspaceController; open?: boolean } =
    $props();

  // `ws` is fixed for the life of the bar.
  const sample = createDiverseSample({
    itemBody: () => ws.itemBody(),
    selectItems: (ids) => ws.selectItems(ids),
  });
  onDestroy(() => sample.dispose());
</script>

{#if open}
  <Modal
    label="Select a diverse sample"
    onclose={() => (open = false)}
    panelClass="w-full max-w-sm rounded-lg border border-line bg-surface-0 p-5 shadow-2xl"
  >
    <form
      class="flex flex-col gap-3"
      data-testid="diverse-sample"
      onsubmit={(e) => {
        e.preventDefault();
        void sample.start();
      }}
    >
      <h3 class="text-title font-semibold text-fg-1">Select a diverse sample</h3>
      <label class="flex items-center justify-between gap-2 text-fg-1">
        How many
        <input
          type="number"
          min="1"
          class="input w-24 text-right"
          data-testid="diverse-sample-k"
          bind:value={sample.k}
        />
      </label>
      <p class="text-body text-fg-2">
        Picks the items that differ most from each other among those the filter matches.
      </p>
      <div class="flex justify-end gap-2">
        {#if sample.phase === 'running'}
          <button
            type="button"
            class="btn"
            data-testid="diverse-sample-cancel"
            onclick={() => void sample.cancel()}
          >
            Cancel
          </button>
        {:else}
          <button type="button" class="btn" onclick={() => (open = false)}>Close</button>
        {/if}
        <button
          type="submit"
          class="btn btn-primary"
          disabled={sample.phase === 'running' || !(sample.k >= 1)}
          data-testid="diverse-sample-start"
        >
          {sample.phase === 'running' ? 'Picking…' : 'Pick'}
        </button>
      </div>
      {#if sample.message}
        <p class="text-fg-1" role="status" data-testid="diverse-sample-message">
          {sample.message}
        </p>
      {/if}
    </form>
  </Modal>
{/if}
