<script lang="ts">
  /**
   * "Diverse sample…": picks k items that cover the Workspace filter as widely as possible and
   * makes them the selection. The server answers at once for a small pool and runs a job for a
   * large one; this shows which, with a Cancel while it runs.
   */
  import { onDestroy } from 'svelte';
  import { createDiverseSample } from '$lib/workspace/diverseSample.svelte';
  import type { WorkspaceController } from '$lib/workspace/workspaceController.svelte';

  let { ws }: { ws: WorkspaceController } = $props();

  // `ws` is fixed for the life of the bar.
  const sample = createDiverseSample({
    itemBody: () => ws.itemBody(),
    selectItems: (ids) => ws.selectItems(ids),
  });
  onDestroy(() => sample.dispose());
  let open = $state(false);
</script>

<div class="inline-flex sm:relative" data-testid="diverse-sample">
  <button
    type="button"
    class="btn"
    aria-expanded={open}
    data-testid="diverse-sample-open"
    onclick={() => (open = !open)}
  >
    Diverse sample…
  </button>
  {#if open}
    <form
      class="absolute top-full right-4 left-4 z-20 mt-1 flex flex-col gap-2 rounded border border-line-strong bg-surface-1 p-3 shadow-lg sm:right-0 sm:left-auto sm:w-64"
      onsubmit={(e) => {
        e.preventDefault();
        void sample.start();
      }}
    >
      <label class="flex items-center justify-between gap-2 text-fg-1">
        How many
        <input
          type="number"
          min="1"
          class="w-24 rounded border border-line-strong bg-surface-0 px-2 py-1 text-right"
          data-testid="diverse-sample-k"
          bind:value={sample.k}
        />
      </label>
      <p class="text-fg-2">
        Picks the items that differ most from each other among those the filter matches.
      </p>
      <div class="flex gap-2">
        <button
          type="submit"
          class="btn btn-primary"
          disabled={sample.phase === 'running' || !(sample.k >= 1)}
          data-testid="diverse-sample-start"
        >
          {sample.phase === 'running' ? 'Picking…' : 'Pick'}
        </button>
        {#if sample.phase === 'running'}
          <button
            type="button"
            class="btn"
            data-testid="diverse-sample-cancel"
            onclick={() => void sample.cancel()}
          >
            Cancel
          </button>
        {/if}
      </div>
      {#if sample.message}
        <p class="text-fg-1" role="status" data-testid="diverse-sample-message">
          {sample.message}
        </p>
      {/if}
    </form>
  {/if}
</div>
