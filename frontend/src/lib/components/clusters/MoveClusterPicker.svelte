<script lang="ts">
  import { resolve } from '$app/paths';
  import Modal from '$components/Modal.svelte';
  import { projectHref } from '$lib/projectPaths';
  import type { ClusterDetailController } from '$lib/clusters/clusterDetailController.svelte';
  import type { ClusterDetailActions } from '$lib/clusters/clusterDetailActions.svelte';

  // The M-key dialog: move the selection to another cluster by id.
  let {
    view,
    actions,
  }: { view: ClusterDetailController; actions: ClusterDetailActions } = $props();
</script>

{#if actions.movePickerOpen}
  <Modal
    label="Move crops to cluster"
    onclose={actions.cancelMovePicker}
    panelClass="w-full max-w-sm rounded-lg border border-zinc-800 bg-zinc-950 p-5 shadow-2xl"
  >
    <h3 class="mb-2 text-base font-semibold">
      Move {view.sel.size} crop{view.sel.size === 1 ? '' : 's'}
    </h3>
    <p class="mb-3 text-xs text-zinc-400">
      Move these from cluster #{view.clusterId} to another cluster, by its id (the #N on each
      <a class="text-blue-400 underline" href={resolve(projectHref('/clusters'))}
        >/clusters</a
      >
      card). Reversible with {actions.kg('cluster.undo')}.
    </p>
    <p class="mb-3 text-xs text-zinc-400" data-testid="move-relabel-hint">
      To change their <strong>class</strong> instead, close this and pick a class, then
      "Assign class to selected" ({actions.kg('cluster.confirm')}), or press the class's
      hotkey.
    </p>
    <label class="mb-3 block text-sm">
      <span class="mb-1 block text-zinc-400">Target cluster id</span>
      <input
        type="number"
        min="0"
        step="1"
        bind:this={actions.movePickerInput}
        bind:value={actions.movePickerValue}
        onkeydown={(e) => {
          if (e.key === 'Enter') {
            e.preventDefault();
            void actions.confirmMovePicker();
          } else if (e.key === 'Escape') {
            e.preventDefault();
            actions.cancelMovePicker();
          }
        }}
        class="w-full rounded-md border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-sm focus:border-blue-500 focus:outline-none"
        placeholder="e.g. 42"
      />
    </label>
    <div class="flex justify-end gap-2">
      <button type="button" class="btn" onclick={actions.cancelMovePicker}>Cancel</button>
      <button
        type="button"
        class="btn btn-primary"
        onclick={() => void actions.confirmMovePicker()}
      >
        Move
      </button>
    </div>
  </Modal>
{/if}
