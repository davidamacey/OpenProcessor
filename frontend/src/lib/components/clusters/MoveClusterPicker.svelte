<script lang="ts">
  import Modal from '$components/Modal.svelte';
  import { displayNoText, dominantShareText } from '$lib/clusters/clusterCardText';
  import type { ClusterDetailController } from '$lib/clusters/clusterDetailController.svelte';
  import type { ClusterDetailActions } from '$lib/clusters/clusterDetailActions.svelte';

  // The M-key dialog: move the selection to another cluster of the serving
  // partition. The choices are the served cluster cards.
  let {
    view,
    actions,
  }: { view: ClusterDetailController; actions: ClusterDetailActions } = $props();
</script>

{#if actions.movePickerOpen}
  <Modal
    label="Move crops to cluster"
    onclose={actions.cancelMovePicker}
    panelClass="w-full max-w-sm rounded-lg border border-line bg-surface-0 p-5 shadow-2xl"
  >
    <h3 class="mb-2 text-title font-semibold">
      Move {view.sel.size} crop{view.sel.size === 1 ? '' : 's'}
    </h3>
    <p class="mb-3 text-meta text-fg-2">
      Move these from this cluster to another one. A move only places the crops in the
      cluster; reversible with {actions.kg('cluster.undo')}.
    </p>
    <p class="mb-3 text-meta text-fg-2" data-testid="move-relabel-hint">
      To change their <strong>class</strong> instead, close this and use "Label as…" in the
      bulk bar, or press the class's hotkey.
    </p>
    <label class="mb-3 block text-body">
      <span class="mb-1 block text-fg-2">Target cluster</span>
      <select
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
        class="w-full rounded-md border border-line-strong bg-surface-1 px-2 py-1.5 text-body focus:border-accent focus:outline-none"
        data-testid="move-target"
      >
        <option value="">Choose a cluster…</option>
        {#each actions.moveCandidates as c (c.uid)}
          <option value={c.uid}>
            {displayNoText(c) ? `${displayNoText(c)} · ` : ''}{c.dominant_class_name ??
              'unlabeled'} · {c.size.toLocaleString()}{dominantShareText(c)
              ? ` · ${dominantShareText(c)}`
              : ''}
          </option>
        {/each}
      </select>
    </label>
    {#if actions.moveCandidatesError}
      <p class="mb-3 text-meta text-danger-text" role="alert">
        Could not load the clusters: {actions.moveCandidatesError}
      </p>
    {/if}
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
