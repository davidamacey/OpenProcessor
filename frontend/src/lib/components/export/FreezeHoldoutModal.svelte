<script lang="ts">
  /** /export's confirm modal for freezing the test holdout. */
  import Modal from '$components/Modal.svelte';
  import NearDupStalePanel from '$components/export/NearDupStalePanel.svelte';
  import type { FreezeHoldoutController } from '$lib/export/freezeHoldoutController.svelte';

  interface Props {
    freeze: FreezeHoldoutController;
  }

  let { freeze }: Props = $props();
</script>

<!-- Freeze test holdout modal -->
{#if freeze.freezeOpen}
  <Modal label="Freeze test holdout" onclose={freeze.closeFreeze}>
    <h3 class="mb-2 text-title font-semibold">
      {freeze.freezeForce ? 'Refreeze test holdout' : 'Freeze test holdout'}
    </h3>
    <div
      class="mb-3 rounded border border-warn/40 bg-warn/10 px-3 py-2 text-meta text-warn-text"
      data-testid="freeze-warning"
    >
      {#if freeze.freezeForce}
        A test set already exists. Refreezing adds a per-class sample of the crops
        validated since the last freeze; crops already frozen stay in the test set. It is
        recorded as a new freeze, and runs trained before it were scored on the smaller
        set.
      {:else}
        Freezes a per-class sample of the validated crops as the test set, never trained
        on. Selection is deterministic (SHA1 of each crop id), so there's no seed to pick.
        It can be refreshed later with Refreeze.
      {/if}
    </div>
    <NearDupStalePanel refresh={freeze.nearDup} />
    <label class="mb-3 block text-body">
      <span class="mb-1 block text-fg-2">Percent of validated crops</span>
      <input
        type="number"
        min="1"
        max="50"
        bind:value={freeze.freezePercent}
        class="input w-full"
      />
    </label>
    <div class="flex justify-end gap-2">
      <button
        type="button"
        class="btn"
        onclick={freeze.closeFreeze}
        disabled={freeze.freezeBusy}
      >
        Cancel
      </button>
      <button
        type="button"
        class="btn btn-primary"
        onclick={() => void freeze.submitFreeze()}
        disabled={freeze.freezeBusy}
      >
        {freeze.freezeBusy ? 'Freezing…' : freeze.freezeForce ? 'Refreeze' : 'Freeze'}
      </button>
    </div>
  </Modal>
{/if}
