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
    <h3 class="mb-2 text-base font-semibold">Freeze test holdout</h3>
    <div
      class="mb-3 rounded border border-warn/40 bg-warn/10 px-3 py-2 text-xs text-warn"
    >
      One-shot per dataset version. Deterministic per-class selection (SHA1 of each crop
      id) — the same cohort always freezes the same set, so there's no seed to pick.
    </div>
    <NearDupStalePanel refresh={freeze.nearDup} />
    <label class="mb-3 block text-sm">
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
        {freeze.freezeBusy ? 'Freezing…' : 'Freeze'}
      </button>
    </div>
  </Modal>
{/if}
