<script lang="ts">
  /**
   * The budget edit of a queued, running or paused loop. The server holds the
   * rule (a limit under what is spent plus reserved is a 409 it words); this
   * form only sends what the person changed and shows the refusal.
   */
  import type { AssistLoopController } from '$lib/assistLoop/assistLoopController.svelte';
  import Modal from '$components/Modal.svelte';
  import type { LoopBudgetLimit } from '$lib/types_assistLoop';

  interface Props {
    ctl: AssistLoopController;
    onclose: () => void;
  }

  let { ctl, onclose }: Props = $props();

  const vlm = $derived(ctl.loop?.budget?.vlm);
  const wall = $derived(ctl.loop?.budget?.max_wall_s ?? null);

  // `undefined` = the person has not touched the field; the shown value is then the
  // loop's current one and it is not sent.
  let editedCrops = $state<number | null | undefined>(undefined);
  let editedRequests = $state<number | null | undefined>(undefined);
  let editedWall = $state<number | null | undefined>(undefined);
  const crops = $derived(
    editedCrops === undefined ? (vlm?.max_crops ?? null) : editedCrops,
  );
  const requests = $derived(
    editedRequests === undefined ? (vlm?.max_requests ?? null) : editedRequests,
  );
  const wallS = $derived(editedWall === undefined ? wall : editedWall);

  async function save(): Promise<void> {
    const limits: LoopBudgetLimit[] = [];
    if (crops !== null && crops !== vlm?.max_crops) {
      limits.push({ pool: 'vlm', unit: 'crops', limit_value: crops });
    }
    if (requests !== null && requests !== vlm?.max_requests) {
      limits.push({ pool: 'vlm', unit: 'requests', limit_value: requests });
    }
    if (wallS !== wall) limits.push({ pool: 'vlm', unit: 'wall_s', limit_value: wallS });
    if (limits.length === 0 || (await ctl.editBudget(limits))) onclose();
  }
</script>

<Modal label="Edit the loop budget" {onclose} data-testid="loop-budget-dialog">
  <h3 class="text-sm font-medium text-fg-1">Edit the loop budget</h3>
  <p class="mt-1 text-xs text-fg-2">
    Hard caps; the loop never goes past one. Editing does not resume a paused loop.
  </p>
  <div class="mt-3 flex flex-col gap-3 text-xs">
    <label class="flex flex-col gap-1">
      <span class="text-fg-2">VLM crops</span>
      <input
        type="number"
        min="1"
        class="input"
        bind:value={() => crops, (v) => (editedCrops = v)}
        data-testid="loop-budget-crops"
      />
    </label>
    <label class="flex flex-col gap-1">
      <span class="text-fg-2">VLM requests</span>
      <input
        type="number"
        min="1"
        class="input"
        bind:value={() => requests, (v) => (editedRequests = v)}
        data-testid="loop-budget-requests"
      />
    </label>
    <label class="flex flex-col gap-1">
      <span class="text-fg-2">Time (seconds, empty for no limit)</span>
      <input
        type="number"
        min="1"
        class="input"
        bind:value={() => wallS, (v) => (editedWall = v)}
        data-testid="loop-budget-wall"
      />
    </label>
  </div>
  {#if ctl.actionError}
    <p class="mt-3 text-xs text-danger-text" role="alert" data-testid="loop-budget-error">
      {ctl.actionError}
    </p>
  {/if}
  <div class="mt-4 flex justify-end gap-2">
    <button type="button" class="btn btn-sm" onclick={onclose}>Close</button>
    <button
      type="button"
      class="btn btn-sm btn-primary"
      disabled={ctl.busy}
      onclick={() => void save()}
      data-testid="loop-budget-save"
    >
      Save the budget
    </button>
  </div>
</Modal>
