<script lang="ts">
  /**
   * The Workspace entry to the same step the Train page offers: run the newest finished run's
   * checkpoint over the project so Disagreements fills. The run is the one `ProbeControl`
   * already gates on; the route is `POST /probe/run`.
   */
  import Modal from '$components/Modal.svelte';
  import ProbeControl from '$components/ProbeControl.svelte';
  import { apiErrorText } from '$lib/api_errors';
  import { getTrainRuns } from '$lib/api_train';
  import { canRunProbe, FIND_DISAGREEMENTS } from '$lib/probe';
  import type { TrainJobStatus } from '$lib/types_train';

  let { onclose }: { onclose: () => void } = $props();

  let run = $state<TrainJobStatus | null>(null);
  let loading = $state(true);
  let error = $state<string | null>(null);

  $effect(() => {
    const ctl = new AbortController();
    getTrainRuns(50, 0, ctl.signal)
      .then((res) => {
        run = res.items.find((r) => canRunProbe(r)) ?? null;
        loading = false;
      })
      .catch((e: unknown) => {
        if (ctl.signal.aborted) return;
        error = apiErrorText(e);
        loading = false;
      });
    return () => ctl.abort();
  });
</script>

<Modal label={FIND_DISAGREEMENTS} {onclose}>
  <h3 class="mb-3 text-title font-semibold text-fg-1">{FIND_DISAGREEMENTS}</h3>
  <div class="mb-4 space-y-2 text-body text-fg-2" data-testid="find-disagreements-body">
    {#if loading}
      <p>Looking for a trained model…</p>
    {:else if error}
      <p class="text-danger-text" role="alert">{error}</p>
    {:else if run}
      <p>
        Runs the newest trained model over your items, so the Disagreements queue lists
        where it differs from your labels.
      </p>
      <ProbeControl status={run} />
    {:else}
      <p>No finished training run has a checkpoint to run. Train a model first.</p>
    {/if}
  </div>
  <div class="flex justify-end">
    <button type="button" class="btn" onclick={onclose}>Close</button>
  </div>
</Modal>
