<script lang="ts">
  import Modal from '$components/Modal.svelte';
  /** Confirm step before `POST /bakeoff/run`: what will be scored on what. */

  interface Props {
    datasets: string[];
    models: string[];
    profile: string;
    submitting: boolean;
    error: string | null;
    onConfirm: () => void;
    onCancel: () => void;
  }
  let { datasets, models, profile, submitting, error, onConfirm, onCancel }: Props =
    $props();
</script>

<Modal
  label="Confirm model comparison"
  onclose={onCancel}
  panelClass="max-h-[80vh] w-full max-w-lg overflow-auto rounded-lg border border-line bg-surface-0 p-5 shadow-2xl"
  data-testid="run-confirm"
>
  <h3 class="mb-2 text-title font-semibold">Run model comparison</h3>
  <p class="mb-3 text-body text-fg-2" data-testid="run-confirm-summary">
    {models.length} model{models.length === 1 ? '' : 's'} × {datasets.length} dataset{datasets.length ===
    1
      ? ''
      : 's'} = {models.length * datasets.length} evaluation{models.length *
      datasets.length ===
    1
      ? ''
      : 's'}, profile <span>{profile || 'server default'}</span>.
  </p>
  <div class="mb-3 grid grid-cols-2 gap-3 text-meta">
    <div>
      <p class="mb-1 text-fg-3">Datasets</p>
      <ul class="space-y-0.5 text-fg-2">
        <code
          >{#each datasets as d (d)}<li>{d}</li>{/each}</code
        >
      </ul>
    </div>
    <div>
      <p class="mb-1 text-fg-3">Models</p>
      <ul class="space-y-0.5 text-fg-2">
        <code
          >{#each models as m (m)}<li>{m}</li>{/each}</code
        >
      </ul>
    </div>
  </div>
  <p class="mb-3 text-meta text-fg-2">
    Runs in the on-demand evaluator container and may pause other GPU services while it
    runs.
  </p>
  {#if error}
    <p
      class="mb-3 rounded border border-danger-line bg-danger-soft p-2 text-meta text-danger-text"
      data-testid="run-error"
    >
      {error}
    </p>
  {/if}
  <div class="flex justify-end gap-2">
    <button type="button" class="btn" onclick={onCancel}>Cancel</button>
    <button
      type="button"
      class="btn btn-primary"
      disabled={submitting}
      onclick={onConfirm}
    >
      {submitting ? 'Enqueuing…' : 'Run'}
    </button>
  </div>
</Modal>
