<script lang="ts">
  /** Confirm step before `POST /bakeoff/run`: what will be scored on what. */
  import { focusOnMount } from '$lib/actions/focusOnMount';
  import { trapFocus } from '$lib/actions/trapFocus';

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

<!-- svelte-ignore a11y_click_events_have_key_events -->
<div
  class="fixed inset-0 z-40 flex items-center justify-center bg-black/60 p-4"
  role="dialog"
  aria-modal="true"
  aria-label="Confirm model comparison"
  tabindex="-1"
  data-testid="run-confirm"
  use:focusOnMount
  use:trapFocus={{ onEscape: onCancel }}
  onclick={(e) => {
    if (e.target === e.currentTarget) onCancel();
  }}
>
  <div
    class="max-h-[80vh] w-full max-w-lg overflow-auto rounded-lg border border-zinc-800 bg-zinc-950 p-5 shadow-2xl"
  >
    <h3 class="mb-2 text-base font-semibold">Run model comparison</h3>
    <p class="mb-3 text-sm text-zinc-300" data-testid="run-confirm-summary">
      {models.length} model{models.length === 1 ? '' : 's'} × {datasets.length} dataset{datasets.length ===
      1
        ? ''
        : 's'} = {models.length * datasets.length} evaluation{models.length *
        datasets.length ===
      1
        ? ''
        : 's'}, profile <span class="font-mono">{profile || 'server default'}</span>.
    </p>
    <div class="mb-3 grid grid-cols-2 gap-3 text-xs">
      <div>
        <p class="mb-1 text-zinc-500">Datasets</p>
        <ul class="space-y-0.5 font-mono text-zinc-300">
          {#each datasets as d (d)}<li>{d}</li>{/each}
        </ul>
      </div>
      <div>
        <p class="mb-1 text-zinc-500">Models</p>
        <ul class="space-y-0.5 font-mono text-zinc-300">
          {#each models as m (m)}<li>{m}</li>{/each}
        </ul>
      </div>
    </div>
    <p class="mb-3 text-xs text-zinc-400">
      Runs in the on-demand evaluator container and may pause other GPU services while it
      runs.
    </p>
    {#if error}
      <p
        class="mb-3 rounded border border-red-800 bg-red-950/60 p-2 text-xs text-red-200"
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
  </div>
</div>
