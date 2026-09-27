<script lang="ts">
  /**
   * /bakeoff — model comparison (OpenProcessor #34 v2 wire).
   *
   * Pick eval datasets (export test splits, external frozen sets), pick
   * contenders (finished training runs, profile baselines, custom refs),
   * run in the on-demand evaluator, poll, then read the served model ×
   * dataset matrix and per-class comparison. Previous runs stay viewable.
   * State lives in `bakeoffController`; this file is layout.
   * See docs/design/bakeoff-v2-ui-plan-2026-09-25.md.
   */
  import { onMount, onDestroy } from 'svelte';
  import { createBakeoffController } from '$lib/bakeoff/bakeoffController.svelte';
  import { formatTime, metricLabel } from '$lib/bakeoff/view';
  import MonitoringLinks from '$lib/components/MonitoringLinks.svelte';
  import DatasetPicker from '$components/bakeoff/DatasetPicker.svelte';
  import ModelPicker from '$components/bakeoff/ModelPicker.svelte';
  import RunConfirmDialog from '$components/bakeoff/RunConfirmDialog.svelte';
  import RunStatusPanel from '$components/bakeoff/RunStatusPanel.svelte';
  import BakeoffMatrixTable from '$components/bakeoff/BakeoffMatrixTable.svelte';
  import ComparisonView from '$components/bakeoff/ComparisonView.svelte';

  const controller = createBakeoffController();
  const s = controller.state;

  let confirming = $state(false);

  const activeProfile = $derived(s.profiles.find((p) => p.name === s.profile) ?? null);
  const nModels = $derived(
    s.selectedRuns.length + s.selectedBaselines.length + s.customRefs.length,
  );
  const canRun = $derived(s.selectedDatasets.length > 0 && nModels > 0);

  function datasetName(id: string): string {
    return s.datasets.find((d) => d.id === id)?.name ?? id;
  }

  const modelLabels = $derived([
    ...s.selectedRuns.map(
      (id) => s.trained.find((t) => t.run_id === id)?.display_name ?? id,
    ),
    ...s.selectedBaselines,
    ...s.customRefs.map((c) => c.display_name ?? c.name),
  ]);

  const resultDatasets = $derived(
    s.matrix?.datasets.map((d) => d.id) ??
      s.runs.find((r) => r.job_id === s.viewJob)?.datasets ??
      [],
  );

  async function confirmRun() {
    if (await controller.submit()) confirming = false;
  }

  onMount(() => void controller.init());
  onDestroy(() => controller.destroy());
</script>

<svelte:head><title>Bake-off · Cropwright</title></svelte:head>

<div class="mx-auto max-w-7xl p-6 text-zinc-200">
  <h1 class="mb-1 text-2xl font-semibold">Bake-off: model comparison</h1>
  <p class="mb-6 text-sm text-zinc-400">
    Score trained models and baselines on frozen test splits, per class, in the on-demand
    evaluator. Classes a model does not cover are shown, not hidden.
  </p>

  <div class="mb-6"><MonitoringLinks /></div>

  {#if s.loadErrors.length}
    <ul
      class="mb-4 space-y-0.5 rounded border border-red-700 bg-red-950 p-3 text-sm text-red-200"
      data-testid="load-errors"
    >
      {#each s.loadErrors as e, i (i)}<li>{e}</li>{/each}
    </ul>
  {/if}

  <section class="mb-6 rounded-lg border border-zinc-800 bg-zinc-900/50 p-4">
    <h2 class="mb-3 text-sm font-medium text-zinc-300">Set up a comparison</h2>

    <div class="mb-4 flex flex-wrap items-center gap-3 text-sm">
      <label>
        <span class="text-zinc-400">Profile</span>
        <select
          value={s.profile}
          onchange={(e) => void controller.setProfile(e.currentTarget.value)}
          class="ml-2 rounded border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs"
          data-testid="profile-select"
        >
          {#if !s.profiles.some((p) => p.name === s.profile)}
            <option value="">server default</option>
          {/if}
          {#each s.profiles as p (p.name)}
            <option value={p.name}>{p.name}{p.default ? ' (default)' : ''}</option>
          {/each}
        </select>
      </label>
      {#if activeProfile}
        <span class="text-xs text-zinc-500" data-testid="profile-summary">
          {activeProfile.description}{activeProfile.description ? ' · ' : ''}ranked by
          {metricLabel(activeProfile.rank_metric)} · op conf {activeProfile.op_conf} · op IoU
          {activeProfile.op_iou} · {activeProfile.imgsz}px
          {#if activeProfile.class_filter.length}
            · classes: {activeProfile.class_filter.join(', ')}{/if}
        </span>
      {/if}
    </div>
    {#if s.profileDefaultError}
      <p class="mb-3 text-xs text-amber-300" data-testid="profile-default-error">
        The configured default profile is invalid: {s.profileDefaultError}
      </p>
    {/if}

    <div class="grid gap-4 md:grid-cols-2">
      <div>
        <h3 class="mb-2 text-sm font-medium text-zinc-300">
          Datasets ({s.selectedDatasets.length}/{s.datasets.length})
        </h3>
        <div class="max-h-80 overflow-auto pr-1">
          <DatasetPicker
            datasets={s.datasets}
            selected={s.selectedDatasets}
            onToggle={(id, on) => void controller.setDatasetSelected(id, on)}
          />
        </div>
      </div>
      <div>
        <h3 class="mb-2 text-sm font-medium text-zinc-300">Models ({nModels})</h3>
        <div class="max-h-80 overflow-auto pr-1">
          <ModelPicker
            trained={s.trained}
            baselines={s.baselines}
            customRefs={s.customRefs}
            selectedRuns={s.selectedRuns}
            selectedBaselines={s.selectedBaselines}
            selectedDatasets={s.selectedDatasets}
            facts={s.facts}
            {datasetName}
            onToggleRun={controller.setRunSelected}
            onToggleBaseline={controller.setBaselineSelected}
            onAddCustom={controller.addCustom}
            onRemoveCustom={controller.removeCustom}
          />
        </div>
      </div>
    </div>

    <div class="mt-4 flex flex-wrap items-center gap-3">
      <button
        type="button"
        class="btn btn-primary"
        disabled={!canRun || s.submitting}
        onclick={() => {
          s.runError = null;
          confirming = true;
        }}
        data-testid="run-open"
      >
        Run comparison
      </button>
      {#if !canRun}
        <span class="text-xs text-zinc-500"
          >Select at least one dataset and one model.</span
        >
      {/if}
    </div>
  </section>

  {#if confirming}
    <RunConfirmDialog
      datasets={s.selectedDatasets.map(datasetName)}
      models={modelLabels}
      profile={s.profile}
      submitting={s.submitting}
      error={s.runError}
      onConfirm={() => void confirmRun()}
      onCancel={() => (confirming = false)}
    />
  {/if}

  {#if s.activeJob}
    <RunStatusPanel
      jobId={s.activeJob}
      status={s.activeStatus}
      accepted={s.accepted}
      {datasetName}
    />
  {/if}

  <div class="grid grid-cols-[220px_minmax(0,1fr)] gap-6">
    <aside>
      <h2 class="mb-2 text-sm font-medium text-zinc-400">Previous runs</h2>
      <ul class="space-y-1" data-testid="runs-list">
        {#each s.runs as r (r.job_id)}
          <li>
            <button
              type="button"
              onclick={() => void controller.viewRun(r.job_id)}
              class="w-full rounded px-2 py-1 text-left text-xs hover:bg-zinc-800 {s.viewJob ===
              r.job_id
                ? 'bg-zinc-800'
                : ''}"
              data-job-id={r.job_id}
            >
              <div class="flex items-center justify-between gap-2">
                <span class="truncate font-mono" title={r.job_id}>{r.job_id}</span>
                <span class="shrink-0 text-zinc-500">{r.state}</span>
              </div>
              <div class="text-[10px] text-zinc-600">
                {r.models.length} model{r.models.length === 1 ? '' : 's'} × {r.datasets
                  .length} dataset{r.datasets.length === 1 ? '' : 's'}{r.profile
                  ? ` · ${r.profile}`
                  : ''}
              </div>
              <div class="text-[10px] text-zinc-600">{formatTime(r.started_at)}</div>
            </button>
          </li>
        {:else}
          <li class="text-xs text-zinc-600">No comparison runs yet.</li>
        {/each}
      </ul>
    </aside>

    <main class="min-w-0" data-testid="results">
      {#if s.viewJob}
        <h2 class="mb-3 text-sm font-medium text-zinc-400">
          Results · <span class="font-mono">{s.viewJob}</span>
        </h2>
        {#if s.matrixError}
          <p
            class="mb-3 rounded border border-red-800 bg-red-950/60 p-3 text-sm text-red-200"
          >
            {s.matrixError}
          </p>
        {/if}
        {#if s.matrix && s.matrix.models.length}
          <div class="mb-6">
            <BakeoffMatrixTable matrix={s.matrix} {datasetName} />
          </div>
        {/if}
        {#if resultDatasets.length > 1}
          <div class="mb-3 flex flex-wrap gap-1" role="tablist">
            {#each resultDatasets as id (id)}
              <button
                type="button"
                role="tab"
                aria-selected={s.viewDataset === id}
                class="rounded px-2 py-1 text-xs {s.viewDataset === id
                  ? 'bg-zinc-700 text-zinc-100'
                  : 'bg-zinc-900 text-zinc-400 hover:bg-zinc-800'}"
                onclick={() => void controller.setViewDataset(id)}
                >{datasetName(id)}</button
              >
            {/each}
          </div>
        {/if}
        <ComparisonView
          comparison={s.comparison}
          legacy={s.comparisonLegacy}
          error={s.comparisonError}
          loading={s.loadingResults}
        />
      {:else}
        <p class="text-sm text-zinc-500">Select a run to view its results.</p>
      {/if}
    </main>
  </div>
</div>
