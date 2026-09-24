<script lang="ts">
  /**
   * /bakeoff — Model x dataset bake-off cockpit.
   *
   * Scores every selected model on every selected frozen dataset (matrix) in
   * the on-demand curation-evaluator container, then renders a model x dataset
   * matrix with the best cell per dataset bolded. Datasets + baseline models are
   * discovered from the backend (auto-discovered frozen dirs + an editable
   * registry) and our trained runs are selectable directly — so adding a model
   * or dataset never needs a UI change.
   */
  import { onMount, onDestroy } from 'svelte';
  import {
    ApiError,
    bakeoffBaselineModels,
    bakeoffEvalDatasets,
    bakeoffMatrix,
    bakeoffProfiles,
    bakeoffResults,
    bakeoffRun,
    bakeoffRuns,
    bakeoffStatus,
    bakeoffTrainedModels,
    type BakeoffComparison,
    type BakeoffEvalDataset,
    type BakeoffFailure,
    type BakeoffMatrix,
    type BakeoffModelSpec,
    type BakeoffProfile,
    type BakeoffRunSummary,
    type BakeoffTrainedModel,
  } from '$lib/api';
  import { bakeoffAvailability } from '$lib/bakeoffAvailability.svelte';
  import { bakeoffFailureWhere } from '$lib/bakeoffStatus';
  import MonitoringLinks from '$lib/components/MonitoringLinks.svelte';
  import QuantizationPanel from '$components/QuantizationPanel.svelte';

  // Inference regime applied to every model: full-frame, parent-crop, or both
  // (the runner expands 'both' into [full] + [crop]).
  let mode = $state<'full' | 'crop' | 'both'>('both');

  interface DatasetChoice extends BakeoffEvalDataset {
    enabled: boolean;
  }
  interface BaselineChoice extends BakeoffModelSpec {
    enabled: boolean;
  }
  interface TrainedChoice extends BakeoffTrainedModel {
    enabled: boolean;
    imgsz: number;
  }

  // '' = omit `profile`, i.e. the evaluator's deployment default.
  let profile = $state<string>('');
  let profiles = $state<BakeoffProfile[]>([]);
  let profileDefaultError = $state<string | null>(null);
  const activeProfile = $derived(profiles.find((p) => p.name === profile) ?? null);

  let evalDatasets = $state<DatasetChoice[]>([]);
  let baselines = $state<BaselineChoice[]>([]);
  let trainedModels = $state<TrainedChoice[]>([]);

  let runs = $state<BakeoffRunSummary[]>([]);
  let selected = $state<string | null>(null);
  let matrix = $state<BakeoffMatrix | null>(null);
  // Fallback for single-dataset / older runs that only wrote comparison.json.
  let comparison = $state<BakeoffComparison | null>(null);
  let metric = $state<string>('map_50');
  let activeJob = $state<string | null>(null);
  let activeState = $state<string | null>(null);
  let activeProgress = $state<{ done: number; total: number } | null>(null);
  // What went wrong in the tracked job: the job-level reason when it ended
  // in `error`, and every stage / dataset x model cell that failed.
  let activeError = $state<string | null>(null);
  let activeFailures = $state<BakeoffFailure[]>([]);
  let error = $state<string | null>(null);
  let busy = $state(false);
  let poll: ReturnType<typeof setInterval> | undefined;

  const METRIC_LABELS: Record<string, string> = {
    map_50: 'mAP@.5',
    map_50_95: 'mAP@.5:.95',
    mean_iou: 'meanIoU',
    ap_small: 'AP small',
    precision: 'Precision',
    recall: 'Recall',
    f1: 'F1',
    latency_ms: 'Latency (ms)',
    size_mb: 'Size (MB)',
  };

  async function refreshRuns() {
    try {
      runs = (await bakeoffRuns()).runs;
      // Adopt an already-running run (e.g. triggered from another tab or the
      // API) so its progress bar shows even when this session didn't start it.
      if (!activeJob) {
        const live = runs.find((r) => r.state === 'running' || r.state === 'enqueued');
        if (live) {
          activeJob = live.job_id;
          activeState = live.state;
          startPolling();
        }
      }
    } catch (e) {
      error = e instanceof ApiError ? e.message : String(e);
    }
  }

  async function refreshDatasets() {
    try {
      const r = await bakeoffEvalDatasets();
      // Default: curated + public on; balanced samples off (opt-in).
      evalDatasets = (r.datasets ?? []).map((d) => ({
        ...d,
        enabled: d.kind !== 'sample',
      }));
    } catch (e) {
      error = e instanceof ApiError ? e.message : String(e);
    }
  }

  async function refreshProfiles() {
    try {
      const r = await bakeoffProfiles();
      profiles = r.profiles ?? [];
      profileDefaultError = r.default_error ?? null;
      // Preselect the deployment's default so the baselines shown are the
      // ones a run would actually use. Only while the operator hasn't
      // picked one themselves.
      if (!profile && r.default_profile) profile = r.default_profile;
    } catch (e) {
      error = e instanceof ApiError ? e.message : String(e);
    }
  }

  async function refreshBaselines() {
    try {
      const r = await bakeoffBaselineModels(profile || undefined);
      baselines = (r.baselines ?? []).map((b) => ({ ...b, enabled: true }));
    } catch (e) {
      error = e instanceof ApiError ? e.message : String(e);
    }
  }

  async function refreshTrainedModels() {
    try {
      const r = await bakeoffTrainedModels();
      trainedModels = (r.models ?? []).map((m) => ({ ...m, enabled: false, imgsz: 640 }));
      if (trainedModels.length > 0) trainedModels[0].enabled = true; // newest on
    } catch (e) {
      error = e instanceof ApiError ? e.message : String(e);
    }
  }

  async function loadMatrix(jobId: string) {
    selected = jobId;
    matrix = null;
    comparison = null;
    try {
      matrix = await bakeoffMatrix(jobId);
    } catch {
      matrix = null; // not a matrix job — fall back to the single-dataset table
    }
    if (!matrix) {
      try {
        comparison = await bakeoffResults(jobId);
      } catch {
        comparison = null; // still running / no results yet
      }
    }
  }

  function pct(v: number): string {
    return (v * 100).toFixed(1);
  }

  async function startRun() {
    error = null;
    busy = true;
    const datasets = evalDatasets
      .filter((d) => d.enabled)
      .map((d) => ({ path: d.path, name: d.name }));
    const ours: BakeoffModelSpec[] = trainedModels
      .filter((t) => t.enabled)
      .map((t) => ({
        backend: 'ultralytics',
        name: t.name,
        weights: t.checkpoint_path,
        imgsz: t.imgsz,
        mode,
        device: 'cuda',
        training_data: `trained here${t.model_size ? ` (${t.model_size})` : ''}`,
      }));
    const base: BakeoffModelSpec[] = baselines
      .filter((b) => b.enabled)
      .map(({ enabled: _e, ...spec }) => ({ ...spec, mode }));
    const models = [...ours, ...base];
    if (datasets.length === 0) {
      error = 'select at least one dataset';
      busy = false;
      return;
    }
    if (models.length === 0) {
      error = 'select at least one model';
      busy = false;
      return;
    }
    try {
      const res = await bakeoffRun({
        datasets,
        models,
        profile: profile || undefined,
        verify_frozen: true,
      });
      activeJob = res.job_id;
      activeState = 'enqueued';
      activeProgress = null;
      activeError = null;
      activeFailures = [];
      await refreshRuns();
      startPolling();
    } catch (e) {
      error =
        e instanceof ApiError ? `${e.message}: ${JSON.stringify(e.body)}` : String(e);
    } finally {
      busy = false;
    }
  }

  function startPolling() {
    stopPolling();
    poll = setInterval(async () => {
      if (!activeJob) return;
      try {
        const st = await bakeoffStatus(activeJob);
        activeState = st.state ?? null;
        activeProgress = st.progress ?? null;
        activeError = st.state === 'error' ? (st.error ?? 'bake-off failed') : null;
        activeFailures = st.failed ?? [];
        if (st.state === 'done' || st.state === 'error') {
          stopPolling();
          await refreshRuns();
          if (st.state === 'done') await loadMatrix(activeJob);
        }
      } catch {
        /* keep polling */
      }
    }, 4000);
  }

  function stopPolling() {
    if (poll) clearInterval(poll);
    poll = undefined;
  }

  function fmt(v: number | null | undefined, m: string): string {
    if (v == null) return '—';
    if (m === 'latency_ms') return v.toFixed(0);
    if (m === 'size_mb') return v.toFixed(1);
    return (v * 100).toFixed(1);
  }

  function cell(model: string, ds: string): number | null {
    return matrix?.cells?.[model]?.[ds]?.[metric] ?? null;
  }
  function isBest(model: string, ds: string): boolean {
    return matrix?.best?.[ds]?.[metric] === model;
  }

  onMount(async () => {
    // Only fire the four discovery GETs once the capability probe says
    // the router exists — firing them unconditionally is the "never 404"
    // violation removed from /train's export panel (see that page's
    // onMount comment). bakeoffAvailability.init() is idempotent and
    // shares its result with +layout.svelte's nav-link gate, so this
    // reuses that same probe rather than doubling it.
    await bakeoffAvailability.init();
    if (bakeoffAvailability.available === false) return;
    void refreshRuns();
    // Baselines depend on the (possibly preselected) profile, so they
    // must wait for it — two concurrent lookups could land out of order.
    void refreshProfiles().then(refreshBaselines);
    void refreshDatasets();
    void refreshTrainedModels();
  });
  onDestroy(stopPolling);
</script>

<svelte:head><title>Bake-off · Cropwright</title></svelte:head>

<div class="mx-auto max-w-6xl p-6 text-zinc-200">
  <h1 class="mb-1 text-2xl font-semibold">Model × Dataset Bake-off</h1>
  <p class="mb-6 text-sm text-zinc-400">
    Every selected model scored on every selected frozen dataset with one IoU metric
    (pycocotools), in the on-demand <code>curation-evaluator</code>. Best per dataset is
    <strong>bold</strong>.
  </p>

  <div class="mb-6"><MonitoringLinks /></div>

  {#if bakeoffAvailability.available === false}
    <p class="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4 text-sm text-zinc-400">
      Model evaluation is not available on this backend.
    </p>
  {:else}
    {#if error}
      <div class="mb-4 rounded border border-red-700 bg-red-950 p-3 text-sm text-red-200">
        {error}
      </div>
    {/if}

    <!-- Run form -->
    <section class="mb-8 rounded-lg border border-zinc-800 bg-zinc-900/50 p-4">
      <div class="mb-4 grid gap-4 md:grid-cols-2">
        <!-- Datasets -->
        <div>
          <h2 class="mb-2 text-sm font-medium text-zinc-300">
            Datasets ({evalDatasets.filter((d) => d.enabled)
              .length}/{evalDatasets.length})
          </h2>
          <div class="max-h-44 space-y-1 overflow-auto pr-1">
            {#each [['curated', 'Curated (ours, frozen split)'], ['public', 'Public — full split'], ['sample', 'Balanced — deduplicated cluster sample']] as [kind, heading] (kind)}
              {@const group = evalDatasets.filter((d) => d.kind === kind)}
              {#if group.length}
                <p class="mt-1 text-[10px] uppercase tracking-wide text-zinc-500">
                  {heading}
                </p>
                {#each group as d (d.path)}
                  <label class="flex items-center gap-2 text-xs">
                    <input type="checkbox" bind:checked={d.enabled} />
                    <span class="font-mono">{d.name}</span>
                    {#if d.n_test}<span class="text-[10px] text-zinc-500"
                        >{d.n_test} frames</span
                      >{/if}
                  </label>
                {/each}
              {/if}
            {/each}
            {#if evalDatasets.length === 0}
              <p class="text-xs text-zinc-600">no frozen datasets discovered</p>
            {/if}
          </div>
        </div>

        <!-- Models -->
        <div>
          <h2 class="mb-2 text-sm font-medium text-zinc-300">Models</h2>
          <div class="max-h-44 space-y-1 overflow-auto pr-1">
            <p class="text-[10px] uppercase tracking-wide text-zinc-500">Our trained</p>
            {#each trainedModels as t (t.run_id)}
              <label class="flex items-center gap-2 text-xs">
                <input type="checkbox" bind:checked={t.enabled} />
                <span class="font-mono">{t.name}</span>
                {#if t.model_size}<span
                    class="rounded bg-blue-900/60 px-1 text-[10px] text-blue-200"
                    >{t.model_size}</span
                  >{/if}
                <input
                  type="number"
                  min="320"
                  step="32"
                  bind:value={t.imgsz}
                  class="w-16 rounded border border-zinc-700 bg-zinc-950 px-1 text-[10px]"
                />
              </label>
            {/each}
            <p class="mt-1 text-[10px] uppercase tracking-wide text-zinc-500">
              Baselines
            </p>
            {#each baselines as b (b.name)}
              <label class="flex items-center gap-2 text-xs">
                <input type="checkbox" bind:checked={b.enabled} />
                <span class="font-mono">{b.name}</span>
                <span class="rounded bg-zinc-800 px-1 text-[10px] text-zinc-400"
                  >{b.backend}</span
                >
              </label>
            {/each}
          </div>
        </div>
      </div>

      <div class="flex flex-wrap items-center gap-4">
        <label class="text-sm">
          <span class="text-zinc-400">Profile</span>
          <select
            bind:value={profile}
            onchange={() => void refreshBaselines()}
            class="ml-2 rounded border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs"
          >
            {#if !profiles.some((p) => p.default)}
              <option value="">deployment default</option>
            {/if}
            {#each profiles as p (p.name)}
              <option value={p.name}
                >{p.name}{p.default
                  ? ' (default)'
                  : p.kind === 'example'
                    ? ' (example)'
                    : ''}</option
              >
            {/each}
          </select>
          {#if activeProfile}
            <span class="ml-2 text-xs text-zinc-500">
              target <span class="font-mono">{activeProfile.target_class_name}</span> ·
              ranked by {METRIC_LABELS[activeProfile.rank_metric] ??
                activeProfile.rank_metric}
            </span>
          {/if}
        </label>
        <label class="text-sm">
          <span class="text-zinc-400">Regime</span>
          <select
            bind:value={mode}
            class="ml-2 rounded border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs"
          >
            <option value="both">both (full + crop)</option>
            <option value="full">full-frame</option>
            <option value="crop">parent-crop</option>
          </select>
        </label>
        <button
          onclick={startRun}
          disabled={busy}
          class="rounded bg-emerald-700 px-4 py-1.5 text-sm font-medium hover:bg-emerald-600 disabled:opacity-50"
        >
          {busy ? 'Enqueuing…' : 'Run matrix bake-off'}
        </button>
        {#if activeJob}
          <span class="text-sm text-zinc-400">
            job <code>{activeJob}</code> — <strong>{activeState}</strong>
            {#if activeProgress}({activeProgress.done}/{activeProgress.total}){/if}
          </span>
        {/if}
      </div>

      {#if profileDefaultError}
        <p class="mt-3 text-xs text-amber-300">
          The configured default bake-off profile is invalid: {profileDefaultError}
        </p>
      {/if}

      {#if activeError || activeFailures.length > 0}
        <div
          class="mt-3 rounded border border-red-800 bg-red-950/60 p-3 text-xs text-red-200"
        >
          {#if activeError}<p class="font-medium">Job failed: {activeError}</p>{/if}
          {#if activeFailures.length > 0}
            <p class="mb-1 {activeError ? 'mt-2' : ''} font-medium">
              {activeFailures.length} failed {activeFailures.length === 1
                ? 'piece'
                : 'pieces'}
            </p>
            <ul class="max-h-40 space-y-0.5 overflow-auto font-mono">
              {#each activeFailures as f, i (i)}
                <li>
                  {bakeoffFailureWhere(f)} — {f.error}
                </li>
              {/each}
            </ul>
          {/if}
        </div>
      {/if}

      {#if activeJob && activeProgress && activeProgress.total > 0 && activeState !== 'done' && activeState !== 'error'}
        <div class="mt-3">
          <div class="h-2 w-full overflow-hidden rounded bg-zinc-800">
            <div
              class="h-full bg-emerald-600 transition-all"
              style="width: {Math.round(
                (activeProgress.done / activeProgress.total) * 100,
              )}%"
            ></div>
          </div>
          <p class="mt-1 text-xs text-zinc-500">
            {activeProgress.done} / {activeProgress.total} evaluations ({Math.round(
              (activeProgress.done / activeProgress.total) * 100,
            )}%)
          </p>
        </div>
      {/if}
    </section>

    <div class="grid grid-cols-[240px_1fr] gap-6">
      <!-- Runs list -->
      <aside>
        <h2 class="mb-2 text-sm font-medium text-zinc-400">Runs</h2>
        <ul class="space-y-1">
          {#each runs as r (r.job_id)}
            <li>
              <button
                onclick={() => loadMatrix(r.job_id)}
                class="w-full rounded px-2 py-1 text-left text-xs hover:bg-zinc-800 {selected ===
                r.job_id
                  ? 'bg-zinc-800'
                  : ''}"
              >
                <div class="flex items-center justify-between gap-2">
                  <span class="truncate font-mono" title={r.job_id}>{r.job_id}</span>
                  <span class="shrink-0 text-zinc-500">{r.state ?? ''}</span>
                </div>
                {#if r.started_at}
                  <div class="text-[10px] text-zinc-600">
                    {new Date(r.started_at).toLocaleString()}
                  </div>
                {/if}
              </button>
            </li>
          {:else}
            <li class="text-xs text-zinc-600">no runs yet</li>
          {/each}
        </ul>
      </aside>

      <!-- Matrix -->
      <main>
        {#if matrix && matrix.models.length}
          <div class="mb-2 flex items-center justify-between">
            <h2 class="text-sm font-medium text-zinc-400">Matrix — {selected}</h2>
            <label class="text-xs text-zinc-400">
              metric
              <select
                bind:value={metric}
                class="ml-1 rounded border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs"
              >
                {#each matrix.metrics as m (m)}
                  <option value={m}>{METRIC_LABELS[m] ?? m}</option>
                {/each}
              </select>
            </label>
          </div>
          <div class="overflow-x-auto rounded-lg border border-zinc-800">
            <table class="w-full text-sm">
              <thead class="bg-zinc-900 text-xs uppercase text-zinc-400">
                <tr>
                  <th class="px-3 py-2 text-left">Model \\ Dataset</th>
                  {#each matrix.datasets as ds (ds)}
                    <th class="px-2 py-2 text-right">{ds}</th>
                  {/each}
                </tr>
              </thead>
              <tbody>
                {#each matrix.models as m (m)}
                  <tr class="border-t border-zinc-800 hover:bg-zinc-800/40">
                    <td class="px-3 py-2 font-mono text-xs">{m}</td>
                    {#each matrix.datasets as ds (ds)}
                      <td
                        class="px-2 py-2 text-right {isBest(m, ds)
                          ? 'font-bold text-emerald-300'
                          : ''}"
                      >
                        {fmt(cell(m, ds), metric)}
                      </td>
                    {/each}
                  </tr>
                {/each}
              </tbody>
            </table>
          </div>
          <p class="mt-2 text-xs text-zinc-500">
            {metric === 'latency_ms'
              ? 'milliseconds (lower better)'
              : metric === 'size_mb'
                ? 'megabytes (lower better)'
                : 'percent'}; best per dataset in
            <span class="font-bold text-emerald-300">bold</span>.
          </p>

          <QuantizationPanel {matrix} dataset={matrix.datasets[0]} />
        {:else if comparison && comparison.models.length}
          <h2 class="mb-2 text-sm font-medium text-zinc-400">
            Results — {selected} (single dataset, ranked by mAP@.5:.95)
          </h2>
          <div class="overflow-x-auto rounded-lg border border-zinc-800">
            <table class="w-full text-sm">
              <thead class="bg-zinc-900 text-xs uppercase text-zinc-400">
                <tr>
                  <th class="px-3 py-2 text-left">Model</th>
                  <th class="px-2 py-2 text-right">mAP@.5</th>
                  <th class="px-2 py-2 text-right">mAP@.5:.95</th>
                  <th class="px-2 py-2 text-right">meanIoU</th>
                  <th class="px-2 py-2 text-right">P</th>
                  <th class="px-2 py-2 text-right">R</th>
                  <th class="px-2 py-2 text-right">F1</th>
                  <th class="px-2 py-2 text-right">ms</th>
                </tr>
              </thead>
              <tbody>
                {#each comparison.models as m, i (m.model)}
                  <tr
                    class="border-t border-zinc-800 {i === 0 ? 'bg-emerald-950/40' : ''}"
                  >
                    <td class="px-3 py-2 font-mono text-xs">{m.model}</td>
                    <td class="px-2 py-2 text-right">{pct(m.map_50)}</td>
                    <td class="px-2 py-2 text-right">{pct(m.map_50_95)}</td>
                    <td class="px-2 py-2 text-right">{pct(m.mean_iou)}</td>
                    <td class="px-2 py-2 text-right">{pct(m.precision)}</td>
                    <td class="px-2 py-2 text-right">{pct(m.recall)}</td>
                    <td class="px-2 py-2 text-right">{pct(m.f1)}</td>
                    <td class="px-2 py-2 text-right text-zinc-400"
                      >{m.latency_ms.toFixed(0)}</td
                    >
                  </tr>
                {/each}
              </tbody>
            </table>
          </div>
          <p class="mt-2 text-xs text-zinc-500">
            Percent; top row (green) leads on mAP@.5:.95.
          </p>
        {:else if selected}
          <p class="text-sm text-zinc-500">
            No results for {selected} yet (still running?).
          </p>
        {:else}
          <p class="text-sm text-zinc-500">Select a run to view its results.</p>
        {/if}
      </main>
    </div>
  {/if}
</div>
