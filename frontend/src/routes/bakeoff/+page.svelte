<script lang="ts">
  /**
   * /bakeoff — LPR model bake-off cockpit.
   *
   * Lists comparison runs, renders the ranked results table for the
   * selected run (the clean leaderboard MLflow can't give cleanly), and
   * enqueues new runs via POST /curation/bakeoff/run (executed by the on-demand
   * legacy-evaluator container). Polls status while a run is active.
   */
  import { onMount, onDestroy } from 'svelte';
  import {
    ApiError,
    bakeoffResults,
    bakeoffRun,
    bakeoffRuns,
    bakeoffStatus,
    type BakeoffComparison,
    type BakeoffModelSpec,
    type BakeoffRunSummary,
  } from '$lib/api';

  // Sensible default contenders. Operator edits the dataset path + toggles.
  const DEFAULT_DATASET = '/data/legacy_train_dataset_v7/lpr_exports/current';
  const LPDNET = '/data/datasets/models/lpdnet_pruned_v2.2.1/lpdnet_pruned_v2.2.1/LPDNet_usa_pruned_tao5.onnx';

  interface Contender extends BakeoffModelSpec {
    enabled: boolean;
  }

  let dataset = $state(DEFAULT_DATASET);
  let contenders = $state<Contender[]>([
    { enabled: true, backend: 'triton', name: 'lpr_nanov11_640', triton_model: 'lpr_nanov11_640',
      triton_url: 'triton-server:4601', training_data: 'andrewmvd Kaggle' } as Contender,
    { enabled: true, backend: 'open-image-models', name: 'open-image-models-yolov9t',
      device: 'cuda', training_data: 'open plate datasets' },
    { enabled: true, backend: 'lpdnet', name: 'lpdnet-usa', weights: LPDNET, lpdnet_variant: 'usa',
      device: '0', training_data: 'NVIDIA TAO (US)' },
    { enabled: false, backend: 'ultralytics', name: 'ours-yolo26', weights: '', imgsz: 1280,
      device: '0', training_data: 'curated vehicles' },
  ]);

  let runs = $state<BakeoffRunSummary[]>([]);
  let selected = $state<string | null>(null);
  let comparison = $state<BakeoffComparison | null>(null);
  let activeJob = $state<string | null>(null);
  let activeState = $state<string | null>(null);
  let error = $state<string | null>(null);
  let busy = $state(false);

  let poll: ReturnType<typeof setInterval> | undefined;

  async function refreshRuns() {
    try {
      runs = (await bakeoffRuns()).runs;
    } catch (e) {
      error = e instanceof ApiError ? e.message : String(e);
    }
  }

  async function loadResults(jobId: string) {
    selected = jobId;
    comparison = null;
    try {
      comparison = await bakeoffResults(jobId);
    } catch (e) {
      comparison = null; // not done yet / no comparison
    }
  }

  async function startRun() {
    error = null;
    busy = true;
    const models = contenders
      .filter((c) => c.enabled)
      .map(({ enabled: _e, ...spec }) => spec);
    if (models.length === 0) {
      error = 'enable at least one model';
      busy = false;
      return;
    }
    try {
      const res = await bakeoffRun({ dataset, models, verify_frozen: true });
      activeJob = res.job_id;
      activeState = 'enqueued';
      await refreshRuns();
      startPolling();
    } catch (e) {
      error = e instanceof ApiError ? `${e.message}: ${JSON.stringify(e.body)}` : String(e);
    } finally {
      busy = false;
    }
  }

  function startPolling() {
    stopPolling();
    poll = setInterval(async () => {
      if (!activeJob) return;
      try {
        const st = (await bakeoffStatus(activeJob)) as { state?: string };
        activeState = st.state ?? null;
        if (st.state === 'done' || st.state === 'error') {
          stopPolling();
          await refreshRuns();
          if (st.state === 'done') await loadResults(activeJob);
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

  function pct(v: number): string {
    return (v * 100).toFixed(1);
  }

  onMount(refreshRuns);
  onDestroy(stopPolling);
</script>

<div class="mx-auto max-w-6xl p-6 text-zinc-200">
  <h1 class="mb-1 text-2xl font-semibold">LPR Model Bake-off</h1>
  <p class="mb-6 text-sm text-zinc-400">
    Every model scored on the same frozen test split with one IoU metric (pycocotools).
    Runs execute in the on-demand <code>legacy-evaluator</code> container and log to MLflow.
  </p>

  {#if error}
    <div class="mb-4 rounded border border-red-700 bg-red-950 p-3 text-sm text-red-200">{error}</div>
  {/if}

  <!-- Run form -->
  <section class="mb-8 rounded-lg border border-zinc-800 bg-zinc-900/50 p-4">
    <h2 class="mb-3 text-lg font-medium">New comparison</h2>
    <label class="mb-3 block text-sm">
      <span class="text-zinc-400">Frozen test dataset (export root)</span>
      <input
        bind:value={dataset}
        class="mt-1 w-full rounded border border-zinc-700 bg-zinc-950 px-2 py-1 font-mono text-xs"
      />
    </label>
    <div class="mb-3 space-y-2">
      {#each contenders as c (c.name)}
        <label class="flex items-center gap-2 text-sm">
          <input type="checkbox" bind:checked={c.enabled} />
          <span class="font-medium">{c.name}</span>
          <span class="rounded bg-zinc-800 px-1.5 py-0.5 text-xs text-zinc-400">{c.backend}</span>
          {#if c.backend === 'ultralytics'}
            <input
              bind:value={c.weights}
              placeholder="weights .pt path"
              class="flex-1 rounded border border-zinc-700 bg-zinc-950 px-2 py-0.5 font-mono text-xs"
            />
          {/if}
          <span class="text-xs text-zinc-500">{c.training_data}</span>
        </label>
      {/each}
    </div>
    <button
      onclick={startRun}
      disabled={busy}
      class="rounded bg-emerald-700 px-4 py-1.5 text-sm font-medium hover:bg-emerald-600 disabled:opacity-50"
    >
      {busy ? 'Enqueuing…' : 'Run bake-off'}
    </button>
    {#if activeJob}
      <span class="ml-3 text-sm text-zinc-400">
        job <code>{activeJob}</code> — <strong>{activeState}</strong>
      </span>
    {/if}
  </section>

  <div class="grid grid-cols-[220px_1fr] gap-6">
    <!-- Runs list -->
    <aside>
      <h2 class="mb-2 text-sm font-medium text-zinc-400">Runs</h2>
      <ul class="space-y-1">
        {#each runs as r (r.job_id)}
          <li>
            <button
              onclick={() => loadResults(r.job_id)}
              class="w-full rounded px-2 py-1 text-left text-xs hover:bg-zinc-800 {selected === r.job_id ? 'bg-zinc-800' : ''}"
            >
              <span class="font-mono">{r.job_id}</span>
              <span class="ml-1 text-zinc-500">{r.state ?? ''}</span>
            </button>
          </li>
        {:else}
          <li class="text-xs text-zinc-600">no runs yet</li>
        {/each}
      </ul>
    </aside>

    <!-- Results table -->
    <main>
      {#if comparison && comparison.models.length}
        <h2 class="mb-2 text-sm font-medium text-zinc-400">
          Results — {selected} (ranked by mAP@.5:.95)
        </h2>
        <div class="overflow-x-auto rounded-lg border border-zinc-800">
          <table class="w-full text-sm">
            <thead class="bg-zinc-900 text-xs uppercase text-zinc-400">
              <tr>
                <th class="px-3 py-2 text-left">Model</th>
                <th class="px-3 py-2 text-left">Training data</th>
                <th class="px-2 py-2 text-right">mAP@.5</th>
                <th class="px-2 py-2 text-right">mAP@.5:.95</th>
                <th class="px-2 py-2 text-right">AP_s</th>
                <th class="px-2 py-2 text-right">meanIoU</th>
                <th class="px-2 py-2 text-right">P</th>
                <th class="px-2 py-2 text-right">R</th>
                <th class="px-2 py-2 text-right">F1</th>
                <th class="px-2 py-2 text-right">ms</th>
              </tr>
            </thead>
            <tbody>
              {#each comparison.models as m, i (m.model)}
                <tr class="border-t border-zinc-800 {i === 0 ? 'bg-emerald-950/40' : ''}">
                  <td class="px-3 py-2 font-medium">{m.model}</td>
                  <td class="px-3 py-2 text-xs text-zinc-400">{m.training_data || '—'}</td>
                  <td class="px-2 py-2 text-right">{pct(m.map_50)}</td>
                  <td class="px-2 py-2 text-right">{pct(m.map_50_95)}</td>
                  <td class="px-2 py-2 text-right">{pct(m.ap_small)}</td>
                  <td class="px-2 py-2 text-right">{pct(m.mean_iou)}</td>
                  <td class="px-2 py-2 text-right">{pct(m.precision)}</td>
                  <td class="px-2 py-2 text-right">{pct(m.recall)}</td>
                  <td class="px-2 py-2 text-right">{pct(m.f1)}</td>
                  <td class="px-2 py-2 text-right text-zinc-400">{m.latency_ms.toFixed(0)}</td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
        <p class="mt-2 text-xs text-zinc-500">Values are %; top row (green) leads on mAP@.5:.95.</p>
      {:else if selected}
        <p class="text-sm text-zinc-500">No comparison for {selected} yet (still running?).</p>
      {:else}
        <p class="text-sm text-zinc-500">Select a run to view its ranked comparison.</p>
      {/if}
    </main>
  </div>
</div>
