<script lang="ts">
  /**
   * Finished-run results — evaluation (labelled by its own `eval.split`)
   * + lineage for a terminal `/train` run. Collapsed by default; the
   * manifest (lineage: dataset/test-split SHAs, class remap, code
   * versions) is fetched lazily the first time the section is opened.
   * Everything else (last-epoch/best-checkpoint training metrics, eval,
   * mlflow, checkpoint) is already on the served `TrainJobStatus` the
   * caller hands in, so it renders immediately.
   *
   * Thin frontend: every value here is rendered exactly as served —
   * no client-side metric computation. `formatMetric`/`formatScalar`
   * (`$lib/trainResults.ts`) turn a missing value into "—", never a
   * false 0.
   */
  import { untrack } from 'svelte';
  import { getTrainManifest, resolveApiUrl } from '$lib/api';
  import {
    evalOverallLabel,
    evalPerClassLabel,
    formatMetric,
    formatScalar,
    isTerminalTrainState,
    metricEpochLabel,
  } from '$lib/trainResults';
  import { TRAINER_PROTOCOL_LABEL } from '$lib/bakeoff/view';
  import { formatCount } from '$lib/formatCount';
  import { focusOnMount } from '$lib/actions/focusOnMount';
  import { trapFocus } from '$lib/actions/trapFocus';
  import ProbeControl from './ProbeControl.svelte';
  import type { TrainJobStatus, TrainManifest } from '$lib/types_train';

  interface Props {
    status: TrainJobStatus;
    /** Starts expanded — used by a "just finished" surface that wants
     *  the section open without a click. Defaults closed (past-runs
     *  table row). */
    startOpen?: boolean;
  }

  let { status, startOpen = false }: Props = $props();

  // Deliberately captures only the initial `startOpen` value — it sets
  // the starting open/closed state, not a live-tracked prop.
  let open = $state(untrack(() => startOpen));
  let manifest = $state<TrainManifest | null>(null);
  // Confusion-matrix lightbox — thumbnail-only in the panel, opened full
  // size on click (see the "Confusion matrix" section below).
  let matrixLightboxOpen = $state(false);
  let manifestLoading = $state(false);
  let manifestError = $state<string | null>(null);
  let manifestRequested = false;

  async function loadManifest(): Promise<void> {
    if (manifestRequested) return;
    manifestRequested = true;
    manifestLoading = true;
    manifestError = null;
    try {
      manifest = await getTrainManifest(status.job_id);
    } catch (e) {
      manifestError = (e as Error).message;
    } finally {
      manifestLoading = false;
    }
  }

  function toggle(): void {
    open = !open;
    if (open) void loadManifest();
  }

  $effect(() => {
    if (startOpen) void loadManifest();
  });

  const isTerminal = $derived(isTerminalTrainState(status.state));
  const evalData = $derived(status.eval ?? manifest?.results?.eval ?? null);
  const perClass = $derived(evalData?.per_class ?? []);
  const lastEpochMetric = $derived(
    status.last_epoch_metric ?? manifest?.results?.last_epoch_metric ?? null,
  );
  const bestCheckpointMetric = $derived(
    status.best_checkpoint_metric ?? manifest?.results?.best_checkpoint_metric ?? null,
  );
  const checkpointSha = $derived(
    status.checkpoint_sha256 ?? manifest?.results?.checkpoint_sha256 ?? null,
  );
  const mlflowRunId = $derived(
    status.mlflow_run_id ?? manifest?.results?.mlflow_run_id ?? null,
  );
  const mlflowRunUrl = $derived(
    status.mlflow_run_url ?? manifest?.results?.mlflow_run_url ?? null,
  );

  const lineage = $derived(manifest?.lineage ?? null);
  const codeVersions = $derived(manifest?.code_versions ?? null);
  const classRemap = $derived(lineage?.class_remap ?? null);

  /** `{new_to_original, names}` -> rows of `{newId, originalId, name}`,
   *  sorted by new id — served verbatim, no client remap math. */
  const remapRows = $derived.by(() => {
    if (!classRemap?.new_to_original) return [];
    return Object.entries(classRemap.new_to_original)
      .map(([newId, originalId]) => ({
        newId: Number(newId),
        originalId,
        name: classRemap.names?.[Number(newId)] ?? null,
      }))
      .sort((a, b) => a.newId - b.newId);
  });
</script>

<div class="border-t border-zinc-800">
  <button
    type="button"
    class="flex w-full items-center gap-1.5 px-3 py-2 text-left text-xs text-zinc-300 hover:bg-zinc-800/50"
    onclick={toggle}
    aria-expanded={open}
  >
    <span class="font-mono">{open ? '▾' : '▸'}</span>
    Results
  </button>

  {#if open}
    <div class="space-y-4 border-t border-zinc-800 bg-zinc-950/40 px-3 py-3 text-xs">
      <!-- T5 (visual audit 2026-09-24): the sticky table header can cover
           the run's own row while this panel is open, so name the run here. -->
      <p class="font-mono text-[11px] text-zinc-300" data-testid="run-results-title">
        Results for {status.job_id}
      </p>
      {#if status.error}
        <p
          class="rounded border border-red-500/30 bg-red-500/10 px-2 py-1.5 text-red-200"
        >
          {status.error}
        </p>
      {/if}

      <!-- #36 item 8: populates probe_pred_* from this run's checkpoint —
           the one prerequisite the Uncertainty/Model-disagreements review
           queues need (item 9's empty_state.has_probe_predictions). -->
      <ProbeControl {status} />

      <!-- Training-time metrics (OpenProcessor #34 W1). Neither of these
           is the run's headline number — that's eval.map50 below,
           labelled by eval.split. `last_epoch_metric` is the true last
           TRAINING epoch's own validation pass; `best_checkpoint_metric`
           is the best checkpoint's (best.pt) own re-validation, which
           Ultralytics only performs once, after training ends — both
           are one coherent row (mAP50 + mAP50-95 from the SAME pass),
           never a per-key max spanning different epochs. `null` on a
           run whose status.json predates these fields (no incorrect
           back-fill) renders "—", never a guessed value. -->
      <section>
        <h3 class="mb-1 text-[11px] uppercase tracking-wide text-zinc-500">
          Metrics — training epochs
        </h3>
        <div class="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <div>
            <p class="mb-1 text-[10px] uppercase tracking-wide text-zinc-600">
              last epoch{#if metricEpochLabel(lastEpochMetric)}
                ({metricEpochLabel(lastEpochMetric)}){/if}
            </p>
            <div class="grid grid-cols-2 gap-2">
              <div>
                <dt class="text-zinc-500">mAP50</dt>
                <dd class="font-mono text-zinc-100">
                  {formatMetric(lastEpochMetric?.map50)}
                </dd>
              </div>
              <div>
                <dt class="text-zinc-500">mAP50-95</dt>
                <dd class="font-mono text-zinc-100">
                  {formatMetric(lastEpochMetric?.map50_95)}
                </dd>
              </div>
            </div>
          </div>
          <div>
            <p
              class="mb-1 text-[10px] uppercase tracking-wide text-zinc-600"
              title="Ultralytics re-validates the best checkpoint once more after training completes — not the same pass as the frozen-holdout eval below"
            >
              best checkpoint{#if metricEpochLabel(bestCheckpointMetric)}
                ({metricEpochLabel(bestCheckpointMetric)}){/if}
            </p>
            <div class="grid grid-cols-2 gap-2">
              <div>
                <dt class="text-zinc-500">mAP50</dt>
                <dd class="font-mono text-zinc-100">
                  {formatMetric(bestCheckpointMetric?.map50)}
                </dd>
              </div>
              <div>
                <dt class="text-zinc-500">mAP50-95</dt>
                <dd class="font-mono text-zinc-100">
                  {formatMetric(bestCheckpointMetric?.map50_95)}
                </dd>
              </div>
            </div>
          </div>
        </div>
      </section>

      <!-- Eval: overall + per-class, labelled by whichever pass actually
           produced each half (see TrainEval's doc comment). -->
      <section>
        <h3 class="mb-1 text-[11px] uppercase tracking-wide text-zinc-500">Evaluation</h3>
        {#if evalData}
          <p class="mb-2 text-zinc-400">
            overall: <span class="text-zinc-200">{evalOverallLabel(evalData)}</span>
            <span
              class="ml-1 text-[10px] text-zinc-500"
              data-testid="eval-protocol"
              title="Ultralytics' own val protocol and defaults. A bake-off scores the same split with its profile's served thresholds, so its mAP differs."
              >({TRAINER_PROTOCOL_LABEL})</span
            >
            <span class="ml-3 font-mono text-zinc-100"
              >mAP50 {formatMetric(evalData.map50)}</span
            >
            <span class="ml-2 font-mono text-zinc-100"
              >mAP50-95 {formatMetric(evalData.map50_95)}</span
            >
            {#if evalData.precision != null}
              <span class="ml-2 font-mono text-zinc-100"
                >precision {formatMetric(evalData.precision)}</span
              >
            {/if}
            {#if evalData.recall != null}
              <span class="ml-2 font-mono text-zinc-100"
                >recall {formatMetric(evalData.recall)}</span
              >
            {/if}
            {#if evalData.head}
              <span class="ml-2 text-zinc-500">head: {evalData.head}</span>
            {/if}
          </p>
          {#if evalData.val_last}
            <p class="mb-2 text-zinc-400" data-testid="eval-val-last">
              validation (last epoch):
              <span class="ml-1 font-mono text-zinc-100"
                >mAP50 {formatMetric(evalData.val_last.map50)}</span
              >
              <span class="ml-2 font-mono text-zinc-100"
                >mAP50-95 {formatMetric(evalData.val_last.map50_95)}</span
              >
            </p>
          {/if}
          {#if perClass.length > 0}
            <p class="mb-1 text-zinc-400">
              per-class: <span class="text-zinc-200">{evalPerClassLabel(evalData)}</span>
            </p>
            <div class="overflow-auto rounded border border-zinc-800">
              <table class="w-full text-xs">
                <thead
                  class="border-b border-zinc-800 bg-zinc-950 text-left uppercase text-zinc-500"
                >
                  <tr>
                    <th class="px-2 py-1 font-medium">Class</th>
                    <th class="px-2 py-1 text-right font-medium">Precision</th>
                    <th class="px-2 py-1 text-right font-medium">Recall</th>
                    <th class="px-2 py-1 text-right font-medium">F1</th>
                    <th class="px-2 py-1 text-right font-medium">AP50</th>
                    <th class="px-2 py-1 text-right font-medium">Support</th>
                  </tr>
                </thead>
                <tbody>
                  {#each perClass as row (row.class_id)}
                    <tr class="border-b border-zinc-900 text-zinc-300 last:border-b-0">
                      <td class="px-2 py-1">{row.name}</td>
                      <td class="px-2 py-1 text-right font-mono"
                        >{formatMetric(row.precision)}</td
                      >
                      <td class="px-2 py-1 text-right font-mono"
                        >{formatMetric(row.recall)}</td
                      >
                      <td class="px-2 py-1 text-right font-mono"
                        >{formatMetric(row.f1)}</td
                      >
                      <td class="px-2 py-1 text-right font-mono"
                        >{formatMetric(row.ap50)}</td
                      >
                      <td class="px-2 py-1 text-right font-mono"
                        >{formatCount(row.support)}</td
                      >
                    </tr>
                  {/each}
                </tbody>
              </table>
            </div>
          {/if}

          <!-- Confusion matrix — image only from the servable URL, never
               the server filesystem path. Rendered as a bounded
               thumbnail (it used to render at full panel width, ~1094×
               821 at 1600px, dwarfing the rest of the past-runs row) —
               click opens it full size in a lightbox. -->
          <div class="mt-2">
            {#if evalData.confusion_matrix_url}
              <button
                type="button"
                class="block max-w-full cursor-zoom-in rounded border border-zinc-800 p-0"
                onclick={() => (matrixLightboxOpen = true)}
                aria-label="Enlarge confusion matrix"
              >
                <img
                  src={resolveApiUrl(evalData.confusion_matrix_url)}
                  alt="Confusion matrix (click to enlarge)"
                  class="max-h-64 max-w-full rounded object-contain"
                />
              </button>
            {:else if evalData.confusion_matrix_path}
              <p class="text-zinc-500">
                confusion matrix (server path, not viewable here):
                <span class="ml-1 break-all font-mono text-zinc-400"
                  >{evalData.confusion_matrix_path}</span
                >
              </p>
            {:else}
              <p class="text-zinc-500">confusion matrix: —</p>
            {/if}
          </div>
        {:else}
          <p class="text-zinc-500">—</p>
        {/if}
      </section>

      <!-- MLflow + checkpoint -->
      <section class="grid grid-cols-1 gap-2 sm:grid-cols-2">
        <div>
          <dt class="text-[11px] uppercase tracking-wide text-zinc-500">MLflow run</dt>
          <dd class="break-all font-mono text-zinc-100">
            {#if mlflowRunUrl}
              <!-- eslint-disable svelte/no-navigation-without-resolve -- external MLflow server URL, not a SvelteKit route -->
              <a
                href={mlflowRunUrl}
                target="_blank"
                rel="noopener"
                class="text-blue-300 underline">{mlflowRunUrl}</a
              >
              <!-- eslint-enable svelte/no-navigation-without-resolve -->
              <!-- TODO: backend is being asked to serve mlflow_run_url as
                   null unless OP_MLFLOW_PUBLIC_URL is set (never the
                   docker-internal hostname) — once that lands every
                   non-null value here is guaranteed browser-reachable. -->
            {:else if mlflowRunId}
              {mlflowRunId}
              {#if !isTerminal}
                <span class="ml-1 text-zinc-500">(url pending)</span>
              {/if}
            {:else if !isTerminal}
              <span class="text-zinc-500">pending</span>
            {:else}
              —
            {/if}
          </dd>
        </div>
        <div>
          <dt class="text-[11px] uppercase tracking-wide text-zinc-500">
            Checkpoint SHA-256
          </dt>
          <dd class="break-all font-mono text-zinc-100">{formatScalar(checkpointSha)}</dd>
        </div>
      </section>

      <!-- Lineage (manifest) -->
      <section>
        <h3 class="mb-1 text-[11px] uppercase tracking-wide text-zinc-500">
          Lineage (manifest)
        </h3>
        {#if manifestLoading}
          <p class="text-zinc-500">Loading manifest…</p>
        {:else if manifestError}
          <p class="text-red-300">Manifest fetch failed: {manifestError}</p>
        {:else if !manifest}
          <p class="text-zinc-500">—</p>
        {:else}
          <dl class="grid grid-cols-1 gap-2 sm:grid-cols-2">
            <div>
              <dt class="text-zinc-500">export_dir</dt>
              <dd class="break-all font-mono text-zinc-100">
                {formatScalar(lineage?.export_dir)}
              </dd>
            </div>
            <div>
              <dt class="text-zinc-500">dataset_sha</dt>
              <dd class="break-all font-mono text-zinc-100">
                {formatScalar(lineage?.dataset_sha)}
              </dd>
            </div>
            <div>
              <dt class="text-zinc-500">dataset_version_tag</dt>
              <dd class="break-all font-mono text-zinc-100">
                {formatScalar(lineage?.dataset_version_tag)}
              </dd>
            </div>
            <div>
              <dt class="text-zinc-500">frozen_test_sha</dt>
              <dd class="break-all font-mono text-zinc-100">
                {formatScalar(lineage?.frozen_test_sha)}
              </dd>
            </div>
            <div>
              <dt class="text-zinc-500">test_label_sha</dt>
              <dd class="break-all font-mono text-zinc-100">
                {formatScalar(lineage?.test_label_sha)}
              </dd>
            </div>
            <div>
              <dt class="text-zinc-500">include_classes</dt>
              <dd class="break-all font-mono text-zinc-100">
                {lineage?.include_classes && lineage.include_classes.length > 0
                  ? lineage.include_classes.join(', ')
                  : '—'}
              </dd>
            </div>
            <div>
              <dt class="text-zinc-500">training_seed</dt>
              <dd class="font-mono text-zinc-100">
                {formatScalar(lineage?.training_seed)}
              </dd>
            </div>
            <div>
              <dt class="text-zinc-500">code: api_sha</dt>
              <dd class="break-all font-mono text-zinc-100">
                {formatScalar(codeVersions?.api_sha)}
              </dd>
            </div>
            <div>
              <dt class="text-zinc-500">code: trainer_sha</dt>
              <dd class="break-all font-mono text-zinc-100">
                {formatScalar(codeVersions?.trainer_sha)}
              </dd>
            </div>
            <div>
              <dt class="text-zinc-500">code: trainer_image_id</dt>
              <dd class="break-all font-mono text-zinc-100">
                {formatScalar(codeVersions?.trainer_image_id)}
              </dd>
            </div>
          </dl>

          {#if remapRows.length > 0}
            <div class="mt-2 overflow-auto rounded border border-zinc-800">
              <table class="w-full text-xs">
                <thead
                  class="border-b border-zinc-800 bg-zinc-950 text-left uppercase text-zinc-500"
                >
                  <tr>
                    <th class="px-2 py-1 font-medium">New ID</th>
                    <th class="px-2 py-1 font-medium">Original ID</th>
                    <th class="px-2 py-1 font-medium">Name</th>
                  </tr>
                </thead>
                <tbody>
                  {#each remapRows as row (row.newId)}
                    <tr class="border-b border-zinc-900 text-zinc-300 last:border-b-0">
                      <td class="px-2 py-1 font-mono">{row.newId}</td>
                      <td class="px-2 py-1 font-mono">{formatScalar(row.originalId)}</td>
                      <td class="px-2 py-1">{formatScalar(row.name)}</td>
                    </tr>
                  {/each}
                </tbody>
              </table>
            </div>
          {/if}
        {/if}
      </section>
    </div>
  {/if}
</div>

{#if matrixLightboxOpen && evalData?.confusion_matrix_url}
  <!-- svelte-ignore a11y_click_events_have_key_events -->
  <div
    class="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4"
    role="dialog"
    aria-modal="true"
    aria-label="Confusion matrix"
    use:focusOnMount
    use:trapFocus={{ onEscape: () => (matrixLightboxOpen = false) }}
    onclick={(e) => {
      if (e.target === e.currentTarget) matrixLightboxOpen = false;
    }}
    tabindex="-1"
  >
    <div class="relative max-h-full max-w-6xl">
      <img
        src={resolveApiUrl(evalData.confusion_matrix_url)}
        alt="Confusion matrix, full size"
        class="max-h-[90vh] max-w-full rounded-md border border-zinc-700 object-contain"
      />
      <button
        type="button"
        class="absolute -top-3 -right-3 rounded-full border border-zinc-700 bg-zinc-900 px-2 py-1 text-sm text-white"
        onclick={(e) => {
          e.stopPropagation();
          matrixLightboxOpen = false;
        }}
        aria-label="Close"
      >
        ×
      </button>
    </div>
  </div>
{/if}
