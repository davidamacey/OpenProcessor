<script lang="ts">
  import { apiErrorText } from '$lib/api_errors';
  /**
   * Contender picker for `/bakeoff`: finished training runs (with the
   * served per-dataset facts for every selected dataset), the profile's
   * baselines, and an optional custom model reference.
   */
  import type {
    BaselineModel,
    CustomModelRef,
    TrainedModel,
    TrainedModelForDataset,
  } from '$lib/types_bakeoff';
  import {
    formatOverlap,
    hasOverlap,
    MISSING,
    parseClassMap,
    TRAINER_PROTOCOL_LABEL,
  } from '$lib/bakeoff/view';

  interface Props {
    trained: TrainedModel[];
    baselines: BaselineModel[];
    customRefs: CustomModelRef[];
    selectedRuns: string[];
    selectedBaselines: string[];
    selectedDatasets: string[];
    facts: Record<string, Record<string, TrainedModelForDataset | null>>;
    datasetName: (id: string) => string;
    onToggleRun: (runId: string, on: boolean) => void;
    onToggleBaseline: (name: string, on: boolean) => void;
    onAddCustom: (ref: CustomModelRef) => void;
    onRemoveCustom: (index: number) => void;
  }
  let {
    trained,
    baselines,
    customRefs,
    selectedRuns,
    selectedBaselines,
    selectedDatasets,
    facts,
    datasetName,
    onToggleRun,
    onToggleBaseline,
    onAddCustom,
    onRemoveCustom,
  }: Props = $props();

  let cName = $state('');
  let cBackend = $state('');
  let cWeights = $state('');
  let cTriton = $state('');
  let cImgsz = $state<number | null>(null);
  let cDisplay = $state('');
  let cClassMap = $state('');
  let customError = $state<string | null>(null);

  function addCustom() {
    customError = null;
    if (!cName.trim() || !cBackend.trim()) {
      customError = 'name and backend are required';
      return;
    }
    let classMap: Record<string, string> | undefined;
    try {
      classMap = parseClassMap(cClassMap);
    } catch (e) {
      customError = apiErrorText(e);
      return;
    }
    const ref: CustomModelRef = {
      source: 'custom',
      name: cName.trim(),
      backend: cBackend.trim(),
    };
    if (cWeights.trim()) ref.weights = cWeights.trim();
    if (cTriton.trim()) ref.triton_model = cTriton.trim();
    if (typeof cImgsz === 'number' && Number.isFinite(cImgsz)) ref.imgsz = cImgsz;
    if (cDisplay.trim()) ref.display_name = cDisplay.trim();
    if (classMap) ref.class_map = classMap;
    onAddCustom(ref);
    cName = cBackend = cWeights = cTriton = cDisplay = cClassMap = '';
    cImgsz = null;
  }

  function sameFrozen(f: TrainedModelForDataset): string {
    if (f.same_frozen_test === null) return 'frozen test unknown';
    return f.same_frozen_test ? 'same frozen test' : 'different frozen test';
  }
</script>

<div class="space-y-3" data-testid="model-picker">
  <div>
    <p class="text-meta text-fg-2 font-semibold">Trained runs</p>
    {#each trained as t (t.run_id)}
      <div
        class="rounded px-1 py-1 text-meta hover:bg-surface-2/50"
        data-testid="run-row"
        data-run-id={t.run_id}
      >
        <label class="flex flex-wrap items-center gap-1.5">
          <input
            type="checkbox"
            checked={selectedRuns.includes(t.run_id)}
            onchange={(e) => onToggleRun(t.run_id, e.currentTarget.checked)}
          />
          <span class="font-mono text-fg-1">{t.display_name}</span>
          {#if t.model_family || t.model_size}
            <span class="rounded bg-accent-soft px-1 text-caption text-accent-text"
              >{t.model_family ?? ''}{t.model_size ?? ''}</span
            >
          {/if}
          <span class="text-caption text-fg-3">{t.imgsz}px</span>
          <span class="text-caption text-fg-3">
            {t.class_names.length} classes{t.single_cls ? ' (single-class)' : ''}
          </span>
          <span
            class="text-caption text-fg-3"
            title="The trainer's own number from its own evaluation (Ultralytics val defaults) — not a comparison metric."
          >
            {TRAINER_PROTOCOL_LABEL} mAP50 {t.trainer_map50 == null
              ? MISSING
              : t.trainer_map50.toFixed(3)}{t.trainer_map50_split
              ? ` (${t.trainer_map50_split})`
              : ''}
          </span>
        </label>
        {#each selectedDatasets as dsId (dsId)}
          {@const f = facts[dsId]?.[t.run_id]}
          <div class="ml-6 text-caption text-fg-3" data-testid="run-facts">
            <span class="font-mono">{datasetName(dsId)}</span>:
            {#if facts[dsId] === undefined}
              loading…
            {:else if !f}
              {MISSING}
            {:else}
              {f.same_export ? 'same export' : 'different export'} · {sameFrozen(f)} ·
              {f.n_classes_mapped} classes mapped
              {#if hasOverlap(f.train_test_overlap)}
                <span
                  class="ml-1 rounded bg-danger-soft px-1 text-danger-text"
                  data-testid="overlap-warning"
                  >overlap: {formatOverlap(f.train_test_overlap)}</span
                >
              {/if}
            {/if}
          </div>
        {/each}
      </div>
    {:else}
      <p class="px-1 text-meta text-fg-3">No finished training run with a checkpoint.</p>
    {/each}
  </div>

  <div>
    <p class="text-meta text-fg-2 font-semibold">Baselines</p>
    {#each baselines as b (b.name)}
      <label
        class="flex items-center gap-1.5 rounded px-1 py-1 text-meta hover:bg-surface-2/50"
        data-testid="baseline-row"
      >
        <input
          type="checkbox"
          checked={selectedBaselines.includes(b.name)}
          onchange={(e) => onToggleBaseline(b.name, e.currentTarget.checked)}
        />
        <span class="font-mono text-fg-1">{b.name}</span>
        <span class="rounded bg-surface-2 px-1 text-caption text-fg-2">{b.backend}</span>
        {#if b.training_data}
          <span class="text-caption text-fg-3">trained on {b.training_data}</span>
        {/if}
      </label>
    {:else}
      <p class="px-1 text-meta text-fg-3">This profile registers no baselines.</p>
    {/each}
  </div>

  {#if customRefs.length}
    <div>
      <p class="text-meta text-fg-2 font-semibold">Custom</p>
      {#each customRefs as c, i (i)}
        <div class="flex items-center gap-1.5 px-1 py-1 text-meta">
          <span class="font-mono text-fg-1">{c.display_name ?? c.name}</span>
          <span class="rounded bg-surface-2 px-1 text-caption text-fg-2">{c.backend}</span
          >
          <button
            type="button"
            class="text-caption text-fg-3 hover:text-fg-2"
            onclick={() => onRemoveCustom(i)}>remove</button
          >
        </div>
      {/each}
    </div>
  {/if}

  <details class="rounded border border-line px-2 py-1 text-meta">
    <summary class="cursor-pointer text-fg-2">Add a custom model</summary>
    <div class="mt-2 grid grid-cols-2 gap-2">
      <label class="flex flex-col gap-0.5"
        ><span class="text-fg-3">Name</span>
        <input class="input-sm" bind:value={cName} /></label
      >
      <label class="flex flex-col gap-0.5"
        ><span class="text-fg-3">Backend</span>
        <input class="input-sm" bind:value={cBackend} /></label
      >
      <label class="flex flex-col gap-0.5"
        ><span class="text-fg-3">Weights path (evaluator)</span>
        <input class="input-sm" bind:value={cWeights} /></label
      >
      <label class="flex flex-col gap-0.5"
        ><span class="text-fg-3">Served model name</span>
        <input class="input-sm" bind:value={cTriton} /></label
      >
      <label class="flex flex-col gap-0.5"
        ><span class="text-fg-3">Image size</span>
        <input
          class="input-sm"
          type="number"
          min="32"
          step="32"
          bind:value={cImgsz}
        /></label
      >
      <label class="flex flex-col gap-0.5"
        ><span class="text-fg-3">Display name</span>
        <input class="input-sm" bind:value={cDisplay} /></label
      >
      <label class="col-span-2 flex flex-col gap-0.5"
        ><span class="text-fg-3"
          >Class map (JSON, model class id → eval class name; empty = match by name)</span
        >
        <textarea class="input-sm font-mono" rows="2" bind:value={cClassMap}
        ></textarea></label
      >
    </div>
    {#if customError}<p class="mt-1 text-danger-text">{customError}</p>{/if}
    <button type="button" class="btn mt-2" onclick={addCustom}>Add</button>
  </details>
</div>

<style>
  .input-sm {
    border-radius: 0.25rem;
    border: 1px solid rgb(63 63 70);
    background: rgb(9 9 11);
    padding: 0.125rem 0.375rem;
    color: rgb(228 228 231);
  }
</style>
