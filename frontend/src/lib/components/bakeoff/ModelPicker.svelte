<script lang="ts">
  import { apiErrorText } from '$lib/api';
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
    <p class="text-[10px] uppercase tracking-wide text-zinc-500">Trained runs</p>
    {#each trained as t (t.run_id)}
      <div
        class="rounded px-1 py-1 text-xs hover:bg-zinc-800/50"
        data-testid="run-row"
        data-run-id={t.run_id}
      >
        <label class="flex flex-wrap items-center gap-1.5">
          <input
            type="checkbox"
            checked={selectedRuns.includes(t.run_id)}
            onchange={(e) => onToggleRun(t.run_id, e.currentTarget.checked)}
          />
          <span class="font-mono text-zinc-200">{t.display_name}</span>
          {#if t.model_family || t.model_size}
            <span class="rounded bg-blue-900/60 px-1 text-[10px] text-blue-200"
              >{t.model_family ?? ''}{t.model_size ?? ''}</span
            >
          {/if}
          <span class="text-[10px] text-zinc-500">{t.imgsz}px</span>
          <span class="text-[10px] text-zinc-500">
            {t.class_names.length} classes{t.single_cls ? ' (single-class)' : ''}
          </span>
          <span
            class="text-[10px] text-zinc-500"
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
          <div class="ml-6 text-[10px] text-zinc-500" data-testid="run-facts">
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
                  class="ml-1 rounded bg-red-900/60 px-1 text-red-200"
                  data-testid="overlap-warning"
                  >overlap: {formatOverlap(f.train_test_overlap)}</span
                >
              {/if}
            {/if}
          </div>
        {/each}
      </div>
    {:else}
      <p class="px-1 text-xs text-zinc-600">
        No finished training run with a checkpoint.
      </p>
    {/each}
  </div>

  <div>
    <p class="text-[10px] uppercase tracking-wide text-zinc-500">Baselines</p>
    {#each baselines as b (b.name)}
      <label
        class="flex items-center gap-1.5 rounded px-1 py-1 text-xs hover:bg-zinc-800/50"
        data-testid="baseline-row"
      >
        <input
          type="checkbox"
          checked={selectedBaselines.includes(b.name)}
          onchange={(e) => onToggleBaseline(b.name, e.currentTarget.checked)}
        />
        <span class="font-mono text-zinc-200">{b.name}</span>
        <span class="rounded bg-zinc-800 px-1 text-[10px] text-zinc-400">{b.backend}</span
        >
        {#if b.training_data}
          <span class="text-[10px] text-zinc-500">trained on {b.training_data}</span>
        {/if}
      </label>
    {:else}
      <p class="px-1 text-xs text-zinc-600">This profile registers no baselines.</p>
    {/each}
  </div>

  {#if customRefs.length}
    <div>
      <p class="text-[10px] uppercase tracking-wide text-zinc-500">Custom</p>
      {#each customRefs as c, i (i)}
        <div class="flex items-center gap-1.5 px-1 py-1 text-xs">
          <span class="font-mono text-zinc-200">{c.display_name ?? c.name}</span>
          <span class="rounded bg-zinc-800 px-1 text-[10px] text-zinc-400"
            >{c.backend}</span
          >
          <button
            type="button"
            class="text-[10px] text-zinc-500 hover:text-zinc-300"
            onclick={() => onRemoveCustom(i)}>remove</button
          >
        </div>
      {/each}
    </div>
  {/if}

  <details class="rounded border border-zinc-800 px-2 py-1 text-xs">
    <summary class="cursor-pointer text-zinc-400">Add a custom model</summary>
    <div class="mt-2 grid grid-cols-2 gap-2">
      <label class="flex flex-col gap-0.5"
        ><span class="text-zinc-500">Name</span>
        <input class="input-sm" bind:value={cName} /></label
      >
      <label class="flex flex-col gap-0.5"
        ><span class="text-zinc-500">Backend</span>
        <input class="input-sm" bind:value={cBackend} /></label
      >
      <label class="flex flex-col gap-0.5"
        ><span class="text-zinc-500">Weights path (evaluator)</span>
        <input class="input-sm" bind:value={cWeights} /></label
      >
      <label class="flex flex-col gap-0.5"
        ><span class="text-zinc-500">Served model name</span>
        <input class="input-sm" bind:value={cTriton} /></label
      >
      <label class="flex flex-col gap-0.5"
        ><span class="text-zinc-500">Image size</span>
        <input
          class="input-sm"
          type="number"
          min="32"
          step="32"
          bind:value={cImgsz}
        /></label
      >
      <label class="flex flex-col gap-0.5"
        ><span class="text-zinc-500">Display name</span>
        <input class="input-sm" bind:value={cDisplay} /></label
      >
      <label class="col-span-2 flex flex-col gap-0.5"
        ><span class="text-zinc-500"
          >Class map (JSON, model class id → eval class name; empty = match by name)</span
        >
        <textarea class="input-sm font-mono" rows="2" bind:value={cClassMap}
        ></textarea></label
      >
    </div>
    {#if customError}<p class="mt-1 text-red-300">{customError}</p>{/if}
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
