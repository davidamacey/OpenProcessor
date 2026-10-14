<script lang="ts">
  /** /train's dataset card (which frozen export the run trains on) and,
   *  when the backend serves it, the single-class dataset export panel. */
  import { formatCount } from '$lib/format/count';
  import { classesStore } from '$stores/classes.svelte';
  import { datasetLabel } from '$lib/train/trainDisplay';
  import {
    MULTI_CLASS,
    type TrainDatasetController,
  } from '$lib/train/trainDatasetController.svelte';
  import type { TrainSingleClassExport } from '$lib/train/trainSingleClassExport.svelte';

  interface Props {
    dataset: TrainDatasetController;
    exporter: TrainSingleClassExport;
  }

  let { dataset, exporter }: Props = $props();
</script>

<!-- Dataset header -->
<section class="rounded-md border border-zinc-800 bg-zinc-900 p-4">
  <div class="flex items-center justify-between gap-3">
    <h2 class="text-[11px] uppercase tracking-wide text-zinc-500">Dataset</h2>
    <div class="flex gap-1 text-xs">
      <button
        type="button"
        class="rounded border px-2 py-0.5 {dataset.datasetKind === MULTI_CLASS
          ? 'border-blue-500 bg-blue-950 text-blue-200'
          : 'border-zinc-700 bg-zinc-950 text-zinc-400 hover:bg-zinc-800'}"
        onclick={() => dataset.selectDatasetKind(MULTI_CLASS)}
      >
        Multi-class
      </button>
      {#if dataset.datasetExportAvailable}
        <button
          type="button"
          class="rounded border px-2 py-0.5 {dataset.datasetKind ===
          dataset.datasetExportSpec?.datasetKind
            ? 'border-blue-500 bg-blue-950 text-blue-200'
            : 'border-zinc-700 bg-zinc-950 text-zinc-400 hover:bg-zinc-800'}"
          onclick={() =>
            dataset.selectDatasetKind(dataset.datasetExportSpec!.datasetKind)}
        >
          {dataset.datasetExportSpec!.label}{dataset.datasetExportSpec!.singleClass
            ? ' (single-class)'
            : ''}
        </button>
      {/if}
    </div>
  </div>
  {#if dataset.kindDatasets.length > 0}
    <label class="mt-3 block">
      <span class="mb-1 block text-xs text-zinc-400">
        dataset version ({dataset.kindDatasets.length} available — pick a sample, subset, or
        the full set)
      </span>
      <select
        bind:value={dataset.selectedExportDir}
        class="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 font-mono text-xs text-zinc-100 focus:border-blue-500 focus:outline-none"
      >
        <option value="">
          current ({dataset.datasetKind === dataset.datasetExportSpec?.datasetKind
            ? `${dataset.datasetExportSpec.profileName} current`
            : 'current'} symlink — latest)
        </option>
        {#each dataset.kindDatasets as d (d.export_dir)}
          <option value={d.export_dir}>{datasetLabel(d)}</option>
        {/each}
      </select>
    </label>
  {/if}
  {#if dataset.datasetExportDir}
    <p class="mt-1 break-all font-mono text-sm text-zinc-200">
      {dataset.datasetExportDir}
    </p>
    {#if dataset.datasetKind === dataset.datasetExportSpec?.datasetKind}
      <p class="mt-2 text-xs text-zinc-400">
        {dataset.datasetExportSpec.blurb}
      </p>
    {:else if dataset.showsCurrentExport && dataset.multiClassExportState?.class_split_counts}
      <!-- Current export's own contents (OpenProcessor df01309's
           GET {API_PREFIX}/export/status) — only valid for the current
           export, so this branch renders for the `current` symlink or an
           explicit pick of that same directory, never a past version. m-train-card (2026-09-24): this used to show the
           classesStore-wide validated total (which counted
           test_holdout crops too), a different — and often much
           bigger — number than what this specific export actually
           contains. -->
      <div class="mt-2 flex flex-wrap gap-2 text-xs text-zinc-400">
        {#if dataset.multiClassExportState.class_count != null}
          <span class="rounded border border-zinc-700 bg-zinc-950 px-1.5 py-0.5">
            {#if dataset.multiClassExportState.classes_with_objects != null}
              {dataset.multiClassExportState.classes_with_objects} of {dataset
                .multiClassExportState.class_count}
              classes with objects
            {:else}
              {dataset.multiClassExportState.class_count} classes
            {/if}
          </span>
        {/if}
        {#if dataset.multiClassExportState.image_count != null || dataset.multiClassExportState.object_count != null}
          <span
            class="rounded border border-zinc-700 bg-zinc-950 px-1.5 py-0.5 font-mono"
          >
            {formatCount(dataset.multiClassExportState.object_count)} objects in {formatCount(
              dataset.multiClassExportState.image_count,
            )} images
            {#if dataset.multiClassExportState.group_key}
              <span class="text-zinc-500"
                >(by {dataset.multiClassExportState.group_key})</span
              >
            {/if}
          </span>
        {/if}
        {#if dataset.multiClassExportState.split_counts}
          <span
            class="rounded border border-zinc-700 bg-zinc-950 px-1.5 py-0.5 font-mono"
            title="Images per split"
          >
            images: train {dataset.multiClassExportState.split_counts.train.toLocaleString()}
            · val
            {dataset.multiClassExportState.split_counts.val.toLocaleString()}
            · test {dataset.multiClassExportState.split_counts.test.toLocaleString()}
          </span>
        {/if}
        {#if dataset.multiClassExportState.split_object_counts}
          <span
            class="rounded border border-zinc-700 bg-zinc-950 px-1.5 py-0.5 font-mono"
            title="Objects (label lines) per split"
          >
            objects: train {dataset.multiClassExportState.split_object_counts.train.toLocaleString()}
            · val {dataset.multiClassExportState.split_object_counts.val.toLocaleString()}
            · test {dataset.multiClassExportState.split_object_counts.test.toLocaleString()}
          </span>
        {/if}
      </div>
      <details class="mt-2 text-xs text-zinc-400">
        <summary class="cursor-pointer hover:text-zinc-200">
          per-class object counts ({dataset.multiClassExportState.class_split_counts
            .length})
        </summary>
        <div class="mt-1 max-h-48 overflow-auto rounded border border-zinc-800">
          <table class="w-full text-xs">
            <thead
              class="sticky top-0 border-b border-zinc-800 bg-zinc-950 text-left uppercase text-zinc-500"
            >
              <tr>
                <th class="px-2 py-1 font-medium">Class</th>
                <th class="px-2 py-1 text-right font-medium">Train (objects)</th>
                <th class="px-2 py-1 text-right font-medium">Val (objects)</th>
                <th class="px-2 py-1 text-right font-medium">Test (objects)</th>
              </tr>
            </thead>
            <tbody>
              {#each dataset.multiClassExportState.class_split_counts as c (c.class_id)}
                {@const missing = c.train === 0 || c.val === 0}
                <tr
                  class="border-b border-zinc-900 {missing
                    ? 'bg-red-500/10 text-red-200'
                    : 'text-zinc-300'}"
                >
                  <td class="px-2 py-1">{c.class_name}</td>
                  <td class="px-2 py-1 text-right font-mono"
                    >{c.train.toLocaleString()}</td
                  >
                  <td class="px-2 py-1 text-right font-mono">{c.val.toLocaleString()}</td>
                  <td class="px-2 py-1 text-right font-mono">{c.test.toLocaleString()}</td
                  >
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      </details>
    {:else}
      <!-- No per-export split data: the operator picked a past export
           version /export/status can't describe (or no export has run
           yet) — clearly labelled as the dataset-wide global pool, not
           this export's own contents. -->
      <p class="mt-2 flex flex-wrap gap-2 text-xs text-zinc-400">
        <span class="rounded border border-zinc-700 bg-zinc-950 px-1.5 py-0.5">
          {classesStore.classes.filter((c) => !c.deprecated).length} classes (global pool)
        </span>
        <span class="rounded border border-zinc-700 bg-zinc-950 px-1.5 py-0.5 font-mono">
          {classesStore.classes
            .reduce((acc, c) => acc + (c.validated_count ?? 0), 0)
            .toLocaleString()} validated crops (global pool, not this export)
        </span>
      </p>
    {/if}
  {:else}
    <p class="mt-1 text-sm text-zinc-300">
      {dataset.datasetKind === dataset.datasetExportSpec?.datasetKind
        ? `No ${dataset.datasetExportSpec?.label} yet — build one below.`
        : (dataset.datasetMessage ?? 'Loading…')}
    </p>
  {/if}
</section>

{#if dataset.datasetExportAvailable && dataset.datasetExportSpec}
  <!-- Single-class dataset export, gated on the /methods `export` axis -->
  <section class="rounded-md border border-zinc-800 bg-zinc-900 p-4">
    <div class="flex items-center justify-between gap-3">
      <h2 class="text-[11px] uppercase tracking-wide text-zinc-500">
        {dataset.datasetExportSpec.label}
      </h2>
      <button
        type="button"
        class="rounded border border-zinc-700 bg-zinc-950 px-2.5 py-1 text-xs text-zinc-200 hover:bg-zinc-800 disabled:opacity-50"
        onclick={exporter.runSingleClassExport}
        disabled={exporter.singleClassExporting}
      >
        {exporter.singleClassExporting
          ? 'Exporting…'
          : `Build ${dataset.datasetExportSpec.label}`}
      </button>
    </div>
    <div class="mt-3 flex flex-wrap items-end gap-4">
      <label class="block">
        <span class="mb-1 block text-xs text-zinc-400">image mode</span>
        <select
          bind:value={exporter.singleClassImageMode}
          disabled={exporter.singleClassExporting}
          class="rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-sm text-zinc-100 focus:border-blue-500 focus:outline-none"
        >
          <option value="whole_frame">whole frame</option>
          <option value="item_crop">parent crop</option>
        </select>
      </label>
      <label class="block">
        <span class="mb-1 block text-xs text-zinc-400">image size</span>
        <select
          bind:value={exporter.singleClassImgSize}
          disabled={exporter.singleClassExporting}
          class="rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-sm text-zinc-100 focus:border-blue-500 focus:outline-none"
        >
          <option value={640}>640 (fast)</option>
          <option value={1280}>1280 (full)</option>
        </select>
      </label>
      <label class="block">
        <span class="mb-1 block text-xs text-zinc-400"
          >sample N positives (blank = all)</span
        >
        <input
          type="number"
          min="0"
          step="500"
          placeholder="all"
          bind:value={exporter.singleClassMaxPositives}
          disabled={exporter.singleClassExporting}
          class="w-32 rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-sm text-zinc-100 focus:border-blue-500 focus:outline-none"
        />
      </label>
      {#if exporter.singleClassDedupThreshold != null}
        <label class="flex items-center gap-2 pb-1.5">
          <input
            type="checkbox"
            bind:checked={exporter.singleClassDedup}
            disabled={exporter.singleClassExporting}
          />
          <span class="text-xs text-zinc-400"
            >dedup near-dup frames (cos ≥ {exporter.singleClassDedupThreshold})</span
          >
        </label>
      {/if}
    </div>
    <p class="mt-2 text-[11px] text-zinc-500">
      N samples positives spread <em>evenly across clusters</em> — build a small set first,
      then a larger one from the same labeled pool for progressive training.
    </p>
    {#if dataset.singleClassExportDir}
      <p class="mt-1 break-all font-mono text-sm text-zinc-200">
        {dataset.singleClassExportDir}
      </p>
    {/if}
    {#if exporter.singleClassExportMessage}
      <p class="mt-2 text-xs text-zinc-400">{exporter.singleClassExportMessage}</p>
    {:else}
      <p class="mt-2 text-xs text-zinc-500">{dataset.datasetExportSpec.blurb}</p>
    {/if}
  </section>
{/if}
