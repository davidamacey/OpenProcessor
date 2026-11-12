<script lang="ts">
  /** /train's dataset card (which frozen export the run trains on) and,
   *  when the backend serves it, the single-class dataset export panel. */
  import ClassSplitTable from '$components/export/ClassSplitTable.svelte';
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
<section class="rounded-md border border-line bg-surface-1 p-4">
  <div class="flex items-center justify-between gap-3">
    <h2 class="text-caption uppercase tracking-wide text-fg-3">Dataset</h2>
    <div class="flex gap-1 text-xs">
      <button
        type="button"
        class="rounded border px-2 py-0.5 {dataset.datasetKind === MULTI_CLASS
          ? 'border-accent bg-accent-soft text-accent-text'
          : 'border-line-strong bg-surface-0 text-fg-2 hover:bg-surface-2'}"
        onclick={() => dataset.selectDatasetKind(MULTI_CLASS)}
      >
        Multi-class
      </button>
      {#if dataset.datasetExportAvailable}
        <button
          type="button"
          class="rounded border px-2 py-0.5 {dataset.datasetKind ===
          dataset.datasetExportSpec?.datasetKind
            ? 'border-accent bg-accent-soft text-accent-text'
            : 'border-line-strong bg-surface-0 text-fg-2 hover:bg-surface-2'}"
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
      <span class="mb-1 block text-xs text-fg-2">
        dataset version ({dataset.kindDatasets.length} available — pick a sample, subset, or
        the full set)
      </span>
      <select
        bind:value={dataset.selectedExportDir}
        class="w-full rounded-md border border-line-strong bg-surface-0 px-2 py-1.5 font-mono text-xs text-fg-1 focus:border-accent focus:outline-none"
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
    <p class="mt-1 break-all font-mono text-sm text-fg-1">
      {dataset.datasetExportDir}
    </p>
    {#if dataset.datasetKind === dataset.datasetExportSpec?.datasetKind}
      <p class="mt-2 text-xs text-fg-2">
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
      <div class="mt-2 flex flex-wrap gap-2 text-xs text-fg-2">
        {#if dataset.multiClassExportState.class_count != null}
          <span class="rounded border border-line-strong bg-surface-0 px-1.5 py-0.5">
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
            class="rounded border border-line-strong bg-surface-0 px-1.5 py-0.5 font-mono"
          >
            {formatCount(dataset.multiClassExportState.object_count)} objects in {formatCount(
              dataset.multiClassExportState.image_count,
            )} images
            {#if dataset.multiClassExportState.group_key}
              <span class="text-fg-3">(by {dataset.multiClassExportState.group_key})</span
              >
            {/if}
          </span>
        {/if}
        {#if dataset.multiClassExportState.split_counts}
          <span
            class="rounded border border-line-strong bg-surface-0 px-1.5 py-0.5 font-mono"
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
            class="rounded border border-line-strong bg-surface-0 px-1.5 py-0.5 font-mono"
            title="Objects (label lines) per split"
          >
            objects: train {dataset.multiClassExportState.split_object_counts.train.toLocaleString()}
            · val {dataset.multiClassExportState.split_object_counts.val.toLocaleString()}
            · test {dataset.multiClassExportState.split_object_counts.test.toLocaleString()}
          </span>
        {/if}
      </div>
      <details class="mt-2 text-xs text-fg-2">
        <summary class="cursor-pointer hover:text-fg-1">
          per-class object counts ({dataset.multiClassExportState.class_split_counts
            .length})
        </summary>
        <ClassSplitTable
          class="mt-1 max-h-48"
          rows={dataset.multiClassExportState.class_split_counts}
        />
      </details>
    {:else}
      <!-- No per-export split data: the operator picked a past export
           version /export/status can't describe (or no export has run
           yet) — clearly labelled as the dataset-wide global pool, not
           this export's own contents. -->
      <p class="mt-2 flex flex-wrap gap-2 text-xs text-fg-2">
        <span class="rounded border border-line-strong bg-surface-0 px-1.5 py-0.5">
          {classesStore.classes.filter((c) => !c.deprecated).length} classes (global pool)
        </span>
        <span
          class="rounded border border-line-strong bg-surface-0 px-1.5 py-0.5 font-mono"
        >
          {classesStore.classes
            .reduce((acc, c) => acc + (c.validated_count ?? 0), 0)
            .toLocaleString()} validated crops (global pool, not this export)
        </span>
      </p>
    {/if}
  {:else}
    <p class="mt-1 text-sm text-fg-2">
      {dataset.datasetKind === dataset.datasetExportSpec?.datasetKind
        ? `No ${dataset.datasetExportSpec?.label} yet — build one below.`
        : (dataset.datasetMessage ?? 'Loading…')}
    </p>
  {/if}
</section>

{#if dataset.datasetExportAvailable && dataset.datasetExportSpec}
  <!-- Single-class dataset export, gated on the /methods `export` axis -->
  <section class="rounded-md border border-line bg-surface-1 p-4">
    <div class="flex items-center justify-between gap-3">
      <h2 class="text-caption uppercase tracking-wide text-fg-3">
        {dataset.datasetExportSpec.label}
      </h2>
      <button
        type="button"
        class="rounded border border-line-strong bg-surface-0 px-2.5 py-1 text-xs text-fg-1 hover:bg-surface-2 disabled:opacity-50"
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
        <span class="mb-1 block text-xs text-fg-2">image mode</span>
        <select
          bind:value={exporter.singleClassImageMode}
          disabled={exporter.singleClassExporting}
          class="rounded-md border border-line-strong bg-surface-0 px-2 py-1.5 text-sm text-fg-1 focus:border-accent focus:outline-none"
        >
          <option value="whole_frame">whole frame</option>
          <option value="item_crop">parent crop</option>
        </select>
      </label>
      <label class="block">
        <span class="mb-1 block text-xs text-fg-2">image size</span>
        <select
          bind:value={exporter.singleClassImgSize}
          disabled={exporter.singleClassExporting}
          class="rounded-md border border-line-strong bg-surface-0 px-2 py-1.5 text-sm text-fg-1 focus:border-accent focus:outline-none"
        >
          <option value={640}>640 (fast)</option>
          <option value={1280}>1280 (full)</option>
        </select>
      </label>
      <label class="block">
        <span class="mb-1 block text-xs text-fg-2">sample N positives (blank = all)</span>
        <input
          type="number"
          min="0"
          step="500"
          placeholder="all"
          bind:value={exporter.singleClassMaxPositives}
          disabled={exporter.singleClassExporting}
          class="w-32 rounded-md border border-line-strong bg-surface-0 px-2 py-1.5 text-sm text-fg-1 focus:border-accent focus:outline-none"
        />
      </label>
      <label class="flex items-center gap-2 pb-1.5">
        <input
          type="checkbox"
          data-testid="single-class-collapse"
          bind:checked={exporter.singleClassCollapse}
          disabled={exporter.singleClassExporting}
        />
        <span class="text-xs text-fg-2">collapse near-duplicate frames</span>
      </label>
    </div>
    <p class="mt-2 text-caption text-fg-3">
      N samples positives spread <em>evenly across clusters</em> — build a small set first,
      then a larger one from the same labeled pool for progressive training.
    </p>
    {#if dataset.singleClassExportDir}
      <p class="mt-1 break-all font-mono text-sm text-fg-1">
        {dataset.singleClassExportDir}
      </p>
    {/if}
    {#if exporter.singleClassExportMessage}
      <p class="mt-2 text-xs text-fg-2">{exporter.singleClassExportMessage}</p>
    {:else}
      <p class="mt-2 text-xs text-fg-3">{dataset.datasetExportSpec.blurb}</p>
    {/if}
  </section>
{/if}
