<script lang="ts">
  /** /export's "Export to staging" section: options, registry downloads,
   *  and the served status of the latest export. */
  import ClassSplitTable from '$components/export/ClassSplitTable.svelte';
  import { formatTimestamp } from '$lib/format/date';
  import { resolve } from '$app/paths';
  import { projectHref } from '$lib/projectPaths';
  import { getClassRegistryUrl, getDataYamlUrl, getManifestUrl } from '$lib/api_export';
  import { formatCount } from '$lib/format/count';
  import NearDupStalePanel from '$components/export/NearDupStalePanel.svelte';
  import ExportItemFilter from '$lib/components/itemFilter/ExportItemFilter.svelte';
  import {
    downloadUrl,
    type ExportRunController,
  } from '$lib/export/exportRunController.svelte';

  interface Props {
    run: ExportRunController;
    /** The served `GET {API_PREFIX}/export/datasets` list holds a current
     *  multi-class (`yolo`) export; gates the registry downloads. */
    hasMulticlassExport: boolean;
  }

  let { run, hasMulticlassExport }: Props = $props();

  // The served counts an export with these options would write, re-read (debounced) when an
  // option or the filter changes.
  $effect(() => {
    void run.requireFullyLabeled;
    void run.collapseNearDuplicates;
    void JSON.stringify(run.exportFilter.toBody());
    const t = setTimeout(() => void run.refreshPreview(), 400);
    return () => clearTimeout(t);
  });
</script>

<!-- Export controls + downloads + training command -->
<section class="surface p-4">
  <h2 class="mb-3 text-body font-semibold text-fg-2">Export to staging</h2>
  <NearDupStalePanel refresh={run.nearDup} />
  <div class="flex flex-wrap items-end gap-3">
    <label class="text-meta">
      <span class="mb-1 block text-fg-2">Version tag (optional)</span>
      <input
        type="text"
        bind:value={run.versionTag}
        placeholder="optional, e.g. baseline-a"
        class="input w-48"
      />
    </label>
    <label
      class="flex cursor-pointer items-center gap-2 pb-1.5 text-meta text-fg-2"
      title="Drop any exported image that still has an unvalidated or otherwise unlabeled object on it, instead of letting the detector learn that object as background."
    >
      <input
        type="checkbox"
        bind:checked={run.requireFullyLabeled}
        class="h-4 w-4 cursor-pointer accent-accent"
      />
      Only images whose every object is labeled
    </label>
    <label
      class="flex cursor-pointer items-center gap-2 pb-1.5 text-meta text-fg-2"
      title="Keep one item per near-duplicate group. Needs a current near-duplicate run."
    >
      <input
        type="checkbox"
        data-testid="export-collapse-near-dup"
        bind:checked={run.collapseNearDuplicates}
        class="h-4 w-4 cursor-pointer accent-accent"
      />
      Collapse near-duplicates
    </label>
    <button
      type="button"
      class="btn btn-primary"
      onclick={() => void run.runExport()}
      disabled={run.exportRunning || run.exportBlocked}
    >
      {run.exportRunning
        ? 'Exporting…'
        : run.exportState?.status === 'success'
          ? 'Re-export'
          : 'Export'}
    </button>

    <span class="grow"></span>

    <div class="flex flex-wrap items-center gap-2">
      <button
        type="button"
        class="btn"
        onclick={() => downloadUrl(getClassRegistryUrl(), 'class_registry.json')}
        disabled={!hasMulticlassExport}
        title={hasMulticlassExport
          ? ''
          : 'No frozen multi-class (yolo) export on disk yet'}
      >
        class_registry.json
      </button>
      <button
        type="button"
        class="btn"
        onclick={() => downloadUrl(getDataYamlUrl(), 'data.yaml')}
        disabled={!hasMulticlassExport}
        title={hasMulticlassExport
          ? ''
          : 'No frozen multi-class (yolo) export on disk yet'}
      >
        data.yaml
      </button>
      <button
        type="button"
        class="btn"
        onclick={() => downloadUrl(getManifestUrl(), 'manifest.json')}
        disabled={!hasMulticlassExport}
        title={hasMulticlassExport
          ? ''
          : 'No frozen multi-class (yolo) export on disk yet'}
      >
        manifest.json
      </button>
      {#if !hasMulticlassExport}
        <span class="text-caption text-fg-3">No frozen multi-class export yet</span>
      {/if}
    </div>
  </div>

  {#if run.previewError}
    <p class="mt-2 text-meta text-warn-text" data-testid="export-preview-error">
      No count: {run.previewError}
    </p>
  {:else if run.preview}
    <p class="mt-2 text-meta text-fg-2" data-testid="export-preview">
      This export would hold
      <span class="text-fg-1 tabular-nums">{formatCount(run.preview.object_count)}</span>
      objects in
      <span class="text-fg-1 tabular-nums">{formatCount(run.preview.image_count)}</span>
      images{run.preview.images_dropped_not_fully_labeled > 0
        ? `, leaving out ${formatCount(run.preview.images_dropped_not_fully_labeled)} not fully labelled`
        : ''}.
    </p>
    {#if run.preview.warning}
      <p class="mt-1 text-meta text-warn-text" data-testid="export-preview-warning">
        {run.preview.warning}
      </p>
    {/if}
  {/if}

  {#if run.exportBlocked && run.blockingReasons.length > 0}
    <ul
      class="mt-2 list-inside list-disc text-meta text-warn-text"
      data-testid="export-blocking-reasons"
    >
      {#each run.blockingReasons as reason, i (i)}
        <li>{reason}</li>
      {/each}
    </ul>
  {/if}

  <ExportItemFilter state={run.exportFilter} />

  {#if run.exportState}
    <div class="mt-3 text-meta text-fg-2">
      Last status: <span class="text-fg-1">{run.exportState.status}</span>
      {#if run.exportState.last_run}
        · {formatTimestamp(run.exportState.last_run)}
      {/if}
      {#if run.exportState.export_dir}
        · <code class="text-fg-2">{run.exportState.export_dir}</code>
      {/if}
      {#if run.exportState.error}
        · <span class="text-danger-text">{run.exportState.error}</span>
      {/if}
    </div>
    {#if run.exportState.status === 'success'}
      <p class="mt-2 text-meta text-fg-2">
        Next: <a
          href={resolve(projectHref('/train'))}
          class="text-accent-text underline hover:text-accent-text"
          >train on this export from the Train cockpit</a
        >.
      </p>
    {/if}

    <!-- Image/object counts + per-class table (OpenProcessor 4c9499a's
         `GET {API_PREFIX}/export/status`, ExportStatusResponse — one
         image + one label file per source image, one line per object).
         A null `image_count`/`object_count` renders "—", never 0
         (formatCount); with none of them served (no export yet) this
         whole block doesn't render rather than showing blanks. -->
    {#if run.exportState.image_count != null || run.exportState.object_count != null || run.exportState.class_count != null || run.exportState.split_counts}
      <div class="mt-3 flex flex-wrap gap-2 text-meta">
        {#if run.exportState.image_count != null || run.exportState.object_count != null}
          <span class="rounded-md border border-line-strong bg-surface-1/40 px-2 py-1">
            <span class="ml-1 text-fg-1 tabular-nums"
              >{formatCount(run.exportState.object_count)}</span
            >
            <span class="text-fg-3">objects in</span>
            <span class="ml-1 text-fg-1 tabular-nums"
              >{formatCount(run.exportState.image_count)}</span
            >
            <span class="text-fg-3">images</span>
            {#if run.exportState.group_key}
              <span class="ml-1 text-fg-3">(grouped by {run.exportState.group_key})</span>
            {/if}
          </span>
        {/if}
        {#if run.exportState.class_count != null}
          <!-- E2 (visual audit 2026-09-24): the served class_count is the
               registry size; say how many classes actually have objects.
               #36 item 6: prefer the served classes_with_objects over the
               client-side count from class_split_counts, which is kept
               only as the fallback for an export/backend written before
               the field existed. -->
          <span
            class="rounded-md border border-line-strong bg-surface-1/40 px-2 py-1"
            data-testid="export-class-count"
          >
            {#if run.exportState.classes_with_objects != null}
              <span class="text-fg-1 tabular-nums"
                >{run.exportState.classes_with_objects}</span
              >
              <span class="text-fg-3">classes with objects</span>
              <span class="ml-1 text-fg-3"
                >({run.exportState.class_count} in registry)</span
              >
            {:else if run.exportState.class_split_counts}
              <span class="text-fg-1 tabular-nums"
                >{run.exportClassSplit.withObjects.length}</span
              >
              <span class="text-fg-3">classes with objects</span>
              <span class="ml-1 text-fg-3"
                >({run.exportState.class_count} in registry)</span
              >
            {:else}
              <span class="text-fg-3">classes in registry</span>
              <span class="ml-1 text-fg-1 tabular-nums"
                >{run.exportState.class_count}</span
              >
            {/if}
          </span>
        {/if}
        {#if run.exportState.split_counts}
          <span
            class="rounded-md border border-line-strong bg-surface-1/40 px-2 py-1 tabular-nums"
            title="Images per split"
          >
            images: train {run.exportState.split_counts.train.toLocaleString()} · val {run.exportState.split_counts.val.toLocaleString()}
            · test {run.exportState.split_counts.test.toLocaleString()}
          </span>
        {/if}
        {#if run.exportState.split_object_counts}
          <span
            class="rounded-md border border-line-strong bg-surface-1/40 px-2 py-1 tabular-nums"
            title="Objects (label lines) per split"
          >
            objects: train {run.exportState.split_object_counts.train.toLocaleString()} · val
            {run.exportState.split_object_counts.val.toLocaleString()}
            · test {run.exportState.split_object_counts.test.toLocaleString()}
          </span>
        {/if}
      </div>
    {/if}

    <!-- Partial-frame policy + counts (4c9499a) — null on an older
         export (formatCount renders "—"); the whole block hides when
         nothing here was ever recorded. -->
    {#if run.exportState.require_fully_labeled_images != null || run.exportState.unlabeled_items_on_exported_images != null || run.exportState.images_with_unlabeled_items != null || run.exportState.images_dropped_not_fully_labeled != null || run.exportState.skipped_items != null}
      <dl
        class="mt-2 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-meta"
        data-testid="export-quality"
      >
        {#if run.exportState.require_fully_labeled_images != null}
          <dt class="text-fg-3">Only fully labelled photos</dt>
          <dd class="text-fg-1">
            {run.exportState.require_fully_labeled_images ? 'on' : 'off'}
          </dd>
        {/if}
        {#if run.exportState.unlabeled_items_on_exported_images != null}
          <dt
            class="text-fg-3"
            title="Objects on exported photos the export did not label; they are learned as background."
          >
            Unlabelled objects on exported photos
          </dt>
          <dd class="text-fg-1 tabular-nums">
            {formatCount(run.exportState.unlabeled_items_on_exported_images)}
          </dd>
        {/if}
        {#if run.exportState.images_with_unlabeled_items != null}
          <dt class="text-fg-3">Photos with an unlabelled object</dt>
          <dd class="text-fg-1 tabular-nums">
            {formatCount(run.exportState.images_with_unlabeled_items)}
          </dd>
        {/if}
        {#if run.exportState.images_dropped_not_fully_labeled != null}
          <dt
            class="text-fg-3"
            title="Photos left out because only fully labelled photos was on (0 when it was off)."
          >
            Photos left out (not fully labelled)
          </dt>
          <dd class="text-fg-1 tabular-nums">
            {formatCount(run.exportState.images_dropped_not_fully_labeled)}
          </dd>
        {/if}
        {#if run.exportState.skipped_items != null}
          <dt class="text-fg-3" title="Validated items the export could not write.">
            Items skipped
          </dt>
          <dd class="text-fg-1 tabular-nums" data-testid="export-skipped-items">
            {formatCount(run.exportState.skipped_items.no_image_id)} without a source image
            id · {formatCount(run.exportState.skipped_items.no_usable_box_or_class)} without
            a usable box or class
          </dd>
        {/if}
      </dl>
    {/if}

    {#if run.exportState.class_split_counts && run.exportState.class_split_counts.length > 0}
      <details class="mt-3 text-meta">
        <summary class="cursor-pointer text-fg-2 hover:text-fg-1">
          Per-class object counts ({run.exportClassSplit.withObjects.length} classes with objects)
        </summary>
        <ClassSplitTable class="mt-2 max-h-64" rows={run.exportClassSplit.withObjects} />
        {#if run.exportClassSplit.empty.length > 0}
          <p class="mt-1 text-fg-3" data-testid="export-empty-classes">
            {run.exportClassSplit.empty.length} class{run.exportClassSplit.empty
              .length === 1
              ? ''
              : 'es'} with no objects in this export
          </p>
        {/if}
      </details>
    {/if}
  {/if}
</section>
