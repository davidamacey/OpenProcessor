<script lang="ts">
  /** /export's "Export to staging" section: options, registry downloads,
   *  and the served status of the latest export. */
  import { resolve } from '$app/paths';
  import { projectHref } from '$lib/projectPaths';
  import { getClassRegistryUrl, getDataYamlUrl, getManifestUrl } from '$lib/api';
  import { formatCount } from '$lib/format/count';
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
</script>

<!-- Export controls + downloads + training command -->
<section class="surface p-4">
  <h2 class="mb-3 text-sm font-semibold text-zinc-300">Export to staging</h2>
  <div class="flex flex-wrap items-end gap-3">
    <label class="text-xs">
      <span class="mb-1 block text-zinc-400">Version tag (optional)</span>
      <input
        type="text"
        bind:value={run.versionTag}
        placeholder="optional, e.g. baseline-a"
        class="input w-48"
      />
    </label>
    <label
      class="flex cursor-pointer items-center gap-2 pb-1.5 text-xs text-zinc-300"
      title="Drop any exported image that still has an unvalidated or otherwise unlabeled object on it, instead of letting the detector learn that object as background."
    >
      <input
        type="checkbox"
        bind:checked={run.requireFullyLabeled}
        class="h-4 w-4 cursor-pointer accent-blue-500"
      />
      Only images whose every object is labeled
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
        <span class="text-[11px] text-zinc-500">No frozen multi-class export yet</span>
      {/if}
    </div>
  </div>

  {#if run.exportBlocked && run.blockingReasons.length > 0}
    <ul
      class="mt-2 list-inside list-disc text-xs text-amber-300"
      data-testid="export-blocking-reasons"
    >
      {#each run.blockingReasons as reason, i (i)}
        <li>{reason}</li>
      {/each}
    </ul>
  {/if}

  <ExportItemFilter state={run.exportFilter} />

  {#if run.exportState}
    <div class="mt-3 text-xs text-zinc-400">
      Last status: <span class="font-mono text-zinc-200">{run.exportState.status}</span>
      {#if run.exportState.last_run}
        · {new Date(run.exportState.last_run).toLocaleString()}
      {/if}
      {#if run.exportState.export_dir}
        · <span class="font-mono text-zinc-300">{run.exportState.export_dir}</span>
      {/if}
      {#if run.exportState.error}
        · <span class="text-red-300">{run.exportState.error}</span>
      {/if}
    </div>
    {#if run.exportState.status === 'success'}
      <p class="mt-2 text-xs text-zinc-400">
        Next: <a
          href={resolve(projectHref('/train'))}
          class="text-blue-400 underline hover:text-blue-300"
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
      <div class="mt-3 flex flex-wrap gap-2 text-xs">
        {#if run.exportState.image_count != null || run.exportState.object_count != null}
          <span class="rounded-md border border-zinc-700 bg-zinc-900/40 px-2 py-1">
            <span class="ml-1 font-mono text-zinc-200"
              >{formatCount(run.exportState.object_count)}</span
            >
            <span class="text-zinc-500">objects in</span>
            <span class="ml-1 font-mono text-zinc-200"
              >{formatCount(run.exportState.image_count)}</span
            >
            <span class="text-zinc-500">images</span>
            {#if run.exportState.group_key}
              <span class="ml-1 text-zinc-500"
                >(grouped by {run.exportState.group_key})</span
              >
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
            class="rounded-md border border-zinc-700 bg-zinc-900/40 px-2 py-1"
            data-testid="export-class-count"
          >
            {#if run.exportState.classes_with_objects != null}
              <span class="font-mono text-zinc-200"
                >{run.exportState.classes_with_objects}</span
              >
              <span class="text-zinc-500">classes with objects</span>
              <span class="ml-1 text-zinc-500"
                >({run.exportState.class_count} in registry)</span
              >
            {:else if run.exportState.class_split_counts}
              <span class="font-mono text-zinc-200"
                >{run.exportClassSplit.withObjects.length}</span
              >
              <span class="text-zinc-500">classes with objects</span>
              <span class="ml-1 text-zinc-500"
                >({run.exportState.class_count} in registry)</span
              >
            {:else}
              <span class="text-zinc-500">classes in registry</span>
              <span class="ml-1 font-mono text-zinc-200"
                >{run.exportState.class_count}</span
              >
            {/if}
          </span>
        {/if}
        {#if run.exportState.split_counts}
          <span
            class="rounded-md border border-zinc-700 bg-zinc-900/40 px-2 py-1 font-mono"
            title="Images per split"
          >
            images: train {run.exportState.split_counts.train.toLocaleString()} · val {run.exportState.split_counts.val.toLocaleString()}
            · test {run.exportState.split_counts.test.toLocaleString()}
          </span>
        {/if}
        {#if run.exportState.split_object_counts}
          <span
            class="rounded-md border border-zinc-700 bg-zinc-900/40 px-2 py-1 font-mono"
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
      <div class="mt-2 flex flex-wrap gap-2 text-xs text-zinc-400">
        {#if run.exportState.require_fully_labeled_images != null}
          <span class="rounded-md border border-zinc-700 bg-zinc-900/40 px-2 py-1">
            require_fully_labeled_images: <span class="font-mono text-zinc-200"
              >{run.exportState.require_fully_labeled_images ? 'true' : 'false'}</span
            >
          </span>
        {/if}
        {#if run.exportState.unlabeled_items_on_exported_images != null}
          <span
            class="rounded-md border border-zinc-700 bg-zinc-900/40 px-2 py-1"
            title="Objects on exported images the export did not label — learned as background."
          >
            unlabeled objects on exported images: <span class="font-mono text-zinc-200"
              >{formatCount(run.exportState.unlabeled_items_on_exported_images)}</span
            >
          </span>
        {/if}
        {#if run.exportState.images_with_unlabeled_items != null}
          <span class="rounded-md border border-zinc-700 bg-zinc-900/40 px-2 py-1">
            images with an unlabeled object: <span class="font-mono text-zinc-200"
              >{formatCount(run.exportState.images_with_unlabeled_items)}</span
            >
          </span>
        {/if}
        {#if run.exportState.images_dropped_not_fully_labeled != null}
          <span
            class="rounded-md border border-zinc-700 bg-zinc-900/40 px-2 py-1"
            title="Images left out by require_fully_labeled_images (0 when it was off)."
          >
            images dropped (not fully labeled): <span class="font-mono text-zinc-200"
              >{formatCount(run.exportState.images_dropped_not_fully_labeled)}</span
            >
          </span>
        {/if}
        {#if run.exportState.skipped_items != null}
          <span
            class="rounded-md border border-zinc-700 bg-zinc-900/40 px-2 py-1"
            title="Validated items the export couldn't write: no source image id, or no usable box/class."
            data-testid="export-skipped-items"
          >
            skipped items: <span class="font-mono text-zinc-200"
              >{formatCount(run.exportState.skipped_items.no_image_id)} no image id · {formatCount(
                run.exportState.skipped_items.no_usable_box_or_class,
              )} no usable box/class</span
            >
          </span>
        {/if}
      </div>
    {/if}

    {#if run.exportState.class_split_counts && run.exportState.class_split_counts.length > 0}
      <details class="mt-3 text-xs">
        <summary class="cursor-pointer text-zinc-400 hover:text-zinc-200">
          Per-class object counts ({run.exportClassSplit.withObjects.length} classes with objects)
        </summary>
        <div class="mt-2 max-h-64 overflow-auto rounded border border-zinc-800">
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
              {#each run.exportClassSplit.withObjects as c (c.class_id)}
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
        {#if run.exportClassSplit.empty.length > 0}
          <p class="mt-1 text-zinc-500" data-testid="export-empty-classes">
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
