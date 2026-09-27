<script lang="ts">
  /**
   * `/datasets/import` — import a labeled dataset into the current project
   * (any_domain_plan.md §7.12 item 1; docs/design/
   * w10-import-reprocess-ui-plan-2026-09-27.md §2). State lives in
   * `ImportWizard`; every count, suggestion, issue and verdict shown here
   * is the served preview's.
   */
  import { goto } from '$app/navigation';
  import { resolve } from '$app/paths';
  import ConfirmDialog from '$components/ConfirmDialog.svelte';
  import DatasetsGate from '$components/datasets/DatasetsGate.svelte';
  import DatasetIssueList from '$components/datasets/DatasetIssueList.svelte';
  import ImportOptions from '$components/datasets/ImportOptions.svelte';
  import MappingTable from '$components/datasets/MappingTable.svelte';
  import PreviewSummary from '$components/datasets/PreviewSummary.svelte';
  import { getDatasetImport, getIngestConfig, listDatasetImports } from '$lib/api';
  import { datasetsAvailability } from '$lib/datasets/datasetsAvailability.svelte';
  import { createImportWizard } from '$lib/datasets/importWizardController.svelte';
  import { datasetUploadMaxBytes } from '$lib/datasets/uploadCap';
  import { projectHref } from '$lib/projectPaths';
  import { classesStore } from '$stores/classes.svelte';
  import type { DatasetFormatsResponse, DatasetImportJob } from '$lib/types_import';

  const wizard = createImportWizard();
  $effect(() => () => wizard.destroy());

  let sourceRoots = $state<string[]>([]);
  let previousImports = $state<DatasetImportJob[]>([]);
  let prefillError = $state<string | null>(null);
  let confirmOpen = $state(false);
  let resumeTarget = $state<string | null>(null);

  const pickableClasses = $derived(
    classesStore.classes
      .filter((c) => !c.deprecated)
      .slice()
      .sort((a, b) => a.name.localeCompare(b.name)),
  );

  // The side reads fire once, only after the gate confirms W10 is served
  // (a backend without it gets no dataset request beyond the probe).
  let sideReadsLoaded = false;
  $effect(() => {
    if (datasetsAvailability.available !== true || sideReadsLoaded) return;
    sideReadsLoaded = true;
    loadSideReads();
  });

  function loadSideReads(): void {
    void getIngestConfig()
      .then((c) => (sourceRoots = c.batch?.source_roots ?? []))
      .catch(() => (sourceRoots = []));
    void listDatasetImports({ page: 1, page_size: 50 })
      .then((l) => (previousImports = l.items))
      .catch(() => (previousImports = []));
  }

  async function prefill(importId: string): Promise<void> {
    if (!importId) return;
    prefillError = null;
    try {
      wizard.prefillFromJob(await getDatasetImport(importId));
    } catch (e) {
      prefillError = (e as Error)?.message ?? String(e);
    }
  }

  async function confirmStart(): Promise<void> {
    const job = await wizard.start();
    confirmOpen = false;
    if (job)
      await goto(
        resolve(projectHref(`/datasets/imports/${encodeURIComponent(job.import_id)}`)),
      );
  }

  async function confirmResume(): Promise<void> {
    const id = resumeTarget;
    resumeTarget = null;
    if (!id) return;
    const job = await wizard.resume(id);
    if (job)
      await goto(
        resolve(projectHref(`/datasets/imports/${encodeURIComponent(job.import_id)}`)),
      );
  }

  function onFile(e: Event, formats: DatasetFormatsResponse): void {
    const input = e.currentTarget as HTMLInputElement;
    const file = input.files?.[0];
    input.value = '';
    if (!file) return;
    const max = datasetUploadMaxBytes(formats.upload.max_bytes);
    void wizard.upload(
      file,
      max,
      `${file.name} is larger than this deployment accepts (${(max / 1024 ** 3).toFixed(1)} GiB).`,
    );
  }
</script>

<div class="mx-auto max-w-6xl space-y-6 p-6">
  <div class="flex flex-wrap items-baseline justify-between gap-2">
    <h1 class="text-lg font-semibold text-zinc-100">Import a labeled dataset</h1>
    <a
      class="text-xs text-blue-300 hover:underline"
      href={resolve(projectHref('/datasets/imports'))}>All imports</a
    >
  </div>

  <DatasetsGate>
    {#snippet children(formats)}
      <p class="text-xs text-zinc-500">
        The dataset is imported into this project. To import into a new project,
        <a class="text-blue-300 hover:underline" href={resolve('/projects')}>create it</a> first
        and open this page there.
      </p>

      <!-- 1. Source -->
      <section
        class="space-y-3 rounded-lg border border-zinc-800 p-4"
        data-testid="import-source"
      >
        <h2 class="text-sm font-semibold text-zinc-200">Source</h2>
        <div class="grid gap-3 sm:grid-cols-[1fr_auto]">
          <label class="block text-xs">
            <span class="mb-0.5 block text-zinc-400">Server path</span>
            <input
              class="input w-full font-mono"
              placeholder="/data/source/my_dataset"
              data-testid="import-path"
              value={wizard.sourcePath}
              oninput={(e) =>
                wizard.setSource((e.currentTarget as HTMLInputElement).value)}
            />
          </label>
          <label class="block text-xs">
            <span class="mb-0.5 block text-zinc-400">Format</span>
            <select
              class="select"
              value={wizard.format}
              onchange={(e) =>
                wizard.setFormat((e.currentTarget as HTMLSelectElement).value)}
            >
              {#each formats.formats as f (f.id)}
                <option value={f.id}>{f.label}</option>
              {/each}
            </select>
          </label>
        </div>
        {#if sourceRoots.length > 0}
          <div class="text-xs text-zinc-500" data-testid="source-roots">
            Allowed roots:
            {#each sourceRoots as r, i (r)}<code class="font-mono text-zinc-300">{r}</code
              >{i < sourceRoots.length - 1 ? ', ' : ''}{/each}
          </div>
        {/if}
        <div class="text-xs">
          <label class="inline-flex items-center gap-2">
            <span class="text-zinc-400">Or upload an archive</span>
            <input
              type="file"
              accept={formats.upload.accepted.join(',')}
              disabled={wizard.uploading}
              onchange={(e) => onFile(e, formats)}
            />
          </label>
          <span class="ml-2 text-zinc-500">
            {formats.upload.accepted.join(', ')}, up to
            {(datasetUploadMaxBytes(formats.upload.max_bytes) / 1024 ** 3).toFixed(1)} GiB.
            Large datasets belong on a server path.
          </span>
          {#if wizard.uploading}<p class="mt-1 text-zinc-400">Uploading…</p>{/if}
          {#if wizard.uploadError}<p class="mt-1 text-red-300">
              {wizard.uploadError}
            </p>{/if}
        </div>
      </section>

      <!-- 2. Preview -->
      <section class="space-y-3 rounded-lg border border-zinc-800 p-4">
        <div class="flex items-baseline justify-between">
          <h2 class="text-sm font-semibold text-zinc-200">Preview</h2>
          {#if wizard.previewing}
            <span class="text-xs text-zinc-500" data-testid="previewing"
              >Checking the dataset…</span
            >
          {/if}
        </div>
        {#if wizard.previewError}
          <p class="text-sm text-red-300" data-testid="preview-error">
            {wizard.previewError}
          </p>
        {:else if wizard.preview}
          <PreviewSummary preview={wizard.preview} {formats} />
        {:else}
          <p class="text-xs text-zinc-500">Enter a server path or upload an archive.</p>
        {/if}
      </section>

      {#if wizard.preview}
        <!-- 3. Class mapping -->
        <section class="space-y-3 rounded-lg border border-zinc-800 p-4">
          <h2 class="text-sm font-semibold text-zinc-200">Class mapping</h2>
          <p class="text-xs text-zinc-500">
            Classes are matched by name. Pick the same class on two rows to merge them.
          </p>
          <div class="flex flex-wrap items-center gap-4 text-xs">
            <label class="inline-flex items-center gap-2">
              <input
                type="checkbox"
                checked={wizard.acceptSuggestions}
                onchange={(e) =>
                  wizard.setAcceptSuggestions(
                    (e.currentTarget as HTMLInputElement).checked,
                  )}
              />
              Accept suggestions
            </label>
            {#if previousImports.length > 0}
              <label class="inline-flex items-center gap-2">
                <span class="text-zinc-400">Use the mapping from</span>
                <select
                  class="select select-sm"
                  onchange={(e) =>
                    void prefill((e.currentTarget as HTMLSelectElement).value)}
                >
                  <option value="">a previous import…</option>
                  {#each previousImports as j (j.import_id)}
                    <option value={j.import_id}>{j.name ?? j.import_id}</option>
                  {/each}
                </select>
              </label>
            {/if}
            {#if prefillError}<span class="text-red-300">{prefillError}</span>{/if}
          </div>
          <MappingTable {wizard} {formats} classes={pickableClasses} />
        </section>

        <!-- 4. Options -->
        <section class="space-y-3 rounded-lg border border-zinc-800 p-4">
          <h2 class="text-sm font-semibold text-zinc-200">Options</h2>
          <ImportOptions {wizard} {formats} />
        </section>

        <!-- 5. Start -->
        <section class="space-y-3" data-testid="import-start">
          {#if wizard.refusal}
            <div
              class="space-y-2 rounded border border-red-900 bg-red-950/30 p-3 text-sm text-red-200"
              data-testid="start-refusal"
            >
              <p>{wizard.refusal.message}</p>
              {#if wizard.refusal.code === 'import_resumable' && wizard.refusal.importId}
                <button
                  type="button"
                  class="btn btn-sm"
                  onclick={() => (resumeTarget = wizard.refusal?.importId ?? null)}
                  >Resume it</button
                >
              {:else if wizard.refusal.importId}
                <a
                  class="text-blue-300 hover:underline"
                  href={resolve(
                    projectHref(
                      `/datasets/imports/${encodeURIComponent(wizard.refusal.importId)}`,
                    ),
                  )}>Open that import</a
                >
              {/if}
              {#if wizard.refusal.issues.length > 0}
                <DatasetIssueList
                  issues={wizard.refusal.issues}
                  catalog={formats.issues}
                />
              {/if}
            </div>
          {/if}
          {#if wizard.reusedJob}
            <p class="text-sm text-zinc-300" data-testid="start-reused">
              Already imported.
              <a
                class="text-blue-300 hover:underline"
                href={resolve(
                  projectHref(
                    `/datasets/imports/${encodeURIComponent(wizard.reusedJob.import_id)}`,
                  ),
                )}>Open the import</a
              >
            </p>
          {/if}
          <button
            type="button"
            class="btn btn-primary"
            disabled={!wizard.canStart}
            onclick={() => (confirmOpen = true)}
          >
            Start import
          </button>
          {#if wizard.unmappedRows.length > 0}
            <span class="ml-2 text-xs text-amber-300">
              {wizard.unmappedRows.length} class{wizard.unmappedRows.length === 1
                ? ''
                : 'es'} still need a mapping.
            </span>
          {:else if wizard.preview.blocking && !wizard.canStart}
            <span class="ml-2 text-xs text-red-300"
              >The issues above block this import.</span
            >
          {/if}
        </section>

        {#if confirmOpen}
          <ConfirmDialog
            title="Start the import"
            confirmLabel="Start import"
            busy={wizard.starting}
            onconfirm={() => void confirmStart()}
            oncancel={() => (confirmOpen = false)}
          >
            <p>
              Import {wizard.preview.totals.images.toLocaleString()} images and
              {wizard.preview.totals.boxes.toLocaleString()} boxes into this project.
            </p>
            <p class="text-xs text-zinc-400">
              {wizard.preview.totals.images_to_ingest.toLocaleString()} images to ingest,
              {wizard.preview.totals.images_already_indexed.toLocaleString()} already here.
              An import can be undone from its page.
            </p>
          </ConfirmDialog>
        {/if}
      {/if}

      {#if resumeTarget}
        <ConfirmDialog
          title="Resume the import"
          confirmLabel="Resume"
          onconfirm={() => void confirmResume()}
          oncancel={() => (resumeTarget = null)}
        >
          <p>
            Resume import <code class="font-mono">{resumeTarget}</code> from its last finished
            chunk.
          </p>
        </ConfirmDialog>
      {/if}
    {/snippet}
  </DatasetsGate>
</div>
