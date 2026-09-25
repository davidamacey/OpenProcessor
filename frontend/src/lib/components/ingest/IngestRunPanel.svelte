<script lang="ts">
  /**
   * Source tag / identifier prefix / skip-lookup controls, Start/Pause/
   * Resume/Cancel, a progress bar, totals chips, and paged per-file
   * results with a CSV download
   * (docs/design/ingest-ui-and-acceptance-plan-2026-09-24.md §A.2/§A.4).
   */
  import { untrack } from 'svelte';
  import { beforeNavigate } from '$app/navigation';
  import { ingestPathLookup, ingestUpload } from '$lib/api';
  import type { IngestFile } from '$lib/ingest/fileSource';
  import {
    DEFAULT_UPLOAD_CONCURRENCY,
    LARGE_SELECTION_HINT,
    type ResolvedIngestConfig,
  } from '$lib/ingest/ingestConfig';
  import {
    createIngestRun,
    type IngestRunState,
  } from '$lib/ingest/ingestRunController.svelte';
  import type { IngestResultKind } from '$lib/ingest/ingestResults.svelte';

  interface Props {
    files: IngestFile[];
    config: ResolvedIngestConfig;
    onChunkDone?: () => void;
    onStateChange?: (state: IngestRunState) => void;
  }
  let { files, config, onChunkDone, onStateChange }: Props = $props();

  let source = $state('upload');
  let identifierPrefix = $state('upload/');
  let identifierPrefixTouched = $state(false);
  let skipLookup = $state(false);
  let concurrency = $state(DEFAULT_UPLOAD_CONCURRENCY);
  let advancedOpen = $state(false);
  let activeTab = $state<IngestResultKind>('failed');
  // BA-7: the Failed tab's error_kind filter chip. `null` = no filter
  // (every failed result shows). Only meaningful on the 'failed' tab.
  let errorKindFilter = $state<string | null>(null);
  const PAGE_SIZE = 100;
  let pageOffset = $state<Record<IngestResultKind, number>>({
    ingested: 0,
    duplicate: 0,
    failed: 0,
    skipped: 0,
    not_sent: 0,
  });

  $effect(() => {
    if (!identifierPrefixTouched) identifierPrefix = `${source}/`;
  });

  // Snapshot-on-construction, deliberately: the controller reads
  // `config`/`concurrency`/`onChunkDone` once, at creation, not
  // reactively on every change (there is exactly one controller per
  // mounted panel, and changing the concurrency knob mid-run only takes
  // effect on the next run). `untrack` documents that intent instead of
  // silencing the compiler's "only captures the initial value" warning.
  const run = untrack(() =>
    createIngestRun({
      lookup: ingestPathLookup,
      upload: ingestUpload,
      config,
      concurrency,
      onChunkDone,
    }),
  );

  $effect(() => {
    onStateChange?.(run.state);
  });

  const isRunning = $derived(run.state === 'uploading' || run.state === 'prefiltering');
  const isPaused = $derived(run.state === 'paused');

  async function start(): Promise<void> {
    pageOffset = { ingested: 0, duplicate: 0, failed: 0, skipped: 0, not_sent: 0 };
    await run.start(files, { source, identifierPrefix, skipLookup });
  }

  async function retryFailed(): Promise<void> {
    await run.retryFailed();
  }

  function downloadCsv(): void {
    const blob = new Blob([run.results.toCsv()], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'ingest-results.csv';
    a.click();
    URL.revokeObjectURL(url);
  }

  function onBeforeUnload(e: BeforeUnloadEvent): void {
    if (isRunning || isPaused) {
      e.preventDefault();
      e.returnValue = '';
    }
  }

  beforeNavigate((nav) => {
    if (isRunning || isPaused) {
      const ok = confirm(
        'An upload is in progress; leaving cancels it. Re-selecting the same folder later resumes.',
      );
      if (!ok) nav.cancel();
    }
  });

  $effect(() => {
    window.addEventListener('beforeunload', onBeforeUnload);
    return () => window.removeEventListener('beforeunload', onBeforeUnload);
  });

  const progressPct = $derived(
    run.totals.queued > 0
      ? Math.round(
          ((run.totals.successful + run.totals.duplicates + run.totals.failed) /
            run.totals.queued) *
            100,
        )
      : 0,
  );

  const TABS: { kind: IngestResultKind; label: string }[] = [
    { kind: 'failed', label: 'Failed' },
    { kind: 'duplicate', label: 'Duplicate' },
    { kind: 'ingested', label: 'Ingested' },
    { kind: 'skipped', label: 'Skipped' },
  ];
</script>

<div class="space-y-3">
  <div class="flex flex-wrap items-end gap-3">
    <label class="text-xs text-zinc-400">
      Source tag
      <input class="input input-sm block" bind:value={source} disabled={isRunning} />
    </label>
    <label class="flex items-center gap-1.5 text-xs text-zinc-400">
      <input type="checkbox" bind:checked={skipLookup} disabled={isRunning} />
      Skip the already-indexed check
    </label>
    <button
      class="btn btn-sm"
      type="button"
      onclick={() => (advancedOpen = !advancedOpen)}
    >
      Advanced
    </button>
  </div>

  {#if advancedOpen}
    <div class="flex flex-wrap items-end gap-3 rounded border border-zinc-800 p-3">
      <label class="text-xs text-zinc-400">
        Identifier prefix
        <input
          class="input input-sm block"
          value={identifierPrefix}
          disabled={isRunning}
          oninput={(e) => {
            identifierPrefixTouched = true;
            identifierPrefix = (e.currentTarget as HTMLInputElement).value;
          }}
        />
      </label>
      <label class="text-xs text-zinc-400">
        Concurrency
        <input
          class="input input-sm block w-16"
          type="number"
          min="1"
          max="4"
          bind:value={concurrency}
          disabled={isRunning}
        />
      </label>
    </div>
  {/if}

  {#if files.length >= LARGE_SELECTION_HINT}
    <p class="text-xs text-amber-300">
      Large selection ({files.length} files) — the command-line uploader (<code
        >scripts/curation/ingest_upload.py</code
      > on the backend) is faster for very large archives.
    </p>
  {/if}

  <div class="flex gap-2">
    {#if isRunning}
      <button class="btn btn-sm" type="button" onclick={() => run.pause()}>Pause</button>
    {:else if isPaused}
      <button class="btn btn-sm" type="button" onclick={() => run.resume()}>Resume</button
      >
    {:else}
      <button
        class="btn btn-primary btn-sm"
        type="button"
        disabled={files.length === 0}
        onclick={() => void start()}
      >
        Start
      </button>
    {/if}
    {#if isRunning || isPaused}
      <button class="btn btn-danger btn-sm" type="button" onclick={() => run.cancel()}>
        Cancel
      </button>
    {/if}
    {#if run.results.retryable().length > 0 && !isRunning}
      <button class="btn btn-sm" type="button" onclick={() => void retryFailed()}>
        Retry failed
      </button>
    {/if}
  </div>

  {#if run.pauseReason}
    <p class="text-xs text-amber-300">Paused — {run.pauseReason}</p>
  {/if}
  {#if run.errorReason}
    <p class="text-xs text-red-300">{run.errorReason}</p>
  {/if}

  {#if run.totals.queued > 0}
    <div class="h-2 w-full overflow-hidden rounded bg-zinc-800">
      <div
        class="h-full bg-blue-600 transition-[width]"
        style="width: {progressPct}%"
      ></div>
    </div>
    <div class="flex flex-wrap gap-2 text-xs">
      <span class="chip">queued {run.totals.queued}</span>
      <span class="chip">skipped {run.totals.skipped_known}</span>
      <span class="chip">successful {run.totals.successful}</span>
      <span class="chip">duplicate {run.totals.duplicates}</span>
      <span class="chip">failed {run.totals.failed}</span>
      <span class="chip">crops indexed {run.totals.crops_indexed}</span>
    </div>

    <div>
      <div class="flex gap-1 border-b border-zinc-800 text-xs">
        {#each TABS as t (t.kind)}
          <button
            class="px-2 py-1 {activeTab === t.kind
              ? 'border-b-2 border-blue-500 text-white'
              : 'text-zinc-400'}"
            type="button"
            onclick={() => {
              activeTab = t.kind;
              errorKindFilter = null;
            }}
          >
            {t.label} ({run.results.countOf(t.kind)})
          </button>
        {/each}
        <span class="grow"></span>
        <button class="btn btn-sm" type="button" onclick={downloadCsv}
          >Download CSV</button
        >
      </div>

      {#if activeTab === 'failed' && run.results.errorKindCounts().length > 0}
        <!-- BA-7: group/filter Failed by the served error_kind. -->
        <div class="flex flex-wrap gap-1 border-b border-zinc-800 py-2 text-xs">
          <button
            class="chip {errorKindFilter === null ? 'bg-blue-900 text-blue-100' : ''}"
            type="button"
            onclick={() => (errorKindFilter = null)}
          >
            all ({run.totals.failed})
          </button>
          {#each run.results.errorKindCounts() as [kind, count] (kind)}
            <button
              class="chip {errorKindFilter === kind ? 'bg-blue-900 text-blue-100' : ''}"
              type="button"
              onclick={() => (errorKindFilter = kind)}
            >
              {kind} ({count})
            </button>
          {/each}
        </div>
      {/if}

      <ul class="max-h-64 overflow-y-auto text-xs">
        {#each run.results.page(activeTab, pageOffset[activeTab], PAGE_SIZE, activeTab === 'failed' ? (errorKindFilter ?? undefined) : undefined) as [id, r] (id)}
          <li class="border-b border-zinc-900 py-1 font-mono">
            {r.identifier}
            {#if r.error_kind}
              <span class="text-zinc-500"> [{r.error_kind}]</span>
            {/if}
            {#if r.error}
              <span class="text-red-300"> — {r.error}</span>
            {/if}
          </li>
        {/each}
      </ul>
      {#if (activeTab === 'failed' && errorKindFilter !== null ? run.results.countOfErrorKind(errorKindFilter) : run.results.countOf(activeTab)) > PAGE_SIZE}
        <div class="flex justify-between text-xs">
          <button
            class="btn btn-sm"
            type="button"
            disabled={pageOffset[activeTab] === 0}
            onclick={() =>
              (pageOffset = {
                ...pageOffset,
                [activeTab]: Math.max(0, pageOffset[activeTab] - PAGE_SIZE),
              })}
          >
            Prev
          </button>
          <button
            class="btn btn-sm"
            type="button"
            disabled={pageOffset[activeTab] + PAGE_SIZE >=
              (activeTab === 'failed' && errorKindFilter !== null
                ? run.results.countOfErrorKind(errorKindFilter)
                : run.results.countOf(activeTab))}
            onclick={() =>
              (pageOffset = {
                ...pageOffset,
                [activeTab]: pageOffset[activeTab] + PAGE_SIZE,
              })}
          >
            Next
          </button>
        </div>
      {/if}
    </div>
  {/if}
</div>
