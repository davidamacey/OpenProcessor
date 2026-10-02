<script lang="ts">
  /**
   * `/ingest` — bring images into the pool
   * (docs/design/ingest-ui-and-acceptance-plan-2026-09-24.md §A). The
   * page shell composes the sections below and owns no upload logic of
   * its own — that lives in `ingestRunController.svelte.ts`, driven
   * through `IngestRunPanel`.
   *
   * Every limit and caveat comes from the served `IngestConfig`
   * (`GET {API_PREFIX}/ingest/config`), fetched once on mount; the
   * upload UI renders only once it has loaded. The served
   * `upload.enabled: false` replaces the browser-upload panel with one
   * line; `batch.enabled: false` (or no `batch.source_roots`) hides the
   * server-path panel. The upload caveat banner renders only when the
   * backend serves `upload.persists_bytes: false`.
   */
  import IngestDropZone from '$lib/components/ingest/IngestDropZone.svelte';
  import IngestRunPanel from '$lib/components/ingest/IngestRunPanel.svelte';
  import IngestStatusTable from '$lib/components/ingest/IngestStatusTable.svelte';
  import RegionDrainPanel from '$lib/components/ingest/RegionDrainPanel.svelte';
  import ClusteringHandoff from '$lib/components/ingest/ClusteringHandoff.svelte';
  import IngestBatchPanel from '$lib/components/ingest/IngestBatchPanel.svelte';
  import { resolve } from '$app/paths';
  import { getIngestConfig } from '$lib/api';
  import { datasetsAvailability } from '$lib/datasets/datasetsAvailability.svelte';
  import { projectHref } from '$lib/projectPaths';
  import {
    resolveIngestConfig,
    serverPathIngestAvailable,
  } from '$lib/ingest/ingestConfig';
  import { regionProfileStore } from '$stores/regionProfile.svelte';
  import type { IngestFile } from '$lib/ingest/fileSource';
  import type { IngestRunState } from '$lib/ingest/ingestRunController.svelte';
  import type { IngestConfig, RegionDrain } from '$lib/types';

  // W10: the labeled-dataset import lives under /datasets; the link shows
  // only when the backend serves it (a one-shot probe per project).
  $effect(() => {
    void datasetsAvailability.init();
  });

  let servedConfig = $state<IngestConfig | null>(null);
  let configError = $state<string | null>(null);

  $effect(() => {
    void getIngestConfig()
      .then((c) => (servedConfig = c))
      .catch((e: unknown) => {
        configError = (e as Error)?.message ?? 'failed to load the ingest config';
      });
  });

  const config = $derived(servedConfig ? resolveIngestConfig(servedConfig) : null);

  let selectedFiles = $state<IngestFile[]>([]);
  let statusRefreshToken = $state(0);
  let runState = $state<IngestRunState>('idle');
  let drain = $state<RegionDrain | null>(null);
  let drainError = $state(false);
  let drainObservedAt = $state<number | null>(null);

  function onDrainUpdate(next: RegionDrain | null, observedAt: number): void {
    drain = next;
    drainError = next === null;
    drainObservedAt = observedAt;
  }
</script>

<!-- max-w-7xl matches /dashboard, not the narrower content pages —
     ClusteringHandoff embeds the real AutoLabelPanel, whose internal
     `justify-between` description/controls row was designed for that
     width and wraps badly in anything narrower (caught by the piece-9
     live-build screenshot review). -->
<div class="mx-auto max-w-7xl space-y-6 p-6">
  <div class="flex flex-wrap items-baseline justify-between gap-2">
    <h1 class="text-lg font-semibold text-zinc-100">Ingest</h1>
    {#if datasetsAvailability.available === true}
      <a
        class="text-sm text-blue-300 hover:underline"
        data-testid="ingest-dataset-import-link"
        href={resolve(projectHref('/datasets/import'))}>Import a labeled dataset</a
      >
    {/if}
  </div>

  {#if configError}
    <p class="text-sm text-red-300">Could not load the ingest config: {configError}</p>
  {:else if !config}
    <p class="text-sm text-zinc-500">Loading…</p>
  {:else}
    {#if !config.uploadEnabled}
      <p class="text-sm text-zinc-400" data-testid="ingest-upload-disabled">
        Browser uploads are disabled on this deployment.
      </p>
    {:else}
      {#if !config.uploadPersistsBytes}
        <p
          class="rounded border border-amber-900 bg-amber-950/30 p-3 text-xs text-amber-200"
        >
          This backend indexes uploads without keeping the image; use server-path ingest
          or the command-line uploader.
        </p>
      {/if}

      <section class="space-y-3 rounded-lg border border-zinc-800 p-4">
        <h2 class="text-sm font-semibold text-zinc-200">Upload</h2>
        <IngestDropZone
          acceptedExtensions={config.acceptedExtensions}
          onselect={(files) => (selectedFiles = files)}
        />
        {#if selectedFiles.length > 0}
          <p class="text-xs text-zinc-400">{selectedFiles.length} files selected</p>
        {/if}
        <IngestRunPanel
          files={selectedFiles}
          {config}
          onChunkDone={() => (statusRefreshToken += 1)}
          onStateChange={(s) => (runState = s)}
        />
      </section>
    {/if}

    {#if serverPathIngestAvailable(config)}
      <!-- Piece 11: server-path ingest, gated on the served `batch.enabled`
           and at least one served `batch.source_roots` entry — a
           deployment with batch ingest off, or no mounted server-side
           root, has nothing to offer here. -->
      <section class="space-y-3 rounded-lg border border-zinc-800 p-4">
        <h2 class="text-sm font-semibold text-zinc-200">Server-path ingest</h2>
        <IngestBatchPanel {config} />
      </section>
    {/if}

    <!-- The region drain only exists with a served region profile; without
         one there is no region worklog to wait on, so the panel (and its
         poll) is absent and the clustering gate sees no drain. -->
    <section class="grid gap-6" class:sm:grid-cols-2={regionProfileStore.configured}>
      <div class="rounded-lg border border-zinc-800 p-4">
        <IngestStatusTable refreshToken={statusRefreshToken} />
      </div>
      {#if regionProfileStore.configured}
        <div class="rounded-lg border border-zinc-800 p-4">
          <RegionDrainPanel
            pollIntervalS={config.regionDrainPollIntervalS}
            onUpdate={onDrainUpdate}
          />
        </div>
      {/if}
    </section>

    <section class="rounded-lg border border-zinc-800 p-4">
      <h2 class="mb-2 text-sm font-semibold text-zinc-200">Clustering</h2>
      <ClusteringHandoff {runState} {drain} {drainError} {drainObservedAt} />
    </section>
  {/if}
</div>
