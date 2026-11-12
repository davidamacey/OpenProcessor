<script lang="ts">
  import { apiErrorText } from '$lib/api_errors';
  /**
   * `/ingest` — bring images into the pool
   * (Ingest UI plan §A). The
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
  import RegionDrainDetails from '$lib/components/ingest/RegionDrainDetails.svelte';
  import IngestPipelineRail from '$lib/components/ingest/IngestPipelineRail.svelte';
  import ClusteringHandoff from '$lib/components/ingest/ClusteringHandoff.svelte';
  import BulkIngestPanel from '$lib/components/ingest/BulkIngestPanel.svelte';
  import IngestSourcesPanel from '$lib/components/ingest/IngestSourcesPanel.svelte';
  import IngestBatchPanel from '$lib/components/ingest/IngestBatchPanel.svelte';
  import IngestDetectorCard from '$lib/components/ingest/IngestDetectorCard.svelte';
  import RegionStagePanel from '$lib/components/ingest/RegionStagePanel.svelte';
  import { page } from '$app/state';
  import Chip from '$components/ui/Chip.svelte';
  import PageHeader from '$components/ui/PageHeader.svelte';
  import PageShell from '$components/ui/PageShell.svelte';
  import Tabs from '$components/ui/Tabs.svelte';
  import { getIngestConfig } from '$lib/api_ingest';
  import { datasetsAvailability } from '$lib/datasets/datasetsAvailability.svelte';
  import { projectHref } from '$lib/projectPaths';
  import {
    resolveIngestConfig,
    serverPathIngestAvailable,
  } from '$lib/ingest/ingestConfig';
  import Card from '$components/ui/Card.svelte';
  import { regionProfileStore } from '$stores/regionProfile.svelte';
  import type { IngestFile } from '$lib/ingest/fileSource';
  import type { IngestRunState } from '$lib/ingest/ingestRunController.svelte';
  import type { IngestConfig, RegionDrain } from '$lib/types_ingest';

  // the labeled-dataset import lives under /datasets; the link shows
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
        configError = apiErrorText(e) ?? 'failed to load the ingest config';
      });
  });

  const config = $derived(servedConfig ? resolveIngestConfig(servedConfig) : null);

  // The tab is the address: `?tab=`, so reload and Back keep it. Import dataset is its own page.
  const TAB_IDS = ['upload', 'folder', 'sources', 'activity'] as const;
  const tab = $derived.by(() => {
    const t = page.url.searchParams.get('tab');
    return TAB_IDS.find((id) => id === t) ?? 'upload';
  });
  const tabItems = $derived([
    { id: 'upload', label: 'Upload', href: projectHref('/ingest') },
    ...(config && serverPathIngestAvailable(config)
      ? [
          {
            id: 'folder',
            label: 'Server folder',
            href: projectHref('/ingest?tab=folder'),
          },
        ]
      : []),
    { id: 'sources', label: 'Sources', href: projectHref('/ingest?tab=sources') },
    ...(datasetsAvailability.available === true
      ? [{ id: 'import', label: 'Import dataset', href: projectHref('/datasets/import') }]
      : []),
    { id: 'activity', label: 'Activity', href: projectHref('/ingest?tab=activity') },
  ]);
  const shownTab = $derived(tabItems.some((t) => t.id === tab) ? tab : 'upload');

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

<PageShell width="wide">
  <PageHeader title="Ingest" description="Bring images into the project.">
    {#snippet meta()}
      {#if servedConfig?.detector}
        <Chip tone="info" title="The detector that proposes boxes at ingest"
          >Detector {servedConfig.detector.model} v{servedConfig.detector.version}</Chip
        >
      {/if}
    {/snippet}
  </PageHeader>

  {#if configError}
    <p class="text-body text-danger-text">
      Could not load the ingest config: {configError}
    </p>
  {:else if !config}
    <p class="text-body text-fg-3">Loading…</p>
  {:else}
    <div class="grid gap-6 lg:grid-cols-[minmax(0,1fr)_20rem]">
      <div class="min-w-0">
        <Tabs label="Ingest" value={shownTab} items={tabItems} />

        {#if shownTab === 'upload'}
          <div class="space-y-4">
            {#if servedConfig}
              <IngestDetectorCard config={servedConfig} />
            {/if}
            {#if !config.uploadEnabled}
              <p class="text-body text-fg-2" data-testid="ingest-upload-disabled">
                Browser uploads are disabled on this deployment.
              </p>
            {:else}
              {#if !config.uploadPersistsBytes}
                <p
                  class="rounded border border-warn-line bg-warn-soft p-3 text-xs text-warn-text"
                >
                  This backend indexes uploads without keeping the image; use server-path
                  ingest or the command-line uploader.
                </p>
              {/if}
              <section class="space-y-3">
                <IngestDropZone
                  acceptedExtensions={config.acceptedExtensions}
                  onselect={(files) => (selectedFiles = files)}
                />
                {#if selectedFiles.length > 0}
                  <p class="text-xs text-fg-2">{selectedFiles.length} files selected</p>
                {/if}
                <IngestRunPanel
                  files={selectedFiles}
                  {config}
                  onChunkDone={() => (statusRefreshToken += 1)}
                  onStateChange={(s) => (runState = s)}
                />
              </section>
            {/if}
          </div>
        {:else if shownTab === 'folder'}
          <div class="space-y-6">
            <section class="space-y-3">
              <h2 class="text-title font-semibold text-fg-1">
                Bulk ingest (folder or manifest)
              </h2>
              <BulkIngestPanel />
            </section>
            <!-- Server-path ingest, gated on the served `batch.enabled` and at least one
                 served `batch.source_roots` entry. -->
            <section class="space-y-3">
              <h2 class="text-title font-semibold text-fg-1">Server-path ingest</h2>
              <IngestBatchPanel {config} />
            </section>
          </div>
        {:else if shownTab === 'sources'}
          <section class="space-y-3">
            <h2 class="text-title font-semibold text-fg-1">
              Sources (S3 buckets and watched folders)
            </h2>
            <IngestSourcesPanel />
          </section>
        {:else}
          <div class="space-y-6">
            {#if regionProfileStore.configured}
              <Card>
                <RegionDrainDetails
                  {drain}
                  error={drainError ? 'worklog read failed' : null}
                  lastChecked={drainObservedAt}
                />
                <RegionStagePanel />
              </Card>
            {/if}
            <ClusteringHandoff {runState} {drain} {drainError} {drainObservedAt} />
          </div>
        {/if}
      </div>

      <IngestPipelineRail
        refreshToken={statusRefreshToken}
        regionConfigured={regionProfileStore.configured}
        drainPollIntervalS={config.regionDrainPollIntervalS}
        {onDrainUpdate}
      />
    </div>
  {/if}
</PageShell>
