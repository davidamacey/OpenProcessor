<script lang="ts">
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
  /**
   * `/ingest` — the tab bodies (the layout owns the header, the tab strip and the Pipeline
   * rail). This page owns no upload logic of its own: that lives in
   * `ingestRunController.svelte.ts`, driven through `IngestRunPanel`.
   *
   * The served `upload.enabled: false` replaces the browser-upload panel with one line;
   * `batch.enabled: false` (or no `batch.source_roots`) hides the server-path panel. The upload
   * caveat banner renders only when the backend serves `upload.persists_bytes: false`.
   */
  import ImportWizardPanel from '$lib/components/datasets/ImportWizardPanel.svelte';
  import IngestDropZone from '$lib/components/ingest/IngestDropZone.svelte';
  import IngestRunPanel from '$lib/components/ingest/IngestRunPanel.svelte';
  import RegionDrainDetails from '$lib/components/ingest/RegionDrainDetails.svelte';
  import ClusteringHandoff from '$lib/components/ingest/ClusteringHandoff.svelte';
  import BulkIngestPanel from '$lib/components/ingest/BulkIngestPanel.svelte';
  import IngestSourcesPanel from '$lib/components/ingest/IngestSourcesPanel.svelte';
  import IngestBatchPanel from '$lib/components/ingest/IngestBatchPanel.svelte';
  import IngestDetectorCard from '$lib/components/ingest/IngestDetectorCard.svelte';
  import RegionStagePanel from '$lib/components/ingest/RegionStagePanel.svelte';
  import Card from '$components/ui/Card.svelte';
  import Section from '$components/ui/Section.svelte';
  import { getIngestShell } from '$lib/ingest/ingestShell.svelte';
  import { regionProfileStore } from '$stores/regionProfile.svelte';
  import type { IngestFile } from '$lib/ingest/fileSource';

  const shell = getIngestShell();
  // The layout renders this page only once the config has loaded.
  const config = $derived(shell.config!);
  const shownTab = $derived(shell.tab);

  let selectedFiles = $state<IngestFile[]>([]);
</script>

{#if config}
  <div class="pt-4">
    {#if shownTab === 'upload'}
      <div class="space-y-4">
        {#if !config.uploadEnabled}
          <p class="text-body text-fg-2" data-testid="ingest-upload-disabled">
            Browser uploads are disabled on this deployment.
          </p>
        {:else}
          {#if !config.uploadPersistsBytes}
            <p
              class="rounded border border-warn-line bg-warn-soft p-3 text-meta text-warn-text"
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
              <p class="text-meta text-fg-2">{selectedFiles.length} files selected</p>
            {/if}
            <IngestRunPanel
              files={selectedFiles}
              {config}
              onChunkDone={() => (shell.statusRefreshToken += 1)}
              onStateChange={(s) => (shell.runState = s)}
            />
          </section>
        {/if}
        {#if shell.servedConfig}
          <details data-testid="ingest-detector-details">
            <summary class="cursor-pointer text-meta text-fg-2">Detector details</summary>
            <div class="mt-2"><IngestDetectorCard config={shell.servedConfig} /></div>
          </details>
        {/if}
      </div>
    {:else if shownTab === 'folder'}
      <div class="space-y-2">
        <Section
          title="Bulk ingest (folder or manifest)"
          description="Index a whole folder or a list of files that already sit on the server."
        >
          <Card><BulkIngestPanel /></Card>
        </Section>
        <!-- Server-path ingest, gated on the served `batch.enabled` and at least one
             served `batch.source_roots` entry. -->
        <Section
          title="Server-path ingest"
          description="Index specific files by their path under a server root."
        >
          <Card><IngestBatchPanel {config} /></Card>
        </Section>
      </div>
    {:else if shownTab === 'sources'}
      <Section
        title="Sources (S3 buckets and watched folders)"
        description="Places the server pulls images from on its own."
      >
        <IngestSourcesPanel />
      </Section>
    {:else if shownTab === 'import'}
      <ImportWizardPanel />
    {:else}
      <div class="space-y-6">
        {#if regionProfileStore.configured}
          <Card>
            <RegionDrainDetails
              drain={shell.drain}
              error={shell.drainError ? 'worklog read failed' : null}
              lastChecked={shell.drainObservedAt}
            />
            <RegionStagePanel />
          </Card>
        {/if}
        <ClusteringHandoff
          runState={shell.runState}
          drain={shell.drain}
          drainError={shell.drainError}
          drainObservedAt={shell.drainObservedAt}
        />
      </div>
    {/if}
  </div>
{/if}
