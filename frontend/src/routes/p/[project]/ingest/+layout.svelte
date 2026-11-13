<script lang="ts">
  /**
   * The Ingest section's layout: the page header with the detector chip, the URL-bound tab strip
   * (Upload, Server folder, Sources, Import dataset, Activity) and the Pipeline rail, for every
   * tab and for the import job page (`/ingest/imports/<id>`). The pages render only their tab
   * body. Every limit and caveat comes from the served `IngestConfig`
   * (`GET {API_PREFIX}/ingest/config`), fetched once on mount.
   */
  import type { Snippet } from 'svelte';
  import { page } from '$app/state';
  import Chip from '$components/ui/Chip.svelte';
  import PageHeader from '$components/ui/PageHeader.svelte';
  import PageShell from '$components/ui/PageShell.svelte';
  import Tabs from '$components/ui/Tabs.svelte';
  import IngestPipelineRail from '$lib/components/ingest/IngestPipelineRail.svelte';
  import { datasetsAvailability } from '$lib/datasets/datasetsAvailability.svelte';
  import { serverPathIngestAvailable } from '$lib/ingest/ingestConfig';
  import { IngestShell, setIngestShell } from '$lib/ingest/ingestShell.svelte';
  import { projectHref } from '$lib/projectPaths';
  import { regionProfileStore } from '$stores/regionProfile.svelte';

  let { children }: { children: Snippet } = $props();

  const shell = setIngestShell(new IngestShell(() => page.url));

  // The Import dataset tab shows only when the backend serves it (a one-shot probe per project).
  $effect(() => {
    void datasetsAvailability.init();
  });
  $effect(() => {
    shell.load();
  });

  const config = $derived(shell.config);
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
      ? [
          {
            id: 'import',
            label: 'Import dataset',
            href: projectHref('/ingest?tab=import'),
          },
        ]
      : []),
    { id: 'activity', label: 'Activity', href: projectHref('/ingest?tab=activity') },
  ]);
  const shownTab = $derived(
    tabItems.some((t) => t.id === shell.tab) ? shell.tab : 'upload',
  );
</script>

<PageShell width="wide">
  <PageHeader title="Ingest" description="Bring images into the project.">
    {#snippet meta()}
      {#if shell.servedConfig?.detector}
        <Chip tone="info" title="The detector that proposes boxes at ingest"
          >Detector {shell.servedConfig.detector.model} v{shell.servedConfig.detector
            .version}</Chip
        >
      {/if}
    {/snippet}
  </PageHeader>

  {#if shell.configError}
    <p class="text-body text-danger-text">
      Could not load the ingest config: {shell.configError}
    </p>
  {:else if !config}
    <p class="text-body text-fg-3">Loading…</p>
  {:else}
    <div class="grid gap-6 lg:grid-cols-[minmax(0,1fr)_20rem]">
      <div class="min-w-0">
        <Tabs label="Ingest" value={shownTab} items={tabItems} />
        {@render children()}
      </div>

      <IngestPipelineRail
        refreshToken={shell.statusRefreshToken}
        regionConfigured={regionProfileStore.configured}
        drainPollIntervalS={config.regionDrainPollIntervalS}
        onDrainUpdate={shell.onDrainUpdate}
      />
    </div>
  {/if}
</PageShell>
