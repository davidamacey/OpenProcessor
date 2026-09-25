<script lang="ts">
  /**
   * `/ingest` — bring images into the pool
   * (docs/design/ingest-ui-and-acceptance-plan-2026-09-24.md §A). The
   * page shell composes the sections below and owns no upload logic of
   * its own — that lives in `ingestRunController.svelte.ts`, driven
   * through `IngestRunPanel`.
   *
   * §A.7 "absent, not disabled": `ingestAvailability.available === false`
   * renders the absence copy with no other requests fired at all.
   *
   * §A.7 / F1 upload caveat: BA-2 (`GET {API_PREFIX}/ingest/config`)
   * isn't served by any backend yet, so `resolveIngestConfig(null)` is
   * always the interim config — upload mode always renders WITH the
   * caveat banner unless the operator has explicitly set
   * `PUBLIC_CROPWRIGHT_INGEST_UPLOAD=1`.
   *
   * Plan deviation (recorded per CLAUDE.md's "trust the code" rule): the
   * plan names the override env var `CROPWRIGHT_INGEST_UPLOAD` (no
   * prefix). `vite.config.ts`'s `envPrefix` only exposes `VITE_`/
   * `PUBLIC_`-prefixed vars to `import.meta.env` in the client bundle —
   * an unprefixed var is simply undefined there. This uses
   * `PUBLIC_CROPWRIGHT_INGEST_UPLOAD` instead, matching every other
   * client-visible env var in this codebase
   * (`PUBLIC_API_PREFIX`/`PUBLIC_TRITON_API_URL`/`PUBLIC_APP_NAME`).
   */
  import IngestDropZone from '$lib/components/ingest/IngestDropZone.svelte';
  import IngestRunPanel from '$lib/components/ingest/IngestRunPanel.svelte';
  import IngestStatusTable from '$lib/components/ingest/IngestStatusTable.svelte';
  import RegionDrainPanel from '$lib/components/ingest/RegionDrainPanel.svelte';
  import ClusteringHandoff from '$lib/components/ingest/ClusteringHandoff.svelte';
  import { ingestAvailability } from '$lib/ingest/ingestAvailability.svelte';
  import { resolveIngestConfig } from '$lib/ingest/ingestConfig';
  import { regionProfileStore } from '$stores/regionProfile.svelte';
  import type { IngestFile } from '$lib/ingest/fileSource';
  import type { IngestRunState } from '$lib/ingest/ingestRunController.svelte';
  import type { RegionDrain } from '$lib/types';

  $effect(() => {
    void ingestAvailability.init();
  });

  const config = resolveIngestConfig(null);

  const uploadOverride =
    (import.meta.env?.PUBLIC_CROPWRIGHT_INGEST_UPLOAD as string | undefined) === '1';

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
  <h1 class="text-lg font-semibold text-zinc-100">Ingest</h1>

  {#if ingestAvailability.available === false}
    <p class="text-sm text-zinc-400">This backend does not provide ingest.</p>
  {:else if ingestAvailability.available === null}
    <!-- Deliberately not the nav link's optimistic render: §A.7 requires
         that visiting /ingest directly when the backend lacks the
         router fires NO other requests at all. Rendering the upload
         section here (which mounts IngestStatusTable/RegionDrainPanel,
         each firing its own GET on mount) before the probe resolves
         would violate that the moment it later turns out `false`. -->
    <p class="text-sm text-zinc-500">Loading…</p>
  {:else}
    {#if config.uploadPersistsBytes === false}
      <p
        class="rounded border border-amber-900 bg-amber-950/30 p-3 text-xs text-amber-200"
      >
        This backend indexes uploads without keeping the image; use server-path ingest or
        the command-line uploader.
      </p>
    {:else if !uploadOverride}
      <p
        class="rounded border border-amber-900 bg-amber-950/30 p-3 text-xs text-amber-200"
      >
        Uploaded images can be browsed only if the backend stores upload bytes (not yet
        advertised by this backend).
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
