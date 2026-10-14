<script lang="ts">
  /**
   * Registered ingest sources: S3 buckets and server folders that are scanned,
   * watched, run on a schedule or fed by webhook, all draining into the one bulk
   * runner. The backend serves the status and decides what can run; this panel
   * renders it, sends the mutations and refreshes on the served interval.
   */
  import { onDestroy, onMount } from 'svelte';
  import SourceCard from '$lib/components/ingest/SourceCard.svelte';
  import SourceForm from '$lib/components/ingest/SourceForm.svelte';
  import { IngestSources } from '$lib/ingest/sourcesController.svelte';

  interface Props {
    /** Injected in tests. */
    controller?: IngestSources;
  }
  let { controller = new IngestSources() }: Props = $props();

  onMount(() => void controller.init());
  onDestroy(() => controller.stop());
</script>

<div class="space-y-4" data-testid="ingest-sources">
  {#if controller.loadError && !controller.list}
    <p class="text-xs text-red-300" data-testid="sources-load-error">
      Could not load sources: {controller.loadError}
    </p>
  {:else if !controller.list}
    <p class="text-xs text-zinc-500">Loading…</p>
  {:else}
    <p class="text-xs text-zinc-400">
      A source is a bucket prefix or a server folder that keeps feeding this project: scan
      it now, watch it, run it on a schedule or let a webhook announce new files. Images
      already stored are skipped.
    </p>

    {#if controller.loadError}
      <p class="text-xs text-amber-300">Could not refresh: {controller.loadError}</p>
    {/if}
    {#if controller.notice}
      <p class="text-xs text-emerald-300" data-testid="sources-notice">
        {controller.notice}
      </p>
    {/if}
    {#if controller.actionError && controller.editing === null}
      <p class="text-xs text-red-300" data-testid="sources-action-error">
        {controller.actionError}
      </p>
    {/if}

    {#each controller.sources as source (source.source_id)}
      {#if controller.editing === source.source_id}
        <SourceForm {controller} />
      {:else}
        <SourceCard {source} {controller} />
      {/if}
    {:else}
      <p class="text-xs text-zinc-500" data-testid="sources-empty">No sources yet.</p>
    {/each}

    {#if controller.editing === 'new'}
      <SourceForm {controller} />
    {:else}
      <button
        class="btn btn-sm"
        type="button"
        data-testid="source-add"
        onclick={() => controller.openCreate()}
      >
        Add a source
      </button>
    {/if}
  {/if}
</div>
