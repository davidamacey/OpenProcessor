<script lang="ts">
  /**
   * Piece 11 (docs/design/ingest-ui-and-acceptance-plan-2026-09-24.md
   * §A.6) — server-path ingest for images already reachable inside the
   * API container. `POST {API_PREFIX}/ingest/batch` predates OpenProcessor
   * #36; what #36 (BA-2/BA-5) added is the served `source_roots` list
   * this panel is gated on, the served `batch.max_items` cap, and the
   * `label_txt_path` root guard so a client-typed path can't escape the
   * configured roots any more than the image path itself could.
   *
   * One synchronous `POST /ingest/batch` call per submit — not chunked
   * like the upload run controller, since a server-path batch has no
   * per-file byte cost to the browser and the backend itself already
   * batches the detector inference internally.
   */
  import { ApiError, ingestBatch } from '$lib/api';
  import type { BatchIngestResponse } from '$lib/types';
  import type { ResolvedIngestConfig } from '$lib/ingest/ingestConfig';
  import { toastStore } from '$stores/toast.svelte';

  interface Props {
    config: ResolvedIngestConfig;
  }
  let { config }: Props = $props();

  let source = $state('batch');
  let pathsText = $state('');
  let labelTxtPathsText = $state('');
  let labelSource = $state('human');
  let detectMismatches = $state(false);
  let submitting = $state(false);
  let result = $state<BatchIngestResponse | null>(null);
  let submitError = $state<string | null>(null);
  let errorKindFilter = $state<string | null>(null);

  function splitLines(text: string): string[] {
    return text
      .split('\n')
      .map((l) => l.trim())
      .filter((l) => l.length > 0);
  }

  const paths = $derived(splitLines(pathsText));
  const labelPaths = $derived(splitLines(labelTxtPathsText));
  const overLimit = $derived(paths.length > config.batchMaxItems);

  const failedResults = $derived(
    result ? result.results.filter((r) => r.status === 'failed') : [],
  );
  const secondaryFailures = $derived(
    result ? result.results.filter((r) => r.secondary_detector_error) : [],
  );
  const errorKindCounts = $derived.by(() => {
    // eslint-disable-next-line svelte/prefer-svelte-reactivity -- local tally map consumed synchronously within this computation, never stored in reactive state
    const counts = new Map<string, number>();
    for (const r of failedResults) {
      const kind = r.error_kind ?? 'unknown';
      counts.set(kind, (counts.get(kind) ?? 0) + 1);
    }
    return [...counts.entries()].sort((a, b) => b[1] - a[1]);
  });
  const visibleFailures = $derived(
    errorKindFilter === null
      ? failedResults
      : failedResults.filter((r) => (r.error_kind ?? 'unknown') === errorKindFilter),
  );

  async function submit(): Promise<void> {
    if (paths.length === 0 || overLimit) return;
    submitting = true;
    submitError = null;
    result = null;
    errorKindFilter = null;
    try {
      result = await ingestBatch({
        items: paths.map((path, i) => ({
          path,
          source,
          label_txt_path: labelPaths[i] || null,
        })),
        label_source: labelSource,
        detect_mismatches: detectMismatches,
      });
      toastStore.success(
        `Batch ingest ${result.status}: ${result.summary.successful} ingested, ` +
          `${result.summary.duplicates} duplicate, ${result.summary.failed} failed`,
      );
    } catch (e) {
      submitError =
        e instanceof ApiError ? (e.detail ?? e.message) : (e as Error).message;
      toastStore.error(`Batch ingest failed: ${submitError}`);
    } finally {
      submitting = false;
    }
  }
</script>

<div class="space-y-3">
  <div>
    <p class="text-xs text-zinc-500">
      Paths must be under one of these folders on the server:
    </p>
    <ul class="mt-1 text-xs text-zinc-300">
      {#each config.batchSourceRoots as root (root)}
        <li class="font-mono">{root}</li>
      {/each}
    </ul>
  </div>

  <div class="flex flex-wrap items-end gap-3">
    <label class="text-xs text-zinc-400">
      Source tag
      <input class="input input-sm block" bind:value={source} disabled={submitting} />
    </label>
    <label class="text-xs text-zinc-400">
      Label source
      <input
        class="input input-sm block"
        bind:value={labelSource}
        disabled={submitting}
      />
    </label>
    <label class="flex items-center gap-1.5 text-xs text-zinc-400">
      <input type="checkbox" bind:checked={detectMismatches} disabled={submitting} />
      Detect label/detector mismatches
    </label>
  </div>

  <label class="block text-xs text-zinc-400">
    Image paths (one per line, absolute paths inside the API container)
    <textarea
      class="input block h-24 w-full font-mono text-xs"
      bind:value={pathsText}
      disabled={submitting}></textarea>
  </label>
  <label class="block text-xs text-zinc-400">
    Companion YOLO .txt label paths (optional, one per line — aligned by line number with
    the image paths above)
    <textarea
      class="input block h-16 w-full font-mono text-xs"
      bind:value={labelTxtPathsText}
      disabled={submitting}></textarea>
  </label>

  {#if overLimit}
    <p class="text-xs text-red-300">
      {paths.length} paths exceeds this backend's per-request limit of {config.batchMaxItems}
      — split into smaller batches.
    </p>
  {/if}

  <div class="flex gap-2">
    <button
      class="btn btn-primary btn-sm"
      type="button"
      disabled={submitting || paths.length === 0 || overLimit}
      onclick={() => void submit()}
    >
      {submitting
        ? 'Ingesting…'
        : `Ingest ${paths.length} path${paths.length === 1 ? '' : 's'}`}
    </button>
  </div>

  {#if submitError}
    <p class="text-xs text-red-300">{submitError}</p>
  {/if}

  {#if result}
    <div class="flex flex-wrap gap-2 text-xs">
      <span class="chip">status {result.status}</span>
      <span class="chip">successful {result.summary.successful}</span>
      <span class="chip">duplicate {result.summary.duplicates}</span>
      <span class="chip">failed {result.summary.failed}</span>
      <span class="chip">crops indexed {result.summary.crops_indexed}</span>
      {#if result.summary.labels_imported > 0}
        <span class="chip">labels imported {result.summary.labels_imported}</span>
      {/if}
      {#if (result.summary.secondary_detector_failures ?? 0) > 0}
        <span
          class="chip border-amber-700 text-amber-200"
          data-testid="batch-secondary-failures"
          >secondary detector failed {result.summary.secondary_detector_failures}</span
        >
      {/if}
    </div>

    {#if secondaryFailures.length > 0}
      <!-- d72cc63: these images ingested with primary-detector crops only. -->
      <ul class="max-h-32 overflow-y-auto text-xs" data-testid="batch-secondary-list">
        {#each secondaryFailures as r (r.image_path)}
          <li class="border-b border-zinc-900 py-1 font-mono">
            {r.image_path}
            <span class="text-amber-300">
              — secondary detector: {r.secondary_detector_error}</span
            >
          </li>
        {/each}
      </ul>
    {/if}

    {#if errorKindCounts.length > 0}
      <div class="flex flex-wrap gap-1 text-xs">
        <button
          class="chip {errorKindFilter === null ? 'bg-blue-900 text-blue-100' : ''}"
          type="button"
          onclick={() => (errorKindFilter = null)}
        >
          all ({failedResults.length})
        </button>
        {#each errorKindCounts as [kind, count] (kind)}
          <button
            class="chip {errorKindFilter === kind ? 'bg-blue-900 text-blue-100' : ''}"
            type="button"
            onclick={() => (errorKindFilter = kind)}
          >
            {kind} ({count})
          </button>
        {/each}
      </div>
      <ul class="max-h-48 overflow-y-auto text-xs">
        {#each visibleFailures as r (r.image_path)}
          <li class="border-b border-zinc-900 py-1 font-mono">
            {r.image_path}
            {#if r.error_kind}
              <span class="text-zinc-500"> [{r.error_kind}]</span>
            {/if}
            {#if r.error}
              <span class="text-red-300"> — {r.error}</span>
            {/if}
          </li>
        {/each}
      </ul>
    {/if}
  {/if}
</div>
