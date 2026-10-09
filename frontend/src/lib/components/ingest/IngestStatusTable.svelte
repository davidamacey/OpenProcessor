<script lang="ts">
  import { apiErrorText } from '$lib/api';
  /**
   * The `by_source` ingest status table plus `total` and `by_day`
   * (docs/design/ingest-ui-and-acceptance-plan-2026-09-24.md §A.2).
   * Refreshes every 10s while the page is visible, plus once whenever
   * `refreshToken` changes (the run panel bumps it, debounced, after
   * each completed chunk).
   */
  import { onDestroy, onMount } from 'svelte';
  import { getIngestStatus } from '$lib/api';
  import type { IngestStatus } from '$lib/types';

  interface Props {
    /** Bump this (e.g. a counter) to trigger an out-of-band refresh. */
    refreshToken?: number;
  }
  let { refreshToken = 0 }: Props = $props();

  let status = $state<IngestStatus | null>(null);
  let error = $state<string | null>(null);
  let intervalId: ReturnType<typeof setInterval> | undefined;

  async function load(): Promise<void> {
    try {
      status = await getIngestStatus();
      error = null;
    } catch (e) {
      error = apiErrorText(e);
    }
  }

  function onVisibilityChange(): void {
    if (document.visibilityState === 'visible') void load();
  }

  // The first read is the refreshToken effect below (it also runs on mount).
  onMount(() => {
    intervalId = setInterval(() => {
      if (document.visibilityState === 'visible') void load();
    }, 10_000);
    document.addEventListener('visibilitychange', onVisibilityChange);
  });

  onDestroy(() => {
    if (intervalId) clearInterval(intervalId);
    document.removeEventListener('visibilitychange', onVisibilityChange);
  });

  $effect(() => {
    void refreshToken;
    void load();
  });
</script>

<div>
  <h3 class="mb-2 text-sm font-semibold text-zinc-200">Ingest status</h3>
  {#if error}
    <p class="text-xs text-red-300">Ingest status unavailable: {error}</p>
  {:else if !status}
    <p class="text-xs text-zinc-500">Loading…</p>
  {:else}
    <p class="mb-2 text-xs text-zinc-400">
      Total images:
      <span class="font-mono text-zinc-200" data-testid="ingest-status-total"
        >{status.total}</span
      >
    </p>
    {#if status.by_source.length === 0}
      <p class="text-xs text-zinc-500">No sources yet.</p>
    {:else}
      <table class="w-full text-xs">
        <thead>
          <tr class="text-left text-zinc-500">
            <th class="pb-1 font-normal">Source</th>
            <th class="pb-1 font-normal">Images</th>
          </tr>
        </thead>
        <tbody>
          {#each status.by_source as bucket (bucket.key)}
            <tr class="border-t border-zinc-800">
              <td class="py-1 font-mono text-zinc-300">{bucket.key}</td>
              <td class="py-1 font-mono text-zinc-300">{bucket.doc_count}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    {/if}
  {/if}
</div>
