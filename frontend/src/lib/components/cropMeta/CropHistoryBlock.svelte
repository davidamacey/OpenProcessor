<script lang="ts">
  import type { CropHistoryEntry } from '$lib/types';
  import { classSourcesStore } from '$stores/classSources.svelte';
  import { formatTimestamp } from '$lib/format/date';

  interface Props {
    historyEntries: CropHistoryEntry[] | null;
    historyError: string | null;
    historyLoading: boolean;
  }

  let { historyEntries, historyError, historyLoading }: Props = $props();
</script>

<div class="mt-4 border-t border-zinc-800 pt-3">
  <div class="mb-1.5 text-[10px] uppercase tracking-wider text-zinc-500">History</div>
  {#if historyLoading}
    <p class="text-[11px] text-zinc-500">Loading…</p>
  {:else if historyError}
    <p class="text-[11px] text-red-300">History unavailable: {historyError}</p>
  {:else if !historyEntries || historyEntries.length === 0}
    <p class="text-[11px] text-zinc-500">No prior writes recorded.</p>
  {:else}
    <ul class="space-y-1 text-[11px]">
      {#each historyEntries as entry, i (i)}
        <!-- K7 (visual audit 2026-09-24): every row names its resulting
             class ("no class" when the write left none) with the source
             labelled, and a short timestamp (raw value in the tooltip). -->
        <li
          class="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-zinc-400"
          data-testid="history-entry"
        >
          <span class="font-mono text-zinc-300">{entry.writer ?? 'unknown'}</span>
          <span class={entry.class_name ? '' : 'text-zinc-500'}
            >→ {entry.class_name ?? 'no class'}</span
          >
          {#if entry.class_source}
            <span class="text-zinc-600"
              >source: {classSourcesStore.labelFor(entry.class_source) ||
                entry.class_source}</span
            >
          {/if}
          {#if entry.at}
            <span class="font-mono text-zinc-600" title={entry.at}
              >{formatTimestamp(entry.at)}</span
            >
          {/if}
        </li>
      {/each}
    </ul>
  {/if}
</div>
