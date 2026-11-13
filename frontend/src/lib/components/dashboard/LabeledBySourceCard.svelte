<script lang="ts">
  /**
   * "Labeled by source": the served count of every label source and the reminder that a label
   * is not ground truth until a person validates it. The dashboard shows it up front; nothing is
   * summed here.
   */
  import type { DatasetStats } from '$lib/api_stats';
  import { fmt, labeledRowsOf } from '$lib/datasetStats';

  let { stats, class: className = '' }: { stats: DatasetStats | null; class?: string } =
    $props();

  const labeledTotal = $derived(stats?.labeled?.total ?? 0);
  const labeledRows = $derived(labeledRowsOf(stats));
  const note = $derived(
    `A label is not ground truth until a human validates it. Only ${fmt(stats?.validated ?? 0)} of ${fmt(stats?.total_crops ?? 0)} crops are validated; VLM and detector labels are suggestions.`,
  );
</script>

{#if stats}
  <div class="surface p-4 {className}" data-testid="labeled-by-source">
    <header class="mb-3 flex items-center justify-between">
      <h3 class="text-body font-semibold text-fg-2">Labeled by source</h3>
      <span class="text-meta text-fg-3" data-testid="labeled-total"
        >{fmt(labeledTotal)} total</span
      >
    </header>
    <p class="mb-2 text-meta text-fg-3" data-testid="labeled-not-ground-truth">
      {note}
    </p>
    {#if labeledRows.length === 0}
      <p class="text-body text-fg-3">No labels yet.</p>
    {:else}
      <ul class="space-y-1.5">
        {#each labeledRows as row (row.key)}
          <li class="flex items-center gap-3 text-meta">
            <span class="w-24 shrink-0 text-fg-2">{row.label}</span>
            <div class="relative h-3 grow overflow-hidden rounded bg-surface-1">
              <div
                class="h-full {row.tone}"
                style:width="{Math.max(0.5, row.pct)}%"
                title="{row.pct}%"
              ></div>
            </div>
            <span class="w-20 shrink-0 text-right text-fg-2 tabular-nums">
              {fmt(row.count)}
            </span>
            <span class="w-12 shrink-0 text-right text-fg-3 tabular-nums">
              {row.pct.toFixed(1)}%
            </span>
          </li>
        {/each}
      </ul>
    {/if}
  </div>
{/if}
