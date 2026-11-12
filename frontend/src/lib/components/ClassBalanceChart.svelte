<script lang="ts">
  /**
   * `/dashboard`'s per-class balance bars (visual audit 2026-09-24, D2).
   * Bars are sized by the served `trainable` count, with the served test
   * holdout alongside; classes at 0 validated collapse into one line and any
   * overflow past `limit` is counted, never silently dropped.
   */
  import { buildClassBalance, type ClassBalanceRow } from '$lib/dashboard/classBalance';
  import { holdoutByClass } from '$lib/holdoutCounts';
  import type { ClassThresholds } from '$lib/types_classes';
  import type { TestHoldoutStats } from '$lib/types_export';

  interface Props {
    rows: Array<ClassBalanceRow & { adequacy: string }>;
    holdout: TestHoldoutStats | null;
    thresholds?: ClassThresholds | null;
    limit?: number;
  }

  let { rows, holdout, thresholds = null, limit = 30 }: Props = $props();

  const view = $derived(buildClassBalance(rows, holdoutByClass(holdout), limit));

  function tierBg(t: string): string {
    if (t === 'ok') return 'bg-success';
    if (t === 'warn') return 'bg-warn';
    return 'bg-danger';
  }
</script>

<header class="mb-3 flex flex-wrap items-center justify-between gap-2">
  <h2 class="text-body font-semibold text-fg-2">Class balance (validated)</h2>
  <!-- m6: the served thresholds, never a hardcoded legend. -->
  <span class="text-meta text-fg-3">
    {#if thresholds}
      green &ge;{thresholds.warn_below} · orange {thresholds.block_below}–{thresholds.warn_below -
        1} · red &lt;{thresholds.block_below}
    {/if}
  </span>
</header>

{#if rows.length === 0}
  <p class="text-body text-fg-3">No classes yet.</p>
{:else}
  {#if view.bars.length > 0}
    <p class="mb-2 text-caption text-fg-3">
      Bars show trainable crops (frozen test holdout and excluded crops left out).
    </p>
    <ul class="space-y-1.5" data-testid="class-balance-bars">
      {#each view.bars as row (row.class_id)}
        <li class="flex items-center gap-3 text-meta">
          <span class="w-32 shrink-0 truncate text-fg-2" title={row.class_name}>
            {row.class_name}
          </span>
          <div class="relative h-4 grow overflow-hidden rounded bg-surface-1">
            {#if row.pct > 0}
              <div class="h-full {tierBg(row.tier)}" style:width="{row.pct}%"></div>
            {/if}
          </div>
          <span
            class="w-44 shrink-0 whitespace-nowrap text-right font-mono text-fg-2"
            title="{row.validated} validated, {row.test} frozen as test holdout"
          >
            {`${row.trainable} trainable`}{#if row.test > 0}<span class="text-fg-3"
                >{` · ${row.test} test`}</span
              >{/if}
          </span>
        </li>
      {/each}
    </ul>
  {/if}
  {#if view.moreCount > 0}
    <p class="mt-2 text-meta text-fg-3" data-testid="class-balance-more">
      +{view.moreCount} more class{view.moreCount === 1 ? '' : 'es'} with validated crops
    </p>
  {/if}
  {#if view.zeroCount > 0}
    <p class="mt-2 text-meta text-fg-3" data-testid="class-balance-zero">
      {view.zeroCount} class{view.zeroCount === 1 ? '' : 'es'} with 0 validated
    </p>
  {/if}
{/if}
