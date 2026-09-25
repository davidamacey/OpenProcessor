<script lang="ts">
  /**
   * One dataset's served comparison: ranked rows (the dataset's
   * `rank_scope` block), coverage, overlap and class-mapping notes, then a
   * per-class table where a class a model does not cover reads
   * "not covered", and each model's unmapped classes. No value here is
   * computed client-side.
   */
  import type { BakeoffComparison, ComparisonRow, PerClassRow } from '$lib/types_bakeoff';
  import {
    formatCount,
    formatMetric,
    formatOverlap,
    hasOverlap,
    LEGACY_RESULTS_MESSAGE,
    metricLabel,
    PER_CLASS_METRICS,
    type PerClassMetric,
  } from '$lib/bakeoff/view';

  interface Props {
    comparison: BakeoffComparison | null;
    legacy: boolean;
    error: string | null;
    loading: boolean;
  }
  let { comparison, legacy, error, loading }: Props = $props();

  let perClassMetric = $state<PerClassMetric>('ap50_95');

  const SCOPE_METRICS = ['map_50_95', 'map_50', 'precision', 'recall', 'f1'] as const;

  function scopeBlock(c: BakeoffComparison, r: ComparisonRow) {
    return c.rank_scope === 'common' ? r.common : r.overall;
  }
  function perClass(r: ComparisonRow, evalClassId: number): PerClassRow | undefined {
    return r.per_class.find((p) => p.eval_class_id === evalClassId);
  }
</script>

<div data-testid="comparison-view">
  {#if legacy}
    <p
      class="rounded border border-zinc-800 bg-zinc-900/50 p-3 text-sm text-zinc-400"
      data-testid="comparison-legacy"
    >
      {LEGACY_RESULTS_MESSAGE}
    </p>
  {:else if error}
    <p
      class="rounded border border-red-800 bg-red-950/60 p-3 text-sm text-red-200"
      data-testid="comparison-error"
    >
      {error}
    </p>
  {:else if loading && !comparison}
    <p class="text-sm text-zinc-500">Loading results…</p>
  {:else if comparison}
    {@const c = comparison}
    <div class="mb-2 text-xs text-zinc-500">
      {formatCount(c.dataset.n_images)} test images · {formatCount(c.dataset.n_objects)} objects
      · ranked by {metricLabel(c.rank_by)} over
      {c.rank_scope === 'common'
        ? `the ${c.common_classes.length} class${c.common_classes.length === 1 ? '' : 'es'} every model covers`
        : "each model's own classes (not comparable)"}
      {#if c.profile}· profile <span class="font-mono">{c.profile}</span>{/if}
    </div>

    {#if c.warnings.length}
      <ul
        class="mb-3 list-disc rounded border border-amber-800/60 bg-amber-950/30 p-2 pl-6 text-xs text-amber-200"
        data-testid="comparison-warnings"
      >
        {#each c.warnings as w, i (i)}<li>{w}</li>{/each}
      </ul>
    {/if}

    <div class="overflow-x-auto rounded-lg border border-zinc-800">
      <table class="w-full text-sm" data-testid="comparison-rows">
        <thead class="bg-zinc-900 text-xs text-zinc-400">
          <tr>
            <th class="px-2 py-2 text-right">Rank</th>
            <th class="px-3 py-2 text-left">Model</th>
            {#each SCOPE_METRICS as m (m)}
              <th class="px-2 py-2 text-right">{metricLabel(m)}</th>
            {/each}
            <th class="px-2 py-2 text-right">Classes covered</th>
            <th
              class="px-2 py-2 text-right"
              title="mAP@.5:.95 over the model's own classes">Own classes mAP@.5:.95</th
            >
            <th class="px-2 py-2 text-right">Latency (ms)</th>
            <th class="px-2 py-2 text-right">Size (MB)</th>
          </tr>
        </thead>
        <tbody>
          {#each c.models as r (r.model)}
            {@const b = scopeBlock(c, r)}
            <tr class="border-t border-zinc-800 align-top" data-model={r.model}>
              <td class="px-2 py-2 text-right tabular-nums"
                >{formatMetric(r.rank, 'rank')}</td
              >
              <td class="min-w-[14rem] px-3 py-2 text-xs">
                <div class="font-mono">{r.display_name}</div>
                <div class="text-[10px] text-zinc-500">
                  {r.source} · {r.runtime}{r.imgsz ? ` · ${r.imgsz}px` : ''} · mapping {r
                    .class_mapping.method}
                </div>
                {#each r.class_mapping.warnings as w, i (i)}
                  <div class="text-[10px] text-amber-300">{w}</div>
                {/each}
                {#if hasOverlap(r.train_test_overlap)}
                  <div
                    class="mt-0.5 text-[10px] text-red-300"
                    data-testid="row-overlap-warning"
                  >
                    overlap: {formatOverlap(r.train_test_overlap)}
                  </div>
                {/if}
              </td>
              {#each SCOPE_METRICS as m (m)}
                <td class="px-2 py-2 text-right tabular-nums">{formatMetric(b[m], m)}</td>
              {/each}
              <td class="px-2 py-2 text-right tabular-nums"
                >{r.coverage.n_covered} / {r.coverage.n_eval_classes}</td
              >
              <td class="px-2 py-2 text-right tabular-nums"
                >{formatMetric(r.overall.map_50_95, 'map_50_95')}</td
              >
              <td class="px-2 py-2 text-right tabular-nums text-zinc-400"
                >{formatMetric(r.latency_ms?.mean, 'latency_ms')}</td
              >
              <td class="px-2 py-2 text-right tabular-nums text-zinc-400"
                >{formatMetric(r.size_mb, 'size_mb')}</td
              >
            </tr>
          {/each}
        </tbody>
      </table>
    </div>

    {#if c.failed.length}
      <ul
        class="mt-3 rounded border border-red-800 bg-red-950/60 p-2 text-xs text-red-200"
        data-testid="comparison-failed"
      >
        {#each c.failed as f (f.model)}
          <li><span class="font-mono">{f.model}</span> — {f.error ?? 'failed'}</li>
        {/each}
      </ul>
    {/if}

    <div class="mt-5 mb-2 flex flex-wrap items-center justify-between gap-2">
      <h4 class="text-sm font-medium text-zinc-300">Per class</h4>
      <label class="text-xs text-zinc-400">
        metric
        <select
          bind:value={perClassMetric}
          class="ml-1 rounded border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs"
          data-testid="per-class-metric"
        >
          {#each PER_CLASS_METRICS as m (m)}
            <option value={m}>{metricLabel(m)}</option>
          {/each}
        </select>
      </label>
    </div>
    <div class="overflow-x-auto rounded-lg border border-zinc-800">
      <table class="w-full text-sm" data-testid="per-class-table">
        <thead class="bg-zinc-900 text-xs text-zinc-400">
          <tr>
            <th class="px-3 py-2 text-left">Class</th>
            <th class="px-2 py-2 text-right">Objects</th>
            {#each c.models as r (r.model)}
              <th class="px-2 py-2 text-right font-mono font-normal">{r.display_name}</th>
            {/each}
          </tr>
        </thead>
        <tbody>
          {#each c.eval_classes as ec (ec.eval_class_id)}
            <tr class="border-t border-zinc-800" data-class-id={ec.eval_class_id}>
              <td class="px-3 py-1.5 text-xs">
                <span class="font-mono">{ec.name}</span>
                {#if c.common_classes.includes(ec.eval_class_id)}
                  <span class="ml-1 text-[10px] text-zinc-500">common</span>
                {/if}
              </td>
              <td class="px-2 py-1.5 text-right tabular-nums text-zinc-400">{ec.n_gt}</td>
              {#each c.models as r (r.model)}
                {@const p = perClass(r, ec.eval_class_id)}
                {#if p && !p.covered}
                  <td
                    class="px-2 py-1.5 text-right text-xs text-zinc-600 italic"
                    data-testid="not-covered">not covered</td
                  >
                {:else}
                  <td class="px-2 py-1.5 text-right tabular-nums"
                    >{formatMetric(p?.[perClassMetric], perClassMetric)}</td
                  >
                {/if}
              {/each}
            </tr>
          {/each}
        </tbody>
      </table>
    </div>

    {#if c.models.some((r) => r.coverage.unmapped_model_classes.length)}
      <div class="mt-3 text-xs text-zinc-400" data-testid="unmapped-classes">
        <p class="mb-1 text-zinc-500">
          Model classes with no class in this dataset (their predictions are not scored):
        </p>
        <ul class="space-y-0.5">
          {#each c.models as r (r.model)}
            {#if r.coverage.unmapped_model_classes.length}
              <li>
                <span class="font-mono">{r.display_name}</span>:
                {r.coverage.unmapped_model_classes
                  .map(
                    (u) =>
                      `${u.name ?? `#${u.model_class_id}`}${u.n_predictions != null ? ` (${u.n_predictions} predictions)` : ''}`,
                  )
                  .join(', ')}
              </li>
            {/if}
          {/each}
        </ul>
      </div>
    {/if}
  {:else}
    <p class="text-sm text-zinc-500">No results for this dataset yet.</p>
  {/if}
</div>
