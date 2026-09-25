<script lang="ts">
  /**
   * Model × dataset matrix. Cells, ranks and winners are served: every
   * model listed in `best[dataset][metric]` is bold, so tied winners are
   * all bold. Accuracy cells are the dataset's served `rank_scope` values.
   */
  import type { BakeoffMatrix, MatrixCell } from '$lib/types_bakeoff';
  import { formatMetric, isBest, metricLabel, metricUnit } from '$lib/bakeoff/view';

  interface Props {
    matrix: BakeoffMatrix;
    datasetName: (id: string) => string;
    /** Initial metric; defaults to the served `rank_by` when it is a column. */
    metric?: string;
  }
  let { matrix, datasetName, metric: initial }: Props = $props();

  let picked = $state<string | null>(null);
  const metric = $derived.by(() => {
    for (const m of [picked, initial, matrix.rank_by]) {
      if (m && matrix.metrics.includes(m)) return m;
    }
    return matrix.metrics[0] ?? '';
  });

  function cellValue(model: string, ds: string): number | null {
    const v = matrix.cells?.[model]?.[ds]?.[metric as keyof MatrixCell];
    return typeof v === 'number' ? v : null;
  }
  function cellRank(model: string, ds: string): number | null {
    return matrix.cells?.[model]?.[ds]?.rank ?? null;
  }
</script>

<div data-testid="bakeoff-matrix">
  <div class="mb-2 flex flex-wrap items-center justify-between gap-2">
    <h3 class="text-sm font-medium text-zinc-300">
      Model × dataset{matrix.rank_by ? ` · ranked by ${metricLabel(matrix.rank_by)}` : ''}
    </h3>
    <label class="text-xs text-zinc-400">
      metric
      <select
        value={metric}
        onchange={(e) => (picked = e.currentTarget.value)}
        class="ml-1 rounded border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs"
        data-testid="matrix-metric"
      >
        {#each matrix.metrics as m (m)}
          <option value={m}>{metricLabel(m)}</option>
        {/each}
      </select>
    </label>
  </div>
  <div class="overflow-x-auto rounded-lg border border-zinc-800">
    <table class="w-full text-sm">
      <thead class="bg-zinc-900 text-xs text-zinc-400">
        <tr>
          <th class="px-3 py-2 text-left">Model</th>
          {#each matrix.datasets as ds (ds.id)}
            <th class="px-2 py-2 text-right font-normal">
              <div class="font-mono text-zinc-300">{datasetName(ds.id)}</div>
              <div class="text-[10px] text-zinc-500">
                {ds.rank_scope === 'common'
                  ? `${ds.n_common_classes} common classes`
                  : ds.rank_scope === 'overall'
                    ? "each model's own classes"
                    : ''}
              </div>
            </th>
          {/each}
        </tr>
      </thead>
      <tbody>
        {#each matrix.models as m (m.model)}
          <tr class="border-t border-zinc-800 hover:bg-zinc-800/40" data-model={m.model}>
            <td class="px-3 py-2 text-xs">
              <span class="font-mono">{m.display_name}</span>
              <span class="ml-1 text-[10px] text-zinc-500">{m.source}</span>
            </td>
            {#each matrix.datasets as ds (ds.id)}
              {@const best = isBest(matrix, ds.id, metric, m.model)}
              <td
                class="px-2 py-2 text-right tabular-nums {best
                  ? 'font-bold text-emerald-300'
                  : ''}"
                data-testid="matrix-cell"
                data-dataset={ds.id}
                data-best={best ? 'true' : 'false'}
              >
                {formatMetric(cellValue(m.model, ds.id), metric)}
                {#if cellRank(m.model, ds.id) != null}
                  <span class="ml-1 text-[10px] font-normal text-zinc-500"
                    >#{cellRank(m.model, ds.id)}</span
                  >
                {/if}
              </td>
            {/each}
          </tr>
        {/each}
      </tbody>
    </table>
  </div>
  <p class="mt-2 text-xs text-zinc-500">
    {metricUnit(metric)}; the best value per dataset is
    <span class="font-bold text-emerald-300">bold</span> (every tied model); #n is the served
    rank.
  </p>
</div>
