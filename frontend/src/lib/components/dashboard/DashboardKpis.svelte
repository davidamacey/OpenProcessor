<script lang="ts">
  /**
   * The dashboard's four headline numbers from the served dataset stats
   * (`GET /stats/dataset`): objects, validated, the detector labels nobody has checked (the
   * served `labeled.rows` entry, with its served label) and objects with no class. Nothing is
   * summed here.
   */
  import { fmt } from '$lib/datasetStats';
  import type { DatasetStats } from '$lib/api_stats';

  let { stats }: { stats: DatasetStats | null } = $props();

  const detector = $derived(stats?.labeled.rows.find((r) => r.key === 'by_detector'));
  const tiles = $derived(
    stats
      ? [
          { id: 'objects', label: 'Objects', value: stats.total_crops },
          { id: 'validated', label: 'Validated', value: stats.validated },
          ...(detector
            ? [{ id: 'detector', label: detector.label, value: detector.count }]
            : []),
          { id: 'no-class', label: 'No class', value: stats.unlabeled.no_label_source },
        ]
      : [],
  );
</script>

{#if tiles.length > 0}
  <dl class="grid grid-cols-2 gap-3 md:grid-cols-4" data-testid="dashboard-kpis">
    {#each tiles as tile (tile.id)}
      <div class="surface p-4" data-kpi={tile.id}>
        <dt class="text-meta text-fg-2">{tile.label}</dt>
        <dd class="mt-1 text-kpi font-semibold text-fg-1 tabular-nums">
          {fmt(tile.value)}
        </dd>
      </div>
    {/each}
  </dl>
{/if}
