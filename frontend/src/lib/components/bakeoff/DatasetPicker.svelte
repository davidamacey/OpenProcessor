<script lang="ts">
  /**
   * Eval-dataset picker for `/bakeoff`: export test splits first (the
   * current export flagged), then external frozen sets grouped by their
   * served `group`. Counts and flags are served values.
   */
  import type { EvalDataset } from '$lib/types_bakeoff';
  import { formatCount } from '$lib/format/count';
  import { groupEvalDatasets } from '$lib/bakeoff/view';

  interface Props {
    datasets: EvalDataset[];
    selected: string[];
    onToggle: (id: string, on: boolean) => void;
  }
  let { datasets, selected, onToggle }: Props = $props();

  const grouped = $derived(groupEvalDatasets(datasets));
  const KIND_LABEL: Record<EvalDataset['dataset_kind'], string> = {
    multi_class: 'multi-class',
    single_class: 'single-class',
    external: 'external',
  };
</script>

{#snippet row(d: EvalDataset)}
  <label
    class="flex items-start gap-2 rounded px-1 py-1 text-meta hover:bg-surface-2/50"
    data-testid="dataset-row"
    data-dataset-id={d.id}
  >
    <input
      type="checkbox"
      class="mt-0.5"
      checked={selected.includes(d.id)}
      onchange={(e) => onToggle(d.id, e.currentTarget.checked)}
    />
    <span class="min-w-0 flex-1">
      <span class="flex flex-wrap items-center gap-1.5">
        <span class="text-fg-1">{d.name}</span>
        {#if d.is_current}
          <span
            class="rounded bg-success-soft px-1 text-caption text-success-text"
            data-testid="dataset-current">current</span
          >
        {/if}
        <span class="rounded bg-surface-2 px-1 text-caption text-fg-2"
          >{KIND_LABEL[d.dataset_kind] ?? d.dataset_kind}</span
        >
        {#if d.frozen_ok === false}
          <span
            class="rounded bg-danger-soft px-1 text-caption text-danger-text"
            title="The test split no longer matches its frozen lock file."
            >frozen check failed</span
          >
        {/if}
      </span>
      <span class="block text-caption text-fg-3">
        {formatCount(d.n_images)} test images · {formatCount(d.n_objects)} objects ·
        {formatCount(d.classes.length)} scored class{d.classes.length === 1
          ? ''
          : 'es'}{#if d.n_background_images > 0}
          · {formatCount(d.n_background_images)} background{/if}
      </span>
      {#if d.unlabeled_items_on_exported_images}
        <span class="block text-caption text-warn-text/80">
          {formatCount(d.unlabeled_items_on_exported_images)} unlabeled items on these images
          (a model detecting them is charged false positives)
        </span>
      {/if}
    </span>
  </label>
{/snippet}

<div class="space-y-2" data-testid="dataset-picker">
  <div>
    <p class="text-meta text-fg-2 font-semibold">Exported test splits</p>
    {#each grouped.exports as d (d.id)}
      {@render row(d)}
    {:else}
      <p class="px-1 text-meta text-fg-3">No export with a test split yet.</p>
    {/each}
  </div>
  {#each grouped.external as g (g.group)}
    <div>
      <p class="text-meta text-fg-2 font-semibold">
        External · {g.group}
      </p>
      {#each g.datasets as d (d.id)}
        {@render row(d)}
      {/each}
    </div>
  {/each}
</div>
