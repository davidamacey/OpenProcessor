<script lang="ts">
  import ChipPicker from '$lib/components/clusters/ChipPicker.svelte';
  import { asSortId, asValidationState } from '$lib/clusters/clusterListFilter';
  import type { ClusterListController } from '$lib/clusters/clusterListController.svelte';
  import { VALIDATION_TEXT } from '$lib/clusters/clusterCardText';
  import { VALIDATION_STATES } from '$lib/types_clusterList';
  import { classesStore } from '$stores/classes.svelte';

  // The sort and card filters of the cluster list. Sort labels, flag labels
  // and flag descriptions are served; the server does the sorting and
  // filtering, this only sends the choice.
  let { list }: { list: ClusterListController } = $props();

  const filter = $derived(list.filter);
  const classNames = $derived(
    classesStore.classes.filter((c) => !c.deprecated).map((c) => c.name),
  );

  function sizeValue(e: Event): number | null {
    const raw = (e.currentTarget as HTMLInputElement).value.trim();
    if (raw === '') return null;
    const n = Number(raw);
    return Number.isInteger(n) && n >= 0 ? n : null;
  }

  function toggleFlag(id: string): void {
    const flag = filter.flag.includes(id)
      ? filter.flag.filter((x) => x !== id)
      : [...filter.flag, id];
    list.setFilter({ ...filter, flag });
  }
</script>

<div
  class="flex flex-wrap items-center gap-x-4 gap-y-2 border-b border-zinc-800 bg-zinc-900/40 px-4 py-2 text-xs"
  role="group"
  aria-label="Cluster sort and filters"
  data-testid="cluster-list-filters"
>
  {#if list.sortOptions.length > 0}
    <label class="flex items-center gap-2 text-zinc-400">
      Sort
      <select
        class="select-sm"
        data-testid="cluster-sort"
        value={list.appliedSort ?? ''}
        onchange={(e) => (list.sort = asSortId(e.currentTarget.value))}
      >
        {#each list.sortOptions as o (o.id)}
          <option value={o.id}>{o.label}</option>
        {/each}
      </select>
    </label>
  {/if}

  <ChipPicker
    label="dominant class"
    options={classNames}
    selected={filter.dominant_class}
    onchange={(dominant_class) => list.setFilter({ ...filter, dominant_class })}
  />
  <ChipPicker
    label="detector class"
    options={list.detectorClassOptions}
    selected={filter.detector_class}
    onchange={(detector_class) => list.setFilter({ ...filter, detector_class })}
  />

  <div class="flex items-center gap-1.5 text-zinc-400">
    <span>Size</span>
    <input
      type="number"
      min="0"
      step="1"
      class="input-sm w-20"
      placeholder="min"
      aria-label="Minimum cluster size"
      value={filter.min_size ?? ''}
      onchange={(e) => list.setFilter({ ...filter, min_size: sizeValue(e) })}
    />
    <span aria-hidden="true">to</span>
    <input
      type="number"
      min="0"
      step="1"
      class="input-sm w-20"
      placeholder="max"
      aria-label="Maximum cluster size"
      value={filter.max_size ?? ''}
      onchange={(e) => list.setFilter({ ...filter, max_size: sizeValue(e) })}
    />
  </div>

  <label class="flex items-center gap-2 text-zinc-400">
    Validation
    <select
      class="select-sm"
      data-testid="cluster-validation-filter"
      value={filter.validation_state ?? ''}
      onchange={(e) =>
        list.setFilter({
          ...filter,
          validation_state: asValidationState(e.currentTarget.value),
        })}
    >
      <option value="">Any</option>
      {#each VALIDATION_STATES as v (v)}
        <option value={v}>{VALIDATION_TEXT[v]}</option>
      {/each}
    </select>
  </label>

  <label class="flex items-center gap-1.5 text-zinc-300">
    <input
      type="checkbox"
      checked={filter.needs_review}
      onchange={(e) =>
        list.setFilter({ ...filter, needs_review: e.currentTarget.checked })}
    />
    Needs review
  </label>

  {#if list.flagVocabulary.length > 0}
    <div class="flex flex-wrap items-center gap-1.5" role="group" aria-label="Flags">
      {#each list.flagVocabulary as f (f.id)}
        <button
          type="button"
          class="btn-sm border text-xs transition-colors {filter.flag.includes(f.id)
            ? 'border-red-500/60 bg-red-500/20 text-red-100'
            : 'border-zinc-700 bg-zinc-900 text-zinc-300 hover:border-red-500/40'}"
          aria-pressed={filter.flag.includes(f.id)}
          title={f.description}
          data-testid="cluster-flag-filter-{f.id}"
          onclick={() => toggleFlag(f.id)}
        >
          {f.label}
        </button>
      {/each}
    </div>
  {/if}

  {#if list.filterActive}
    <button
      type="button"
      class="btn-sm ml-auto border border-zinc-700 bg-zinc-800 text-zinc-200 hover:bg-zinc-700"
      data-testid="cluster-filters-clear"
      onclick={() => list.clearFilter()}
    >
      Clear filters
    </button>
  {/if}
</div>
