<script lang="ts">
  /**
   * What the cluster list's Refine popover holds: where to search, the item-text search, what to
   * show (the ignored bucket, every matching item), the subject scope and clarity, the class and
   * size filters and the status group. Every control the old toolbar had is here, in groups.
   */
  import BlurSlider from '$lib/components/BlurSlider.svelte';
  import ChipPicker from '$lib/components/clusters/ChipPicker.svelte';
  import SubjectScopeToggle from '$lib/components/SubjectScopeToggle.svelte';
  import { asValidationState } from '$lib/clusters/clusterListFilter';
  import { VALIDATION_TEXT } from '$lib/clusters/clusterCardText';
  import type { ClusterListController } from '$lib/clusters/clusterListController.svelte';
  import type { ClusterSearchModes } from '$lib/clusters/clusterSearchModes.svelte';
  import { VALIDATION_STATES } from '$lib/types_clusterList';
  import { classesStore } from '$stores/classes.svelte';

  interface Props {
    list: ClusterListController;
    modes: ClusterSearchModes;
    searchAvailable: boolean;
    matchingModeActive: boolean;
    enterMatchingMode: () => void;
    exitMatchingMode: () => void;
    close: () => void;
  }
  let {
    list,
    modes,
    searchAvailable,
    matchingModeActive,
    enterMatchingMode,
    exitMatchingMode,
    close,
  }: Props = $props();

  const filter = $derived(list.filter);
  const classNames = $derived(
    classesStore.classes.filter((c) => !c.deprecated).map((c) => c.name),
  );
  const gridModes = $derived(!modes.ignoredModeActive);

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

  const GROUP = 'flex flex-wrap items-center gap-x-3 gap-y-2';
  const LABEL = 'w-28 shrink-0 text-meta font-semibold text-fg-2';
</script>

<div class="flex flex-col gap-3 text-meta" data-testid="cluster-list-filters">
  {#if searchAvailable}
    <div class={GROUP}>
      <span class={LABEL}>Search in</span>
      <div
        class="inline-flex overflow-hidden rounded-md border border-line-strong"
        role="group"
        aria-label="Search target"
        data-testid="search-target"
      >
        {#each [{ id: 'item', label: 'Crops' }, { id: 'box', label: 'Regions' }] as opt (opt.id)}
          <button
            type="button"
            class="px-2 py-1 {modes.searchTarget === opt.id
              ? 'bg-accent/20 text-accent-text'
              : 'bg-surface-1 text-fg-2 hover:bg-surface-2'}"
            aria-pressed={modes.searchTarget === opt.id}
            onclick={() => void modes.setSearchTarget(opt.id as 'item' | 'box')}
          >
            {opt.label}
          </button>
        {/each}
      </div>
    </div>
  {/if}

  <form
    class={GROUP}
    onsubmit={(e) => {
      e.preventDefault();
      void modes.runItemTextSearch();
      close();
    }}
  >
    <label class={LABEL} for="refine-item-text">Text in the item</label>
    <input
      id="refine-item-text"
      type="text"
      bind:value={modes.itemTextQuery}
      placeholder="Text read from the image"
      class="input-sm w-48"
    />
    <button type="submit" class="btn-sm">Search text</button>
    {#if modes.itemTextError}
      <span class="text-caption text-warn-text">{modes.itemTextError}</span>
    {/if}
  </form>

  <div class={GROUP}>
    <span class={LABEL}>Show</span>
    <button
      type="button"
      class="btn-sm {modes.ignoredModeActive
        ? 'border-warn/60 bg-warn/20 text-warn-text'
        : ''}"
      aria-pressed={modes.ignoredModeActive}
      title="Crops excluded from training and clustering"
      data-testid="ignored-mode-toggle"
      onclick={() => {
        if (modes.ignoredModeActive) modes.exitIgnoredMode();
        else void modes.loadIgnored();
        close();
      }}
    >
      Ignored bucket
    </button>
    <button
      type="button"
      class="btn-sm {matchingModeActive
        ? 'border-accent/60 bg-accent/20 text-accent-text'
        : ''}"
      aria-pressed={matchingModeActive}
      data-testid="matching-mode-toggle"
      title="List every item the filter matches, and act on all of them"
      onclick={() => {
        if (matchingModeActive) exitMatchingMode();
        else enterMatchingMode();
        close();
      }}
    >
      Every matching item
    </button>
  </div>

  {#if gridModes}
    <div class={GROUP}>
      <span class={LABEL}>Subject</span>
      <SubjectScopeToggle bind:value={list.subjectScope} />
    </div>
    <div class={GROUP}>
      <span class={LABEL}>Minimum clarity</span>
      <BlurSlider
        bind:value={list.blurSlider}
        oncommit={list.commitClusterBlur}
        max={list.BLUR_MAX}
        width="w-40"
        title="Hide crops blurrier than this (blur_lap_ratio)"
      />
    </div>
    <div class={GROUP}>
      <span class={LABEL}>Class</span>
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
    </div>
    <div class={GROUP}>
      <span class={LABEL}>Cluster size</span>
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
    <div class={GROUP} role="group" aria-label="Status">
      <span class={LABEL}>Status</span>
      <select
        class="select-sm"
        aria-label="Validation"
        data-testid="cluster-validation-filter"
        value={filter.validation_state ?? ''}
        onchange={(e) =>
          list.setFilter({
            ...filter,
            validation_state: asValidationState(e.currentTarget.value),
          })}
      >
        <option value="">Any validation</option>
        {#each VALIDATION_STATES as v (v)}
          <option value={v}>{VALIDATION_TEXT[v]}</option>
        {/each}
      </select>
      <label class="flex items-center gap-1.5 text-fg-2">
        <input
          type="checkbox"
          checked={filter.needs_review}
          onchange={(e) =>
            list.setFilter({ ...filter, needs_review: e.currentTarget.checked })}
        />
        Needs review
      </label>
      {#each list.flagVocabulary as f (f.id)}
        <button
          type="button"
          class="btn-sm {filter.flag.includes(f.id)
            ? 'border-accent/60 bg-accent/20 text-accent-text'
            : ''}"
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
</div>
