<!--
  The /review filter bar: Source, the shared item filter, Conf, the slot text
  search, the served per-tab enum filters, subject scope and clarity, the
  URL-seeded filter chips, the All tab's preset chips, and the key hints.
-->
<script lang="ts">
  import BlurSlider from '$lib/components/BlurSlider.svelte';
  import SubjectScopeToggle from '$lib/components/SubjectScopeToggle.svelte';
  import ItemFilterBar from '$lib/components/itemFilter/ItemFilterBar.svelte';
  import ServedFilterField from '$lib/components/itemFilter/ServedFilterField.svelte';
  import type { ReviewDiverseController } from '$lib/review/reviewDiverseController.svelte';
  import type { ReviewFilterController } from '$lib/review/reviewFilterController.svelte';
  import { kg } from '$lib/review/reviewKeys';
  import type { ReviewTabsController } from '$lib/review/reviewTabsController.svelte';
  import { rejectKeyGlyph } from '$lib/review/slotKeymap';
  import { REVIEW_PRESETS } from '$lib/reviewTabs';
  import { reviewTabsVocabularyStore } from '$stores/reviewTabsVocabulary.svelte';

  interface Props {
    tabs: ReviewTabsController;
    diverse: ReviewDiverseController;
    filters: ReviewFilterController;
    isMultiBoxSlot: boolean;
    editMode: boolean;
  }

  let { tabs, diverse, filters, isMultiBoxSlot, editMode }: Props = $props();

  const tab = $derived(tabs.tab);
  const preset = $derived(tabs.preset);
  const activeSlot = $derived(tabs.activeSlot);
  const filterVisible = $derived(tabs.filterVisible);
  const servedMaxRankDefault = $derived(tabs.servedMaxRankDefault);
  const togglePreset = $derived(tabs.togglePreset);
  const diverseMode = $derived(diverse.diverseMode);
  const itemFilter = $derived(filters.itemFilter);
  const itemFilterVisible = $derived(filters.itemFilterVisible);
  const servedSpecs = $derived(filters.servedSpecs);
  const persistItemFilter = $derived(filters.persistItemFilter);
  const activeFilterSpecs = $derived(filters.activeFilterSpecs);
  const enumFilterValues = $derived(filters.enumFilterValues);
  const setEnumFilter = $derived(filters.setEnumFilter);
  const commitBlur = $derived(filters.commitBlur);
  const BLUR_MAX = $derived(filters.BLUR_MAX);
  const importIdFilter = $derived(filters.importIdFilter);
  const combineConflictFilter = $derived(filters.combineConflictFilter);
  const urlFilterServed = $derived(filters.urlFilterServed);
  const clearUrlFilter = $derived(filters.clearUrlFilter);
</script>

<!-- Filter bar — flex children keep their width via flex-shrink-0; hotkey hint
     hides below md so it doesn't collide with controls on narrow viewports
     (same content is on the ~ overlay). -->
<div
  class="flex min-w-0 flex-wrap items-center gap-3 border-b border-zinc-800 bg-zinc-900/40 px-4 py-2 text-xs"
>
  <!-- GET /review/{tab} accepts class_id/source/conf_min/conf_max as of
       the 2026-09-24 logic-moves cutover (item 14/G3) — re-enabled,
       server-side, unconditionally (no client-side filtering here). -->
  {#if filterVisible('source')}
    <label class="flex shrink-0 items-center gap-1.5">
      <span class="text-zinc-400">Source</span>
      <input
        type="text"
        bind:value={filters.sourceFilter}
        placeholder="any"
        class="input-sm w-32"
      />
    </label>
  {/if}

  <ItemFilterBar
    state={itemFilter}
    visible={itemFilterVisible}
    served={servedSpecs}
    onchange={persistItemFilter}
    rootClass="contents"
  />

  {#if filterVisible('conf_min') || filterVisible('conf_max')}
    <label
      class="flex shrink-0 items-center gap-1.5"
      class:opacity-40={diverseMode}
      title={diverseMode ? 'not applied to diverse selection' : undefined}
    >
      <span class="text-zinc-400">Conf</span>
      <input
        type="number"
        min="0"
        max="1"
        step="0.05"
        bind:value={filters.confMin}
        disabled={diverseMode}
        aria-label="Minimum confidence"
        class="input-sm w-16"
      />
      <span class="text-zinc-500">..</span>
      <input
        type="number"
        min="0"
        max="1"
        step="0.05"
        bind:value={filters.confMax}
        disabled={diverseMode}
        aria-label="Maximum confidence"
        class="input-sm w-16"
      />
    </label>
  {/if}

  {#if activeSlot?.capabilities.queue?.textFilter && filterVisible(activeSlot.capabilities.queue.textFilter.param)}
    <label
      class="flex shrink-0 items-center gap-1.5"
      class:opacity-40={diverseMode}
      title={diverseMode ? 'not applied to diverse selection' : undefined}
    >
      <span class="text-zinc-400">{activeSlot.capabilities.queue.textFilter.label}</span>
      <input
        type="text"
        bind:value={filters.slotTextQuery}
        disabled={diverseMode}
        placeholder={activeSlot.capabilities.queue.textFilter.placeholder}
        class="input-sm w-28"
      />
    </label>
  {/if}

  <!-- Generic served-enum filter bar (3f1a11e adoption) — one <select>
       per ReviewFilterSpec the active tab's GET {API_PREFIX}/review/tabs entry
       declares (e.g. Regions' region_status: all / detected only /
       verifier-rejected candidates only). No param-specific markup —
       a future spec on any tab renders here unchanged. -->
  {#each activeFilterSpecs as spec (spec.param)}
    <ServedFilterField
      {spec}
      value={enumFilterValues[spec.param]}
      onchange={setEnumFilter}
    />
  {/each}

  <!-- Always available, on every tab and preset — matches Conf/Class/HDD
       source below/above, and the backend's own query builder already
       treats max_rank/min_blur_ratio as tab-agnostic. Used to be gated to
       only primary_low_conf/classifier_blind_spots, which made these controls
       appear and disappear depending on which tab or chip was active.
       Disabled (not hidden) while diverseMode is active — scope.filters
       on POST {API_PREFIX}/select/diverse doesn't support max_rank/min_blur_ratio
       (P2-10), so applying either here would silently do nothing. -->
  <div
    class="contents"
    class:opacity-40={diverseMode}
    class:pointer-events-none={diverseMode}
    title={diverseMode ? 'not applied to diverse selection' : undefined}
  >
    {#if filterVisible('max_rank')}
      <SubjectScopeToggle
        bind:value={filters.subjectScope}
        labels={[
          servedMaxRankDefault != null ? `Top ${servedMaxRankDefault}` : 'All ranks',
          'Largest',
          '+2nd',
        ]}
        titles={[
          servedMaxRankDefault != null
            ? `This tab's own default: only the ${servedMaxRankDefault} largest subjects in each image`
            : 'Every subject in the image, whatever its size',
          'Only the largest subject in each image',
          'The largest and second-largest subject in each image',
        ]}
        label="subject"
      />
    {/if}
    {#if filterVisible('min_blur_ratio')}
      <BlurSlider
        bind:value={filters.blurSlider}
        oncommit={commitBlur}
        max={BLUR_MAX}
        title="Hide crops blurrier than this"
      />
    {/if}
  </div>

  {#if importIdFilter && urlFilterServed('import_id')}
    <button
      type="button"
      class="chip border-blue-500/60 bg-blue-500/15 text-blue-100"
      data-testid="filter-chip-import-id"
      title="Remove this filter"
      onclick={() => clearUrlFilter('import_id')}
    >
      Import <code class="font-mono">{importIdFilter}</code> ×
    </button>
  {/if}
  {#if combineConflictFilter && urlFilterServed('combine_conflict')}
    <button
      type="button"
      class="chip border-blue-500/60 bg-blue-500/15 text-blue-100"
      data-testid="filter-chip-combine-conflict"
      title="Remove this filter"
      onclick={() => clearUrlFilter('combine_conflict')}
    >
      Combine conflicts only ×
    </button>
  {/if}

  {#if tab === 'all'}
    <!-- Quick-filter preset chips (2026-09 tab consolidation) — Mismatches
         / VLM Low-Conf / Primary·Low-Conf collapsed from top-level tabs
         into these, since live counts showed each was too big (11-97% of
         the dataset) to be a curated queue. Each chip reuses that former
         tab's exact backend query unchanged (see $lib/reviewTabs.ts);
         radio-style — picking a second chip swaps the first, clicking the
         active one (or "clear") returns to plain All. -->
    <div class="flex shrink-0 flex-wrap items-center gap-1.5">
      <!-- m31 (2026-09-24 interactive pass): "Quick filter" implied
           each chip narrows All — Primary · low-conf alone returns
           374 rows, more than All's 114, because both apply different
           ranking/eligibility rules server-side, not a subset relation.
           "Queue:" doesn't claim either one is smaller. -->
      <span class="text-zinc-400">Queue:</span>
      {#each REVIEW_PRESETS as p (p.id)}
        <button
          type="button"
          class="chip {preset === p.id
            ? 'border-blue-500/60 bg-blue-500/15 text-blue-100'
            : 'border-zinc-700 bg-zinc-900 text-zinc-300 hover:bg-zinc-800'}"
          aria-pressed={preset === p.id}
          title={reviewTabsVocabularyStore.descriptionFor(p.id) ?? p.description}
          onclick={() => togglePreset(p.id)}
        >
          {reviewTabsVocabularyStore.labelFor(p.id, p.label)}
        </button>
      {/each}
      {#if preset}
        <button
          type="button"
          class="chip bg-zinc-800 text-zinc-300 hover:bg-zinc-700"
          onclick={() => {
            if (preset) togglePreset(preset);
          }}
        >
          clear
        </button>
      {/if}
    </div>
  {/if}

  <span class="grow"></span>

  <span class="hidden text-[11px] text-zinc-500 md:inline">
    {#if isMultiBoxSlot && activeSlot}
      <kbd>{kg('review.region.confirm')}</kbd> confirm proposed ·
      <kbd>{kg('review.region.accept_box')}</kbd> accept box ·
      <kbd>{kg('review.region.reject_box')}</kbd> reject box ·
      <kbd>{kg('box_edit.next_box')}</kbd> next box
      {#if editMode}
        · <kbd>{kg('box_edit.delete_box')}</kbd> delete box · drag empty area to add
      {:else}
        · <kbd>{kg('review.region.edit_box')}</kbd> edit
      {/if}
      · <kbd>{kg('review.skip')}</kbd> skip
    {:else if activeSlot}
      <kbd>{kg('review.region.confirm')}</kbd> confirm ·
      <kbd>{rejectKeyGlyph(activeSlot)}</kbd> reject
      {#if activeSlot.capabilities.lifecycle?.falsePositiveState}
        · <kbd>{kg('review.region.false_positive')}</kbd> false-pos
      {/if}
      · <kbd>{kg('review.skip')}</kbd> skip · <kbd>{kg('review.region.back')}</kbd> back
    {:else}
      per-class letter assigns · <kbd>{kg('review.queue.class_picker')}</kbd> search all
      classes ·
      <kbd>{kg('review.queue.confirm')}</kbd> confirm · <kbd>{kg('review.skip')}</kbd>
      skip · <kbd>{kg('review.queue.discard')}</kbd> discard ·
      <kbd>{kg('review.undo')}</kbd> undo
    {/if}
  </span>
</div>
