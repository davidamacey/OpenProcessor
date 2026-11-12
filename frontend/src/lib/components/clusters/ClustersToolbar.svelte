<script lang="ts">
  import Icon from '$components/ui/Icon.svelte';
  import { page } from '$app/state';
  import BlurSlider from '$lib/components/BlurSlider.svelte';
  import ImageSearchPicker from '$components/clusters/ImageSearchPicker.svelte';
  import SemanticSearchBox from '$lib/components/SemanticSearchBox.svelte';
  import ShortcutsButton from '$lib/components/ShortcutsButton.svelte';
  import SubjectScopeToggle from '$lib/components/SubjectScopeToggle.svelte';
  import type { ClusterListController } from '$lib/clusters/clusterListController.svelte';
  import type { ClusterSearchModes } from '$lib/clusters/clusterSearchModes.svelte';
  import type { ItemFilterQuery } from '$lib/types_itemFilter';
  import { classesStore } from '$stores/classes.svelte';

  let {
    list,
    modes,
    classFilter,
    isSlotFilter,
    gridItemFilter,
    matchingModeActive,
    enterMatchingMode,
    exitMatchingMode,
  }: {
    list: ClusterListController;
    modes: ClusterSearchModes;
    classFilter: number | null;
    isSlotFilter: boolean;
    gridItemFilter: () => ItemFilterQuery;
    matchingModeActive: boolean;
    enterMatchingMode: () => void;
    exitMatchingMode: () => void;
  } = $props();
</script>

<!-- Toolbar -->
<div class="flex flex-wrap items-center gap-3 border-b border-line px-4 py-2.5">
  <h1 class="text-lg font-semibold">Clusters</h1>
  <ShortcutsButton />

  {#if classFilter != null}
    <!-- C2 (visual audit 2026-09-24): name the class, not its id. -->
    <span
      class="rounded-md border border-accent/40 bg-accent/10 px-2 py-0.5 text-xs text-accent-text"
      title="class id {classFilter}"
      data-testid="class-filter-chip"
    >
      class: {classesStore.classes.find((c) => c.id === classFilter)?.name ??
        `#${classFilter}`}
    </span>
  {/if}

  <!-- Dataset-wide semantic search — left-aligned, first control after
       the title, so it reads as the primary way in rather than a
       control squeezed between unrelated toggles. Deliberately no
       cluster_id/tab scope in `filter` — unscoped-across-the-whole-dataset
       is the entire point of this control, unlike the cluster_id-scoped
       SemanticSearchBox on /clusters/[id]. -->
  {#if modes.semanticSearchAvailable && !isSlotFilter}
    <!-- Re-mounted on a target switch: the box re-runs the typed query with the new target. -->
    {#key modes.searchTarget}
      <SemanticSearchBox
        pageSize={200}
        filter={{
          ...gridItemFilter(),
          ...(modes.searchTarget === 'box' ? { target: 'box' } : {}),
        }}
        initialQuery={modes.similarTo ? null : page.url.searchParams.get('q')}
        onQueryChange={(q) => (modes.searchQuery = q)}
        onResults={(res) => {
          modes.searchModeActive = true;
          modes.searchScores = new Map(
            res.items.map((it) => [it.id, it.similarity_score]),
          );
          modes.noteMatches(res.items);
          modes.searchResults = res.items;
          modes.searchTotal = res.total;
          modes.syncSearchUrl(modes.searchQuery);
          void modes.ensureClusterMeta([
            ...new Set(
              res.items
                .map((it) => it.cluster?.uid)
                .filter((uid): uid is string => uid != null),
            ),
          ]);
        }}
        onClear={modes.exitSearchMode}
      />
    {/key}
    <div
      class="inline-flex overflow-hidden rounded-md border border-line-strong text-xs"
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
    <ImageSearchPicker onSearch={(file, box) => void modes.runImage({ file, box })} />
  {/if}

  <!-- Item-text search (G8) — literal OCR text search over
       item_text_lines, distinct from the semantic search box above. -->
  <form
    class="flex items-center gap-1"
    onsubmit={(e) => {
      e.preventDefault();
      void modes.runItemTextSearch();
    }}
  >
    <input
      type="text"
      bind:value={modes.itemTextQuery}
      placeholder="Item text…"
      class="w-28 rounded border border-line-strong bg-surface-1 px-2 py-1 text-xs text-fg-1 focus:border-accent focus:outline-none"
    />
    <button type="submit" class="btn-sm border border-line-strong bg-surface-1 text-xs">
      Search text
    </button>
  </form>
  {#if modes.itemTextError}
    <span class="text-caption text-warn-text">{modes.itemTextError}</span>
  {/if}

  <!-- Ignored bucket (G7) -->
  <button
    type="button"
    onclick={() => {
      if (modes.ignoredModeActive) modes.exitIgnoredMode();
      else void modes.loadIgnored();
    }}
    class="btn-sm border text-xs transition-colors {modes.ignoredModeActive
      ? 'border-warn/60 bg-warn/20 text-warn-text'
      : 'border-line-strong bg-surface-1 text-fg-2 hover:border-warn/40'}"
    title="Crops excluded from training + clustering"
  >
    {#if modes.ignoredModeActive}<Icon
        name="check"
        size={12}
        class="mr-1 -mt-0.5 inline-block"
      />{/if}Ignored
  </button>

  <!-- Every item the shared filter matches, with run-on-selection actions. -->
  <button
    type="button"
    onclick={() => (matchingModeActive ? exitMatchingMode() : enterMatchingMode())}
    class="btn-sm border text-xs transition-colors {matchingModeActive
      ? 'border-accent/60 bg-accent/20 text-accent-text'
      : 'border-line-strong bg-surface-1 text-fg-2 hover:border-accent/40'}"
    data-testid="matching-mode-toggle"
    title="List every item the filter below matches, and act on all of them"
  >
    {#if matchingModeActive}<Icon
        name="check"
        size={12}
        class="mr-1 -mt-0.5 inline-block"
      />{/if}Matching items
  </button>

  <span class="grow"></span>

  <!-- C5 (visual audit 2026-09-24): the cluster-grid controls below
       (subject, clarity) don't apply to the Ignored bucket's crop list, so they
       are hidden in that mode rather than left live and inert. -->
  {#if !modes.ignoredModeActive}
    <!-- Primary-subject grid filters: scope cards to the largest / clear
       crops. Card size + reps reflect only passing crops, so a filtered
       grid is ready to drag-drop + AHC-refine on the subjects that matter. -->
    <div class="text-xs">
      <SubjectScopeToggle bind:value={list.subjectScope} />
    </div>
    <div class="text-xs">
      <BlurSlider
        bind:value={list.blurSlider}
        oncommit={list.commitClusterBlur}
        max={list.BLUR_MAX}
        width="w-28"
        title="Hide crops blurrier than this (blur_lap_ratio)"
      />
    </div>
  {/if}
</div>
