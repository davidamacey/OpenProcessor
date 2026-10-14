<script lang="ts">
  import { page } from '$app/state';
  import BlurSlider from '$lib/components/BlurSlider.svelte';
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
    embeddingVizAvailable,
    showEmbeddingViz = $bindable(),
  }: {
    list: ClusterListController;
    modes: ClusterSearchModes;
    classFilter: number | null;
    isSlotFilter: boolean;
    gridItemFilter: () => ItemFilterQuery;
    matchingModeActive: boolean;
    enterMatchingMode: () => void;
    exitMatchingMode: () => void;
    embeddingVizAvailable: boolean;
    showEmbeddingViz: boolean;
  } = $props();
</script>

<!-- Toolbar -->
<div class="flex flex-wrap items-center gap-3 border-b border-zinc-800 px-4 py-2.5">
  <h1 class="text-lg font-semibold">Clusters</h1>
  <ShortcutsButton />

  {#if classFilter != null}
    <!-- C2 (visual audit 2026-09-24): name the class, not its id. -->
    <span
      class="rounded-md border border-blue-500/40 bg-blue-500/10 px-2 py-0.5 text-xs text-blue-200"
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
    <SemanticSearchBox
      pageSize={200}
      filter={{ ...gridItemFilter() }}
      initialQuery={page.url.searchParams.get('q')}
      onQueryChange={(q) => (modes.searchQuery = q)}
      onResults={(res) => {
        modes.searchModeActive = true;
        modes.searchScores = new Map(res.items.map((it) => [it.id, it.similarity_score]));
        modes.searchResults = res.items;
        modes.searchTotal = res.total;
        modes.syncSearchUrl(modes.searchQuery);
        void modes.ensureClusterMeta([
          ...new Set(
            res.items.map((it) => it.cluster_id).filter((id): id is number => id != null),
          ),
        ]);
      }}
      onClear={modes.exitSearchMode}
    />
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
      class="w-28 rounded border border-zinc-700 bg-zinc-900 px-2 py-1 text-xs text-zinc-100 focus:border-blue-500 focus:outline-none"
    />
    <button type="submit" class="btn-sm border border-zinc-700 bg-zinc-900 text-xs">
      Search text
    </button>
  </form>
  {#if modes.itemTextError}
    <span class="text-[11px] text-amber-300">{modes.itemTextError}</span>
  {/if}

  <!-- Ignored bucket (G7) -->
  <button
    type="button"
    onclick={() => {
      if (modes.ignoredModeActive) modes.exitIgnoredMode();
      else void modes.loadIgnored();
    }}
    class="btn-sm border text-xs transition-colors {modes.ignoredModeActive
      ? 'border-amber-500/60 bg-amber-500/20 text-amber-200'
      : 'border-zinc-700 bg-zinc-900 text-zinc-300 hover:border-amber-500/40'}"
    title="Crops excluded from training + clustering"
  >
    {modes.ignoredModeActive ? '✓ ' : ''}Ignored
  </button>

  <!-- Every item the shared filter matches, with run-on-selection actions. -->
  <button
    type="button"
    onclick={() => (matchingModeActive ? exitMatchingMode() : enterMatchingMode())}
    class="btn-sm border text-xs transition-colors {matchingModeActive
      ? 'border-blue-500/60 bg-blue-500/20 text-blue-200'
      : 'border-zinc-700 bg-zinc-900 text-zinc-300 hover:border-blue-500/40'}"
    data-testid="matching-mode-toggle"
    title="List every item the filter below matches, and act on all of them"
  >
    {matchingModeActive ? '✓ ' : ''}Matching items
  </button>

  <span class="grow"></span>

  <!-- C5 (visual audit 2026-09-24): the cluster-grid controls below
       (purity legend, unlabeled filter, embedding plot, sort, subject,
       clarity) don't apply to the Ignored bucket's crop list, so they
       are hidden in that mode rather than left live and inert. -->
  {#if !modes.ignoredModeActive}
    <!-- Color legend for the card border. The cluster grid uses border
       color to encode purity at a glance; without this strip the user
       has to mouse over each card to figure out what the colors mean. -->
    <div
      class="flex items-center gap-2 text-[10px] text-zinc-500"
      title="Card border color encodes cluster cohesion (the served bands): share of measured members whose nearest cluster centre is this one — not label agreement. See each card's cohesion chip tooltip for n, label agreement and labelled share."
    >
      <span class="flex items-center gap-1">
        <span class="inline-block h-2 w-3 rounded-sm border-2 border-green-500/60"></span>
        {list.purityThresholds
          ? `≥${Math.round(list.purityThresholds.pure_min * 100)}%`
          : 'pure'}
      </span>
      <span class="flex items-center gap-1">
        <span class="inline-block h-2 w-3 rounded-sm border-2 border-orange-500/60"
        ></span>
        {list.purityThresholds
          ? `≥${Math.round(list.purityThresholds.mixed_min * 100)}%`
          : 'mixed'}
      </span>
      <span class="flex items-center gap-1">
        <span class="inline-block h-2 w-3 rounded-sm border-2 border-red-500/60"></span>
        {list.purityThresholds
          ? `<${Math.round(list.purityThresholds.mixed_min * 100)}%`
          : 'noisy'}
      </span>
      <span class="flex items-center gap-1">
        <span class="inline-block h-2 w-3 rounded-sm border-2 border-blue-500/60"></span>
        sub-clustered
      </span>
    </div>

    <!-- Unlabeled-only filter: AHC clusters that didn't reach class-majority
       consensus surface as "Unlabeled #N". This toggle narrows the grid
       to just those, so operators can drain the unlabeled backlog in
       one pass (drag a cluster's reps into a class on the sidebar). -->
    <button
      type="button"
      onclick={() => {
        list.unlabeledOnly = !list.unlabeledOnly;
        if (list.unlabeledOnly) list.sort = 'size_desc';
      }}
      class="btn-sm border text-xs transition-colors {list.unlabeledOnly
        ? 'border-amber-500/60 bg-amber-500/20 text-amber-200'
        : 'border-zinc-700 bg-zinc-900 text-zinc-300 hover:border-amber-500/40'}"
      title="Show only clusters without a dominant class (need labeling)"
    >
      {list.unlabeledOnly ? '✓ ' : ''}Unlabeled only
      <span class="ml-1 font-mono text-[10px] text-zinc-500">({list.unlabeledCount})</span
      >
    </button>

    <!-- Embedding-plot toggle (curation-strategy plan §5.6): fully absent
       unless {API_PREFIX}/methods actually reports the overlay, same convention
       as the diverse overlay in <StrategyBar>. Replaces the card grid
       when active (never overlays it) — see the {#if showEmbeddingViz}
       branch in ClustersGrid. Hidden on the slot-gallery view, which has its own
       grid + toolbar and no per-crop cluster_id to color by. -->
    {#if embeddingVizAvailable && !isSlotFilter}
      <button
        type="button"
        onclick={() => (showEmbeddingViz = !showEmbeddingViz)}
        class="btn-sm border text-xs transition-colors {showEmbeddingViz
          ? 'border-blue-500/60 bg-blue-500/20 text-blue-200'
          : 'border-zinc-700 bg-zinc-900 text-zinc-300 hover:border-blue-500/40'}"
        title="Toggle a 2-d embedding-projection scatter plot (replaces the grid); lasso-select feeds the same label/move actions as the grid"
      >
        {showEmbeddingViz ? '✓ ' : ''}Embedding plot
      </button>
    {/if}

    <label class="flex items-center gap-2 text-xs text-zinc-400">
      Sort
      <select bind:value={list.sort} class="select-sm">
        <option value="purity_asc">cohesion asc</option>
        <option value="purity_desc">cohesion desc</option>
        <option value="size_desc">size desc</option>
        <option value="size_asc">size asc</option>
        <option value="dominant_class">dominant class</option>
      </select>
    </label>

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
