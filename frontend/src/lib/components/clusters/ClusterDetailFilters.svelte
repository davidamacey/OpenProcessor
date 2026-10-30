<script lang="ts">
  import BlurSlider from '$components/BlurSlider.svelte';
  import UnembeddedBanner from '$components/embedding/UnembeddedBanner.svelte';
  import SemanticSearchBox from '$components/SemanticSearchBox.svelte';
  import StrategyBar from '$components/StrategyBar.svelte';
  import SubjectScopeToggle from '$components/SubjectScopeToggle.svelte';
  import { classSourcesStore } from '$stores/classSources.svelte';
  import { strategiesStore } from '$stores/strategies.svelte';
  import type { ClusterDetailController } from '$lib/clusters/clusterDetailController.svelte';

  // The label-source filter and the search / subject / clarity / order row.
  let { view }: { view: ClusterDetailController } = $props();

  // The served per-source member count, when the facets have loaded.
  function sourceCount(id: string): string {
    const n = view.facets?.class_sources.find((s) => s.value === id)?.count;
    return n == null ? '' : ` (${n.toLocaleString()})`;
  }
</script>

<!-- class_source filter chips. Narrows the grid to one source bucket
     (model / vlm / human / low_conf / ...) without changing
     the cluster card stats. Active filter is reflected in the URL-free
     reactive state so re-mounting the page resets to "all". -->
<div
  class="flex flex-wrap items-center gap-1.5 border-b border-zinc-800 px-4 py-1.5 text-xs"
>
  <span class="text-zinc-500">source:</span>
  <label class="flex items-center gap-1.5">
    <select
      aria-label="Filter by label source"
      class="select-sm"
      value={view.classSourceFilter ?? ''}
      onchange={(e) => {
        const v = (e.currentTarget as HTMLSelectElement).value;
        view.classSourceFilter = v === '' ? null : v;
      }}
    >
      <option value="">All sources</option>
      {#each view.classSourceOptions as opt (opt.id)}
        <option value={opt.id}>{opt.label}{sourceCount(opt.id)}</option>
      {/each}
    </select>
  </label>
  {#if view.classSourceFilter !== null}
    <span class="ml-auto text-zinc-500">
      showing {view.cropPager.total.toLocaleString()} from {classSourcesStore.labelFor(
        view.classSourceFilter,
      )}
    </span>
  {/if}
</div>

<!-- Primary-subject controls: focus on the largest subject(s) in frame
     (what the business sorts on) and hide too-blurry crops. View-only —
     no data is deleted; the slider commits on release to avoid a reload
     per pixel. -->
<div
  class="flex flex-wrap items-center gap-3 border-b border-zinc-800 px-4 py-1.5 text-xs"
>
  <!-- Cluster-scoped semantic search — left-aligned, first control in
       the row, so it reads as the primary way in rather than a control
       squeezed wherever there happens to be leftover flex space. -->
  {#if view.semanticSearchAvailable}
    <SemanticSearchBox
      filter={{ cluster_id: view.clusterId }}
      pageSize={200}
      onResults={(res) => {
        view.searchModeActive = true;
        view.searchScores = new Map(res.items.map((it) => [it.id, it.similarity_score]));
        view.cropPager.items = res.items;
        view.cropPager.total = res.total;
      }}
      onClear={() => {
        view.searchModeActive = false;
        view.searchScores = new Map();
        void view.cropPager.loadFirst();
      }}
    />
  {/if}

  <SubjectScopeToggle
    bind:value={view.subjectScope}
    labels={['All', 'Largest', 'Largest + 2nd']}
    label="subject:"
    labelClass="text-zinc-500"
    dense
  />

  <BlurSlider
    bind:value={view.blurSlider}
    oncommit={view.commitBlur}
    max={view.BLUR_MAX}
    labelClass="ml-2 text-zinc-500"
    width="w-40"
    stops
    title="Hide crops blurrier than this (blur_lap_ratio). 1.1/1.3/1.4 are the v1.1.9 sale-quality stops; training tolerance is lower."
  />

  <!-- Outliers-first: float the members least like the cluster centroid to
       the top, so wrong/atypical items are easy to cherry-pick out. Kept
       as a direct shortcut into strategyBar.sort — same muscle memory as
       before this phase, just backed by the generalized order id now. -->
  <button
    type="button"
    class="btn-sm {view.outliersFirst
      ? 'border-amber-500/60 bg-amber-500/20 text-amber-100'
      : 'border-zinc-700 bg-zinc-800 text-zinc-300 hover:bg-zinc-700'}"
    onclick={() => (view.strategyBar.sort = view.outliersFirst ? 'default' : 'outliers')}
    title="Sort by distance from the cluster centroid (most atypical first) to spot mislabels/junk"
  >
    {view.outliersFirst ? '◤ Outliers first' : 'Outliers first'}
  </button>

  <!-- Fuller order selector — additive alongside the shortcut above.
       Filters are hidden here: the crop query doesn't forward
       min-mistakenness / hide-near-dup params (Phase 3 review-only).
       'diverse' (Phase 4) is included in allowedOrderIds unconditionally
       — StrategyBar only actually renders it once {API_PREFIX}/methods reports
       the overlay, so this is harmless against a backend that hasn't
       shipped it yet. -->
  <StrategyBar
    bar={view.strategyBar}
    allowedIds={view.allowedOrderIds}
    showFilters={false}
    diverseKDefault={view.pageSize}
    diverseKMax={strategiesStore.methods.overlays.find((o) => o.id === 'diverse')
      ?.max_k ?? undefined}
    diverseMeta={view.orderMeta}
  />

  <UnembeddedBanner
    count={view.unembedded?.count}
    context="ordered"
    suggestedReprocess={view.unembedded?.request}
    onapplied={() => void view.loadFirst()}
  />

  {#if view.subjectScope !== 0 || view.minBlurRatio !== null || !view.strategyBar.isDefault || view.facetFilterActive}
    <button
      type="button"
      class="btn-sm ml-auto bg-zinc-800 text-zinc-300 hover:bg-zinc-700"
      onclick={() => {
        view.subjectScope = 0;
        view.blurSlider = 0;
        view.minBlurRatio = null;
        view.strategyBar.reset();
        view.resetFacetFilters();
      }}
    >
      reset
    </button>
  {/if}
</div>
