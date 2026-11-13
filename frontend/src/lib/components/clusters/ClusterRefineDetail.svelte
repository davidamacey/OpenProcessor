<script lang="ts">
  import BlurSlider from '$components/BlurSlider.svelte';
  import UnembeddedBanner from '$components/embedding/UnembeddedBanner.svelte';
  import SemanticSearchBox from '$components/SemanticSearchBox.svelte';
  import StrategyBar from '$components/StrategyBar.svelte';
  import SubjectScopeToggle from '$components/SubjectScopeToggle.svelte';
  import { classSourcesStore } from '$stores/classSources.svelte';
  import { strategiesStore } from '$stores/strategies.svelte';
  import ClusterFacetBar from '$components/clusters/ClusterFacetBar.svelte';
  import type { ClusterDetailController } from '$lib/clusters/clusterDetailController.svelte';

  // The panel the open cluster's Refine button opens.
  let { view }: { view: ClusterDetailController } = $props();

  const LABEL = 'w-36 shrink-0 text-meta font-semibold text-fg-2';

  // The served per-source member count, when the facets have loaded.
  function sourceCount(id: string): string {
    const n = view.facets?.class_sources.find((s) => s.value === id)?.count;
    return n == null ? '' : ` (${n.toLocaleString()})`;
  }
</script>

<!-- The open cluster's Refine panel: label source, search in the cluster, subject, clarity,
     the fuller order selector and the facet chips. The one Sort is in the header. -->
<div class="flex flex-col gap-3 text-meta" data-testid="cluster-detail-refine">
  <div class="flex flex-wrap items-center gap-x-3 gap-y-2">
    <span class={LABEL}>Label source</span>
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
    {#if view.classSourceFilter !== null}
      <span class="text-fg-3">
        showing {view.cropPager.total.toLocaleString()} from {classSourcesStore.labelFor(
          view.classSourceFilter,
        )}
      </span>
    {/if}
  </div>

  {#if view.semanticSearchAvailable}
    <div class="flex flex-wrap items-center gap-x-3 gap-y-2">
      <span class={LABEL}>Search in this cluster</span>
      <SemanticSearchBox
        filter={{ cluster_uid: view.clusterUid }}
        pageSize={200}
        onResults={(res) => {
          view.searchModeActive = true;
          view.searchScores = new Map(
            res.items.map((it) => [it.id, it.similarity_score]),
          );
          view.cropPager.items = res.items;
          view.cropPager.total = res.total;
        }}
        onClear={() => {
          view.searchModeActive = false;
          view.searchScores = new Map();
          void view.cropPager.loadFirst();
        }}
      />
    </div>
  {/if}

  <div class="flex flex-wrap items-center gap-x-3 gap-y-2">
    <span class={LABEL}>Subject</span>
    <SubjectScopeToggle
      bind:value={view.subjectScope}
      labels={['All', 'Largest', 'Largest + 2nd']}
      dense
    />
  </div>

  <div class="flex flex-wrap items-center gap-x-3 gap-y-2">
    <span class={LABEL}>Minimum clarity</span>
    <BlurSlider
      bind:value={view.blurSlider}
      oncommit={view.commitBlur}
      max={view.BLUR_MAX}
      width="w-40"
      stops
      title="Hide crops blurrier than this (blur_lap_ratio). 1.1/1.3/1.4 are the v1.1.9 sale-quality stops; training tolerance is lower."
    />
  </div>

  <div class="flex flex-wrap items-center gap-x-3 gap-y-2">
    <span class={LABEL}>Order</span>
    <StrategyBar
      bar={view.strategyBar}
      allowedIds={view.allowedOrderIds}
      showFilters={false}
      diverseKDefault={view.pageSize}
      diverseKMax={strategiesStore.methods.overlays.find((o) => o.id === 'diverse')
        ?.max_k ?? undefined}
      diverseMeta={view.orderMeta}
    />
  </div>

  <UnembeddedBanner
    count={view.unembedded?.count}
    context="ordered"
    suggestedReprocess={view.unembedded?.request}
    onapplied={() => void view.loadFirst()}
  />

  <ClusterFacetBar {view} />

  {#if view.subjectScope !== 0 || view.minBlurRatio !== null || !view.strategyBar.isDefault || view.facetFilterActive || view.classSourceFilter !== null}
    <button
      type="button"
      class="btn-sm self-start"
      data-testid="cluster-refine-reset"
      onclick={() => {
        view.subjectScope = 0;
        view.blurSlider = 0;
        view.minBlurRatio = null;
        view.classSourceFilter = null;
        view.strategyBar.reset();
        view.resetFacetFilters();
      }}
    >
      Reset
    </button>
  {/if}
</div>
