<script lang="ts">
  import type { ClusterDetailController } from '$lib/clusters/clusterDetailController.svelte';
  import type { ConfidenceBin } from '$lib/types_clusterList';
  import { reviewTabsVocabularyStore } from '$stores/reviewTabsVocabulary.svelte';

  // Filter chips for one cluster. Every count is served by the facets route
  // and equals the total the matching filtered crop list returns; a chip
  // sends the matching `GET /crops` parameter and nothing is counted here.
  let { view }: { view: ClusterDetailController } = $props();

  const facets = $derived(view.facets);
  // Open on a wide screen; on a narrow one the chip rows would push the grid
  // out of view, so they start folded.
  let open = $state(
    typeof window === 'undefined' ||
      typeof window.matchMedia !== 'function' ||
      window.matchMedia('(min-width: 768px)').matches,
  );

  function binLabel(b: ConfidenceBin): string {
    return b.max == null
      ? `${b.min.toFixed(1)}+`
      : `${b.min.toFixed(1)}–${b.max.toFixed(1)}`;
  }
  const bandActive = (b: ConfidenceBin): boolean =>
    view.facetBand?.min === b.min && view.facetBand?.max === b.max;

  const chip = (on: boolean): string =>
    `btn-sm border text-xs transition-colors ${
      on
        ? 'border-blue-500/60 bg-blue-500/20 text-blue-100'
        : 'border-zinc-700 bg-zinc-900 text-zinc-300 hover:border-blue-500/40'
    }`;
</script>

{#if view.facetsError}
  <p
    class="border-b border-zinc-800 px-4 py-1.5 text-xs text-amber-300"
    role="alert"
    data-testid="cluster-facets-error"
  >
    Facet counts unavailable: {view.facetsError}
  </p>
{:else if facets}
  <details
    bind:open
    class="border-b border-zinc-800 px-4 py-2 text-xs"
    data-testid="cluster-facets"
  >
    <summary class="cursor-pointer text-zinc-300">
      Facets{view.facetFilterActive ? ' (filtered)' : ''}
    </summary>
    <div class="mt-2 flex flex-col gap-1.5" role="group" aria-label="Cluster facets">
      {#if facets.detector_classes.length > 0}
        <div class="flex flex-wrap items-center gap-1.5">
          <span class="w-28 shrink-0 text-zinc-500">Detector class</span>
          {#each facets.detector_classes as f (f.value)}
            <button
              type="button"
              class={chip(view.facetDetectorClass === f.value)}
              aria-pressed={view.facetDetectorClass === f.value}
              data-testid="facet-detector-class"
              onclick={() => view.toggleDetectorClass(f.value)}
            >
              {f.value}
              <span class="ml-1 font-mono text-zinc-400">{f.count.toLocaleString()}</span>
            </button>
          {/each}
        </div>
      {/if}

      <div class="flex flex-wrap items-center gap-1.5">
        <span class="w-28 shrink-0 text-zinc-500">Validation</span>
        <button
          type="button"
          class={chip(view.facetValidated === true)}
          aria-pressed={view.facetValidated === true}
          data-testid="facet-validated"
          onclick={() => view.toggleValidated(true)}
        >
          validated
          <span class="ml-1 font-mono text-zinc-400"
            >{facets.validation.validated.toLocaleString()}</span
          >
        </button>
        <button
          type="button"
          class={chip(view.facetValidated === false)}
          aria-pressed={view.facetValidated === false}
          data-testid="facet-unvalidated"
          onclick={() => view.toggleValidated(false)}
        >
          not validated
          <span class="ml-1 font-mono text-zinc-400"
            >{facets.validation.unvalidated.toLocaleString()}</span
          >
        </button>
      </div>

      {#if Object.keys(facets.review_tabs).length > 0}
        <div class="flex flex-wrap items-center gap-1.5">
          <span class="w-28 shrink-0 text-zinc-500">Review</span>
          {#each Object.entries(facets.review_tabs) as [tab, count] (tab)}
            <button
              type="button"
              class={chip(view.facetReviewTab === tab)}
              aria-pressed={view.facetReviewTab === tab}
              title={reviewTabsVocabularyStore.descriptionFor(tab) ?? undefined}
              data-testid="facet-review-tab"
              onclick={() => view.toggleReviewTab(tab)}
            >
              {reviewTabsVocabularyStore.labelFor(tab, tab)}
              <span class="ml-1 font-mono text-zinc-400">{count.toLocaleString()}</span>
            </button>
          {/each}
        </div>
      {/if}

      {#if facets.detector_confidence.length > 0}
        <div class="flex flex-wrap items-center gap-1.5">
          <span class="w-28 shrink-0 text-zinc-500">Detector confidence</span>
          {#each facets.detector_confidence as b (b.min)}
            <button
              type="button"
              class={chip(bandActive(b))}
              aria-pressed={bandActive(b)}
              data-testid="facet-confidence-bin"
              onclick={() => view.toggleBand(b)}
            >
              {binLabel(b)}
              <span class="ml-1 font-mono text-zinc-400">{b.count.toLocaleString()}</span>
            </button>
          {/each}
          <span class="text-zinc-500" data-testid="facet-confidence-missing">
            none: {facets.detector_confidence_missing.toLocaleString()}
          </span>
        </div>
      {/if}

      <div class="flex flex-wrap items-center gap-1.5">
        <label class="flex items-center gap-2 text-zinc-500">
          <span class="w-28 shrink-0">Sort by</span>
          <select
            class="select-sm"
            data-testid="facet-sort"
            value={view.detectorSort}
            onchange={(e) =>
              (view.detectorSort = e.currentTarget.value as 'default' | 'desc' | 'asc')}
          >
            <option value="default">Cluster order</option>
            <option value="desc">Detector confidence, high first</option>
            <option value="asc">Detector confidence, low first</option>
          </select>
        </label>
      </div>
    </div>
  </details>
{/if}
