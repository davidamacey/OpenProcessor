<script lang="ts">
  import type { ClusterDetailController } from '$lib/clusters/clusterDetailController.svelte';
  import type { ConfidenceBin } from '$lib/types_clusterList';
  import { reviewTabsVocabularyStore } from '$stores/reviewTabsVocabulary.svelte';

  // Filter chips for one cluster. Every count is served by the facets route
  // and equals the total the matching filtered crop list returns; a chip
  // sends the matching `GET /crops` parameter and nothing is counted here.
  let { view }: { view: ClusterDetailController } = $props();

  const facets = $derived(view.facets);
  function binLabel(b: ConfidenceBin): string {
    return b.max == null
      ? `${b.min.toFixed(1)}+`
      : `${b.min.toFixed(1)}–${b.max.toFixed(1)}`;
  }
  const bandActive = (b: ConfidenceBin): boolean =>
    view.facetBand?.min === b.min && view.facetBand?.max === b.max;

  const chip = (on: boolean): string =>
    `btn-sm border text-meta transition-colors ${
      on
        ? 'border-accent/60 bg-accent/20 text-accent-text'
        : 'border-line-strong bg-surface-1 text-fg-2 hover:border-accent/40'
    }`;
</script>

{#if view.facetsError}
  <p class="text-meta text-warn-text" role="alert" data-testid="cluster-facets-error">
    Facet counts unavailable: {view.facetsError}
  </p>
{:else if facets}
  <section class="text-meta" data-testid="cluster-facets">
    <p class="mb-1.5 font-semibold text-fg-2">
      Facets{view.facetFilterActive ? ' (filtered)' : ''}
    </p>
    <div class="flex flex-col gap-1.5" role="group" aria-label="Cluster facets">
      {#if facets.detector_classes.length > 0}
        <div class="flex flex-wrap items-center gap-1.5">
          <span class="w-28 shrink-0 text-fg-3">Detector class</span>
          {#each facets.detector_classes as f (f.value)}
            <button
              type="button"
              class={chip(view.facetDetectorClass === f.value)}
              aria-pressed={view.facetDetectorClass === f.value}
              data-testid="facet-detector-class"
              onclick={() => view.toggleDetectorClass(f.value)}
            >
              {f.value}
              <span class="ml-1 text-fg-2 tabular-nums">{f.count.toLocaleString()}</span>
            </button>
          {/each}
        </div>
      {/if}

      <div class="flex flex-wrap items-center gap-1.5">
        <span class="w-28 shrink-0 text-fg-3">Validation</span>
        <button
          type="button"
          class={chip(view.facetValidated === true)}
          aria-pressed={view.facetValidated === true}
          data-testid="facet-validated"
          onclick={() => view.toggleValidated(true)}
        >
          validated
          <span class="ml-1 text-fg-2 tabular-nums"
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
          <span class="ml-1 text-fg-2 tabular-nums"
            >{facets.validation.unvalidated.toLocaleString()}</span
          >
        </button>
      </div>

      {#if facets.review_tabs.length > 0}
        <div class="flex flex-wrap items-center gap-1.5">
          <span class="w-28 shrink-0 text-fg-3">Review</span>
          {#each facets.review_tabs as { value: tab, count, label } (tab)}
            <button
              type="button"
              class={chip(view.facetReviewTab === tab)}
              aria-pressed={view.facetReviewTab === tab}
              title={reviewTabsVocabularyStore.descriptionFor(tab) ?? undefined}
              data-testid="facet-review-tab"
              onclick={() => view.toggleReviewTab(tab)}
            >
              {label ?? tab}
              <span class="ml-1 text-fg-2 tabular-nums">{count.toLocaleString()}</span>
            </button>
          {/each}
        </div>
      {/if}

      {#if facets.detector_confidence.length > 0}
        <div class="flex flex-wrap items-center gap-1.5">
          <span class="w-28 shrink-0 text-fg-3">Detector confidence</span>
          {#each facets.detector_confidence as b (b.min)}
            <button
              type="button"
              class={chip(bandActive(b))}
              aria-pressed={bandActive(b)}
              data-testid="facet-confidence-bin"
              onclick={() => view.toggleBand(b)}
            >
              {binLabel(b)}
              <span class="ml-1 text-fg-2 tabular-nums">{b.count.toLocaleString()}</span>
            </button>
          {/each}
          <span class="text-fg-3" data-testid="facet-confidence-missing">
            none: {facets.detector_confidence_missing.toLocaleString()}
          </span>
        </div>
      {/if}
    </div>
  </section>
{/if}
