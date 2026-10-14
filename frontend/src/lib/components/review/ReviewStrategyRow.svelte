<!--
  The /review strategy row: semantic search, the strategy bar, and the
  diverse-selection job chip.
-->
<script lang="ts">
  import SemanticSearchBox from '$lib/components/SemanticSearchBox.svelte';
  import StrategyBar from '$lib/components/StrategyBar.svelte';
  import type { ReviewDiverseController } from '$lib/review/reviewDiverseController.svelte';
  import type { ReviewFilterController } from '$lib/review/reviewFilterController.svelte';
  import type { ReviewQueueLoader } from '$lib/review/reviewQueueLoader.svelte';
  import type { ReviewTabsController } from '$lib/review/reviewTabsController.svelte';
  import { endpointForTab } from '$lib/reviewTabs';

  interface Props {
    tabs: ReviewTabsController;
    diverse: ReviewDiverseController;
    filters: ReviewFilterController;
    loader: ReviewQueueLoader;
  }

  let { tabs, diverse, filters, loader }: Props = $props();

  const effectiveTab = $derived(tabs.effectiveTab);
  const _filter = $derived(filters._filter);
  const semanticSearchAvailable = $derived(loader.semanticSearchAvailable);
  const applySearchResults = $derived(loader.applySearchResults);
  const clearSearch = $derived(loader.clearSearch);
  const sortApplied = $derived(loader.sortApplied);
  const sortFallbackReason = $derived(loader.sortFallbackReason);
  const strategyBar = $derived(diverse.strategyBar);
  const diverseAvailable = $derived(diverse.diverseAvailable);
  const pinnedSortId = $derived(diverse.pinnedSortId);
  const DIVERSE_K_DEFAULT = $derived(diverse.DIVERSE_K_DEFAULT);
  const diverseKMax = $derived(diverse.diverseKMax);
  const diverseSelection = $derived(diverse.diverseSelection);
  const diverseMode = $derived(diverse.diverseMode);
  const diverseJobId = $derived(diverse.diverseJobId);
  const diverseJobStatus = $derived(diverse.diverseJobStatus);
  const diverseError = $derived(diverse.diverseError);
  const cancelDiverseJob = $derived(diverse.cancelDiverseJob);
</script>

<!-- Strategy bar — collapsed one-line sort/filter chip by default (see
     StrategyBar.svelte); the fallback note only appears when the
     server couldn't honor the requested sort. -->
<div class="flex items-center gap-3 border-b border-zinc-800 px-4 py-1.5">
  <!-- Left-aligned, first control in the row, so it reads as the
       primary way in rather than a control squeezed after the strategy
       bar. -->
  {#if semanticSearchAvailable}
    <SemanticSearchBox
      filter={{ tab: endpointForTab(effectiveTab), ..._filter() }}
      pageSize={200}
      onResults={applySearchResults}
      onClear={clearSearch}
    />
  {/if}
  <StrategyBar
    bar={strategyBar}
    offerDiverse={diverseAvailable}
    appliedSort={sortApplied}
    fallbackReason={sortFallbackReason}
    {pinnedSortId}
    diverseKDefault={DIVERSE_K_DEFAULT}
    {diverseKMax}
    diverseMeta={diverseSelection
      ? {
          method: diverseSelection.method,
          version: diverseSelection.version,
          n_pool: diverseSelection.n_pool,
        }
      : null}
  />
  <!-- M11: sort_fallback_reason now renders inside StrategyBar's own
       summary chip, next to sort_applied, instead of a separate banner
       here — see fallbackReason above. -->
  {#if diverseMode}
    {#if diverseJobId}
      <span class="text-[11px] text-blue-300">
        selecting… ({diverseJobStatus ?? 'running'})
        <button
          type="button"
          class="ml-1 underline hover:text-blue-100"
          onclick={() => void cancelDiverseJob()}
        >
          cancel
        </button>
      </span>
    {/if}
    {#if diverseError}
      <span class="text-[11px] text-red-300" title={diverseError}>
        {diverseError}
      </span>
    {/if}
  {/if}
</div>
