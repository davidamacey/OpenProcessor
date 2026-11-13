<script lang="ts">
  /**
   * The open cluster's header, two rows. Row one: the way back, the served name and counts, and
   * the previous and next cluster. Row two: the cluster's one primary ("Label all N as…"), the
   * Assist menu, More, Refine and the one Sort. With a selection the bulk bar takes row two's
   * place (the host renders it), so a view never shows two primaries.
   */
  import Icon from '$components/ui/Icon.svelte';
  import Kbd from '$components/ui/Kbd.svelte';
  import Menu, { type MenuItem } from '$components/ui/Menu.svelte';
  import ClusterRefineDetail from '$components/clusters/ClusterRefineDetail.svelte';
  import DiverseSample from '$components/workspace/DiverseSample.svelte';
  import RefinePopover from '$components/workspace/RefinePopover.svelte';
  import SortSelect from '$components/workspace/SortSelect.svelte';
  import { displayNoText } from '$lib/clusters/clusterCardText';
  import type { ClusterDetailController } from '$lib/clusters/clusterDetailController.svelte';
  import { applyScopeSort, SCOPE_SORTS, scopeSortOf } from '$lib/clusters/scopeSort';
  import type { ClusterAssist } from '$lib/assistRun/clusterAssist.svelte';
  import type { WorkspaceController } from '$lib/workspace/workspaceController.svelte';
  import { keymapStore } from '$stores/keymap.svelte';
  import { undoStore } from '$stores/undo.svelte';

  interface Props {
    ws: WorkspaceController;
    view: ClusterDetailController;
    clusterAssist: ClusterAssist;
    /** Step to the previous (-1) or next (1) cluster of the served list. */
    step: (by: 1 | -1) => void;
    /** The one primary is hidden while a selection shows the bulk bar. */
    showPrimary: boolean;
    /** The cluster's own rare actions (flag, reprocess, accept the VLM), first under More. */
    more: MenuItem[];
  }
  let { ws, view, clusterAssist, step, showPrimary, more }: Props = $props();

  const card = $derived(view.cluster);
  const size = $derived(card?.size ?? view.cropPager.total);
  const facts = $derived(
    [
      `${size.toLocaleString()} items`,
      card?.labelled_share != null
        ? `${Math.round(card.labelled_share * 100)}% labelled`
        : null,
      card?.label_agreement != null
        ? `agreement ${Math.round(card.label_agreement * 100)}%`
        : null,
    ]
      .filter(Boolean)
      .join(' · '),
  );
  const sortId = $derived(scopeSortOf(view.outliersFirst, view.detectorSort));
  const refineCount = $derived(
    [
      view.subjectScope !== 0,
      view.minBlurRatio !== null,
      !view.strategyBar.isDefault,
      view.facetFilterActive,
      view.classSourceFilter !== null,
    ].filter(Boolean).length,
  );

  let diverseOpen = $state(false);

  const moreItems = $derived<MenuItem[]>([
    ...more,
    {
      id: 'diverse-sample',
      label: 'Select a diverse sample…',
      testid: 'diverse-sample-open',
      onselect: () => (diverseOpen = true),
    },
    {
      id: 'undo',
      label: 'Undo last change',
      icon: 'undo-2' as const,
      onselect: () => void undoStore.undoLast(),
    },
  ]);
</script>

<div class="border-b border-line px-4 py-2.5" data-testid="cluster-scope-header">
  <div class="flex flex-wrap items-center gap-x-3 gap-y-2">
    <button
      type="button"
      class="btn shrink-0"
      title="Back to all clusters"
      data-testid="scope-back"
      onclick={() => ws.openCluster(null)}
    >
      <Icon name="arrow-left" size={14} />All clusters
    </button>
    <h1 class="sr-only">Cluster {view.clusterName ?? displayNoText(card ?? {})}</h1>
    <div
      class="flex items-baseline gap-2 text-title font-semibold"
      data-testid="scope-title"
    >
      {#if view.clusterName}
        <span class="capitalize text-fg-1">{view.clusterName}</span>
        <span class="text-body font-normal text-fg-3 tabular-nums" title={view.clusterUid}
          >{displayNoText(card ?? {})}</span
        >
      {:else}
        <span class="text-fg-1" title={view.clusterUid}
          >Cluster {displayNoText(card ?? {})}</span
        >
        {#if card?.is_unlabeled}
          <span class="text-body font-normal text-warn-text">unlabeled</span>
        {/if}
      {/if}
    </div>
    <span class="text-meta text-fg-2 tabular-nums" data-testid="scope-facts">{facts}</span
    >
    {#if view.cardError}
      <span class="text-meta text-warn-text" data-testid="cluster-card-error"
        >Cluster details unavailable: {view.cardError}</span
      >
    {/if}
    <span class="grow"></span>
    <button
      type="button"
      class="btn"
      data-testid="scope-prev"
      title="Previous cluster"
      onclick={() => step(-1)}
    >
      <Icon name="chevron-left" size={14} />Prev
      <Kbd combo={keymapStore.combo('workspace.scope_prev')} />
    </button>
    <button
      type="button"
      class="btn"
      data-testid="scope-next"
      title="Next cluster"
      onclick={() => step(1)}
    >
      Next <Kbd combo={keymapStore.combo('workspace.scope_next')} /><Icon
        name="chevron-right"
        size={14}
      />
    </button>
  </div>
  {#if showPrimary}
    <div class="mt-2 flex flex-wrap items-center gap-2" data-testid="cluster-actions">
      <button
        type="button"
        class="btn btn-primary"
        data-testid="label-cluster"
        disabled={size === 0}
        onclick={() => ws.startLabelCluster()}
      >
        Label all {size.toLocaleString()} as…
        <Kbd combo={keymapStore.combo('workspace.label_cluster')} />
      </button>
      <button
        type="button"
        class="btn"
        title="Ask the VLM about the selected crops, this cluster, or a representative sample of it"
        data-testid="assist-open"
        onclick={clusterAssist.openAssist}
      >
        Assist…
      </button>
      <Menu
        label="More"
        items={moreItems}
        triggerClass="btn"
        triggerTestid="scope-more"
      />
      <span class="grow"></span>
      <SortSelect
        testid="scope-sort"
        options={[...SCOPE_SORTS]}
        applied={sortId}
        onchange={(id) =>
          applyScopeSort(id as (typeof SCOPE_SORTS)[number]['id'], {
            setOrder: (o) => (view.strategyBar.sort = o),
            setDetectorSort: (v) => (view.detectorSort = v),
          })}
      />
      <RefinePopover count={refineCount} testid="scope-refine">
        <ClusterRefineDetail {view} />
      </RefinePopover>
    </div>
  {/if}
</div>
<DiverseSample {ws} bind:open={diverseOpen} />
