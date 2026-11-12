<script lang="ts">
  import Icon from '$components/ui/Icon.svelte';
  /**
   * One open cluster inside the Clusters view: its crops, the facet chips and
   * the bulk bar. The cluster is `ws.url.scope`; the filter, queue and view
   * stay as they were, and the crop selection is the Workspace's.
   */
  import CropDetailModal from '$components/CropDetailModal.svelte';
  import ReprocessControl from '$components/datasets/ReprocessControl.svelte';
  import BulkBar from '$components/workspace/BulkBar.svelte';
  import ClusterDetailFilters from '$lib/components/clusters/ClusterDetailFilters.svelte';
  import ClusterFacetBar from '$lib/components/clusters/ClusterFacetBar.svelte';
  import ClusterDetailGrid from '$lib/components/clusters/ClusterDetailGrid.svelte';
  import MoveClusterPicker from '$lib/components/clusters/MoveClusterPicker.svelte';
  import { displayNoText } from '$lib/clusters/clusterCardText';
  import { createClusterDetailActions } from '$lib/clusters/clusterDetailActions.svelte';
  import { createClusterDetailController } from '$lib/clusters/clusterDetailController.svelte';
  import { createClusterDetailLive } from '$lib/clusters/clusterDetailLive.svelte';
  import ClusterProposalPanel from '$components/assistRun/ClusterProposalPanel.svelte';
  import AssistRunDialog from '$components/assistRun/AssistRunDialog.svelte';
  import { createClusterAssist } from '$lib/assistRun/clusterAssist.svelte';
  import SelectionActionDialog from '$components/itemFilter/SelectionActionDialog.svelte';
  import { createSelectionActionController } from '$lib/itemFilter/selectionActionController.svelte';
  import { getClusters } from '$lib/api_clusters';
  import { keyboardStore } from '$stores/keyboard.svelte';
  import { dropOnClassStore } from '$stores/dropOnClass.svelte';
  import type { Crop } from '$lib/types';
  import { NO_SELECTION } from '$lib/workspace/selection';
  import type { WorkspaceController } from '$lib/workspace/workspaceController.svelte';

  let { ws }: { ws: WorkspaceController } = $props();

  const view = createClusterDetailController(() => ws.url.scope?.id ?? '');
  const actions = createClusterDetailActions(view);
  const live = createClusterDetailLive(view);
  const clusterAssist = createClusterAssist(view);

  // The selection survives a view switch: seed the grid's own from it, then mirror it back.
  // `ws` is fixed for the life of the view, so reading it once is right.
  // svelte-ignore state_referenced_locally
  if (ws.selection.kind === 'items') view.sel.ids = new Set(ws.selection.cropIds);
  $effect(() => {
    const ids = [...view.sel.ids];
    ws.select(ids.length > 0 ? { kind: 'items', cropIds: ids } : NO_SELECTION);
  });

  $effect(() =>
    ws.registerPage({
      kind: 'items',
      ids: () => view.filteredCrops.map((c) => c.id),
      total: () => view.cropPager.total,
      apply: (ids) => view.sel.selectAll(ids),
    }),
  );

  $effect(() => {
    ws.scopeCard = view.cluster;
    return () => {
      ws.scopeCard = null;
    };
  });

  // A class letter with nothing selected labels the whole open cluster: the
  // server's dry run states the count, Enter applies it.
  const wholeCluster = createSelectionActionController(() => ({ filter: ws.itemBody() }));
  $effect(() =>
    dropOnClassStore.register((cls, droppedIds) => {
      if (view.sel.size === 0 && droppedIds.length === 0) {
        wholeCluster.classId = cls.id;
        return wholeCluster.open('label');
      }
      return actions.controller.handleClassDrop(cls, droppedIds);
    }),
  );

  // `,` and `.` step through the served cluster list.
  async function step(by: 1 | -1): Promise<void> {
    const uids = (await getClusters({ per_cluster: 0 })).items.map((c) => c.uid);
    const at = uids.indexOf(view.clusterUid);
    const next = uids[at + by];
    if (at >= 0 && next) ws.openCluster(next);
  }
  $effect(() => {
    const offs = [
      keyboardStore.registerAction(
        'workspace.scope_prev',
        () => void step(-1),
        'workspace',
      ),
      keyboardStore.registerAction(
        'workspace.scope_next',
        () => void step(1),
        'workspace',
      ),
      keyboardStore.registerAction(
        'workspace.label_cluster',
        () => ws.startLabelCluster(),
        'workspace',
      ),
    ];
    return () => offs.forEach((off) => off());
  });

  // The proposal on screen belongs to the cluster being viewed.
  $effect(() => {
    void view.clusterUid;
    void clusterAssist.load();
  });

  // Crop opened in the read-only details modal (info button on each card).
  let detailCrop = $state<Crop | null>(null);
</script>

<div class="flex h-full flex-col">
  <div class="flex flex-wrap items-center gap-3 border-b border-line px-4 py-2.5">
    <button
      type="button"
      class="btn shrink-0"
      title="Back to all clusters"
      data-testid="scope-back"
      onclick={() => ws.openCluster(null)}
    >
      <Icon name="arrow-left" size={14} class="mr-1 -mt-0.5 inline-block" />All clusters
    </button>
    <h1
      class="flex items-baseline gap-2 text-title font-semibold"
      data-testid="scope-title"
    >
      {#if view.clusterName}
        <span class="capitalize text-fg-1">{view.clusterName}</span>
        <span class="font-mono text-body font-normal text-fg-3" title={view.clusterUid}
          >{displayNoText(view.cluster ?? {})}</span
        >
      {:else}
        <span class="text-fg-1" title={view.clusterUid}
          >Cluster {displayNoText(view.cluster ?? {})}</span
        >
        {#if view.cluster?.is_unlabeled}
          <span class="text-body font-normal text-warn-text/80">unlabeled</span>
        {/if}
      {/if}
    </h1>
    {#if view.cardError}
      <span class="text-meta text-warn-text" data-testid="cluster-card-error"
        >Cluster details unavailable: {view.cardError}</span
      >
    {/if}
  </div>
  <BulkBar
    {ws}
    onapplied={() => {
      void view.loadFirst();
      if (!ws.labelThenNext) return;
      ws.labelThenNext = false;
      ws.clearSelection();
      void step(1);
    }}
  >
    {#snippet extra()}
      <button
        class="btn"
        type="button"
        onclick={() => void actions.flagSelectedForNewClass()}
        disabled={view.sel.size === 0}
        title="{actions.kg(
          'cluster.flag_new_class',
        )} — flag selected as needing a new class"
      >
        Flag
      </button>
      <ReprocessControl
        target={{ kind: 'crops', cropIds: [...view.sel.ids] }}
        disabled={view.sel.size === 0}
        buttonClass="btn"
        onapplied={() => void view.loadFirst()}
      />
      <button
        class="btn"
        type="button"
        onclick={actions.controller.acceptAllVlmOnPage}
        title="{actions.kg(
          'cluster.accept_all_vlm',
        )} — accept all VLM suggestions on this page"
      >
        Accept VLM
      </button>
      <button
        class="btn"
        type="button"
        onclick={clusterAssist.openAssist}
        title="Ask the VLM about the selected crops, this cluster, or a representative sample of it"
        data-testid="assist-open"
      >
        Assist…
      </button>
    {/snippet}
  </BulkBar>
  <ClusterDetailFilters {view} />
  <ClusterFacetBar {view} />
  {#if clusterAssist.loadError}
    <p
      class="border-b border-line px-4 py-1.5 text-meta text-warn-text"
      data-testid="cluster-assist-error"
    >
      VLM proposals unavailable: {clusterAssist.loadError}
    </p>
  {/if}
  {#if clusterAssist.latest}
    {#key clusterAssist.latest.run_id}
      <div class="border-b border-line px-4 py-2">
        <ClusterProposalPanel
          run={clusterAssist.latest}
          onapplied={() => {
            void view.loadFirst();
            void clusterAssist.load();
          }}
        />
      </div>
    {/key}
  {/if}
  <ClusterDetailGrid {view} {actions} {live} ondetail={(c) => (detailCrop = c)} />
</div>

<MoveClusterPicker {view} {actions} />

<SelectionActionDialog
  controller={wholeCluster}
  onapplied={() => void view.loadFirst()}
/>

<AssistRunDialog
  ctl={clusterAssist.assist}
  onclosed={() => {
    void view.loadFirst();
    void clusterAssist.load();
  }}
/>

{#if detailCrop}
  <CropDetailModal crop={detailCrop} onclose={() => (detailCrop = null)} />
{/if}
