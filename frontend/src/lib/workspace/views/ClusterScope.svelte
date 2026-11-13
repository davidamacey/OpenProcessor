<script lang="ts">
  /**
   * One open cluster inside the Clusters view: its crops, the facet chips and
   * the bulk bar. The cluster is `ws.url.scope`; the filter, queue and view
   * stay as they were, and the crop selection is the Workspace's.
   */
  import CropDetailModal from '$components/CropDetailModal.svelte';
  import ReprocessControl from '$components/datasets/ReprocessControl.svelte';
  import type { MenuItem } from '$components/ui/Menu.svelte';
  import BulkBar from '$components/workspace/BulkBar.svelte';
  import ClusterScopeHeader from '$lib/components/clusters/ClusterScopeHeader.svelte';
  import ClusterDetailGrid from '$lib/components/clusters/ClusterDetailGrid.svelte';
  import MoveClusterPicker from '$lib/components/clusters/MoveClusterPicker.svelte';
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
  async function stepAsync(by: 1 | -1): Promise<void> {
    const uids = (await getClusters({ per_cluster: 0 })).items.map((c) => c.uid);
    const at = uids.indexOf(view.clusterUid);
    const next = uids[at + by];
    if (at >= 0 && next) ws.openCluster(next);
  }
  const step = (by: 1 | -1): void => void stepAsync(by);

  // After a whole-cluster label the next cluster opens, once: when the write is queued (the
  // dialog closed) or, when it was not detached, when it has landed.
  function advance(): void {
    if (!ws.labelThenNext) return;
    ws.labelThenNext = false;
    if (!ws.openNextAfterLabel) return;
    ws.clearSelection();
    step(1);
  }
  function onLabelWritten(): void {
    void view.loadFirst();
    advance();
  }

  let reprocess = $state<ReturnType<typeof ReprocessControl> | null>(null);
  const scopeMore = $derived<MenuItem[]>([
    {
      id: 'assist',
      label: 'Assist…',
      testid: 'assist-open',
      onselect: clusterAssist.openAssist,
    },
    {
      id: 'flag',
      label: 'Flag selected as a new class',
      disabled: view.sel.size === 0,
      onselect: () => void actions.flagSelectedForNewClass(),
    },
    ...(reprocess?.isAvailable() === true
      ? [
          {
            id: 'reprocess',
            label: 'Reprocess…',
            testid: 'reprocess-open',
            disabled: view.sel.size === 0,
            onselect: () => void reprocess?.openFlow(),
          },
        ]
      : []),
    {
      id: 'accept-vlm',
      label: "Accept the VLM's suggestions on this page",
      onselect: () => actions.controller.acceptAllVlmOnPage(),
    },
  ]);
  $effect(() => {
    const offs = [
      keyboardStore.registerAction('workspace.scope_prev', () => step(-1), 'workspace'),
      keyboardStore.registerAction('workspace.scope_next', () => step(1), 'workspace'),
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
  <ClusterScopeHeader
    {ws}
    {view}
    {clusterAssist}
    {step}
    showPrimary={view.sel.size === 0}
    more={scopeMore}
  />
  <BulkBar
    {ws}
    more={scopeMore}
    onapplied={onLabelWritten}
    ondetached={advance}
    split={{
      classes: view.facets?.detector_classes ?? [],
      onsplit: (value) => view.toggleDetectorClass(value),
    }}
  />
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

<ReprocessControl
  bind:this={reprocess}
  hideButton
  target={{ kind: 'crops', cropIds: [...view.sel.ids] }}
  onapplied={() => void view.loadFirst()}
/>

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
