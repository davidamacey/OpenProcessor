<script lang="ts">
  import CropDetailModal from '$components/CropDetailModal.svelte';
  import ClusterDetailFilters from '$lib/components/clusters/ClusterDetailFilters.svelte';
  import ClusterFacetBar from '$lib/components/clusters/ClusterFacetBar.svelte';
  import ClusterDetailGrid from '$lib/components/clusters/ClusterDetailGrid.svelte';
  import ClusterDetailToolbar from '$lib/components/clusters/ClusterDetailToolbar.svelte';
  import MoveClusterPicker from '$lib/components/clusters/MoveClusterPicker.svelte';
  import SubClusterTabs from '$lib/components/clusters/SubClusterTabs.svelte';
  import { createClusterDetailActions } from '$lib/clusters/clusterDetailActions.svelte';
  import { createClusterDetailController } from '$lib/clusters/clusterDetailController.svelte';
  import { createClusterDetailLive } from '$lib/clusters/clusterDetailLive.svelte';
  import ClusterProposalPanel from '$components/assistRun/ClusterProposalPanel.svelte';
  import AssistRunDialog from '$components/assistRun/AssistRunDialog.svelte';
  import { createClusterAssist } from '$lib/assistRun/clusterAssist.svelte';
  import type { Crop } from '$lib/types';

  const view = createClusterDetailController();
  const actions = createClusterDetailActions(view);
  const live = createClusterDetailLive(view);
  const clusterAssist = createClusterAssist(view);

  // The proposal on screen belongs to the cluster or sub-cluster being viewed.
  $effect(() => {
    void view.clusterId;
    void view.subTab;
    void clusterAssist.load();
  });

  // Crop opened in the read-only details modal (info button on each card).
  let detailCrop = $state<Crop | null>(null);
</script>

<div class="flex h-full flex-col">
  <ClusterDetailToolbar {view} {actions} {clusterAssist} />
  <ClusterDetailFilters {view} />
  <ClusterFacetBar {view} />
  <SubClusterTabs {view} />
  {#if clusterAssist.loadError}
    <p
      class="border-b border-zinc-800 px-4 py-1.5 text-xs text-amber-300"
      data-testid="cluster-assist-error"
    >
      VLM proposals unavailable: {clusterAssist.loadError}
    </p>
  {/if}
  {#if clusterAssist.latest}
    {#key clusterAssist.latest.run_id}
      <div class="border-b border-zinc-800 px-4 py-2">
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
