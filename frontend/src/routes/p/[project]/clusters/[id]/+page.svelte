<script lang="ts">
  import CropDetailModal from '$components/CropDetailModal.svelte';
  import ClusterDetailFilters from '$lib/components/clusters/ClusterDetailFilters.svelte';
  import ClusterDetailGrid from '$lib/components/clusters/ClusterDetailGrid.svelte';
  import ClusterDetailToolbar from '$lib/components/clusters/ClusterDetailToolbar.svelte';
  import MoveClusterPicker from '$lib/components/clusters/MoveClusterPicker.svelte';
  import SubClusterTabs from '$lib/components/clusters/SubClusterTabs.svelte';
  import { createClusterDetailActions } from '$lib/clusters/clusterDetailActions.svelte';
  import { createClusterDetailController } from '$lib/clusters/clusterDetailController.svelte';
  import { createClusterDetailLive } from '$lib/clusters/clusterDetailLive.svelte';
  import type { Crop } from '$lib/types';

  const view = createClusterDetailController();
  const actions = createClusterDetailActions(view);
  const live = createClusterDetailLive(view);

  // Crop opened in the read-only details modal (info button on each card).
  let detailCrop = $state<Crop | null>(null);
</script>

<div class="flex h-full flex-col">
  <ClusterDetailToolbar {view} {actions} />
  <ClusterDetailFilters {view} />
  <SubClusterTabs {view} />
  <ClusterDetailGrid {view} {actions} {live} ondetail={(c) => (detailCrop = c)} />
</div>

<MoveClusterPicker {view} {actions} />

{#if detailCrop}
  <CropDetailModal crop={detailCrop} onclose={() => (detailCrop = null)} />
{/if}
