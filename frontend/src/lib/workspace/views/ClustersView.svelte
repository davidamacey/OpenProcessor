<script lang="ts">
  /** Clusters: the cluster cards, or one open cluster when the URL scopes to it. */
  import type { WorkspaceController } from '$lib/workspace/workspaceController.svelte';
  import ClusterListView from './ClusterListView.svelte';
  import ClusterScope from './ClusterScope.svelte';

  let { ws }: { ws: WorkspaceController } = $props();
</script>

{#if ws.url.scope?.kind === 'cluster'}
  {#key ws.url.scope.id}
    <ClusterScope {ws} />
  {/key}
{:else}
  <ClusterListView {ws} />
{/if}
