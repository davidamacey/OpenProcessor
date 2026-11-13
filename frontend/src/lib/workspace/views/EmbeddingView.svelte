<script lang="ts">
  /**
   * The Embedding view: the project's embedding scatter under the Workspace filter. Points the
   * filter leaves out are dimmed (the server marks them), colours are the served class legend,
   * a lasso selects items for the bulk bar, a click opens one in the details panel, and a box
   * drawn on the plot becomes a `viz_box` chip that filters every view.
   */
  import EmbeddingPlot from '$components/EmbeddingPlot.svelte';
  import BulkBar from '$components/workspace/BulkBar.svelte';
  import EmptyNextSteps from '$components/workspace/EmptyNextSteps.svelte';
  import { isEmbeddingVizAvailable, isEmbeddingVizBannerRequired } from '$lib/strategies';
  import { NO_SELECTION } from '$lib/workspace/selection';
  import type { WorkspaceController } from '$lib/workspace/workspaceController.svelte';
  import { strategiesStore } from '$stores/strategies.svelte';

  let { ws }: { ws: WorkspaceController } = $props();

  let plot = $state<{ reload: () => Promise<void> } | null>(null);

  const overlays = $derived(strategiesStore.methods.overlays);
  const available = $derived(isEmbeddingVizAvailable(overlays));
  const bannerRequired = $derived(isEmbeddingVizBannerRequired(overlays));
  const coverageTotal = $derived(
    overlays.find((o) => o.id === 'viz_projection')?.field_coverage_total ?? null,
  );
  const selectedIds = $derived(ws.selection.kind === 'items' ? ws.selection.cropIds : []);
  const query = $derived(ws.itemQuery());

  function select(ids: string[]): void {
    ws.select(ids.length > 0 ? { kind: 'items', cropIds: ids } : NO_SELECTION);
  }

  function drawBox(box: [number, number, number, number]): void {
    ws.filter.vizBox = box;
    ws.applyFilter();
  }
</script>

<div class="flex min-h-0 flex-1 flex-col" data-testid="embedding-view">
  <h1 class="sr-only">Embedding</h1>
  <BulkBar {ws} onapplied={() => void plot?.reload()} />
  <div class="min-h-0 flex-1 overflow-auto p-4">
    {#if strategiesStore.loaded && !available}
      <p class="text-body text-fg-2" data-testid="embedding-unavailable">
        This deployment does not offer the embedding projection (the server reports it as
        off).
      </p>
    {:else}
      <EmbeddingPlot
        bind:this={plot}
        {query}
        vizBox={ws.filter.vizBox}
        {selectedIds}
        {bannerRequired}
        coveragePoolTotal={coverageTotal}
        onselect={select}
        onbox={drawBox}
        onfocus={(id) => ws.setFocus({ kind: 'item', id })}
      >
        {#snippet empty()}
          <EmptyNextSteps />
        {/snippet}
      </EmbeddingPlot>
    {/if}
  </div>
</div>
