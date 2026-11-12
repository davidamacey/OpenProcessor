<script lang="ts">
  /**
   * The Map view: where the photos matching the Workspace filter were taken. The server decides
   * everything shown: whether a map exists, the located / without-location counts, the cells and
   * single-photo pins of the viewport. A cell click focuses it (its summary is in the details
   * panel); Shift+drag or the lasso turns an area into a filter chip for every view.
   */
  import { untrack } from 'svelte';
  import MapCanvas from '$components/map/MapView.svelte';
  import { activeMap } from '$lib/map/activeMap.svelte';
  import { AreaTools } from '$lib/map/areaTools.svelte';
  import { MapController } from '$lib/map/mapController.svelte';
  import type { MapViewQuery } from '$lib/types_map';
  import type { WorkspaceController } from '$lib/workspace/workspaceController.svelte';

  let { ws }: { ws: WorkspaceController } = $props();

  // `ws` is fixed for the life of the view.
  // svelte-ignore state_referenced_locally
  const ctl = new MapController(ws);
  // svelte-ignore state_referenced_locally
  const area = new AreaTools(ws);

  $effect(() => {
    const ac = new AbortController();
    void ctl.loadMeta(ac.signal);
    activeMap.ctl = ctl;
    return () => {
      ac.abort();
      ctl.dispose();
      activeMap.ctl = null;
    };
  });

  const queryKey = $derived(JSON.stringify(ws.itemQuery()));
  let firstKey = true;
  $effect(() => {
    void queryKey;
    if (firstKey) {
      firstKey = false;
      return;
    }
    untrack(() => void ctl.refresh());
  });

  // The URL focus drives the cell: a cell focus (a reload, Back) is restored once its cell is in
  // the loaded set, and leaving a cell focus drops the cell. Only the focus changing clears it, so
  // a click (cell set before the URL catches up) is not undone by an unrelated re-run.
  let hadCellFocus = false;
  $effect(() => {
    const focus = ws.url.focus;
    const features = ctl.cells?.features;
    if (focus?.kind !== 'cell') {
      if (hadCellFocus) untrack(() => ctl.clearLocal());
      hadCellFocus = false;
      return;
    }
    hadCellFocus = true;
    if (untrack(() => ctl.cell?.tile) === focus.id) return;
    const hit = features?.find((f) => f.properties.tile === focus.id);
    if (hit)
      untrack(
        () =>
          void ctl.focusCell({
            tile: focus.id,
            count: hit.properties.count,
            bounds: hit.properties.bounds,
          }),
      );
  });

  /** A focus is useless behind a closed drawer (under 1280 px). */
  function showDetails(): void {
    if (!ws.detailsOpen) ws.toggleDetails();
  }

  function onview(view: MapViewQuery): void {
    void ctl.loadCells(view);
  }
</script>

<div class="flex min-h-0 flex-1 flex-col" data-testid="map-view">
  <h1 class="sr-only">Map</h1>
  {#if ctl.meta?.available}
    <p class="flex flex-wrap items-center gap-x-3 px-4 py-2 text-xs text-fg-2">
      <span data-testid="map-located">{ctl.meta.n_located.toLocaleString()} located</span>
      <span aria-hidden="true">·</span>
      {#if ctl.meta.n_without_location > 0}
        <button
          type="button"
          class="underline decoration-dotted underline-offset-2 hover:text-white"
          data-testid="map-without-location"
          onclick={() => ctl.showWithoutLocation()}
          >{ctl.meta.n_without_location.toLocaleString()} without location</button
        >
      {:else}
        <span data-testid="map-without-location">0 without location</span>
      {/if}
      <span class="chip text-fg-2" data-testid="map-policy"
        >Location: {ctl.meta.location_policy === 'store_coarse'
          ? 'coarse (about 1 km)'
          : 'exact'}</span
      >
      <span class="text-fg-3">Shift+drag selects an area, or</span>
      <button
        type="button"
        class="btn-sm"
        aria-pressed={area.armed}
        data-testid="map-lasso"
        onclick={() => area.toggleLasso()}>draw a lasso</button
      >
    </p>
  {/if}
  {#if ctl.loadingMeta}
    <p class="p-4 text-sm text-fg-3" role="status">Loading…</p>
  {:else if ctl.metaError}
    <div class="flex flex-col items-start gap-3 p-4 text-sm">
      <p class="text-danger" role="alert" data-testid="map-error">{ctl.metaError}</p>
      <button type="button" class="btn" onclick={() => void ctl.loadMeta()}>Retry</button>
    </div>
  {:else if ctl.meta && !ctl.meta.available}
    <p class="p-6 text-sm text-fg-2" data-testid="map-unavailable">
      {#if ctl.meta.location_policy === 'drop'}
        This project does not store where photos were taken (its location policy is
        <code>drop</code>).
      {:else}
        None of this project's photos has a GPS position.
      {/if}
    </p>
  {:else if ctl.meta}
    {#if ctl.cellsError}
      <p class="px-4 text-xs text-danger" role="alert">{ctl.cellsError}</p>
    {/if}
    {#if ctl.cells?.truncated}
      <p class="px-4 text-xs text-warn" data-testid="map-truncated">
        More cells match than fit on the map; zoom in to see them.
      </p>
    {/if}
    {#if ctl.cells?.join_truncated}
      <p class="px-4 text-xs text-warn" data-testid="map-join-truncated">
        The filter matches more photos than the map can use at once; narrow it to see all
        of them.
      </p>
    {/if}
    <div class="relative min-h-[420px] flex-1 p-4 pt-2">
      <MapCanvas
        meta={ctl.meta}
        cells={ctl.cells}
        {area}
        selectedTile={ctl.cell?.tile ?? null}
        {onview}
        oncell={(cell) => {
          showDetails();
          void ctl.focusCell(cell);
        }}
        onpin={(id) => {
          showDetails();
          ctl.focusPin(id);
        }}
      />
      {#if ctl.loadingCells}
        <span class="chip absolute left-6 top-4 bg-surface-1/90 text-fg-2">Loading…</span>
      {/if}
    </div>
  {/if}
</div>
