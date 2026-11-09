<script lang="ts">
  /**
   * Where the project's photos were taken. The server decides everything shown: whether a map
   * exists (a stored position and a location policy that serves it), the grid cells for the
   * viewport and the photos of a cell. The filters are the shared item filter plus the capture
   * days; a click on a small cell lists its photos, on a large one zooms in.
   */
  import { onMount } from 'svelte';
  import ItemFilterBar from '$lib/components/itemFilter/ItemFilterBar.svelte';
  import MatchingItemsView from '$lib/components/itemFilter/MatchingItemsView.svelte';
  import MapView from '$lib/components/map/MapView.svelte';
  import { MapController } from '$lib/map/mapController.svelte';
  import { withoutOpenVocab } from '$lib/itemFilter/itemFilterState.svelte';
  import type { MapViewQuery } from '$lib/types_map';

  const ctl = new MapController();

  onMount(() => {
    const ac = new AbortController();
    void ctl.loadMeta(ac.signal);
    return () => {
      ac.abort();
      ctl.dispose();
    };
  });

  function onview(view: MapViewQuery): void {
    void ctl.loadCells(view);
  }

  /** True when the click listed the cell's photos; false asks the map to zoom in. */
  function oncell(cell: { tile: string; count: number; zoom: number }): boolean {
    if (!ctl.opensList(cell.count, cell.zoom)) return false;
    void ctl.openCell(cell.tile);
    return true;
  }

  const queryKey = $derived(JSON.stringify(ctl.query()));
  let firstKey = true;
  $effect(() => {
    void queryKey;
    if (firstKey) {
      firstKey = false;
      return;
    }
    void ctl.refresh();
  });
</script>

<div class="mx-auto flex h-full max-w-7xl flex-col gap-3 p-4 sm:p-6">
  <header class="flex flex-wrap items-center gap-3">
    <h1 class="text-2xl font-semibold tracking-tight">Map</h1>
    {#if ctl.meta?.available}
      <span class="chip text-zinc-300" data-testid="map-located"
        >{ctl.meta.n_located} located photo{ctl.meta.n_located === 1 ? '' : 's'}</span
      >
      <span class="chip text-zinc-400" data-testid="map-policy"
        >Location: {ctl.meta.location_policy === 'store_coarse'
          ? 'coarse (about 1 km)'
          : 'exact'}</span
      >
    {/if}
  </header>

  {#if ctl.loadingMeta}
    <section class="surface p-6 text-sm text-zinc-500">Loading…</section>
  {:else if ctl.metaError}
    <section class="surface flex flex-col gap-3 p-6 text-sm">
      <p class="text-red-300" role="alert" data-testid="map-error">{ctl.metaError}</p>
      <button type="button" class="btn w-fit" onclick={() => void ctl.loadMeta()}
        >Retry</button
      >
    </section>
  {:else if ctl.meta && !ctl.meta.available}
    <section class="surface p-6 text-sm text-zinc-400" data-testid="map-unavailable">
      {#if ctl.meta.location_policy === 'drop'}
        This project does not store where photos were taken (its location policy is
        <code>drop</code>).
      {:else}
        None of this project's photos has a GPS position.
      {/if}
    </section>
  {:else if ctl.meta}
    <div class="flex flex-wrap items-end gap-3 text-xs" data-testid="map-filters">
      <ItemFilterBar
        state={ctl.filter}
        visible={withoutOpenVocab}
        error={ctl.cellsError}
      />
    </div>
    {#if ctl.cells?.truncated}
      <p class="text-xs text-amber-300" data-testid="map-truncated">
        More cells match than fit on the map; zoom in to see them.
      </p>
    {/if}
    {#if ctl.cells?.join_truncated}
      <p class="text-xs text-amber-300" data-testid="map-join-truncated">
        The filter matches more photos than the map can use at once; narrow it to see all
        of them.
      </p>
    {/if}

    <div class="grid min-h-0 grow gap-3 lg:grid-cols-[minmax(0,1fr)_minmax(0,28rem)]">
      <div class="relative min-h-[420px] lg:min-h-0">
        <MapView meta={ctl.meta} cells={ctl.cells} {onview} {oncell} />
        {#if ctl.loadingCells}
          <span class="chip absolute left-2 top-2 bg-zinc-900/90 text-zinc-300"
            >Loading…</span
          >
        {/if}
      </div>
      {#if ctl.openTile}
        <aside
          class="surface flex min-h-0 flex-col gap-2 overflow-auto p-3"
          data-testid="map-cell-panel"
        >
          <div class="flex items-center gap-2 text-sm">
            <span class="font-medium"
              >{ctl.openTile.count} photo{ctl.openTile.count === 1 ? '' : 's'} here</span
            >
            <span class="grow"></span>
            <button type="button" class="btn-sm" onclick={() => ctl.closeCell()}
              >Close</button
            >
          </div>
          {#if ctl.openTile.truncated}
            <p class="text-xs text-amber-300">
              Showing the first {ctl.openTile.image_ids.length}; zoom in to narrow the
              area.
            </p>
          {/if}
          {#key ctl.openTile.tile}
            <MatchingItemsView
              query={() => ctl.listQuery()}
              body={() => ctl.listBody()}
              onexit={() => ctl.closeCell()}
            />
          {/key}
        </aside>
      {:else if ctl.openError}
        <p class="text-sm text-red-300" role="alert" data-testid="map-open-error">
          {ctl.openError}
        </p>
      {/if}
    </div>
  {/if}
</div>
