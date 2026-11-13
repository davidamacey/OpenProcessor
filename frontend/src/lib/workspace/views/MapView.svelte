<script lang="ts">
  /**
   * The Map view: where the photos matching the Workspace filter were taken. The server decides
   * everything shown: whether a map exists, the located / without-location counts, the cells and
   * single-photo pins of the viewport. A cell click focuses it (its summary is in the details
   * panel); Shift+drag or the lasso turns an area into a filter chip for every view.
   */
  import { untrack } from 'svelte';
  import Icon from '$components/ui/Icon.svelte';
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

  // The counts under the filter are the server's; before a cell answer the totals stand.
  const matching = $derived(ctl.cells?.n_located_matching ?? null);
  const locatedText = $derived.by(() => {
    const total = ctl.meta?.n_located ?? 0;
    return matching != null && matching < total
      ? `${matching.toLocaleString()} of ${total.toLocaleString()} located photos match`
      : `${total.toLocaleString()} located`;
  });
  const withoutLocation = $derived(
    ctl.cells?.n_unlocated_matching ?? ctl.meta?.n_without_location ?? 0,
  );
  const noneMatch = $derived(
    matching === 0 && (ctl.meta?.n_located ?? 0) > 0 && !ctl.loadingCells,
  );

  function onview(view: MapViewQuery): void {
    void ctl.loadCells(view);
  }
</script>

<div class="flex h-full min-h-0 flex-1 flex-col" data-testid="map-view">
  <h1 class="sr-only">Map</h1>
  {#if ctl.meta?.available}
    <div
      class="flex flex-wrap items-center gap-x-3 gap-y-1 px-4 py-2 text-meta text-fg-2"
    >
      <span class="tabular-nums" data-testid="map-located">{locatedText}</span>
      <span aria-hidden="true">·</span>
      {#if withoutLocation > 0}
        <button
          type="button"
          class="underline decoration-dotted underline-offset-2 hover:text-white"
          data-testid="map-without-location"
          onclick={() => ctl.showWithoutLocation()}
          >{withoutLocation.toLocaleString()} without location</button
        >
      {:else}
        <span data-testid="map-without-location">0 without location</span>
      {/if}
      <span class="chip text-fg-2" data-testid="map-policy"
        >Location: {ctl.meta.location_policy === 'store_coarse'
          ? 'coarse (about 1 km)'
          : 'exact'}</span
      >
      <span class="grow"></span>
      <button
        type="button"
        class="btn btn-ghost btn-sm"
        aria-pressed={area.armed}
        title="Shift+drag on the map draws an area"
        data-testid="map-lasso"
        onclick={() => area.toggleLasso()}
      >
        <Icon name="square-dashed" size={14} />Draw an area
      </button>
      <button
        type="button"
        class="btn btn-primary"
        data-testid="map-show-gallery"
        onclick={() => ws.setView('gallery')}
      >
        Show in Gallery
      </button>
    </div>
  {/if}
  {#if ctl.loadingMeta}
    <p class="p-4 text-body text-fg-3" role="status">Loading…</p>
  {:else if ctl.metaError}
    <div class="flex flex-col items-start gap-3 p-4 text-body">
      <p class="text-danger-text" role="alert" data-testid="map-error">{ctl.metaError}</p>
      <button type="button" class="btn" onclick={() => void ctl.loadMeta()}>Retry</button>
    </div>
  {:else if ctl.meta && !ctl.meta.available}
    <p class="p-6 text-body text-fg-2" data-testid="map-unavailable">
      {#if ctl.meta.location_policy === 'drop'}
        This project does not store where photos were taken (its location policy is
        <code>drop</code>).
      {:else}
        None of this project's photos has a GPS position.
      {/if}
    </p>
  {:else if ctl.meta}
    {#if ctl.cellsError}
      <p class="px-4 text-meta text-danger-text" role="alert">{ctl.cellsError}</p>
    {/if}
    {#if ctl.cells?.truncated}
      <p class="px-4 text-meta text-warn-text" data-testid="map-truncated">
        More cells match than fit on the map; zoom in to see them.
      </p>
    {/if}
    {#if ctl.cells?.join_truncated}
      <p class="px-4 text-meta text-warn-text" data-testid="map-join-truncated">
        The filter matches more photos than the map can use at once; narrow it to see all
        of them.
      </p>
    {/if}
    {#if ctl.cells?.features.length}
      <!-- The canvas is a pointing device; this is the same cells for the keyboard and screen readers. -->
      <details class="px-4 pb-1 text-meta text-fg-2" data-testid="map-cell-list">
        <summary class="cursor-pointer">
          Cells in view ({ctl.cells.features.length.toLocaleString()})
        </summary>
        <ul class="mt-1 flex max-h-40 flex-wrap gap-1 overflow-auto">
          {#each ctl.cells.features as f (f.properties.tile)}
            {@const p = f.properties}
            <li>
              <button
                type="button"
                class="btn-sm"
                data-testid="map-cell-item"
                onclick={() => {
                  showDetails();
                  if (p.image_id) ctl.focusPin(p.image_id);
                  else
                    void ctl.focusCell({
                      tile: p.tile,
                      count: p.count,
                      bounds: p.bounds,
                    });
                }}
              >
                {p.count.toLocaleString()} photo{p.count === 1 ? '' : 's'} at {f.geometry.coordinates[1]!.toFixed(
                  2,
                )}, {f.geometry.coordinates[0]!.toFixed(2)}
              </button>
            </li>
          {/each}
        </ul>
      </details>
    {/if}
    <div class="relative flex min-h-[420px] flex-1 flex-col p-4 pt-2">
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
      {#if noneMatch && ctl.meta}
        <div
          class="absolute inset-x-6 top-6 z-10 flex flex-wrap items-center gap-3 rounded-card border border-line-strong bg-surface-2/95 px-4 py-3 text-body text-fg-1 shadow-e3"
          role="status"
          data-testid="map-none-match"
        >
          <span
            >None of the {ctl.meta.n_located.toLocaleString()} located photo{ctl.meta
              .n_located === 1
              ? ''
              : 's'} match the filter.</span
          >
          <button
            type="button"
            class="btn"
            data-testid="map-clear-filters"
            onclick={() => {
              ws.filter.clear();
              ws.applyFilter();
            }}
          >
            Clear filters
          </button>
        </div>
      {/if}
      {#if ctl.loadingCells}
        <span class="chip absolute left-6 top-4 bg-surface-1/90 text-fg-2">Loading…</span>
      {/if}
    </div>
  {/if}
</div>
