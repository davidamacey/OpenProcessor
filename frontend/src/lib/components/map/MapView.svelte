<script lang="ts">
  /**
   * The map: a MapLibre GL JS canvas (https://maplibre.org/maplibre-gl-js/docs/) with a raster
   * basemap from the served tile template and one GeoJSON source of served grid cells. The
   * browser fetches basemap tiles from `meta.tile_url`; cells and counts are the server's, a
   * click is reported to the host, and nothing here aggregates or rounds a position.
   */
  import { onMount } from 'svelte';
  import {
    GPUInitializationError,
    Map as MapLibreMap,
    NavigationControl,
    Popup,
    setWorkerUrl,
    type GeoJSONSource,
  } from 'maplibre-gl';
  // MapLibre 6 looks for its worker file next to the bundled library chunk, where Vite does
  // not emit it; handing Vite the file makes the build copy it and gives its served URL.
  import workerUrl from 'maplibre-gl/dist/maplibre-gl-worker.mjs?url';
  import 'maplibre-gl/dist/maplibre-gl.css';
  import { viewQuery } from '$lib/map/viewport';
  import type { MapCellCollection, MapMeta, MapViewQuery } from '$lib/types_map';

  interface Props {
    meta: MapMeta;
    cells: MapCellCollection | null;
    /** The viewport changed (also once at start): load the cells for it. */
    onview: (view: MapViewQuery) => void;
    /** A cell was clicked; the host decides between listing its photos and zooming in. */
    oncell: (cell: { tile: string; count: number; zoom: number }) => boolean;
  }

  let { meta, cells, onview, oncell }: Props = $props();

  /** What a GeoJSON source takes; the served cells are already a FeatureCollection. */
  type GeoData = Parameters<GeoJSONSource['setData']>[0];

  let container: HTMLDivElement;
  let map: MapLibreMap | null = null;
  let ready = false;
  /** Set when MapLibre cannot get a WebGL2 context; the card then explains instead of drawing. */
  let noWebgl = $state(false);
  /** MapLibre's own words when the worker or style fails to load; the canvas stays blank then. */
  let loadError = $state<string | null>(null);
  const gpsNote = $derived(
    `${meta.n_located} photo${meta.n_located === 1 ? ' has' : 's have'} GPS`,
  );

  const EMPTY: MapCellCollection = {
    type: 'FeatureCollection',
    features: [],
    precision: 0,
    truncated: false,
    join_truncated: false,
  };

  function pointOf(geometry: { type: string }): [number, number] {
    return (geometry as unknown as { coordinates: [number, number] }).coordinates;
  }

  function emitView(): void {
    if (!map) return;
    const b = map.getBounds();
    onview(
      viewQuery(
        {
          west: b.getWest(),
          south: b.getSouth(),
          east: b.getEast(),
          north: b.getNorth(),
        },
        map.getZoom(),
        meta.max_zoom,
      ),
    );
  }

  function push(): void {
    const source = map?.getSource('cells') as GeoJSONSource | undefined;
    if (ready && source) source.setData((cells ?? EMPTY) as GeoData);
  }

  onMount(() => {
    setWorkerUrl(workerUrl);
    const bounds = meta.bounds;
    try {
      map = new MapLibreMap({
        container,
        style: {
          version: 8,
          sources: {
            basemap: {
              type: 'raster',
              tiles: [meta.tile_url],
              tileSize: 256,
              attribution: meta.tile_attribution,
            },
          },
          layers: [{ id: 'basemap', type: 'raster', source: 'basemap' }],
        },
        bounds: bounds
          ? [
              [bounds.west, bounds.south],
              [bounds.east, bounds.north],
            ]
          : undefined,
        fitBoundsOptions: { padding: 48, maxZoom: Math.max(1, meta.max_zoom - 2) },
        center: bounds ? undefined : [0, 20],
        zoom: bounds ? undefined : 1,
        maxZoom: meta.max_zoom + 3,
      });
    } catch (err) {
      // MapLibre throws GPUInitializationError from the constructor when no WebGL2 context exists.
      if (!(err instanceof GPUInitializationError)) throw err;
      noWebgl = true;
      return;
    }
    map.addControl(new NavigationControl({ showCompass: false }), 'top-right');

    map.on('load', () => {
      if (!map) return;
      map.addSource('cells', {
        type: 'geojson',
        data: EMPTY as GeoData,
      });
      map.addLayer({
        id: 'cells',
        type: 'circle',
        source: 'cells',
        paint: {
          'circle-radius': [
            'interpolate',
            ['linear'],
            ['get', 'count'],
            1,
            5,
            10,
            9,
            100,
            15,
            1000,
            22,
            10000,
            30,
          ],
          'circle-color': [
            'step',
            ['get', 'count'],
            '#38bdf8',
            10,
            '#facc15',
            100,
            '#fb923c',
            1000,
            '#f43f5e',
          ],
          'circle-opacity': 0.75,
          'circle-stroke-width': 1.5,
          'circle-stroke-color': '#0b0b0f',
        },
      });
      ready = true;
      push();
      emitView();
    });
    map.on('moveend', emitView);
    map.on('error', (e) => {
      if (e.error instanceof GPUInitializationError) noWebgl = true;
      // A failed tile is routine (offline, 404) and carries `tile`; anything else blanks the map.
      else if (!('tile' in e)) loadError = e.error?.message ?? 'unknown error';
    });

    const popup = new Popup({ closeButton: false, closeOnClick: false });
    map.on('mouseenter', 'cells', (e) => {
      if (!map) return;
      map.getCanvas().style.cursor = 'pointer';
      const f = e.features?.[0];
      if (!f) return;
      const count = Number(f.properties?.count ?? 0);
      popup
        .setLngLat(pointOf(f.geometry))
        .setText(`${count} photo${count === 1 ? '' : 's'}`)
        .addTo(map);
    });
    map.on('mouseleave', 'cells', () => {
      if (!map) return;
      map.getCanvas().style.cursor = '';
      popup.remove();
    });
    map.on('click', 'cells', (e) => {
      const f = e.features?.[0];
      if (!f || !map) return;
      const zoom = Math.floor(map.getZoom());
      const opened = oncell({
        tile: String(f.properties?.tile),
        count: Number(f.properties?.count ?? 0),
        zoom,
      });
      if (!opened) {
        map.easeTo({
          center: pointOf(f.geometry),
          zoom: Math.min(map.getZoom() + 2, meta.max_zoom + 3),
        });
      }
    });

    return () => {
      map?.remove();
      map = null;
      ready = false;
    };
  });

  $effect(() => {
    void cells;
    push();
  });
</script>

{#if noWebgl}
  <p
    class="surface p-6 text-sm text-amber-300"
    role="status"
    data-testid="map-webgl-unsupported"
  >
    Map needs WebGL2, which this browser could not start; {gpsNote}. The rest of the page
    still works.
  </p>
{/if}
{#if loadError}
  <p
    class="surface p-6 text-sm text-amber-300"
    role="status"
    data-testid="map-load-error"
  >
    Map could not load: {loadError}
  </p>
{/if}
<div
  bind:this={container}
  class:hidden={noWebgl}
  class="h-full min-h-[320px] w-full rounded border border-zinc-800"
  data-testid="map-canvas"
  role="application"
  aria-label="Map of where the photos were taken"
></div>

<style>
  /* The popup box is white (MapLibre's stylesheet); the page's light text colour is unreadable on it. */
  :global(.maplibregl-popup-content) {
    color: #18181b;
  }
</style>
