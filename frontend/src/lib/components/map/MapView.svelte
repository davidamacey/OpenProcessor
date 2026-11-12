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
  import type { AreaTools, LngLat } from '$lib/map/areaTools.svelte';
  import type { GeoBox, MapCellCollection, MapMeta, MapViewQuery } from '$lib/types_map';

  interface Props {
    meta: MapMeta;
    cells: MapCellCollection | null;
    /** The viewport changed (also once at start): load the cells for it. */
    onview: (view: MapViewQuery) => void;
    /** A cell was clicked, with the bounds the server gave it. */
    oncell: (cell: { tile: string; count: number; bounds: GeoBox }) => void;
    /** A single-photo pin was clicked. */
    onpin?: (imageId: string) => void;
    /** The focused cell's tile, drawn with a highlight ring. */
    selectedTile?: string | null;
    /** Shift+drag draws a rectangle and the lasso button a polygon; both end in the filter. */
    area?: AreaTools;
  }

  let { meta, cells, onview, oncell, onpin, selectedTile = null, area }: Props = $props();

  /** What a GeoJSON source takes; the served cells are already a FeatureCollection. */
  type GeoData = Parameters<GeoJSONSource['setData']>[0];

  let container: HTMLDivElement;
  let map: MapLibreMap | null = null;
  let ready = false;
  /** Set when MapLibre cannot get a WebGL2 context; the card then explains instead of drawing. */
  let noWebgl = $state(false);
  /** The lasso button was pressed: the next drag draws instead of panning. */
  let lassoArmed = $state(false);
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

  /** MapLibre hands a nested feature property back as a JSON string. */
  function boundsOf(raw: unknown): GeoBox {
    return (typeof raw === 'string' ? JSON.parse(raw) : raw) as GeoBox;
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

  /** The gesture in progress as a GeoJSON polygon for the `area` layer (empty when idle). */
  function areaShape(): GeoData {
    const pts = area?.points ?? [];
    let ring: LngLat[] = [];
    if (area?.mode === 'rectangle' && pts.length === 2) {
      const [[w, s], [e, n]] = pts;
      ring = [
        [w, s],
        [e, s],
        [e, n],
        [w, n],
        [w, s],
      ];
    } else if (pts.length >= 2) ring = [...pts, pts[0]];
    return {
      type: 'FeatureCollection',
      features: ring.length
        ? [
            {
              type: 'Feature',
              properties: {},
              geometry: { type: 'Polygon', coordinates: [ring] },
            },
          ]
        : [],
    } as GeoData;
  }

  function pushArea(): void {
    const source = map?.getSource('area') as GeoJSONSource | undefined;
    if (ready && source) source.setData(areaShape());
  }

  /** Pointer gestures for the area tools; plain drags keep panning the map. */
  function wireArea(m: MapLibreMap): void {
    if (!area) return;
    m.boxZoom.disable();
    const at = (e: { lngLat: { lng: number; lat: number } }): LngLat => [
      e.lngLat.lng,
      e.lngLat.lat,
    ];
    m.on('mousedown', (e) => {
      if (lassoArmed) {
        m.dragPan.disable();
        area.begin('lasso', at(e));
      } else if (e.originalEvent.shiftKey) {
        e.preventDefault();
        m.dragPan.disable();
        area.begin('rectangle', at(e));
      }
    });
    m.on('mousemove', (e) => {
      if (area.drawing) area.extend(at(e));
    });
    m.on('mouseup', () => {
      if (!area.drawing) return;
      area.finish();
      lassoArmed = false;
      m.dragPan.enable();
    });
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
    wireArea(map);

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
        filter: ['!', ['to-boolean', ['get', 'image_id']]],
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
      // A cell with exactly one photo at the finest zoom is a pin, never a count.
      map.addLayer({
        id: 'pins',
        type: 'circle',
        source: 'cells',
        filter: ['to-boolean', ['get', 'image_id']],
        paint: {
          'circle-radius': 7,
          'circle-color': '#a78bfa',
          'circle-stroke-width': 2,
          'circle-stroke-color': '#ffffff',
        },
      });
      map.addLayer({
        id: 'cell-selected',
        type: 'circle',
        source: 'cells',
        filter: ['==', ['get', 'tile'], selectedTile ?? ''],
        paint: {
          'circle-radius': 34,
          'circle-color': 'rgba(0,0,0,0)',
          'circle-stroke-width': 3,
          'circle-stroke-color': '#ffffff',
        },
      });
      map.addSource('area', { type: 'geojson', data: areaShape() });
      map.addLayer({
        id: 'area-fill',
        type: 'fill',
        source: 'area',
        paint: { 'fill-color': '#38bdf8', 'fill-opacity': 0.15 },
      });
      map.addLayer({
        id: 'area-line',
        type: 'line',
        source: 'area',
        paint: { 'line-color': '#38bdf8', 'line-width': 2 },
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
    map.on('mouseenter', 'pins', () => {
      if (map) map.getCanvas().style.cursor = 'pointer';
    });
    map.on('mouseleave', 'pins', () => {
      if (map) map.getCanvas().style.cursor = '';
    });
    map.on('mouseleave', 'cells', () => {
      if (!map) return;
      map.getCanvas().style.cursor = '';
      popup.remove();
    });
    map.on('click', (e) => {
      if (!map || area?.drawing) return;
      const f = map.queryRenderedFeatures(e.point, { layers: ['pins', 'cells'] })[0];
      if (!f) return;
      const props = f.properties ?? {};
      if (props.image_id) onpin?.(String(props.image_id));
      else
        oncell({
          tile: String(props.tile),
          count: Number(props.count ?? 0),
          bounds: boundsOf(props.bounds),
        });
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

  $effect(() => {
    const tile = selectedTile ?? '';
    if (ready && map?.getLayer('cell-selected'))
      map.setFilter('cell-selected', ['==', ['get', 'tile'], tile]);
  });

  $effect(() => {
    void area?.points;
    pushArea();
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
{#if area && !noWebgl}
  <button
    type="button"
    class="btn-sm absolute right-2 top-24 z-10"
    aria-pressed={lassoArmed}
    data-testid="map-lasso"
    onclick={() => (lassoArmed = !lassoArmed)}>Lasso</button
  >
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
