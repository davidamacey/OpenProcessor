<script lang="ts">
  /**
   * The map: a MapLibre GL JS canvas (https://maplibre.org/maplibre-gl-js/docs/) with a raster
   * basemap from the served tile template and one GeoJSON source of served grid cells. The
   * browser fetches basemap tiles from `meta.tile_url`; cells and counts are the server's, a
   * click is reported to the host, and nothing here aggregates or rounds a position.
   */
  import { onMount } from 'svelte';
  import {
    Map as MapLibreMap,
    NavigationControl,
    Popup,
    type GeoJSONSource,
  } from 'maplibre-gl';
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
    const bounds = meta.bounds;
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

<div
  bind:this={container}
  class="h-full min-h-[320px] w-full rounded border border-zinc-800"
  data-testid="map-canvas"
  role="application"
  aria-label="Map of where the photos were taken"
></div>
