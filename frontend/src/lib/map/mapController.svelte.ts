/**
 * `/map`: where the project's photos were taken. Thin: the server says whether a map exists
 * (`GET /map`), aggregates the photos into grid cells for the viewport (`GET /map/cells`, GeoJSON)
 * and lists the photos of a cell (`GET /map/images`). Nothing here counts, rounds or hides a
 * position; under the `drop` policy there is nothing to draw and the page says so.
 */
import { apiErrorText } from '$lib/api_errors';
import { getMapCells, getMapMeta, getMapTileImages } from '$lib/api_map';
import {
  ItemFilterState,
  withoutOpenVocab,
} from '$lib/itemFilter/itemFilterState.svelte';
import type { ItemFilterQuery } from '$lib/types_itemFilter';
import type {
  MapCellCollection,
  MapMeta,
  MapTileImages,
  MapViewQuery,
} from '$lib/types_map';

/** A cell that holds this many photos or fewer opens the list; a larger one zooms in. */
export const OPEN_CELL_MAX = 200;

export interface MapDeps {
  meta: typeof getMapMeta;
  cells: typeof getMapCells;
  images: typeof getMapTileImages;
}

const DEFAULT_DEPS: MapDeps = {
  meta: getMapMeta,
  cells: getMapCells,
  images: getMapTileImages,
};

export class MapController {
  meta = $state<MapMeta | null>(null);
  metaError = $state<string | null>(null);
  loadingMeta = $state(true);

  cells = $state<MapCellCollection | null>(null);
  cellsError = $state<string | null>(null);
  loadingCells = $state(false);

  /** The shared item filter, capture days included; the same names the lists take. */
  readonly filter = new ItemFilterState();

  /** The cell whose photos are listed in the side panel. */
  openTile = $state<MapTileImages | null>(null);
  openError = $state<string | null>(null);

  #deps: MapDeps;
  #cellsAbort: AbortController | null = null;
  #lastView: MapViewQuery | null = null;

  constructor(deps: MapDeps = DEFAULT_DEPS) {
    this.#deps = deps;
  }

  /** The filter as the query parameters every map route takes. */
  query(): ItemFilterQuery {
    return this.filter.toQuery(withoutOpenVocab);
  }

  async loadMeta(signal?: AbortSignal): Promise<void> {
    this.loadingMeta = true;
    this.metaError = null;
    try {
      this.meta = await this.#deps.meta(signal);
    } catch (e) {
      if (signal?.aborted) return;
      this.meta = null;
      this.metaError = apiErrorText(e);
    } finally {
      this.loadingMeta = false;
    }
  }

  /** Loads the cells of a viewport; a newer call replaces an older one still in flight. */
  async loadCells(view: MapViewQuery): Promise<void> {
    this.#lastView = view;
    this.#cellsAbort?.abort();
    const abort = new AbortController();
    this.#cellsAbort = abort;
    this.loadingCells = true;
    this.cellsError = null;
    try {
      const got = await this.#deps.cells(view, this.query(), abort.signal);
      if (!abort.signal.aborted) this.cells = got;
    } catch (e) {
      if (abort.signal.aborted) return;
      this.cellsError = apiErrorText(e);
    } finally {
      if (this.#cellsAbort === abort) this.loadingCells = false;
    }
  }

  /** A filter change reloads the last viewport. */
  refresh(): Promise<void> {
    this.openTile = null;
    return this.#lastView ? this.loadCells(this.#lastView) : Promise.resolve();
  }

  /** Whether a click on a cell lists its photos (true) or zooms in (false). */
  opensList(count: number, zoom: number): boolean {
    const deepest = this.meta ? zoom >= this.meta.max_zoom : false;
    return count <= OPEN_CELL_MAX || deepest;
  }

  async openCell(tile: string): Promise<void> {
    this.openError = null;
    try {
      this.openTile = await this.#deps.images(tile, this.query());
    } catch (e) {
      this.openTile = null;
      this.openError = apiErrorText(e);
    }
  }

  closeCell(): void {
    this.openTile = null;
    this.openError = null;
  }

  /** The list query of the open cell: the filter, narrowed to the cell's photos. */
  listQuery(): ItemFilterQuery {
    return { ...this.query(), image_id: this.openTile?.image_ids ?? [] };
  }

  /** The same, as the `ItemFilter` body the selection actions send. */
  listBody() {
    const body = this.filter.toBody();
    body.image_ids = this.openTile?.image_ids ?? [];
    return body;
  }

  dispose(): void {
    this.#cellsAbort?.abort();
  }
}
