/**
 * The Workspace Map view: where the project's photos were taken. Thin: the server says whether a
 * map exists (`GET /map`), aggregates the photos that match the Workspace filter and queue into
 * grid cells for the viewport (`GET /map/cells`, GeoJSON), and summarises a cell
 * (`POST /selection/summary`, `GET /images`). Nothing here counts, rounds or hides a position.
 * The filter is the Workspace's (`ws.itemQuery()`); a pan or zoom never changes it.
 */
import { apiErrorText } from '$lib/api_errors';
import { listImages } from '$lib/api_images';
import { getMapCells, getMapMeta } from '$lib/api_map';
import { getSelectionSummary } from '$lib/api_select';
import type {
  ItemFilter,
  ItemFilterQuery,
  SelectionSummary,
} from '$lib/types_itemFilter';
import type { ImagePage } from '$lib/types_images';
import type { GeoBox, MapCellCollection, MapMeta, MapViewQuery } from '$lib/types_map';
import type { WorkspaceController } from '$lib/workspace/workspaceController.svelte';

/** A cell's panel shows this many image cards; the summary counts all of its images. */
export const CELL_CARDS = 12;

export type MapHost = Pick<
  WorkspaceController,
  'itemQuery' | 'itemBody' | 'setFocus' | 'applyFilter'
> & { filter: Pick<WorkspaceController['filter'], 'geoBbox' | 'hasLocation'> };

export interface MapDeps {
  meta: typeof getMapMeta;
  cells: typeof getMapCells;
  summary: typeof getSelectionSummary;
  images: typeof listImages;
}

const DEFAULT_DEPS: MapDeps = {
  meta: getMapMeta,
  cells: getMapCells,
  summary: getSelectionSummary,
  images: listImages,
};

/** The cell that has focus, with the bounds the server gave it. */
export interface FocusedCell {
  tile: string;
  count: number;
  bounds: GeoBox;
}

export class MapController {
  meta = $state<MapMeta | null>(null);
  metaError = $state<string | null>(null);
  loadingMeta = $state(true);

  cells = $state<MapCellCollection | null>(null);
  cellsError = $state<string | null>(null);
  loadingCells = $state(false);

  cell = $state<FocusedCell | null>(null);
  summary = $state<SelectionSummary | null>(null);
  cards = $state<ImagePage | null>(null);
  /** The failure of the last cell read, as thrown (it carries the request id); null when fine. */
  cellFailure = $state<unknown>(null);

  #host: MapHost;
  #deps: MapDeps;
  #cellsAbort: AbortController | null = null;
  #cellAbort: AbortController | null = null;
  #lastView: MapViewQuery | null = null;

  constructor(host: MapHost, deps: MapDeps = DEFAULT_DEPS) {
    this.#host = host;
    this.#deps = deps;
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

  /** Loads the cells of a viewport for the Workspace filter; a newer call replaces an older one. */
  async loadCells(view: MapViewQuery): Promise<void> {
    this.#lastView = view;
    this.#cellsAbort?.abort();
    const abort = new AbortController();
    this.#cellsAbort = abort;
    this.loadingCells = true;
    this.cellsError = null;
    try {
      const got = await this.#deps.cells(view, this.#host.itemQuery(), abort.signal);
      if (!abort.signal.aborted) this.cells = got;
    } catch (e) {
      if (abort.signal.aborted) return;
      this.cellsError = apiErrorText(e);
    } finally {
      if (this.#cellsAbort === abort) this.loadingCells = false;
    }
  }

  /** The Workspace filter changed: reload the last viewport and the focused cell. */
  refresh(): Promise<void> {
    const cells = this.#lastView ? this.loadCells(this.#lastView) : Promise.resolve();
    return Promise.all([cells, this.cell ? this.#loadCell(this.cell) : undefined]).then(
      () => undefined,
    );
  }

  /** A cell click: focus it and ask once for its served summary and first cards. */
  focusCell(cell: FocusedCell): Promise<void> {
    if (this.cell?.tile === cell.tile) return Promise.resolve();
    this.#host.setFocus({ kind: 'cell', id: cell.tile });
    return this.#loadCell(cell);
  }

  /** Reads the focused cell's summary and cards again after a failure. */
  retryCell(): Promise<void> {
    return this.cell ? this.#loadCell(this.cell) : Promise.resolve();
  }

  /** A pin click: the single photo opens in the image details. */
  focusPin(imageId: string): void {
    this.clearLocal();
    this.#host.setFocus({ kind: 'image', id: imageId });
  }

  /** The URL left the cell: drop it without writing the URL again. */
  clearLocal(): void {
    this.#abortCell();
    this.cell = null;
  }

  clearCell(): void {
    this.clearLocal();
    this.#host.setFocus(null);
  }

  /** The Workspace filter narrowed to the focused cell's served bounds. */
  cellBody(cell: FocusedCell): ItemFilter {
    return { ...this.#host.itemBody(), geo_bbox: { ...cell.bounds } };
  }

  /** "Show in <view>": the cell's bounds become the area filter and the view switches. */
  showCell(cell: FocusedCell, view: Parameters<MapHost['applyFilter']>[0]): void {
    this.#host.filter.geoBbox = { ...cell.bounds };
    this.#host.applyFilter(view);
  }

  /** The "no location" link: narrow the Workspace to photos without a position. */
  showWithoutLocation(): void {
    this.#host.filter.hasLocation = false;
    this.#host.applyFilter();
  }

  #cellQuery(bounds: GeoBox): ItemFilterQuery {
    const { west, south, east, north } = bounds;
    return { ...this.#host.itemQuery(), geo_bbox: `${west},${south},${east},${north}` };
  }

  #abortCell(): void {
    this.#cellAbort?.abort();
    this.#cellAbort = null;
  }

  async #loadCell(cell: FocusedCell): Promise<void> {
    this.#abortCell();
    const abort = new AbortController();
    this.#cellAbort = abort;
    this.cell = cell;
    this.summary = null;
    this.cards = null;
    this.cellFailure = null;
    try {
      const [summary, cards] = await Promise.all([
        this.#deps.summary({ filter: this.cellBody(cell) }, abort.signal),
        this.#deps.images(
          { ...this.#cellQuery(cell.bounds), limit: CELL_CARDS },
          null,
          abort.signal,
        ),
      ]);
      if (abort.signal.aborted) return;
      this.summary = summary;
      this.cards = cards;
    } catch (e) {
      if (!abort.signal.aborted) this.cellFailure = e;
    }
  }

  dispose(): void {
    this.#cellsAbort?.abort();
    this.#abortCell();
  }
}
