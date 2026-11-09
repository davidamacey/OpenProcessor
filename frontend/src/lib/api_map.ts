/**
 * Map wrappers: whether the project has a map, its grid cells as GeoJSON and the images of one
 * cell. Import from `$lib/api_map` directly; types live in `$lib/types_map`. Every route takes
 * the shared item filter (`ItemFilterQuery`) plus `taken_from` / `taken_to`.
 */
import { apiFetch, qs, scoped } from '$lib/api';
import type { ItemFilterQuery } from '$lib/types_itemFilter';
import type {
  MapCellCollection,
  MapMeta,
  MapTileImages,
  MapViewQuery,
} from '$lib/types_map';

/** `GET /map`: `available` is false under the `drop` policy or without a stored position. */
export function getMapMeta(signal?: AbortSignal): Promise<MapMeta> {
  return apiFetch<MapMeta>(`${scoped()}/map`, {}, signal);
}

/** `GET /map/cells`: the matching images in grid cells at `zoom`, inside `bbox`
 *  (`west,south,east,north`). 400 on a malformed box. */
export function getMapCells(
  view: MapViewQuery,
  filter: ItemFilterQuery,
  signal?: AbortSignal,
): Promise<MapCellCollection> {
  return apiFetch<MapCellCollection>(
    `${scoped()}/map/cells${qs({ ...filter, zoom: view.zoom, bbox: view.bbox })}`,
    {},
    signal,
  );
}

/** `GET /map/images`: the images in the cell `tile` (`z/x/y`), at most 200. */
export function getMapTileImages(
  tile: string,
  filter: ItemFilterQuery,
  signal?: AbortSignal,
): Promise<MapTileImages> {
  return apiFetch<MapTileImages>(
    `${scoped()}/map/images${qs({ ...filter, tile })}`,
    {},
    signal,
  );
}
