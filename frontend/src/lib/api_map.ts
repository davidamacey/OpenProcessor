/**
 * Map wrappers: whether the project has a map, and its grid cells as GeoJSON. Import from
 * `$lib/api_map` directly; types live in `$lib/types_map`. Both routes take the shared item
 * filter (`ItemFilterQuery`).
 */
import { apiFetch, qs, scoped } from '$lib/api';
import type { ItemFilterQuery } from '$lib/types_itemFilter';
import type { MapCellCollection, MapMeta, MapViewQuery } from '$lib/types_map';

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
