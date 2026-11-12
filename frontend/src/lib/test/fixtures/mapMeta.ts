import type { MapMeta } from '$lib/types_map';

/** A served `GET /map` answer for a project with three located photos. */
export function mapMeta(over: Partial<MapMeta> = {}): MapMeta {
  return {
    available: true,
    location_policy: 'store',
    n_located: 3,
    n_without_location: 2,
    bounds: null,
    tile_url: '/t/{z}/{x}/{y}.png',
    tile_attribution: '',
    max_zoom: 12,
    ...over,
  };
}
