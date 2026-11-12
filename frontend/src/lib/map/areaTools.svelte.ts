/**
 * Area selection on the map: a rectangle (Shift+drag) or a polygon lasso becomes the shared
 * item filter's `geo_bbox` or `geo_polygon`, so every Workspace view narrows to the area. The
 * tools only turn pointer positions into a filter value; panning and zooming never call them,
 * and one area is active at a time (a new rectangle replaces a lasso and the reverse).
 */
import type { GeoBox } from '$lib/types_map';

export type LngLat = [number, number];
export type AreaMode = 'rectangle' | 'lasso';

/** The part of the Workspace controller the tools write to. */
export interface AreaHost {
  filter: { geoBbox: GeoBox | null; geoPolygon: LngLat[] | null };
  applyFilter(): void;
}

/** A lasso needs three corners to enclose anything. */
export const MIN_LASSO_POINTS = 3;

export class AreaTools {
  mode = $state<AreaMode | null>(null);
  /** The points of the gesture in progress: two for a rectangle, many for a lasso. */
  points = $state<LngLat[]>([]);

  /** The lasso button was pressed: the next drag draws instead of panning. */
  armed = $state(false);

  #host: AreaHost;

  constructor(host: AreaHost) {
    this.#host = host;
  }

  get drawing(): boolean {
    return this.mode !== null;
  }

  begin(mode: AreaMode, at: LngLat): void {
    this.mode = mode;
    this.points = mode === 'rectangle' ? [at, at] : [at];
  }

  extend(to: LngLat): void {
    if (!this.mode) return;
    this.points = this.mode === 'rectangle' ? [this.points[0], to] : [...this.points, to];
  }

  /** Commits the gesture to the filter; a degenerate one (a click, two points) changes nothing. */
  finish(): void {
    const { mode, points } = this;
    this.cancel();
    if (mode === 'rectangle') this.#commitRectangle(points);
    else if (mode === 'lasso') this.#commitLasso(points);
  }

  toggleLasso(): void {
    this.armed = !this.armed;
  }

  cancel(): void {
    this.armed = false;
    this.mode = null;
    this.points = [];
  }

  #commitRectangle([a, b]: LngLat[]): void {
    if (a[0] === b[0] || a[1] === b[1]) return;
    this.#host.filter.geoPolygon = null;
    this.#host.filter.geoBbox = {
      west: Math.min(a[0], b[0]),
      south: Math.min(a[1], b[1]),
      east: Math.max(a[0], b[0]),
      north: Math.max(a[1], b[1]),
    };
    this.#host.applyFilter();
  }

  #commitLasso(points: LngLat[]): void {
    if (points.length < MIN_LASSO_POINTS) return;
    this.#host.filter.geoBbox = null;
    this.#host.filter.geoPolygon = points.map(([lon, lat]) => [lon, lat]);
    this.#host.applyFilter();
  }
}
