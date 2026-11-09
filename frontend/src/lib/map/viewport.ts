/**
 * The viewport as the server's `bbox` (`west,south,east,north`) and `zoom`. A view that spans the
 * world (or more, when the map wraps) sends no box; a longitude outside -180..180 is wrapped, so a
 * box across the antimeridian goes as west > east, which the server accepts.
 */
import type { MapViewQuery } from '$lib/types_map';

export interface ViewBounds {
  west: number;
  south: number;
  east: number;
  north: number;
}

const wrapLongitude = (lon: number): number => ((((lon + 180) % 360) + 360) % 360) - 180;
const clampLatitude = (lat: number): number => Math.max(-90, Math.min(90, lat));
const round = (v: number): number => Math.round(v * 1e6) / 1e6;

export function viewQuery(
  bounds: ViewBounds,
  zoom: number,
  maxZoom: number,
): MapViewQuery {
  const zoomLevel = Math.max(0, Math.min(Math.floor(zoom), maxZoom));
  if (bounds.east - bounds.west >= 360) return { zoom: zoomLevel };
  const south = clampLatitude(bounds.south);
  const north = clampLatitude(bounds.north);
  const west = wrapLongitude(bounds.west);
  const east = wrapLongitude(bounds.east);
  return {
    zoom: zoomLevel,
    bbox: [west, south, east, north].map(round).join(','),
  };
}
