// Wire types of the project map. Pinned to the generated contract by
// `contract/mapContract.test.ts`.

export interface GeoBox {
  west: number;
  south: number;
  east: number;
  north: number;
}

/** `GET /map`: whether the page exists for this project, and what it draws with. */
export interface MapMeta {
  available: boolean;
  location_policy: 'drop' | 'store_coarse' | 'store';
  n_located: number;
  /** Images with no stored position, or one the location policy does not serve. */
  n_without_location: number;
  bounds: GeoBox | null;
  tile_url: string;
  tile_attribution: string;
  max_zoom: number;
}

export interface MapCellProperties {
  count: number;
  /** The cell as `z/x/y`. */
  tile: string;
  /** The cell as a degrees box, rounded outward under a coarse policy. */
  bounds: GeoBox;
  /** The photo, when the cell holds exactly one at the finest zoom the policy allows. */
  image_id: string | null;
  thumbnail_url: string | null;
}

/** One grid cell: a GeoJSON Point at the centroid of its images (RFC 7946). */
export interface MapCellFeature {
  type: 'Feature';
  geometry: { type: 'Point'; coordinates: [number, number] };
  properties: MapCellProperties;
}

/** `GET /map/cells` (`application/geo+json`). */
export interface MapCellCollection {
  type: 'FeatureCollection';
  features: MapCellFeature[];
  precision: number;
  truncated: boolean;
  join_truncated: boolean;
  /** Photos with a position that match the filter, across the whole map. */
  n_located_matching: number;
  /** Photos without a position that match the filter. */
  n_unlocated_matching: number;
}

/** The filter the map shares with the lists: the shared item filter plus the capture days. */
export interface MapViewQuery {
  zoom: number;
  bbox?: string;
}
