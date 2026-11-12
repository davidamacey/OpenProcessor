import { describe, expect, it } from 'vitest';
import { ItemFilterState } from './itemFilterState.svelte';

const BOX = { west: 7, south: 46, east: 9, north: 48 };
const RING: [number, number][] = [
  [7, 46],
  [9, 46],
  [8, 48],
];
const UID_A = '6f1c1d3e-0a7e-4c55-9d5e-2f0f8f2a9b11';
const UID_B = '0b1c1d3e-0a7e-4c55-9d5e-2f0f8f2a9b22';
const GROUP = '11111111-1111-1111-1111-111111111111';

function filled(): ItemFilterState {
  const s = new ItemFilterState();
  s.clusterUids = [UID_A, UID_B];
  s.geoBbox = BOX;
  s.geoPolygon = RING;
  s.hasLocation = true;
  s.nearDupGroup = GROUP;
  s.imageIds = ['img-a'];
  return s;
}

describe('ItemFilterState area, location and cluster keys', () => {
  it('round-trips geo_bbox, geo_polygon, has_location, cluster_uid', () => {
    const params = new URLSearchParams();
    filled().toUrl(params);
    expect(params.getAll('cluster_uid')).toEqual([UID_A, UID_B]);
    expect(params.get('geo_bbox')).toBe('7,46,9,48');
    expect(params.get('geo_polygon')).toBe('7,46;9,46;8,48');
    expect(params.get('has_location')).toBe('true');
    // The group and the images are set by the scope, not by a bar control or the URL keys.
    expect(params.has('near_dup_group')).toBe(false);
    expect(params.has('image_id')).toBe(false);

    const back = new ItemFilterState();
    back.fromUrl(params);
    expect(back.clusterUids).toEqual([UID_A, UID_B]);
    expect(back.geoBbox).toEqual(BOX);
    expect(back.geoPolygon).toEqual(RING);
    expect(back.hasLocation).toBe(true);
  });

  it('reads has_location=false and drops an unparsable box or ring', () => {
    const s = new ItemFilterState();
    s.fromUrl(new URLSearchParams('has_location=false&geo_bbox=1,2&geo_polygon=a,b;c,d'));
    expect(s.hasLocation).toBe(false);
    expect(s.geoBbox).toBeNull();
    expect(s.geoPolygon).toBeNull();
  });

  it('query and body forms agree', () => {
    const s = filled();
    expect(s.toQuery()).toMatchObject({
      cluster_uid: [UID_A, UID_B],
      geo_bbox: '7,46,9,48',
      geo_polygon: '7,46;9,46;8,48',
      has_location: true,
      near_dup_group: GROUP,
      image_id: ['img-a'],
    });
    expect(s.toBody()).toMatchObject({
      cluster_uids: [UID_A, UID_B],
      geo_bbox: BOX,
      geo_polygon: RING,
      has_location: true,
      near_dup_group: GROUP,
      image_ids: ['img-a'],
    });
  });

  it('is not empty with any of them, and clear() resets them', () => {
    for (const set of [
      (s: ItemFilterState) => (s.clusterUids = [UID_A]),
      (s: ItemFilterState) => (s.geoBbox = BOX),
      (s: ItemFilterState) => (s.geoPolygon = RING),
      (s: ItemFilterState) => (s.hasLocation = false),
      (s: ItemFilterState) => (s.nearDupGroup = GROUP),
      (s: ItemFilterState) => (s.imageIds = ['x']),
    ]) {
      const s = new ItemFilterState();
      set(s);
      expect(s.isEmpty).toBe(false);
    }
    const s = filled();
    s.clear();
    expect(s.isEmpty).toBe(true);
    expect(s.toQuery()).toEqual({});
  });

  it('has a removable chip for each', () => {
    const s = filled();
    const params = s.chips().map((c) => c.param);
    expect(params).toEqual(
      expect.arrayContaining([
        'cluster_uid',
        'geo_bbox',
        'geo_polygon',
        'has_location',
        'near_dup_group',
        'image_id',
      ]),
    );
    for (const chip of s.chips()) chip.clear();
    expect(s.isEmpty).toBe(true);
  });
});
