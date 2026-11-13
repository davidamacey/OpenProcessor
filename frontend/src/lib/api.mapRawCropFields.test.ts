/**
 * The fields of the mapped `Crop` that the Workspace reads, spelled out for a full item, an item
 * with only the required wire keys, an all-null item and one with wrong-typed values. A change to
 * a default or a coercion shows as a failing field here, with its expected value in the diff.
 * The set of wire keys is guarded by `contract/wireKeys.test.ts`.
 */
import { describe, expect, it } from 'vitest';
import { mapRawCrop, RAW_CROP_KEYS, type RawCrop } from '$lib/api_cropMap';
import type { Crop } from '$lib/types';
import { makeItem } from './test/makeItem';

const REQUIRED = ['crop_id', 'image_path', 'bbox_norm'];
const MINIMAL = {
  crop_id: 'c1',
  image_path: '/i.jpg',
  bbox_norm: [0.1, 0.2, 0.5, 0.9],
} as RawCrop;
const NULLS = Object.fromEntries(
  RAW_CROP_KEYS.filter((k) => !REQUIRED.includes(k)).map((k) => [k, null]),
);

/** The identity, box, class, validation, cluster, geometry, duplicate and proposal fields. */
function read(c: Crop) {
  return {
    id: c.id,
    image_id: c.image_id,
    bbox_norm: c.bbox_norm,
    class_id: c.class_id,
    class_name: c.class_name,
    class_validated: c.class_validated,
    label_validated: c.label_validated,
    cluster: c.cluster,
    geometry_locked: c.geometry_locked,
    geometry_source: c.geometry_source,
    geometry_revision: c.geometry_revision,
    near_dup_group_id: c.near_dup_group_id,
    proposed_class_id: c.proposed_class_id,
    proposed_class_name: c.proposed_class_name,
    detector_class_id: c.detector_class_id,
    detector_class_name: c.detector_class_name,
  };
}

const EMPTY = {
  id: 'c1',
  image_id: undefined,
  bbox_norm: { cx: 0.3, cy: 0.55, w: 0.4, h: 0.7 },
  class_id: null,
  class_name: null,
  class_validated: false,
  label_validated: false,
  cluster: null,
  geometry_locked: false,
  geometry_source: 'detector',
  geometry_revision: 0,
  near_dup_group_id: null,
  proposed_class_id: null,
  proposed_class_name: null,
  detector_class_id: null,
  detector_class_name: null,
};

describe('mapRawCrop, the fields the Workspace reads', () => {
  it('full item', () => {
    const got = read(mapRawCrop(makeItem()));
    expect(got).toEqual({
      id: 'crop-fixture-001',
      image_id: 'image-fixture-777',
      bbox_norm: { cx: 0.35, cy: 0.5, w: 0.5, h: expect.closeTo(0.6, 10) },
      class_id: 42,
      class_name: 'widget_a',
      class_validated: true,
      label_validated: true,
      cluster: {
        assigned_by: 'run',
        display_no: 17,
        margin: 0.12,
        partition_id: '5e1c0000-0000-4000-8000-000000000001',
        score: 0.81,
        state: 'assigned',
        uid: '0b9f9a38-0000-4000-8000-000000000017',
      },
      geometry_locked: true,
      geometry_source: 'human',
      geometry_revision: 3,
      near_dup_group_id: 'grp-1',
      proposed_class_id: 101,
      proposed_class_name: 'widget_b',
      detector_class_id: 8,
      detector_class_name: 'widget_h',
    });
  });

  it('an item with only the required wire keys has the defaults', () => {
    expect(read(mapRawCrop(MINIMAL))).toEqual(EMPTY);
  });

  it('every optional key null gives the same defaults', () => {
    expect(read(mapRawCrop({ ...MINIMAL, ...NULLS } as unknown as RawCrop))).toEqual(
      EMPTY,
    );
  });

  it('a missing box is a zero box; a wrong-typed duplicate group and lock are coerced', () => {
    const raw = {
      ...MINIMAL,
      bbox_norm: undefined,
      image_id: '',
      near_dup_group_id: 7,
      geometry_locked: 0,
    } as unknown as RawCrop;
    expect(read(mapRawCrop(raw))).toEqual({
      ...EMPTY,
      bbox_norm: { cx: 0, cy: 0, w: 0, h: 0 },
      near_dup_group_id: null,
      geometry_locked: false,
    });
  });
});
