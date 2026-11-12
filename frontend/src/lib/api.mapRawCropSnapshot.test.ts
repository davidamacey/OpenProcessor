/**
 * Whole-object characterisation of `mapRawCrop`: the complete mapped `Crop`
 * for a full item, a minimal item, an all-null item and a legacy item that
 * omits every optional key. Pins the mapping so a refactor cannot change a
 * field, a default or a coercion branch unnoticed.
 */
import { describe, expect, it } from 'vitest';
import { mapRawCrop, type RawCrop } from '$lib/api_cropMap';
import { RAW_CROP_KEYS } from '$lib/api_cropMap';
import { makeItem } from './test/makeItem';

const MINIMAL = {
  crop_id: 'c1',
  image_path: '/i.jpg',
  bbox_norm: [0.1, 0.2, 0.5, 0.9],
} as RawCrop;

const NULLS = Object.fromEntries(
  RAW_CROP_KEYS.filter((k) => !['crop_id', 'image_path', 'bbox_norm'].includes(k)).map(
    (k) => [k, null],
  ),
);

describe('mapRawCrop whole-object characterisation', () => {
  it('full item', () => {
    expect(mapRawCrop(makeItem())).toMatchSnapshot();
  });
  it('minimal item (legacy-absent fields)', () => {
    expect(mapRawCrop(MINIMAL)).toMatchSnapshot();
  });
  it('item with no bbox, empty image id and wrong-typed text fields', () => {
    const raw = {
      ...MINIMAL,
      bbox_norm: undefined,
      image_id: '',
      label_source: '',
      near_dup_group_id: 7,
      updated_at: 5,
      item_text_lines: 'x',
      import_ids: 'x',
      proposal_chain: [1, 'a'],
      geometry_locked: 0,
    } as unknown as RawCrop;
    expect(mapRawCrop(raw)).toMatchSnapshot();
  });
  it('every optional key null', () => {
    expect(mapRawCrop({ ...MINIMAL, ...NULLS } as unknown as RawCrop)).toMatchSnapshot();
  });
});
