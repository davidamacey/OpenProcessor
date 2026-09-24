/**
 * The license-plate slot reads and writes the backend's generic region
 * keys (OpenProcessor docs/design/curation_api_contract.md, "Item wire
 * format" — 31 region keys, pinned server-side by
 * tests/curation/test_wire_contract.py). Every wire field the slot
 * declares must be one of them, so a backend rename fails here instead of
 * silently rendering blanks.
 */
import { describe, expect, it } from 'vitest';
import { licensePlateSlot } from './profiles/licensePlate';

const REGION_WIRE_KEYS = new Set([
  'region_bbox_norm',
  'region_bbox_frame',
  'region_bbox_correct',
  'region_status',
  'region_score',
  'region_confidence',
  'region_reason',
  'region_rejection_reason',
  'region_text',
  'region_text_raw',
  'region_text_confidence',
  'region_text_source',
  'region_text_engine_version',
  'region_validated',
  'region_verified',
  'region_verified_at',
  'region_verifier',
  'region_verifier_version',
  'region_visible',
  'region_detector',
  'region_detector_version',
  'region_detector_chain',
  'region_detected_at',
  'region_cluster_id',
  'region_cluster_subid',
  'region_cluster_distance',
  'region_class_id',
  'region_label_source',
  'region_source',
  'region_pairing',
  'region_skip_verify',
]);

/** Every `*Field` string anywhere in the slot's capabilities. */
function declaredWireFields(v: unknown, out: string[] = []): string[] {
  if (Array.isArray(v)) v.forEach((x) => declaredWireFields(x, out));
  else if (v && typeof v === 'object') {
    for (const [k, x] of Object.entries(v)) {
      if (k.endsWith('Field') && typeof x === 'string') out.push(x);
      else declaredWireFields(x, out);
    }
  }
  return out;
}

describe('license-plate slot vs the backend region wire contract', () => {
  const fields = declaredWireFields(licensePlateSlot.capabilities);

  it('declares wire fields at all (guards a vacuous pass)', () => {
    expect(fields.length).toBeGreaterThan(10);
  });

  it('uses only documented region keys', () => {
    expect(fields.filter((f) => !REGION_WIRE_KEYS.has(f))).toEqual([]);
  });

  it('has exactly 31 known region keys (drift in the copied list shows here)', () => {
    expect(REGION_WIRE_KEYS.size).toBe(31);
  });
});
