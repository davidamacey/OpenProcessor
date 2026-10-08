/**
 * Item-wire contract tests. Every fact here is read from
 * `contracts/openprocessor/json/item_wire.json` — vendored verbatim
 * from OpenProcessor's generated `contracts/json/item_wire.json`
 * (`npm run contract:sync`, `contracts/openprocessor/SOURCE.md`) — never
 * hand-copied. Supersedes `src/lib/annotations/regionWireContract.test.ts`,
 * which pinned its own 31-key literal (deleted alongside this file).
 *
 * A backend rename shows up here first: refresh the vendored snapshot
 * (`npm run contract:sync`), and whichever assertion below reads the
 * renamed key fails, naming the frontend site that still expects the
 * old name.
 */
import { describe, expect, it } from 'vitest';
import itemWire from '../../../contracts/openprocessor/json/item_wire.json';
import { RAW_CROP_KEYS } from '../api';
import openapi from '../../../contracts/openprocessor/openapi/curation.json';
import { mapRegionBoxWire } from '$lib/annotations/readSlot';
import { widgetTagServedSlot } from '$lib/test/fixtures/regionSlot';
import { regionContractSlots } from '$lib/test/fixtures/exampleProfiles';

const regionSlots = regionContractSlots();

const ITEM_KEYS = new Set<string>(itemWire.item_keys);
const REGION_KEYS = new Set<string>(itemWire.region_keys);
const ALL_WIRE_KEYS = new Set<string>([...ITEM_KEYS, ...REGION_KEYS]);

describe('vendored item-wire snapshot sanity', () => {
  it('loaded a non-trivial key set (guards a vacuous pass)', () => {
    expect(ITEM_KEYS.size).toBeGreaterThan(10);
    expect(REGION_KEYS.size).toBeGreaterThan(10);
  });
});

/**
 * `RawCrop`'s declared keys (src/lib/api.ts) vs the backend's item wire.
 *
 * `RAW_CROP_KEYS` is generated from `RawCrop` by a compile-time
 * exhaustiveness check in api.ts (`_RawCropKeysExhaustive`) — a field
 * added to `RawCrop` without a matching entry in `RAW_CROP_KEYS` fails
 * `npm run check`, so this array can't silently drift from the type it
 * documents.
 *
 * KNOWN_STALE: keys the frontend reads that the backend does not (yet,
 * or any longer) emit on this wire. Today this is empty — every
 * `RawCrop` key the audit found (2026-09-24) is a live item-wire key.
 * One previously-flagged stale field, `Class.added_at`, is NOT a
 * `RawCrop` field at all — it's a `GET {API_PREFIX}/classes` response field
 * (`RawClass`, a different endpoint's wire shape, not covered by
 * `item_wire.json`). If a genuinely stale `RawCrop` field turns up later, list it
 * here with a one-line reason instead of silently excluding it.
 */
const KNOWN_STALE: readonly string[] = [
  // Dropped from item_wire.json by OpenProcessor's OpenSearch perf/
  // correctness merge (main cbbca42, 2026-09-24) — out of scope for this
  // frontend pass. `mapRawCrop` already defaults it to null and every
  // reader treats it as optional, so this is inert, not a crash risk.
  'classifier_raw_confidence',
];

describe('RawCrop (src/lib/api.ts) vs the backend item wire', () => {
  it('every non-stale RawCrop key is a real item-wire key', () => {
    const unknown = RAW_CROP_KEYS.filter(
      (k) => !ALL_WIRE_KEYS.has(k) && !KNOWN_STALE.includes(k),
    );
    expect(unknown).toEqual([]);
  });

  it('KNOWN_STALE does not silently accumulate keys that are actually live', () => {
    const nowLive = KNOWN_STALE.filter((k) => ALL_WIRE_KEYS.has(k));
    expect(nowLive).toEqual([]);
  });
});

/**
 * Every registered slot's `*Field` wire references (bboxField,
 * valueField, statusField, ...) must be a real backend item-wire key.
 * Checked against `ALL_WIRE_KEYS`, not just `REGION_KEYS` — the vendored
 * contract's `region_keys` array covers the region *write* vocabulary,
 * but a capability field can also name a server-computed, read-only
 * item field (e.g. `listField: 'region_boxes'`, which is classified
 * under `item_keys` upstream rather than `region_keys`).
 * Replaces regionWireContract.test.ts's hand-copied `REGION_WIRE_KEYS`
 * literal.
 */
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

describe('region slots (served synthesis + region example profiles) vs the backend region wire', () => {
  for (const slot of regionSlots) {
    describe(slot.key, () => {
      const fields = declaredWireFields(slot.capabilities);

      it('declares wire fields at all (guards a vacuous pass)', () => {
        expect(fields.length).toBeGreaterThan(0);
      });

      it('uses only documented item-wire keys', () => {
        expect(fields.filter((f) => !ALL_WIRE_KEYS.has(f))).toEqual([]);
      });
    });
  }
});

/**
 * A region box is one element of `region_boxes`; its keys are not in
 * `item_keys` (the item wire only says `Record<string, unknown>[]`) but the
 * vendored OpenAPI documents the full element as `RegionTestCandidate`
 * (the test-on-crop route serves the very same per-box shape). Every key
 * `mapRegionBoxWire` reads must be one of them.
 */
describe('region box element keys (mapRegionBoxWire) vs the vendored box schema', () => {
  const boxKeys = new Set(
    Object.keys(
      (
        openapi as unknown as {
          components: {
            schemas: Record<string, { properties: Record<string, unknown> }>;
          };
        }
      ).components.schemas.RegionTestCandidate.properties,
    ),
  );

  it('the vendored box schema loaded (guards a vacuous pass)', () => {
    expect(boxKeys.has('box_id')).toBe(true);
    expect(boxKeys.has('bbox_norm')).toBe(true);
  });

  it('every wire key mapRegionBoxWire reads is a documented box key', () => {
    const read = new Set<string>();
    const probe = new Proxy(
      {},
      {
        get: (_t, k) => {
          if (typeof k === 'string') read.add(k);
          return undefined;
        },
      },
    );
    mapRegionBoxWire(probe);
    expect(read.size).toBeGreaterThan(20);
    expect([...read].filter((k) => !boxKeys.has(k))).toEqual([]);
  });

  it('the item summary fields the region slot declares are item-wire keys', () => {
    const sb = widgetTagServedSlot.capabilities.subBox!;
    for (const f of [
      sb.listField,
      sb.countField,
      sb.rejectedCountField,
      sb.maxScoreField,
      sb.setCompleteField,
      sb.revisionField,
    ]) {
      expect(ITEM_KEYS.has(f as string), `${f} is not an item-wire key`).toBe(true);
    }
  });
});
