/**
 * The region slot synthesized from the served `/health.region_profile`
 * (docs/design/domain-neutral-audit-2026-09-24.md §5.3, §7.2).
 */
import { describe, expect, it } from 'vitest';
import itemWire from '../../../contracts/openprocessor/json/item_wire.json';
import {
  REGION_ENDPOINTS,
  REGION_TAB_ID,
  REGION_WIRE_CAPABILITIES,
  regionSlotFromServedProfile,
} from './servedRegionSlot';
import { datasetExportForSlot } from './datasetExport';
import { WIDGET_TAG_PROFILE } from '$lib/test/fixtures/regionSlot';

const WIRE_KEYS = new Set<string>([...itemWire.item_keys, ...itemWire.region_keys]);

function wireFields(v: unknown, out: string[] = []): string[] {
  if (Array.isArray(v)) v.forEach((x) => wireFields(x, out));
  else if (v && typeof v === 'object') {
    for (const [k, x] of Object.entries(v)) {
      if (k.endsWith('Field') && typeof x === 'string') out.push(x);
      else wireFields(x, out);
    }
  }
  return out;
}

describe('regionSlotFromServedProfile', () => {
  const slot = regionSlotFromServedProfile(WIDGET_TAG_PROFILE);

  it('uses only wire keys the backend documents (or the named pending-W8 key)', () => {
    // 'region_boxes' is W8's SubBoxCapability.listField — see
    // src/lib/contract/wireKeys.test.ts's PENDING_BACKEND_W8, which is
    // the tracked, emptied-at-lockstep-sync allow-list for this.
    const PENDING_BACKEND_W8 = ['region_boxes'];
    const fields = wireFields(slot.capabilities);
    expect(fields.length).toBeGreaterThan(20);
    expect(
      fields.filter((f) => !WIRE_KEYS.has(f) && !PENDING_BACKEND_W8.includes(f)),
    ).toEqual([]);
  });

  it('keys the slot by the profile name and binds it to the region class', () => {
    expect(slot.key).toBe(WIDGET_TAG_PROFILE.name);
    expect(slot.bind).toEqual({ className: WIDGET_TAG_PROFILE.region_class_name });
  });

  it('display_name is the tab label, the plural noun and the stats panel title', () => {
    const noun = WIDGET_TAG_PROFILE.display_name;
    expect(slot.capabilities.queue?.tabLabel).toBe(noun);
    expect(slot.label.plural).toBe(noun);
    // The singular-context title uses the served singular noun
    // ("Confirm Widget tag", #36 item 10) rather than a hardcoded "Region".
    expect(slot.label.title).toBe(WIDGET_TAG_PROFILE.display_name_singular);
    expect(slot.label.singular).toBe(
      WIDGET_TAG_PROFILE.display_name_singular.toLowerCase(),
    );
    expect(slot.stats?.panelTitle).toBe(noun);
  });

  it('falls back to the generic singular title when display_name_singular is empty', () => {
    const s = regionSlotFromServedProfile({
      ...WIDGET_TAG_PROFILE,
      display_name_singular: '  ',
    });
    expect(s.label.title).toBe('Region');
    expect(s.label.singular).toBe('region');
  });

  it('serves its review queue under the backend region tab id and browse route', () => {
    expect(slot.capabilities.queue).toMatchObject({
      endpointId: REGION_TAB_ID,
      urlId: REGION_TAB_ID,
      browsePath: '/regions',
    });
    expect(REGION_TAB_ID).toBe('regions');
  });

  it('writes through the region routes (W8: no setBox/clearBox — PUT /crops/{id}/region is a removed route)', () => {
    expect(slot.endpoints).toBe(REGION_ENDPOINTS);
    expect(slot.endpoints.setBox).toBeUndefined();
    expect(slot.endpoints.clearBox).toBeUndefined();
    expect(slot.endpoints.patchMeta!('a')).toBe('/crops/a/region_meta');
    expect(slot.endpoints.batchStatus!()).toBe('/regions/batch_status');
    expect(slot.capabilities.subBox).toBe(REGION_WIRE_CAPABILITIES.subBox);
  });

  it('declares no client-side training cohorts (the backend serves them)', () => {
    expect(slot.capabilities.trainingCohorts).toBeUndefined();
  });

  it('declares the single-class export under the profile name, so past exports stay where they are', () => {
    const spec = datasetExportForSlot(slot);
    expect(spec).toMatchObject({
      kind: 'single_class',
      profileName: WIDGET_TAG_PROFILE.name,
      datasetKind: WIDGET_TAG_PROFILE.name,
      regionClassName: WIDGET_TAG_PROFILE.region_class_name,
      boxSource: 'region',
      singleClass: true,
    });
    expect(spec!.label).toContain(WIDGET_TAG_PROFILE.display_name);
  });

  it('has a text capability and text filter when the profile reads text', () => {
    expect(slot.capabilities.text?.valueField).toBe('region_text');
    expect(slot.capabilities.queue?.textFilter?.param).toBe('text');
  });

  it('has no text capability or text filter when text_reader is empty', () => {
    const s = regionSlotFromServedProfile({ ...WIDGET_TAG_PROFILE, text_reader: '' });
    expect(s.capabilities.text).toBeUndefined();
    expect(s.capabilities.queue?.textFilter).toBeUndefined();
  });

  it('has no text capability when text_reader is the text-free sentinel "none" (OpenProcessor W1)', () => {
    // No `reads_text` served (pre-W1-shaped call site) — falls back to
    // the text_reader-based heuristic, which must treat 'none' as
    // "doesn't read text", not as a truthy non-empty reader id.
    const s = regionSlotFromServedProfile({ ...WIDGET_TAG_PROFILE, text_reader: 'none' });
    expect(s.capabilities.text).toBeUndefined();
    expect(s.capabilities.queue?.textFilter).toBeUndefined();
  });

  it('gates on the served reads_text flag over text_reader when both are present', () => {
    // A profile could in principle serve reads_text: false alongside a
    // non-empty/non-'none' text_reader (e.g. mid-migration data); the
    // served boolean is authoritative.
    const off = regionSlotFromServedProfile({
      ...WIDGET_TAG_PROFILE,
      text_reader: 'ocr',
      reads_text: false,
    });
    expect(off.capabilities.text).toBeUndefined();

    const on = regionSlotFromServedProfile({
      ...WIDGET_TAG_PROFILE,
      text_reader: 'none',
      reads_text: true,
    });
    expect(on.capabilities.text).toBeDefined();
  });

  it('a legacy backend with no reads_text field still gets text from text_reader', () => {
    // WIDGET_TAG_PROFILE itself carries no reads_text/text_hint_enabled
    // — the pre-W1 shape.
    expect(WIDGET_TAG_PROFILE.reads_text).toBeUndefined();
    expect(slot.capabilities.text?.valueField).toBe('region_text');
  });

  it('falls back to a generic noun when display_name is empty', () => {
    const s = regionSlotFromServedProfile({ ...WIDGET_TAG_PROFILE, display_name: '  ' });
    expect(s.capabilities.queue?.tabLabel).toBe('Regions');
    expect(s.label.plural).toBe('Regions');
  });

  it('binds to the profile name when region_class_name is empty', () => {
    const s = regionSlotFromServedProfile({
      ...WIDGET_TAG_PROFILE,
      region_class_name: '',
    });
    expect(s.bind.className).toBe(WIDGET_TAG_PROFILE.name);
  });
});
