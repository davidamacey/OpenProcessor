import { describe, expect, it } from 'vitest';
import { datasetExportForSlot } from './datasetExport';
import { licensePlateSlot } from './profiles/licensePlate';
import { defectCodeSlot } from './profiles/defectCode';
import type { SlotSpec } from './types';

describe('datasetExportForSlot', () => {
  it("reads the license_plate profile's declared export", () => {
    const spec = datasetExportForSlot(licensePlateSlot);
    expect(spec).toEqual({
      kind: 'single_class',
      label: 'LPR plate dataset',
      buildPath: '/export/single_class',
      statusPath: '/export/single_class/status',
      datasetKind: 'license_plate',
      singleClass: true,
      blurb:
        'Single-class plate dataset (positives + human FP hard-negatives + a sample of plate-free backgrounds).',
      profileName: 'license_plate',
      boxSource: 'region',
      regionClassName: 'license_plate',
      classIds: [],
    });
  });

  it('returns undefined for a slot with no extras at all', () => {
    expect(datasetExportForSlot(defectCodeSlot)).toBeUndefined();
  });

  it('returns undefined for a slot with extras but no datasetExport key', () => {
    const slot: SlotSpec = { ...licensePlateSlot, extras: { somethingElse: true } };
    expect(datasetExportForSlot(slot)).toBeUndefined();
  });

  it('returns undefined when extras.datasetExport is not an object', () => {
    const slot: SlotSpec = { ...licensePlateSlot, extras: { datasetExport: 'nope' } };
    expect(datasetExportForSlot(slot)).toBeUndefined();
  });

  it('returns undefined when kind is missing or empty', () => {
    const base = licensePlateSlot.extras!.datasetExport as Record<string, unknown>;
    for (const kind of [undefined, '']) {
      const slot: SlotSpec = {
        ...licensePlateSlot,
        extras: { datasetExport: { ...base, kind } },
      };
      expect(datasetExportForSlot(slot)).toBeUndefined();
    }
  });

  it('returns undefined when label is missing', () => {
    const base = licensePlateSlot.extras!.datasetExport as Record<string, unknown>;
    const { label: _label, ...rest } = base;
    const slot: SlotSpec = { ...licensePlateSlot, extras: { datasetExport: rest } };
    expect(datasetExportForSlot(slot)).toBeUndefined();
  });

  it('returns undefined when blurb is missing', () => {
    const base = licensePlateSlot.extras!.datasetExport as Record<string, unknown>;
    const { blurb: _blurb, ...rest } = base;
    const slot: SlotSpec = { ...licensePlateSlot, extras: { datasetExport: rest } };
    expect(datasetExportForSlot(slot)).toBeUndefined();
  });

  // `/train` now renders `spec.blurb` directly (bakeoff-train-genericization
  // plan §3.2/§6 commit 2) — until that landed, this field was declared and
  // validated but rendered nowhere, so nothing would have noticed if this
  // check were silently dropped. Pinning both the missing and empty-string
  // cases keeps that from happening again now that a real consumer exists.
  it('returns undefined when blurb is an empty string', () => {
    const base = licensePlateSlot.extras!.datasetExport as Record<string, unknown>;
    const slot: SlotSpec = {
      ...licensePlateSlot,
      extras: { datasetExport: { ...base, blurb: '' } },
    };
    expect(datasetExportForSlot(slot)).toBeUndefined();
  });

  it('returns undefined when buildPath is absolute', () => {
    const base = licensePlateSlot.extras!.datasetExport as Record<string, unknown>;
    const slot: SlotSpec = {
      ...licensePlateSlot,
      extras: {
        datasetExport: { ...base, buildPath: 'https://evil.example/export/lpr' },
      },
    };
    expect(datasetExportForSlot(slot)).toBeUndefined();
  });

  it('returns undefined when statusPath has no leading slash', () => {
    const base = licensePlateSlot.extras!.datasetExport as Record<string, unknown>;
    const slot: SlotSpec = {
      ...licensePlateSlot,
      extras: { datasetExport: { ...base, statusPath: 'export/lpr/status' } },
    };
    expect(datasetExportForSlot(slot)).toBeUndefined();
  });

  it('defaults singleClass to false when absent or non-boolean', () => {
    const base = licensePlateSlot.extras!.datasetExport as Record<string, unknown>;
    for (const singleClass of [undefined, 'true', 1]) {
      const slot: SlotSpec = {
        ...licensePlateSlot,
        extras: { datasetExport: { ...base, singleClass } },
      };
      expect(datasetExportForSlot(slot)?.singleClass).toBe(false);
    }
  });

  it('returns undefined when profileName is missing', () => {
    const base = licensePlateSlot.extras!.datasetExport as Record<string, unknown>;
    const { profileName: _p, ...rest } = base;
    const slot: SlotSpec = { ...licensePlateSlot, extras: { datasetExport: rest } };
    expect(datasetExportForSlot(slot)).toBeUndefined();
  });

  it('returns undefined for an unknown boxSource', () => {
    const base = licensePlateSlot.extras!.datasetExport as Record<string, unknown>;
    for (const boxSource of [undefined, 'plate', 1]) {
      const slot: SlotSpec = {
        ...licensePlateSlot,
        extras: { datasetExport: { ...base, boxSource } },
      };
      expect(datasetExportForSlot(slot)).toBeUndefined();
    }
  });

  it('rejects non-integer or negative classIds', () => {
    const base = licensePlateSlot.extras!.datasetExport as Record<string, unknown>;
    for (const classIds of ['0', [1.5], [-1], [0, 'x']]) {
      const slot: SlotSpec = {
        ...licensePlateSlot,
        extras: { datasetExport: { ...base, classIds } },
      };
      expect(datasetExportForSlot(slot)).toBeUndefined();
    }
  });

  // Mirrors the backend's 422: an item-box export needs its vocabulary.
  it("rejects boxSource 'item' with no classIds, accepts it with some", () => {
    const base = licensePlateSlot.extras!.datasetExport as Record<string, unknown>;
    const withIds = (classIds: number[]): SlotSpec => ({
      ...licensePlateSlot,
      extras: { datasetExport: { ...base, boxSource: 'item', classIds } },
    });
    expect(datasetExportForSlot(withIds([]))).toBeUndefined();
    expect(datasetExportForSlot(withIds([3, 7]))?.classIds).toEqual([3, 7]);
  });
});
