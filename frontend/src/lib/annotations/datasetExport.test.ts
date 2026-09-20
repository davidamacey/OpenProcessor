import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { datasetExportForSlot } from './datasetExport';
import { licensePlateSlot } from './profiles/licensePlate';
import { defectCodeSlot } from './profiles/defectCode';
import type { SlotSpec } from './types';

const here = path.dirname(fileURLToPath(import.meta.url));

describe('datasetExportForSlot', () => {
  it("reads the license_plate profile's declared export", () => {
    const spec = datasetExportForSlot(licensePlateSlot);
    expect(spec).toEqual({
      kind: 'lpr',
      label: 'LPR plate dataset',
      buildPath: '/export/lpr',
      statusPath: '/export/lpr/status',
      datasetKind: 'lpr',
      singleClass: true,
      blurb:
        'Single-class plate dataset (positives + human FP hard-negatives + a sample of plate-free backgrounds).',
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

  // The drift ratchet (Phase C plan §3.6h): api.ts's exportLpr/
  // exportLprStatus keep their own hardcoded '/export/lpr' /
  // '/export/lpr/status' literals rather than threading spec.buildPath
  // through them (a deliberate deferral — see the plan). This test pins
  // the two independently-declared copies together so they cannot
  // silently diverge.
  it("the profile's declared paths match api.ts's exportLpr/exportLprStatus literals", () => {
    const spec = datasetExportForSlot(licensePlateSlot)!;
    const apiSrc = readFileSync(path.resolve(here, '../api.ts'), 'utf-8');
    expect(apiSrc).toContain(`\${API_PREFIX}${spec.buildPath}\``);
    expect(apiSrc).toContain(`\${API_PREFIX}${spec.statusPath}\``);
  });
});
