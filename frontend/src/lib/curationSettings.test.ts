/**
 * Pure-module tests for the shared curation-defaults record: parsing,
 * the axis policy table, and the derived helpers. No fetch, no store —
 * see docs/design/curation-settings-ui-plan-2026-09-21.md §6.2.
 */

import { describe, expect, it } from 'vitest';
import {
  advisoryAxes,
  axisOptions,
  axisSpec,
  EMPTY_CURATION_SETTINGS,
  effectiveDefaultId,
  isPinned,
  parseCurationSettings,
  SETTINGS_AXES,
  settableAxes,
} from './curationSettings';
import type { OpMethodsResponse } from './strategies';

describe('parseCurationSettings', () => {
  it('parses the real documented body', () => {
    const raw = {
      defaults: { cluster: 'ivf' },
      updated_at: '2026-09-20T23:04:39+00:00',
      updated_by: null,
    };
    expect(parseCurationSettings(raw)).toEqual({
      defaults: { cluster: 'ivf' },
      updated_at: '2026-09-20T23:04:39+00:00',
      updated_by: null,
    });
  });

  it('treats the first-run body as normal, not an error', () => {
    const raw = { defaults: {}, updated_at: null, updated_by: null };
    expect(parseCurationSettings(raw)).toEqual(EMPTY_CURATION_SETTINGS);
  });

  it('never throws on garbage, yields empty defaults', () => {
    for (const raw of [null, [], 'x', { defaults: 5 }]) {
      expect(() => parseCurationSettings(raw)).not.toThrow();
      expect(parseCurationSettings(raw).defaults).toEqual({});
    }
  });

  it('drops non-string values per-entry, siblings survive', () => {
    const raw = { defaults: { cluster: 'ivf', sort: 7 } };
    expect(parseCurationSettings(raw).defaults).toEqual({ cluster: 'ivf' });
  });

  it('preserves an axis id this build has never heard of (open-map round trip)', () => {
    const raw = { defaults: { future_axis: 'x' } };
    expect(parseCurationSettings(raw).defaults).toEqual({ future_axis: 'x' });
  });
});

describe('SETTINGS_AXES', () => {
  it('covers exactly the backend four SETTABLE_DEFAULT_AXES, no duplicates', () => {
    const ids = SETTINGS_AXES.map((a) => a.axis);
    expect(new Set(ids).size).toBe(ids.length);
    expect(ids.sort()).toEqual(
      ['cluster', 'detection_profile', 'prompt_pack', 'sort'].sort(),
    );
  });

  it('settableAxes() ids are exactly cluster/sort — the honesty ratchet', () => {
    expect(settableAxes().map((a) => a.axis)).toEqual(['cluster', 'sort']);
  });

  it('advisoryAxes() ids are exactly detection_profile/prompt_pack, each blurb says Display only', () => {
    const advisory = advisoryAxes();
    expect(advisory.map((a) => a.axis)).toEqual(['detection_profile', 'prompt_pack']);
    for (const a of advisory) {
      expect(a.blurb).toContain('Display only');
    }
  });

  it('sort has a non-null irreversibleWarning (H-1); cluster does not', () => {
    expect(axisSpec('sort')?.irreversibleWarning).not.toBeNull();
    expect(axisSpec('cluster')?.irreversibleWarning).toBeNull();
  });
});

describe('axisOptions', () => {
  it('drops shadow/disabled entries and appends no synthetic sentinel', () => {
    const methods: OpMethodsResponse = {
      cluster_methods: [],
      review_sorts: [
        { id: 'recent', label: 'Recent first', status: 'stable' },
        { id: 'uncertainty_entropy', label: 'Uncertainty', status: 'experimental' },
        { id: 'hdbscan_probe', label: 'HDBSCAN probe', status: 'shadow' },
      ],
      overlays: [],
      scores: [],
      dataset_exports: [],
      detection_profiles: [],
      prompt_packs: [],
    };
    const spec = axisSpec('sort')!;
    const opts = axisOptions(methods, spec);
    expect(opts.map((o) => o.id)).toEqual(['recent', 'uncertainty_entropy']);
    // §1.3.4 regression guard: 'default' must never be synthesized here.
    expect(opts.some((o) => o.id === 'default')).toBe(false);
  });
});

describe('effectiveDefaultId / isPinned', () => {
  const spec = axisSpec('sort')!;
  const methods: OpMethodsResponse = {
    cluster_methods: [],
    review_sorts: [
      { id: 'recent', label: 'Recent first', status: 'stable', default: true },
      { id: 'uncertainty_entropy', label: 'Uncertainty', status: 'experimental' },
    ],
    overlays: [],
    scores: [],
    dataset_exports: [],
    detection_profiles: [],
    prompt_packs: [],
  };

  it('a stored value wins over the /methods default flag', () => {
    const settings = {
      ...EMPTY_CURATION_SETTINGS,
      defaults: { sort: 'uncertainty_entropy' },
    };
    expect(effectiveDefaultId(settings, methods, spec)).toBe('uncertainty_entropy');
    expect(isPinned(settings, spec)).toBe(true);
  });

  it('falls back to the /methods default flag when nothing stored', () => {
    expect(effectiveDefaultId(EMPTY_CURATION_SETTINGS, methods, spec)).toBe('recent');
    expect(isPinned(EMPTY_CURATION_SETTINGS, spec)).toBe(false);
  });

  it('falls back to null when neither exists', () => {
    const bareMethods: OpMethodsResponse = { ...methods, review_sorts: [] };
    expect(effectiveDefaultId(EMPTY_CURATION_SETTINGS, bareMethods, spec)).toBeNull();
  });
});
