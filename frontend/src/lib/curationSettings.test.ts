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
import type { MethodsResponse } from './strategies';

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

  it('no axis has an irreversibleWarning now that the backend supports clearing', () => {
    // H-1 (plan §1.4) is closed — see putCurationDefaults' null-clear docstring.
    for (const spec of SETTINGS_AXES) {
      expect(spec.irreversibleWarning).toBeNull();
    }
  });
});

// Settable-ness is the server's per-entry `settable` flag, never this
// build's opinion — mirrors OpenProcessor main after (B): cluster, sort
// and prompt_pack settable; detection_profile display-only.
describe('settableAxes / advisoryAxes', () => {
  const entry = (id: string, settable?: boolean) => ({
    id,
    label: id,
    status: 'stable' as const,
    ...(settable === undefined ? {} : { settable }),
  });
  const methods = {
    cluster_methods: [entry('ivf', true)],
    review_sorts: [entry('recent', true)],
    detection_profiles: [entry('license_plate', false)],
    prompt_packs: [entry('generic_item_v1', true)],
  } as unknown as MethodsResponse;

  it('follows the server flag', () => {
    expect(settableAxes(methods).map((a) => a.axis)).toEqual([
      'cluster',
      'sort',
      'prompt_pack',
    ]);
    expect(advisoryAxes(methods).map((a) => a.axis)).toEqual(['detection_profile']);
  });

  it('treats an absent flag as not settable', () => {
    const old = {
      ...methods,
      prompt_packs: [entry('generic_item_v1')],
    } as MethodsResponse;
    expect(settableAxes(old).map((a) => a.axis)).not.toContain('prompt_pack');
    expect(advisoryAxes(old).map((a) => a.axis)).toContain('prompt_pack');
  });

  it('flips with the server, with no table edit', () => {
    const flipped = {
      ...methods,
      detection_profiles: [entry('license_plate', true)],
    } as MethodsResponse;
    expect(settableAxes(flipped).map((a) => a.axis)).toContain('detection_profile');
  });
});

describe('axisOptions', () => {
  it('drops shadow/disabled entries and appends no synthetic sentinel', () => {
    const methods: MethodsResponse = {
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
  const methods: MethodsResponse = {
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
    const bareMethods: MethodsResponse = { ...methods, review_sorts: [] };
    expect(effectiveDefaultId(EMPTY_CURATION_SETTINGS, bareMethods, spec)).toBeNull();
  });
});
