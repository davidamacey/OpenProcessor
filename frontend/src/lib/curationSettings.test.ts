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
  axisCopy,
  settableAxes,
  settingsOptionView,
  vlmNeedsAcknowledgement,
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
  it('covers exactly the axes the backend can accept, no duplicates', () => {
    const ids = SETTINGS_AXES.map((a) => a.axis);
    expect(new Set(ids).size).toBe(ids.length);
    expect(ids.sort()).toEqual(
      ['cluster', 'detection_profile', 'prompt_pack', 'sort', 'vlm'].sort(),
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
    detection_profiles: [entry('widget_tag', false)],
    prompt_packs: [entry('generic_item_v1', true)],
    vlm: [entry('endpoint_a', true)],
    axes: [],
  } as unknown as MethodsResponse;

  it('follows the server flag', () => {
    expect(settableAxes(methods).map((a) => a.axis)).toEqual([
      'cluster',
      'sort',
      'prompt_pack',
      'vlm',
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
      detection_profiles: [entry('widget_tag', true)],
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
      vlm: [],
      axes: [],
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
    vlm: [],
    axes: [],
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

describe('sort axis blurb (m10, 2026-09-24 interactive pass)', () => {
  it('does not claim the pinned default REPLACES a tab’s own tuned sort', () => {
    const sortSpec = axisSpec('sort')!;
    expect(sortSpec.blurb).not.toMatch(/REPLACES/i);
  });

  it('states the actual precedence: only tabs with no tuned default of their own use it', () => {
    const sortSpec = axisSpec('sort')!;
    expect(sortSpec.blurb).toMatch(/no tuned default of their own/i);
  });
});

describe('vlm axis (W9)', () => {
  const vlm = [
    {
      id: 'local_vlm',
      label: 'Local VLM',
      status: 'stable' as const,
      settable: true,
      default: true,
      endpoint_status_label: 'Ready',
      sends_images_externally: false,
      default_ack_recorded: null,
    },
    {
      id: 'cloud_vlm',
      label: 'Cloud VLM',
      status: 'stable' as const,
      settable: true,
      endpoint_status_label: 'Not probed yet',
      sends_images_externally: true,
      warning: 'Crops leave the deployment.',
      default_ack_recorded: false,
    },
    {
      id: 'cloud_ack',
      label: 'Cloud (acknowledged)',
      status: 'stable' as const,
      settable: true,
      sends_images_externally: true,
      warning: 'Crops leave the deployment.',
      default_ack_recorded: true,
    },
    { id: 'off', label: 'Off', status: 'stable' as const, settable: true },
    { id: 'gone', label: 'Gone', status: 'disabled' as const, settable: true },
  ];
  const methods = {
    cluster_methods: [],
    review_sorts: [],
    overlays: [],
    scores: [],
    dataset_exports: [],
    detection_profiles: [],
    prompt_packs: [],
    vlm,
    axes: [],
  } as unknown as MethodsResponse;
  const spec = axisSpec('vlm')!;

  it('has an axis spec that reads the vlm bucket and is settable by the served flag', () => {
    expect(spec.bucket).toBe('vlm');
    expect(settableAxes(methods).map((a) => a.axis)).toEqual(['vlm']);
  });

  it('offers every served entry that is not disabled, off included', () => {
    expect(axisOptions(methods, spec).map((o) => o.id)).toEqual([
      'local_vlm',
      'cloud_vlm',
      'cloud_ack',
      'off',
    ]);
  });

  it('the effective default is the pinned id, else the served default flag', () => {
    expect(effectiveDefaultId(EMPTY_CURATION_SETTINGS, methods, spec)).toBe('local_vlm');
    expect(
      effectiveDefaultId(
        { ...EMPTY_CURATION_SETTINGS, defaults: { vlm: 'off' } },
        methods,
        spec,
      ),
    ).toBe('off');
  });

  it('only an external entry with an unrecorded acknowledgement needs one', () => {
    const by = (id: string) => vlm.find((e) => e.id === id)!;
    expect(vlmNeedsAcknowledgement(by('cloud_vlm'))).toBe(true);
    // null (not external) and true (recorded) never block; neither does an absent flag.
    expect(vlmNeedsAcknowledgement(by('local_vlm'))).toBe(false);
    expect(vlmNeedsAcknowledgement(by('cloud_ack'))).toBe(false);
    expect(vlmNeedsAcknowledgement(by('off'))).toBe(false);
    // An external entry whose acknowledgement state is unknown (null) is the
    // server's call, not blocked here.
    expect(
      vlmNeedsAcknowledgement({ ...by('cloud_vlm'), default_ack_recorded: null }),
    ).toBe(false);
  });

  it('the option view carries the served status and warning; only the unacknowledged one is disabled', () => {
    const view = (id: string) =>
      settingsOptionView(
        spec,
        vlm.find((e) => e.id === id)!,
      );
    expect(view('local_vlm')).toEqual({
      suffix: ' · Ready',
      disabled: false,
      warning: null,
    });
    expect(view('cloud_vlm')).toEqual({
      suffix:
        ' · Not probed yet · warning: Crops leave the deployment. · activate it on Settings → Models first',
      disabled: true,
      warning: 'Crops leave the deployment.',
    });
    expect(view('cloud_ack').disabled).toBe(false);
    expect(view('off')).toEqual({ suffix: '', disabled: false, warning: null });
  });

  it('every other axis renders as before', () => {
    expect(
      settingsOptionView(axisSpec('sort')!, {
        id: 'recent',
        label: 'R',
        status: 'stable',
      }),
    ).toEqual({
      suffix: '',
      disabled: false,
      warning: null,
    });
  });

  it('the served axes[] copy replaces the spec words; absent, the spec words stay', () => {
    expect(axisCopy({ axes: [] }, spec)).toEqual({
      label: spec.label,
      blurb: spec.blurb,
    });
    expect(
      axisCopy(
        { axes: [{ axis: 'vlm', label: 'Served label', description: 'Served words.' }] },
        spec,
      ),
    ).toEqual({ label: 'Served label', blurb: 'Served words.' });
    expect(
      axisCopy({ axes: [{ axis: 'sort', label: 'x', description: 'y' }] }, spec).label,
    ).toBe(spec.label);
  });
});
