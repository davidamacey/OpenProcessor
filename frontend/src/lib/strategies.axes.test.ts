import { describe, expect, it } from 'vitest';
import {
  EMPTY_METHODS,
  isPromptPackAvailable,
  isScopedAssistAvailable,
  isVlmSelectable,
  parseMethodsResponse,
  pickableVlmEntries,
  selectableAxisEntries,
} from '$lib/strategies';
import type { DetectionProfileInfo, PromptPackInfo } from '$lib/strategies';
import {
  METHODS_TODAY,
  METHODS_WITH_ASSIST_AXES,
} from '$lib/test/fixtures/methodsPayloads';

describe('selectableAxisEntries', () => {
  it('keeps a stable entry', () => {
    const entries: DetectionProfileInfo[] = [{ id: 'a', label: 'A', status: 'stable' }];
    expect(selectableAxisEntries(entries)).toEqual(entries);
  });

  it('keeps an experimental entry', () => {
    const entries: DetectionProfileInfo[] = [
      { id: 'a', label: 'A', status: 'experimental' },
    ];
    expect(selectableAxisEntries(entries)).toEqual(entries);
  });

  it('drops a shadow entry', () => {
    expect(
      selectableAxisEntries([
        { id: 'a', label: 'A', status: 'shadow' },
      ] as DetectionProfileInfo[]),
    ).toEqual([]);
  });

  it('drops a disabled entry', () => {
    expect(
      selectableAxisEntries([
        { id: 'a', label: 'A', status: 'disabled' },
      ] as DetectionProfileInfo[]),
    ).toEqual([]);
  });

  it('is empty in, empty out', () => {
    expect(selectableAxisEntries([])).toEqual([]);
  });

  it('preserves input order', () => {
    const entries: DetectionProfileInfo[] = [
      { id: 'b', label: 'B', status: 'experimental' },
      { id: 'a', label: 'A', status: 'stable' },
    ];
    expect(selectableAxisEntries(entries).map((e) => e.id)).toEqual(['b', 'a']);
  });
});

describe('isPromptPackAvailable', () => {
  it('is false for an empty list', () => {
    expect(isPromptPackAvailable([])).toBe(false);
  });

  it('is true with one stable entry', () => {
    expect(isPromptPackAvailable([{ id: 'a', label: 'A', status: 'stable' }])).toBe(true);
  });

  it('is true with one experimental entry', () => {
    expect(isPromptPackAvailable([{ id: 'a', label: 'A', status: 'experimental' }])).toBe(
      true,
    );
  });

  it('is false with only shadow entries', () => {
    expect(isPromptPackAvailable([{ id: 'a', label: 'A', status: 'shadow' }])).toBe(
      false,
    );
  });

  it('is false with only disabled entries', () => {
    expect(isPromptPackAvailable([{ id: 'a', label: 'A', status: 'disabled' }])).toBe(
      false,
    );
  });

  it('is true for a mixed list with at least one usable entry', () => {
    expect(
      isPromptPackAvailable([
        { id: 'a', label: 'A', status: 'disabled' },
        { id: 'b', label: 'B', status: 'experimental' },
      ]),
    ).toBe(true);
  });

  it('is false for EMPTY_METHODS.prompt_packs', () => {
    expect(isPromptPackAvailable(EMPTY_METHODS.prompt_packs)).toBe(false);
  });
});

/**
 * The gate that matters most: whether `AutoLabelPanel` renders
 * `<AssistScopeBar>` at all. Written against the two real fixtures
 * (§6 of the plan) rather than synthetic entries, because the
 * "must degrade to fully invisible against today's real backend" case
 * is the regression guard for the whole feature.
 */
describe('isScopedAssistAvailable', () => {
  it('is false for EMPTY_METHODS (the /methods-404 path)', () => {
    expect(isScopedAssistAvailable(EMPTY_METHODS)).toBe(false);
  });

  it("is false for today's real backend shape (METHODS_TODAY) — must degrade to fully invisible", () => {
    const parsed = parseMethodsResponse(METHODS_TODAY);
    expect(isScopedAssistAvailable(parsed)).toBe(false);
  });

  it('is true once the backend advertises the assist axes (METHODS_WITH_ASSIST_AXES)', () => {
    const parsed = parseMethodsResponse(METHODS_WITH_ASSIST_AXES);
    expect(isScopedAssistAvailable(parsed)).toBe(true);
  });

  // Detection profiles never gate the bar: a deployment advertising
  // profiles but no usable prompt pack gets no scope bar.
  it('is false when no prompt pack is usable', () => {
    const packs: PromptPackInfo[] = [{ id: 'legacy', label: 'Legacy', status: 'shadow' }];
    expect(isScopedAssistAvailable({ prompt_packs: packs, vlm: [] })).toBe(false);
    expect(isScopedAssistAvailable({ prompt_packs: [], vlm: [] })).toBe(false);
  });

  it('is true when a prompt pack is usable', () => {
    const packs: PromptPackInfo[] = [
      { id: 'warehouse_v1', label: 'Warehouse', status: 'stable' },
    ];
    expect(isScopedAssistAvailable({ prompt_packs: packs, vlm: [] })).toBe(true);
  });
});

describe('vlm axis (W9)', () => {
  const wire = {
    strategies: [
      {
        id: 'local_vlm',
        axis: 'vlm',
        label: 'Local VLM',
        status: 'stable',
        default: true,
        settable: true,
        endpoint_status: 'ready',
        endpoint_status_label: 'Ready',
        sends_images_externally: false,
        warning: null,
        default_ack_recorded: null,
        per_run_ack_required: false,
      },
      {
        id: 'cloud_vlm',
        axis: 'vlm',
        label: 'Cloud VLM',
        status: 'experimental',
        endpoint_status: 'unprobed',
        endpoint_status_label: 'Not probed yet',
        sends_images_externally: true,
        warning: 'Crops leave the deployment.',
        default_ack_recorded: false,
        per_run_ack_required: true,
      },
      { id: 'off', axis: 'vlm', label: 'Off', status: 'stable' },
      { id: 'broken', axis: 'vlm', label: 'Broken', status: 'disabled' },
    ],
    axes: [
      { axis: 'vlm', label: 'VLM endpoint', description: 'Which endpoint reads crops.' },
      { axis: '', label: 'dropped' },
      'garbage',
    ],
  };

  it('normalizes the served per-entry facts verbatim, with null kept distinct from absent', () => {
    const parsed = parseMethodsResponse(wire);
    expect(parsed.vlm.map((e) => e.id)).toEqual([
      'local_vlm',
      'cloud_vlm',
      'off',
      'broken',
    ]);
    const cloud = parsed.vlm.find((e) => e.id === 'cloud_vlm')!;
    expect(cloud).toMatchObject({
      endpoint_status: 'unprobed',
      endpoint_status_label: 'Not probed yet',
      sends_images_externally: true,
      warning: 'Crops leave the deployment.',
      default_ack_recorded: false,
      per_run_ack_required: true,
    });
    const local = parsed.vlm.find((e) => e.id === 'local_vlm')!;
    expect(local.default_ack_recorded).toBeNull();
    expect(local.per_run_ack_required).toBe(false);
    expect(parsed.vlm.find((e) => e.id === 'off')!.default_ack_recorded).toBeUndefined();
  });

  it('serves the axes[] copy and drops malformed entries', () => {
    expect(parseMethodsResponse(wire).axes).toEqual([
      { axis: 'vlm', label: 'VLM endpoint', description: 'Which endpoint reads crops.' },
    ]);
    expect(parseMethodsResponse({ strategies: [] }).axes).toEqual([]);
  });

  it('isVlmSelectable needs at least one entry that is not disabled', () => {
    const parsed = parseMethodsResponse(wire);
    expect(isVlmSelectable(parsed)).toBe(true);
    expect(
      isVlmSelectable({ vlm: parsed.vlm.filter((e) => e.status === 'disabled') }),
    ).toBe(false);
    expect(isVlmSelectable({ vlm: [] })).toBe(false);
    expect(pickableVlmEntries(parsed.vlm).map((e) => e.id)).toEqual([
      'local_vlm',
      'cloud_vlm',
      'off',
    ]);
  });

  it('the scoped-assist bar is available on the vlm axis alone', () => {
    const parsed = parseMethodsResponse(wire);
    expect(isScopedAssistAvailable({ prompt_packs: [], vlm: parsed.vlm })).toBe(true);
    expect(isScopedAssistAvailable({ prompt_packs: [], vlm: [] })).toBe(false);
  });
});
