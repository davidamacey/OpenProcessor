import { describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api_errors';
import type { ActivateResponse, ValidationReport } from '$lib/types_config';
import type { IngestProfileList } from '$lib/types_detector';
import {
  IngestProfilePicker,
  type IngestProfilePickerDeps,
} from './ingestProfilePickerController.svelte';

const LIST: IngestProfileList = {
  profiles: [
    {
      name: 'yolo-primary',
      source: 'stored',
      read_only: false,
      revision: 3,
      etag: 'a',
      primary_model: 'yolo11_small_trt',
      active: true,
      active_revision: 3,
    },
    {
      name: 'yolo-large',
      source: 'stored',
      read_only: false,
      revision: 2,
      etag: 'b',
      primary_model: 'yolo11_large_trt',
      active: false,
    },
  ],
  active: { name: 'yolo-primary', revision: 3 },
  config_revision: 9,
};

const OK_REPORT: ValidationReport = {
  ok: true,
  errors: [],
  warnings: [],
  force_allowed: false,
};

function activated(name: string, revision: number): ActivateResponse {
  return {
    axis: 'ingest_profile',
    active: { name, revision },
    source: 'stored',
    activated_at: null,
    previous: { name: 'yolo-primary', revision: 3 },
    config_revision: 10,
    stale: false,
    applied: [],
    validation: OK_REPORT,
  };
}

function refusal(code: string, message: string, forceAllowed: boolean): ApiError {
  const report: ValidationReport = {
    ok: false,
    errors: [
      {
        code,
        id: code,
        severity: 'error',
        field: 'primary_model',
        message,
        detail: {},
        bypassable: forceAllowed,
      },
    ],
    warnings: [],
    force_allowed: forceAllowed,
  };
  return new ApiError(
    422,
    '/x',
    { detail: { error: 'validation_failed', message: `refused: ${message}`, report } },
    undefined,
  );
}

function setup(over: Partial<IngestProfilePickerDeps> = {}) {
  const deps: IngestProfilePickerDeps = {
    list: vi.fn(async () => structuredClone(LIST)),
    activate: vi.fn(async (name: string) => activated(name, 2)),
    onactivated: vi.fn(async () => {}),
    ...over,
  };
  return { picker: new IngestProfilePicker(deps), deps };
}

describe('IngestProfilePicker', () => {
  it('loads the served list', async () => {
    const { picker } = setup();
    await picker.load();
    expect(picker.list?.profiles.map((p) => p.name)).toEqual([
      'yolo-primary',
      'yolo-large',
    ]);
    expect(picker.loadError).toBeNull();
  });

  it('shows a failed list read as its error, with no list', async () => {
    const { picker } = setup({
      list: vi.fn(async () => {
        throw new ApiError(404, '/x', { detail: 'Not Found' });
      }),
    });
    await picker.load();
    expect(picker.list).toBeNull();
    expect(picker.loadError).toContain('Not Found');
  });

  it('activates the pinned revision with expected_active, re-reads the list and notifies', async () => {
    const { picker, deps } = setup();
    await picker.load();
    const ok = await picker.activate('yolo-large');
    expect(ok).toBe(true);
    expect(deps.activate).toHaveBeenCalledWith('yolo-large', {
      revision: 2,
      expected_active: { name: 'yolo-primary', revision: 3 },
      force: false,
    });
    expect(deps.list).toHaveBeenCalledTimes(2);
    expect(deps.onactivated).toHaveBeenCalledTimes(1);
    expect(picker.refusal).toBeNull();
    expect(picker.lastValidation).toEqual(OK_REPORT);
  });

  it.each([
    ['detector_model_not_ready', 'model yolo11_large_trt is not ready in Triton'],
    ['detector_model_unusable', 'model yolo11_large_trt has an unusable output shape'],
    ['triton_unreachable', 'Triton is not reachable'],
  ])('renders the served %s refusal verbatim and does not notify', async (code, msg) => {
    const { picker, deps } = setup({
      activate: vi.fn(async () => {
        throw refusal(code, msg, false);
      }),
    });
    await picker.load();
    const ok = await picker.activate('yolo-large');
    expect(ok).toBe(false);
    expect(picker.refusal?.name).toBe('yolo-large');
    expect(picker.refusal?.message).toBe(`refused: ${msg}`);
    expect(picker.refusal?.report?.errors[0].code).toBe(code);
    expect(picker.refusal?.report?.errors[0].message).toBe(msg);
    expect(deps.onactivated).not.toHaveBeenCalled();
    expect(deps.list).toHaveBeenCalledTimes(1);
  });

  it('sends force only when the retry is asked for', async () => {
    const activate = vi
      .fn()
      .mockRejectedValueOnce(refusal('detector_model_not_ready', 'not ready', true))
      .mockResolvedValueOnce(activated('yolo-large', 2));
    const { picker } = setup({ activate });
    await picker.load();
    await picker.activate('yolo-large');
    expect(picker.refusal?.report?.force_allowed).toBe(true);
    expect(activate.mock.calls[0][1].force).toBe(false);
    await picker.activate('yolo-large', true);
    expect(activate.mock.calls[1][1].force).toBe(true);
    expect(picker.refusal).toBeNull();
  });

  it('shows a non-validation refusal as its served text and re-reads on active_conflict', async () => {
    const { picker, deps } = setup({
      activate: vi.fn(async () => {
        throw new ApiError(409, '/x', {
          detail: { error: 'active_conflict', message: 'the active profile changed' },
        });
      }),
    });
    await picker.load();
    await picker.activate('yolo-large');
    expect(picker.refusal?.message).toContain('the active profile changed');
    expect(picker.refusal?.report).toBeNull();
    expect(deps.list).toHaveBeenCalledTimes(2);
  });

  it('refuses a second activation while one is in flight', async () => {
    let release: (v: ActivateResponse) => void = () => {};
    const activate = vi.fn(() => new Promise<ActivateResponse>((r) => (release = r)));
    const { picker } = setup({ activate });
    await picker.load();
    const first = picker.activate('yolo-large');
    expect(picker.activating).toBe('yolo-large');
    expect(await picker.activate('yolo-primary')).toBe(false);
    release(activated('yolo-large', 2));
    await first;
    expect(activate).toHaveBeenCalledTimes(1);
    expect(picker.activating).toBeNull();
  });
});
