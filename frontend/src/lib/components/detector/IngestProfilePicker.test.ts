import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import IngestProfilePicker from './IngestProfilePicker.svelte';
import { IngestProfilePicker as Ctl } from '$lib/detector/ingestProfilePickerController.svelte';
import { ApiError } from '$lib/api_errors';
import type { IngestProfileList } from '$lib/types_detector';

let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | undefined;

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
});

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
    },
    {
      name: 'yolo-large',
      source: 'stored',
      read_only: false,
      revision: 2,
      etag: 'b',
      primary_model: 'yolo11_large_trt',
      secondary_model: 'face_scrfd',
      active: false,
    },
  ],
  active: { name: 'yolo-primary', revision: 3 },
  config_revision: 9,
};

async function render(ctl: Ctl): Promise<HTMLDivElement> {
  await ctl.load();
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(IngestProfilePicker, { target, props: { ctl } });
  flushSync();
  return target;
}

const tick = () => new Promise((r) => setTimeout(r, 0));

function refusal(force: boolean): ApiError {
  return new ApiError(422, '/x', {
    detail: {
      error: 'validation_failed',
      message: 'cannot activate',
      report: {
        ok: false,
        force_allowed: force,
        warnings: [],
        errors: [
          {
            code: 'detector_model_not_ready',
            id: 'detector_model_not_ready',
            severity: 'error',
            field: 'primary_model',
            message: 'yolo11_large_trt is not loaded in Triton',
            detail: {},
            bypassable: force,
          },
        ],
      },
    },
  });
}

describe('IngestProfilePicker', () => {
  it('lists profiles, marks the active one and offers Activate on the others only', async () => {
    const ctl = new Ctl({
      list: async () => LIST,
      activate: vi.fn(),
      onactivated: async () => {},
    });
    const el = await render(ctl);
    const rows = el.querySelectorAll('[data-testid="ingest-profile-row"]');
    expect(rows).toHaveLength(2);
    expect(rows[0].textContent).toContain('yolo-primary');
    expect(rows[0].textContent).toContain('yolo11_small_trt');
    expect(rows[0].textContent).toContain('Active');
    expect(rows[0].querySelector('[data-testid="ingest-profile-activate"]')).toBeNull();
    expect(rows[1].textContent).toContain('yolo11_large_trt');
    expect(rows[1].textContent).toContain('face_scrfd');
    expect(
      rows[1].querySelector('[data-testid="ingest-profile-activate"]'),
    ).not.toBeNull();
  });

  it('is absent with a served empty list (nothing to choose)', async () => {
    const ctl = new Ctl({
      list: async () => ({ ...LIST, profiles: [] }),
      activate: vi.fn(),
      onactivated: async () => {},
    });
    const el = await render(ctl);
    expect(el.querySelector('[data-testid="ingest-profile-row"]')).toBeNull();
    expect(el.textContent).toContain('No ingest profiles');
  });

  it('shows the list read error', async () => {
    const ctl = new Ctl({
      list: async () => {
        throw new ApiError(404, '/x', { detail: 'Not Found' });
      },
      activate: vi.fn(),
      onactivated: async () => {},
    });
    const el = await render(ctl);
    expect(
      el.querySelector('[data-testid="ingest-profiles-error"]')?.textContent,
    ).toContain('Not Found');
  });

  it('renders a refusal with the served issue message and no override when not allowed', async () => {
    const ctl = new Ctl({
      list: async () => LIST,
      activate: vi.fn(async () => {
        throw refusal(false);
      }),
      onactivated: async () => {},
    });
    const el = await render(ctl);
    (
      el.querySelector('[data-testid="ingest-profile-activate"]') as HTMLButtonElement
    ).click();
    await tick();
    flushSync();
    const box = el.querySelector('[data-testid="ingest-profile-refusal"]');
    expect(box?.textContent).toContain('cannot activate');
    expect(box?.textContent).toContain('yolo11_large_trt is not loaded in Triton');
    expect(box?.textContent).toContain('detector_model_not_ready');
    expect(el.querySelector('[data-testid="ingest-profile-force"]')).toBeNull();
  });

  it('offers Activate anyway only when the served report allows it, and sends force', async () => {
    const activate = vi
      .fn()
      .mockRejectedValueOnce(refusal(true))
      .mockResolvedValueOnce({
        axis: 'ingest_profile',
        active: { name: 'yolo-large', revision: 2 },
        source: 'stored',
        activated_at: null,
        previous: null,
        config_revision: 10,
        stale: false,
        applied: [],
        validation: { ok: true, errors: [], warnings: [], force_allowed: false },
      });
    const ctl = new Ctl({
      list: async () => LIST,
      activate,
      onactivated: async () => {},
    });
    const el = await render(ctl);
    (
      el.querySelector('[data-testid="ingest-profile-activate"]') as HTMLButtonElement
    ).click();
    await tick();
    flushSync();
    const force = el.querySelector(
      '[data-testid="ingest-profile-force"]',
    ) as HTMLButtonElement;
    expect(force).not.toBeNull();
    force.click();
    await tick();
    flushSync();
    expect(activate.mock.calls[1][1].force).toBe(true);
    expect(el.querySelector('[data-testid="ingest-profile-refusal"]')).toBeNull();
  });
});
