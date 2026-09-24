/**
 * CurationSettingsStore — mock-backed tests, structure copied from
 * `stores/strategies.svelte.test.ts`. See docs/design/
 * curation-settings-ui-plan-2026-09-21.md §6.3.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { curationSettingsStore } from './curationSettings.svelte';
import { EMPTY_CURATION_SETTINGS } from '$lib/curationSettings';
import { API_PREFIX } from '$lib/api';

function jsonResponse(body: unknown, init: ResponseInit = {}): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'content-type': 'application/json' },
    ...init,
  });
}

beforeEach(() => {
  curationSettingsStore.reset();
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  curationSettingsStore.reset();
});

describe('curationSettingsStore', () => {
  it('registers zero window/document event listeners during init()', async () => {
    vi.stubGlobal(
      'fetch',
      vi
        .fn()
        .mockResolvedValue(
          jsonResponse({ defaults: {}, updated_at: null, updated_by: null }),
        ),
    );
    const windowSpy = vi.spyOn(window, 'addEventListener');
    const docSpy = vi.spyOn(document, 'addEventListener');

    await curationSettingsStore.init();

    expect(windowSpy).not.toHaveBeenCalled();
    expect(docSpy).not.toHaveBeenCalled();
  });

  it('starts at EMPTY_CURATION_SETTINGS with loaded=false, supported=null', () => {
    expect(curationSettingsStore.settings).toEqual(EMPTY_CURATION_SETTINGS);
    expect(curationSettingsStore.loaded).toBe(false);
    expect(curationSettingsStore.supported).toBeNull();
  });

  it('200 -> record loaded, supported true, error null', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        jsonResponse({
          defaults: { cluster: 'ivf' },
          updated_at: '2026-09-20T23:04:39+00:00',
          updated_by: null,
        }),
      ),
    );

    await curationSettingsStore.init();

    expect(curationSettingsStore.supported).toBe(true);
    expect(curationSettingsStore.error).toBeNull();
    expect(curationSettingsStore.settings.defaults).toEqual({ cluster: 'ivf' });
  });

  it('404 -> supported false, error null, no throw, settings stays empty', async () => {
    vi.stubGlobal(
      'fetch',
      vi
        .fn()
        .mockResolvedValue(
          new Response(JSON.stringify({ detail: 'not found' }), { status: 404 }),
        ),
    );

    await expect(curationSettingsStore.init()).resolves.toBeUndefined();

    expect(curationSettingsStore.supported).toBe(false);
    expect(curationSettingsStore.error).toBeNull();
    expect(curationSettingsStore.settings).toEqual(EMPTY_CURATION_SETTINGS);
  });

  it('422 on load -> supported null, error contains the server detail', async () => {
    vi.stubGlobal(
      'fetch',
      vi
        .fn()
        .mockResolvedValue(
          new Response(
            JSON.stringify({ detail: "axis 'bogus' does not accept a shared default" }),
            { status: 422 },
          ),
        ),
    );

    await curationSettingsStore.init();

    expect(curationSettingsStore.supported).toBeNull();
    expect(curationSettingsStore.error).toContain(
      "axis 'bogus' does not accept a shared default",
    );
  });

  it('init() is idempotent — three calls, one fetch', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        jsonResponse({ defaults: {}, updated_at: null, updated_by: null }),
      );
    vi.stubGlobal('fetch', fetchMock);

    await curationSettingsStore.init();
    await curationSettingsStore.init();
    await curationSettingsStore.init();

    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('refresh() re-fetches after a reset()', async () => {
    // A fresh Response per call — a Response body can only be read once,
    // and mockResolvedValue would hand back the same consumed instance
    // on the second call, causing a spurious retry storm.
    const fetchMock = vi
      .fn()
      .mockImplementation(() =>
        jsonResponse({ defaults: {}, updated_at: null, updated_by: null }),
      );
    vi.stubGlobal('fetch', fetchMock);

    await curationSettingsStore.init();
    await curationSettingsStore.refresh();

    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it('saveDefault sends exactly {"defaults":{"sort":"uncertainty_entropy"}} via PUT to ${API_PREFIX}/settings', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        defaults: { sort: 'uncertainty_entropy' },
        updated_at: '2026-09-21T00:00:00+00:00',
        updated_by: null,
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    await curationSettingsStore.saveDefault('sort', 'uncertainty_entropy');

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [calledUrl, calledInit] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(calledUrl).toContain(`${API_PREFIX}/settings`);
    expect(calledInit.method).toBe('PUT');
    expect(calledInit.body).toBe(
      JSON.stringify({ defaults: { sort: 'uncertainty_entropy' } }),
    );
  });

  it('adopts the response, not an optimistic guess', async () => {
    // The mock returns a record the client never sent (cluster + future_axis
    // alongside the sort key that was actually PUT) — an optimistic
    // `defaults[axis] = id` implementation would only ever show `sort`.
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        jsonResponse({
          defaults: { cluster: 'ivf', sort: 'uncertainty_entropy', future_axis: 'z' },
          updated_at: '2026-09-21T00:00:00+00:00',
          updated_by: null,
        }),
      ),
    );

    await curationSettingsStore.saveDefault('sort', 'uncertainty_entropy');

    expect(curationSettingsStore.settings.defaults).toEqual({
      cluster: 'ivf',
      sort: 'uncertainty_entropy',
      future_axis: 'z',
    });
  });

  it('422 on save -> rethrows, error contains the server detail, settings unchanged', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            detail:
              "'uncertainty_entropy' is not a currently-advertised id for axis 'sort'; valid ids: ['recent']",
          }),
          { status: 422 },
        ),
      ),
    );

    const before = curationSettingsStore.settings;

    await expect(
      curationSettingsStore.saveDefault('sort', 'uncertainty_entropy'),
    ).rejects.toThrow(/is not a currently-advertised id for axis/);

    expect(curationSettingsStore.error).toContain(
      'is not a currently-advertised id for axis',
    );
    expect(curationSettingsStore.settings).toEqual(before);
  });

  it('clearDefault sends exactly {"defaults":{"sort":null}} via PUT to ${API_PREFIX}/settings', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        defaults: {},
        updated_at: '2026-09-21T00:00:00+00:00',
        updated_by: null,
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    await curationSettingsStore.clearDefault('sort');

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [calledUrl, calledInit] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(calledUrl).toContain(`${API_PREFIX}/settings`);
    expect(calledInit.method).toBe('PUT');
    expect(calledInit.body).toBe(JSON.stringify({ defaults: { sort: null } }));
  });

  it('clearDefault adopts the response — a cleared axis disappears from defaults', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        jsonResponse({
          defaults: { cluster: 'ivf' },
          updated_at: '2026-09-21T00:00:00+00:00',
          updated_by: null,
        }),
      ),
    );

    await curationSettingsStore.clearDefault('sort');

    expect(curationSettingsStore.settings.defaults).toEqual({ cluster: 'ivf' });
  });

  it('clearDefault for an unknown axis throws "unknown settings axis"', async () => {
    await expect(curationSettingsStore.clearDefault('nonsense')).rejects.toThrow(
      'unknown settings axis: nonsense',
    );
  });

  // The server owns settable-ness: a non-settable axis comes back as its
  // 422, surfaced with the server's own detail.
  it('saveDefault surfaces the server 422 for a non-settable axis', async () => {
    vi.stubGlobal(
      'fetch',
      vi
        .fn()
        .mockResolvedValue(
          jsonResponse(
            { detail: "axis 'detection_profile' is not settable" },
            { status: 422 },
          ),
        ),
    );

    await expect(
      curationSettingsStore.saveDefault('detection_profile', 'license_plate'),
    ).rejects.toThrow(/not settable/);
    expect(curationSettingsStore.error).toMatch(/not settable/);
  });

  it('saveDefault for prompt_pack reaches the wire', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        defaults: { prompt_pack: 'generic_item_v1' },
        updated_at: '2026-09-23T00:00:00Z',
        updated_by: null,
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    await curationSettingsStore.saveDefault('prompt_pack', 'generic_item_v1');

    const [, init] = fetchMock.mock.calls[0]!;
    expect(init.method).toBe('PUT');
    expect(JSON.parse(init.body as string)).toEqual({
      defaults: { prompt_pack: 'generic_item_v1' },
    });
  });

  it('saveDefault for an unknown axis throws "unknown settings axis"', async () => {
    await expect(curationSettingsStore.saveDefault('nonsense', 'x')).rejects.toThrow(
      'unknown settings axis: nonsense',
    );
  });

  it('saving is the axis id during the call and null after, on both success and failure', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        jsonResponse({
          defaults: { sort: 'uncertainty_entropy' },
          updated_at: null,
          updated_by: null,
        }),
      ),
    );
    const p = curationSettingsStore.saveDefault('sort', 'uncertainty_entropy');
    expect(curationSettingsStore.saving).toBe('sort');
    await p;
    expect(curationSettingsStore.saving).toBeNull();

    vi.stubGlobal(
      'fetch',
      vi
        .fn()
        .mockResolvedValue(
          new Response(JSON.stringify({ detail: 'bad' }), { status: 422 }),
        ),
    );
    const failing = curationSettingsStore.saveDefault('sort', 'nope').catch(() => {});
    expect(curationSettingsStore.saving).toBe('sort');
    await failing;
    expect(curationSettingsStore.saving).toBeNull();
  });
});
