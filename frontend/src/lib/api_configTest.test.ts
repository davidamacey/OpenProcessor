/**
 * W5 test-on-crop wrappers (`api_configTest.ts`): scoped URLs, bodies sent
 * as given, the preview item mapped through `mapRawCrop` (raw kept), and
 * a refusal's served detail surfacing. The request bodies' keys are
 * pinned to the contract in `contract/configTestContract.test.ts`.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { API_PREFIX, ApiError, configErrorDetail, configErrorText } from './api';
import { testPromptPack, testRegionProfile } from './api_configTest';
import {
  packTestResponseFixture,
  regionTestResponseFixture,
} from '$lib/test/fixtures/configTest';

afterEach(() => vi.unstubAllGlobals());

function capture(body: unknown, status = 200) {
  const fetchMock = vi.fn().mockResolvedValue(
    new Response(JSON.stringify(body), {
      status,
      headers: { 'content-type': 'application/json' },
    }),
  );
  vi.stubGlobal('fetch', fetchMock);
  return () => {
    const [url, init] = fetchMock.mock.calls[0]! as [string, RequestInit];
    return {
      url: String(url),
      method: init.method ?? 'GET',
      body: init.body ? JSON.parse(String(init.body)) : undefined,
    };
  };
}

describe('testPromptPack', () => {
  it('posts the body to the scoped route and maps each preview item', async () => {
    const sent = capture(packTestResponseFixture());
    const res = await testPromptPack({
      draft: { a: 'b' },
      call: 'classify',
      crop_ids: ['c_123'],
    });
    expect(sent()).toEqual({
      url: `${API_PREFIX}/prompt_packs/test`,
      method: 'POST',
      body: { draft: { a: 'b' }, call: 'classify', crop_ids: ['c_123'] },
    });
    expect(res.results[0]!.preview?.id).toBe('c_123');
    expect(res.results[0]!.preview?.proposed_class_name).toBe('widget');
    // The raw doc is kept for the JSON view.
    expect(res.results[0]!.preview_item).toMatchObject({ crop_id: 'c_123' });
    expect(res.vlm.endpoint).toBe('env@abc123');
  });

  it('leaves preview null for a result with no preview_item (skipped)', async () => {
    const body = packTestResponseFixture();
    body.results = [
      { crop_id: 'c_9', skipped: 'no box on this crop', preview_item: null },
    ];
    capture(body);
    const res = await testPromptPack({ call: 'region_verify', crop_ids: ['c_9'] });
    expect(res.results[0]!.preview).toBeNull();
    expect(res.results[0]!.skipped).toBe('no box on this crop');
  });

  it('carries no stale per-result parse keys', async () => {
    capture(packTestResponseFixture());
    const res = await testPromptPack({ call: 'classify', crop_ids: ['c_123'] });
    const keys = Object.keys(res.results[0]!);
    for (const stale of ['parse_ok', 'parse_error', 'parsed_class', 'parsed_combined']) {
      expect(keys).not.toContain(stale);
    }
  });
});

describe('testRegionProfile', () => {
  it('posts one crop to the scoped route and maps preview_item', async () => {
    const sent = capture(regionTestResponseFixture());
    const res = await testRegionProfile({
      crop_id: 'c_123',
      profile_name: 'widget_tag',
      profile_revision: 2,
    });
    expect(sent()).toEqual({
      url: `${API_PREFIX}/region_profiles/test`,
      method: 'POST',
      body: { crop_id: 'c_123', profile_name: 'widget_tag', profile_revision: 2 },
    });
    expect(res.preview.id).toBe('c_123');
    expect(res.preview_item).toMatchObject({ crop_id: 'c_123' });
    expect(res.legs).toHaveLength(2);
    expect(res.profile).toEqual({ name: 'widget_tag', revision: 2, draft: false });
  });

  it('surfaces the served detail of a crop_not_found refusal', async () => {
    capture(
      {
        detail: {
          error: 'crop_not_found',
          message: 'No crop with id c_x.',
          crop_ids: ['c_x'],
        },
      },
      404,
    );
    const err = await testRegionProfile({ crop_id: 'c_x' }).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(configErrorText(err)).toBe('No crop with id c_x.');
    expect(configErrorDetail(err)?.crop_ids).toEqual(['c_x']);
  });
});
