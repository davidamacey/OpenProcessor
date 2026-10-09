import { afterEach, describe, expect, it, vi } from 'vitest';
import { API_PREFIX, setScopedPrefix } from '$lib/api';
import {
  getRegionStage,
  pauseRegionStage,
  resumeRegionStage,
} from '$lib/api_regionStage';

const PROJECT = `${API_PREFIX}/projects/alpha`;

afterEach(() => {
  vi.unstubAllGlobals();
  setScopedPrefix(API_PREFIX);
});

function capture() {
  const fetchMock = vi.fn().mockResolvedValue(
    new Response('{}', {
      status: 200,
      headers: { 'content-type': 'application/json' },
    }),
  );
  vi.stubGlobal('fetch', fetchMock);
  return () => {
    const [url, init] = fetchMock.mock.calls[0]! as [string, RequestInit];
    return { url: String(url), method: init.method ?? 'GET', body: init.body };
  };
}

describe('region stage wrappers', () => {
  it('reads and writes through the project prefix, writes with no body', async () => {
    setScopedPrefix(PROJECT);
    let call = capture();
    await getRegionStage();
    expect(call()).toEqual({
      url: `${PROJECT}/region_stage`,
      method: 'GET',
      body: undefined,
    });
    call = capture();
    await pauseRegionStage();
    expect(call()).toEqual({
      url: `${PROJECT}/region_stage/pause`,
      method: 'POST',
      body: undefined,
    });
    call = capture();
    await resumeRegionStage();
    expect(call()).toEqual({
      url: `${PROJECT}/region_stage/resume`,
      method: 'POST',
      body: undefined,
    });
  });
});
