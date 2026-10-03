/**
 * v0.4.0 detector wrappers: method, project-scoped URL, query and body
 * through the real `apiFetch` (fetch stubbed).
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { API_PREFIX, ApiError, setScopedPrefix } from '$lib/api';
import {
  detectorErrorLines,
  getDetectionsSummary,
  getIngestPolicy,
  previewIngestPolicy,
  putIngestPolicy,
  seedFromDetector,
} from '$lib/api_detector';

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

const PROJECT = `${API_PREFIX}/projects/alpha`;

afterEach(() => {
  vi.unstubAllGlobals();
  setScopedPrefix(API_PREFIX);
});

function capture(body: unknown = {}, status = 200) {
  const fetchMock = vi.fn().mockResolvedValue(json(body, status));
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

describe('seedFromDetector', () => {
  it('always sends dry_run and omits names / group unless set', async () => {
    setScopedPrefix(PROJECT);
    const last = capture();
    await seedFromDetector({ dry_run: true });
    expect(last()).toEqual({
      url: `${PROJECT}/classes/seed_from_detector`,
      method: 'POST',
      body: { dry_run: true },
    });
  });

  it('sends the chosen names and the group', async () => {
    setScopedPrefix(PROJECT);
    const last = capture();
    await seedFromDetector({ dry_run: false, names: ['widget'], group: 'g' });
    expect(last().body).toEqual({ dry_run: false, names: ['widget'], group: 'g' });
  });
});

describe('ingest policy', () => {
  it('GET reads the project policy', async () => {
    setScopedPrefix(PROJECT);
    const last = capture({ revision: 0 });
    await getIngestPolicy();
    expect(last()).toMatchObject({ url: `${PROJECT}/ingest/policy`, method: 'GET' });
  });

  it('PUT carries expected_revision with the draft', async () => {
    setScopedPrefix(PROJECT);
    const last = capture({ revision: 3 });
    await putIngestPolicy({ embedding: { mode: 'lazy' }, expected_revision: 2 });
    expect(last()).toEqual({
      url: `${PROJECT}/ingest/policy`,
      method: 'PUT',
      body: { embedding: { mode: 'lazy' }, expected_revision: 2 },
    });
  });

  it('preview POSTs the draft body only', async () => {
    setScopedPrefix(PROJECT);
    const last = capture({});
    await previewIngestPolicy({ embedding: { mode: 'all' } });
    expect(last()).toEqual({
      url: `${PROJECT}/ingest/policy/preview`,
      method: 'POST',
      body: { embedding: { mode: 'all' } },
    });
  });
});

describe('getDetectionsSummary', () => {
  it('sends the shared filter as repeated keys', async () => {
    setScopedPrefix(PROJECT);
    const last = capture({});
    await getDetectionsSummary({ class_name: ['a', 'b'], embedding_state: ['failed'] });
    expect(last().url).toBe(
      `${PROJECT}/detections/summary?class_name=a&class_name=b&embedding_state=failed`,
    );
  });

  it('sends no query without a filter', async () => {
    setScopedPrefix(PROJECT);
    const last = capture({});
    await getDetectionsSummary();
    expect(last().url).toBe(`${PROJECT}/detections/summary`);
  });
});

describe('detectorErrorLines', () => {
  it('shows the served message then each reason verbatim', () => {
    const e = new ApiError(422, 'u', {
      detail: {
        error: 'detector_not_servable',
        message: 'This detector cannot be served.',
        reasons: ['no labels file', 'input size must be 640'],
      },
    });
    expect(detectorErrorLines(e)).toEqual([
      'This detector cannot be served.',
      'no labels file',
      'input size must be 640',
    ]);
  });

  it('shows a reason that repeats the message once (the backend sends message == its only reason)', () => {
    const e = new ApiError(422, 'u', {
      detail: {
        error: 'detector_not_servable',
        message: 'det_v2 is not ready',
        reasons: ['det_v2 is not ready'],
      },
    });
    expect(detectorErrorLines(e)).toEqual(['det_v2 is not ready']);
  });

  it('names the unknown detector labels of a seed refusal', () => {
    const e = new ApiError(422, 'u', {
      detail: {
        error: 'unknown_detector_names',
        message: 'Some names are not detector labels.',
        unknown_names: ['gizmo'],
      },
    });
    expect(detectorErrorLines(e)).toEqual([
      'Some names are not detector labels.',
      'Unknown: gizmo',
    ]);
  });

  it('falls back to a plain string detail', () => {
    expect(detectorErrorLines(new ApiError(503, 'u', { detail: 'down' }))).toEqual([
      'down',
    ]);
  });
});
