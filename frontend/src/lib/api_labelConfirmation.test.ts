/**
 * Label-confirmation wrappers (#119): method, project-scoped URL, query and
 * body through the real `apiFetch` (fetch stubbed).
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { API_PREFIX, setScopedPrefix } from '$lib/api';
import {
  getAuditQueue,
  getAuditReport,
  getVlmPolicy,
  putVlmPolicy,
  startAudit,
} from '$lib/api_labelConfirmation';
import { makeItem } from '$lib/test/makeItem';

const PROJECT = `${API_PREFIX}/projects/alpha`;

const json = (body: unknown, status = 200): Response =>
  new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });

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

describe('vlm policy', () => {
  it('reads GET /vlm/policy', async () => {
    setScopedPrefix(PROJECT);
    const last = capture({ scope: 'all', revision: 2 });
    expect(await getVlmPolicy()).toEqual({ scope: 'all', revision: 2 });
    expect(last()).toMatchObject({ url: `${PROJECT}/vlm/policy`, method: 'GET' });
  });

  it('PUTs the policy with the revision it read', async () => {
    setScopedPrefix(PROJECT);
    const last = capture({ scope: 'off', revision: 3 });
    await putVlmPolicy({ scope: 'off', max_crops_per_day: 0, expected_revision: 2 });
    expect(last()).toEqual({
      url: `${PROJECT}/vlm/policy`,
      method: 'PUT',
      body: { scope: 'off', max_crops_per_day: 0, expected_revision: 2 },
    });
  });
});

describe('audit', () => {
  it('starts with only the fields the operator set', async () => {
    setScopedPrefix(PROJECT);
    const last = capture({ batch_id: 'b', sampled: 0, requested: 0, strata: [] });
    await startAudit({});
    expect(last()).toEqual({
      url: `${PROJECT}/audit/start`,
      method: 'POST',
      body: {},
    });
  });

  it('sends sample_size and min_per_class when set', async () => {
    setScopedPrefix(PROJECT);
    const last = capture({ batch_id: 'b', sampled: 0, requested: 0, strata: [] });
    await startAudit({ sample_size: 120, min_per_class: 10 });
    expect(last().body).toEqual({ sample_size: 120, min_per_class: 10 });
  });

  it('reads the report, min_per_class only when given', async () => {
    setScopedPrefix(PROJECT);
    let last = capture({});
    await getAuditReport();
    expect(last().url).toBe(`${PROJECT}/audit/report`);
    last = capture({});
    await getAuditReport(25);
    expect(last().url).toBe(`${PROJECT}/audit/report?min_per_class=25`);
  });

  it('reads the queue page and maps each item like any crop', async () => {
    setScopedPrefix(PROJECT);
    const last = capture({
      items: [makeItem({ crop_id: 'q1', detector_class_name: 'widget_h' })],
      total: 41,
      page: 2,
      page_size: 30,
    });
    const page = await getAuditQueue(2, 30, 'batch-9');
    expect(last().url).toBe(
      `${PROJECT}/audit/queue?page=2&page_size=30&batch_id=batch-9`,
    );
    expect(page.total).toBe(41);
    expect(page.pageSize).toBe(30);
    expect(page.items[0]?.id).toBe('q1');
    expect(page.items[0]?.detector_class_name).toBe('widget_h');
  });
});
