/**
 * OpenProcessor d72cc63 made several request bodies `extra='forbid'`
 * (`additionalProperties: false` in the vendored OpenAPI): an unknown key
 * is a 422. Drive each wrapper with every option it accepts and check the
 * JSON it actually sends only uses keys the served schema declares.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import openapi from '../../../contracts/openprocessor/openapi/curation.json';
import { exportYolo, freezeTestHoldout, ingestBatch } from '$lib/api';

type Schema = { properties?: Record<string, unknown>; additionalProperties?: boolean };
const schemas = (
  openapi as unknown as { components: { schemas: Record<string, Schema> } }
).components.schemas;

function declared(name: string): string[] {
  const s = schemas[name];
  if (!s?.properties) throw new Error(`schema not found: ${name}`);
  expect(s.additionalProperties).toBe(false);
  return Object.keys(s.properties);
}

function captureBody(): { calls: () => Record<string, unknown>[] } {
  const payload = { status: 'success', summary: {}, results: [], disagreements: [] };
  const headers = { 'content-type': 'application/json' };
  const respond = async () =>
    new Response(JSON.stringify(payload), { status: 200, headers });
  const fetchMock = vi.fn().mockImplementation(respond);
  vi.stubGlobal('fetch', fetchMock);
  return {
    calls: () =>
      fetchMock.mock.calls.map((c) => JSON.parse(String((c[1] as RequestInit).body))),
  };
}

afterEach(() => vi.unstubAllGlobals());

describe('strict request bodies send only declared keys', () => {
  it('POST /export/yolo (ExportYoloRequest)', async () => {
    const cap = captureBody();
    await exportYolo({ version_tag: 'v1', require_fully_labeled_images: true });
    const body = cap.calls()[0]!;
    const allowed = declared('ExportYoloRequest');
    for (const k of Object.keys(body)) expect(allowed).toContain(k);
    expect(body).not.toHaveProperty('classes');
  });

  it('POST /test_holdout/freeze (TestHoldoutFreezeRequest)', async () => {
    const cap = captureBody();
    await freezeTestHoldout({ percent: 10, force: true });
    const allowed = declared('TestHoldoutFreezeRequest');
    for (const k of Object.keys(cap.calls()[0]!)) expect(allowed).toContain(k);
  });

  it('POST /ingest/batch (IngestBatchRequest + IngestBatchItem)', async () => {
    const cap = captureBody();
    await ingestBatch({
      items: [{ path: '/data/source/a.jpg', source: 'batch', label_txt_path: null }],
      label_source: 'human',
      detect_mismatches: false,
    });
    const body = cap.calls()[0]!;
    const allowed = declared('IngestBatchRequest');
    for (const k of Object.keys(body)) expect(allowed).toContain(k);
    const itemAllowed = declared('IngestBatchItem');
    for (const item of body.items as Record<string, unknown>[]) {
      for (const k of Object.keys(item)) expect(itemAllowed).toContain(k);
    }
  });
});
