/**
 * W10 wrappers (any_domain_plan.md W10.14): every route is scoped, every
 * body is sent as given (no `project` key, delta 15), and the structured
 * `ConfigErrorDetail` refusal surfaces its served `message`.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  API_PREFIX,
  ApiError,
  cancelDatasetImport,
  cancelReprocessJob,
  datasetErrorDetail,
  apiErrorText,
  getDatasetFormats,
  getDatasetImport,
  getDatasetImportEntries,
  getDatasetImportIssues,
  getReprocessJob,
  listDatasetImports,
  previewDataset,
  reprocessBatch,
  reprocessCrop,
  resumeDatasetImport,
  runServedNextStep,
  startDatasetImport,
  undoDatasetImport,
  uploadDatasetArchive,
} from './api';

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
});

function capture(body: unknown = {}, status = 200) {
  const fetchMock = vi.fn().mockResolvedValue(json(body, status));
  vi.stubGlobal('fetch', fetchMock);
  return () => {
    const [url, init] = fetchMock.mock.calls[0]! as [string, RequestInit];
    return {
      url: String(url),
      method: init.method ?? 'GET',
      body:
        init.body instanceof FormData
          ? init.body
          : init.body
            ? JSON.parse(String(init.body))
            : undefined,
    };
  };
}

const P = API_PREFIX;
const ID = 'imp_1';

describe('W10 wrappers hit the scoped routes', () => {
  it('formats / list / get / issues / entries are GETs with their query', async () => {
    let sent = capture();
    await getDatasetFormats();
    expect(sent()).toMatchObject({ url: `${P}/datasets/formats`, method: 'GET' });

    sent = capture({ items: [], total: 0, page: 1, page_size: 50 });
    await listDatasetImports({ page: 2, page_size: 25, status: 'failed' });
    expect(sent().url).toBe(`${P}/datasets/imports?page=2&page_size=25&status=failed`);

    sent = capture();
    await getDatasetImport('imp/1');
    expect(sent().url).toBe(`${P}/datasets/imports/imp%2F1`);

    sent = capture();
    await getDatasetImportIssues(ID, {
      code: 'label_file_missing',
      page: 1,
      page_size: 20,
    });
    expect(sent().url).toBe(
      `${P}/datasets/imports/${ID}/issues?code=label_file_missing&page=1&page_size=20`,
    );

    sent = capture();
    await getDatasetImportEntries(ID, {
      split: 'test',
      label_state: 'negative',
      status: null,
    });
    expect(sent().url).toBe(
      `${P}/datasets/imports/${ID}/entries?split=test&label_state=negative`,
    );
  });

  it('preview and start POST the body verbatim, with no project key', async () => {
    const body = {
      source: { path: '/data/x', format: 'auto' },
      mapping: [{ dataset_class: 'Widget', action: 'map', class_id: 2 }],
      accept_suggestions: false,
      options: { processing: 'none' },
    };
    let sent = capture();
    await previewDataset(body);
    expect(sent()).toEqual({ url: `${P}/datasets/preview`, method: 'POST', body });
    expect(sent().body).not.toHaveProperty('project');

    sent = capture();
    await startDatasetImport({ ...body, expected_import_key: 'k' });
    expect(sent()).toEqual({
      url: `${P}/datasets/imports`,
      method: 'POST',
      body: { ...body, expected_import_key: 'k' },
    });
  });

  it('cancel / resume / undo', async () => {
    let sent = capture();
    await cancelDatasetImport(ID);
    expect(sent()).toMatchObject({
      url: `${P}/datasets/imports/${ID}/cancel`,
      method: 'POST',
    });
    sent = capture();
    await resumeDatasetImport(ID);
    expect(sent()).toMatchObject({
      url: `${P}/datasets/imports/${ID}/resume`,
      method: 'POST',
    });
    sent = capture();
    await undoDatasetImport(ID, {
      dry_run: true,
      remove_images: true,
      deprecate_created_classes: false,
    });
    expect(sent()).toEqual({
      url: `${P}/datasets/imports/${ID}/undo`,
      method: 'POST',
      body: { dry_run: true, remove_images: true, deprecate_created_classes: false },
    });
  });

  it('upload sends one multipart `file` part', async () => {
    const sent = capture({ upload_id: 'u', dataset_path: '/up/x', bytes: 3, files: 1 });
    const file = new File(['abc'], 'set.zip');
    await uploadDatasetArchive(file);
    const s = sent();
    expect(s.url).toBe(`${P}/datasets/uploads`);
    expect(s.method).toBe('POST');
    expect((s.body as FormData).get('file')).toBeInstanceOf(File);
  });

  it('a served next step runs its method against its path under the prefix', async () => {
    const sent = capture();
    await runServedNextStep({
      action: 'cluster_regions',
      method: 'post',
      path: '/regions/cluster',
      reason: 'r',
    });
    expect(sent()).toMatchObject({ url: `${P}/regions/cluster`, method: 'POST' });
  });

  it('reprocess batch, single crop, job read and cancel', async () => {
    let sent = capture();
    await reprocessBatch({
      targets: { crop_ids: ['c1'] },
      scopes: ['region'],
      region_mode: 'redetect',
      dry_run: true,
    });
    expect(sent()).toEqual({
      url: `${P}/reprocess`,
      method: 'POST',
      body: {
        targets: { crop_ids: ['c1'] },
        scopes: ['region'],
        region_mode: 'redetect',
        dry_run: true,
      },
    });
    sent = capture();
    await reprocessCrop('c 1', { scopes: ['vlm'], dry_run: false });
    expect(sent()).toEqual({
      url: `${P}/crops/c%201/reprocess`,
      method: 'POST',
      body: { scopes: ['vlm'], dry_run: false },
    });
    sent = capture();
    await getReprocessJob('j1');
    expect(sent()).toMatchObject({ url: `${P}/reprocess/jobs/j1`, method: 'GET' });
    sent = capture();
    await cancelReprocessJob('j1');
    expect(sent()).toMatchObject({
      url: `${P}/reprocess/jobs/j1/cancel`,
      method: 'POST',
    });
  });
});

describe('datasetErrorDetail / apiErrorText', () => {
  it('reads the structured detail and prefers its served message', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        json(
          {
            detail: {
              error: 'class_mapping_incomplete',
              message: 'Two classes still need a mapping.',
              unmapped: ['sprocket', 'cog'],
            },
          },
          422,
        ),
      ),
    );
    const err = await startDatasetImport({
      source: { path: '/x', format: 'auto' },
      mapping: [],
      accept_suggestions: false,
      options: {},
    }).catch((e: unknown) => e);
    expect(datasetErrorDetail(err)).toMatchObject({
      error: 'class_mapping_incomplete',
      unmapped: ['sprocket', 'cog'],
    });
    expect(apiErrorText(err)).toBe('Two classes still need a mapping.');
  });

  it('falls back to the generic detail for an unstructured error', () => {
    const e = new ApiError(400, '/x', { detail: 'bad body' });
    expect(datasetErrorDetail(e)).toBeNull();
    expect(apiErrorText(e)).toBe('bad body');
    expect(apiErrorText(new Error('network down'))).toBe('network down');
  });
});
