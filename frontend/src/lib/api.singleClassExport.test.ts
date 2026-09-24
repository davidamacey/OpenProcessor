/**
 * The single-class export goes out on the slot's declared paths with the
 * slot's profile identity — the only way two single-class profiles keep
 * separate output roots and `current` symlinks on the backend.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  API_PREFIX,
  exportSingleClass,
  exportSingleClassStatus,
  listDatasets,
} from './api';
import { datasetExportForSlot } from './annotations/datasetExport';
import { licensePlateSlot } from './annotations/profiles/licensePlate';

const ok = (body: unknown) =>
  new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'content-type': 'application/json' },
  });

afterEach(() => {
  vi.unstubAllGlobals();
});

const spec = datasetExportForSlot(licensePlateSlot)!;

describe('single-class export wiring', () => {
  it("POSTs to the spec's buildPath with the slot's profile fields plus the options", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(ok({ export_dir: '/x', dataset_sha: 'abc' }));
    vi.stubGlobal('fetch', fetchMock);

    await exportSingleClass(spec, {
      image_mode: 'item_crop',
      img_max_side: 640,
      dedup_threshold: null,
      max_positive_images: undefined,
    });

    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe(`${API_PREFIX}/export/single_class`);
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body as string)).toEqual({
      profile_name: 'license_plate',
      box_source: 'region',
      region_class_name: 'license_plate',
      class_ids: [],
      image_mode: 'item_crop',
      img_max_side: 640,
      dedup_threshold: null,
    });
  });

  it("asks for the status of the slot's own profile", async () => {
    const fetchMock = vi.fn().mockResolvedValue(ok({ status: 'idle', last_run: null }));
    vi.stubGlobal('fetch', fetchMock);
    await exportSingleClassStatus(spec);
    expect(fetchMock.mock.calls[0]![0]).toBe(
      `${API_PREFIX}/export/single_class/status?profile_name=license_plate`,
    );
  });

  it('passes dataset listing filters through, and none when unfiltered', async () => {
    const fetchMock = vi
      .fn()
      .mockImplementation(() => Promise.resolve(ok({ datasets: [], count: 0 })));
    vi.stubGlobal('fetch', fetchMock);
    await listDatasets({ kind: 'single_class', profile_name: 'license_plate' });
    await listDatasets();
    expect(fetchMock.mock.calls[0]![0]).toBe(
      `${API_PREFIX}/export/datasets?kind=single_class&profile_name=license_plate`,
    );
    expect(fetchMock.mock.calls[1]![0]).toBe(`${API_PREFIX}/export/datasets`);
  });
});
