import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import IngestRunPanel from './IngestRunPanel.svelte';
import { ingestPathLookup, ingestUpload } from '$lib/api';
import { resolveIngestConfig } from '$lib/ingest/ingestConfig';
import type { IngestFile } from '$lib/ingest/fileSource';

vi.mock('$lib/api', async () => {
  const actual = await vi.importActual<typeof import('$lib/api')>('$lib/api');
  return { ...actual, ingestPathLookup: vi.fn(), ingestUpload: vi.fn() };
});

let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | undefined;

function mkFile(id: string): IngestFile {
  return { id, relPath: id, file: new File(['x'], id), size: 1 };
}

beforeEach(() => {
  target = document.createElement('div');
  document.body.appendChild(target);
});

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target.remove();
  vi.restoreAllMocks();
});

describe('IngestRunPanel', () => {
  it('shows the served per-file error in the Failed tab', async () => {
    vi.mocked(ingestPathLookup).mockResolvedValue({ known_paths: {} });
    vi.mocked(ingestUpload).mockResolvedValue({
      status: 'partial',
      summary: {
        successful: 0,
        duplicates: 0,
        failed: 1,
        mismatches: 0,
        missed_labels: 0,
        unmatched_detections: 0,
        labels_imported: 0,
        crops_indexed: 0,
      },
      results: [
        {
          status: 'failed',
          image_id: '',
          image_path: 'upload/a.jpg',
          imohash: '',
          n_crops: 0,
          n_regions: 0,
          error: 'decode failed',
          error_kind: 'decode_failed',
          source_identifier: 'upload/a.jpg',
        },
      ],
      disagreements: [],
    });
    instance = mount(IngestRunPanel, {
      target,
      props: { files: [mkFile('a.jpg')], config: resolveIngestConfig(null) },
    });
    flushSync();
    const startBtn = [...target.querySelectorAll('button')].find(
      (b) => b.textContent?.trim() === 'Start',
    );
    startBtn?.click();
    await vi.waitFor(() => {
      flushSync();
      expect(target.textContent).toContain('decode failed');
    });
  });

  it('renders CSV content including the header and a failed row', async () => {
    vi.mocked(ingestPathLookup).mockResolvedValue({ known_paths: {} });
    vi.mocked(ingestUpload).mockResolvedValue({
      status: 'partial',
      summary: {
        successful: 0,
        duplicates: 0,
        failed: 1,
        mismatches: 0,
        missed_labels: 0,
        unmatched_detections: 0,
        labels_imported: 0,
        crops_indexed: 0,
      },
      results: [
        {
          status: 'failed',
          image_id: '',
          image_path: 'upload/a.jpg',
          imohash: '',
          n_crops: 0,
          n_regions: 0,
          error: 'decode failed',
          error_kind: 'decode_failed',
          source_identifier: 'upload/a.jpg',
        },
      ],
      disagreements: [],
    });
    let capturedBlob: Blob | null = null;
    const originalCreateObjectURL = URL.createObjectURL;
    URL.createObjectURL = vi.fn((b: Blob) => {
      capturedBlob = b;
      return 'blob:test';
    });
    URL.revokeObjectURL = vi.fn();
    instance = mount(IngestRunPanel, {
      target,
      props: { files: [mkFile('a.jpg')], config: resolveIngestConfig(null) },
    });
    flushSync();
    const startBtn = [...target.querySelectorAll('button')].find(
      (b) => b.textContent?.trim() === 'Start',
    );
    startBtn?.click();
    await vi.waitFor(() => {
      flushSync();
      expect(target.textContent).toContain('Download CSV');
    });
    const dlBtn = [...target.querySelectorAll('button')].find(
      (b) => b.textContent?.trim() === 'Download CSV',
    );
    dlBtn?.click();
    expect(capturedBlob).toBeTruthy();
    const text = await (capturedBlob as unknown as Blob).text();
    expect(text).toContain('identifier,status,error,error_kind,image_id,n_crops');
    expect(text).toContain('decode failed');
    URL.createObjectURL = originalCreateObjectURL;
  });

  it('F-59: a duplicate-only re-upload says so instead of showing nothing', async () => {
    vi.mocked(ingestPathLookup).mockResolvedValue({
      known_paths: { 'upload/a.jpg': 'img-1' },
    });
    vi.mocked(ingestUpload).mockReset();
    instance = mount(IngestRunPanel, {
      target,
      props: { files: [mkFile('a.jpg')], config: resolveIngestConfig(null) },
    });
    flushSync();
    [...target.querySelectorAll('button')]
      .find((b) => b.textContent?.trim() === 'Start')
      ?.click();
    await vi.waitFor(() => {
      flushSync();
      expect(
        target.querySelector('[data-testid="ingest-all-skipped"]')?.textContent,
      ).toContain('1 already indexed');
    });
    expect(ingestUpload).not.toHaveBeenCalled();
    // The per-file list is on the Skipped tab, with the row visible.
    expect(target.textContent).toContain('Skipped (1)');
    expect(target.textContent).toContain('upload/a.jpg');
  });

  it('d72cc63: shows the served secondary-detector failures on ingested files', async () => {
    vi.mocked(ingestPathLookup).mockResolvedValue({ known_paths: {} });
    vi.mocked(ingestUpload).mockResolvedValue({
      status: 'success',
      summary: {
        successful: 1,
        duplicates: 0,
        failed: 0,
        mismatches: 0,
        missed_labels: 0,
        unmatched_detections: 0,
        labels_imported: 0,
        crops_indexed: 2,
        secondary_detector_failures: 1,
      },
      results: [
        {
          status: 'success',
          image_id: 'img-1',
          image_path: '/uploads/ab.jpg',
          imohash: 'h',
          n_crops: 2,
          n_regions: 0,
          error: null,
          error_kind: null,
          source_identifier: 'upload/a.jpg',
          secondary_detector_error: 'DEADLINE_EXCEEDED after 30s',
        },
      ],
      disagreements: [],
    });
    instance = mount(IngestRunPanel, {
      target,
      props: { files: [mkFile('a.jpg')], config: resolveIngestConfig(null) },
    });
    flushSync();
    [...target.querySelectorAll('button')]
      .find((b) => b.textContent?.trim() === 'Start')
      ?.click();
    await vi.waitFor(() => {
      flushSync();
      expect(
        target.querySelector('[data-testid="ingest-secondary-failures"]')?.textContent,
      ).toContain('secondary detector failed 1');
    });
    expect(target.textContent).toContain(
      'secondary detector: DEADLINE_EXCEEDED after 30s',
    );
  });
});
