import { describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api';
import type {
  BatchIngestResponse,
  IngestPathLookupResponse,
  IngestUploadRequest,
} from '$lib/types';
import type { IngestFile } from './fileSource';
import { resolveIngestConfig } from './ingestConfig';
import { servedIngestConfig } from '$lib/test/fixtures/ingestConfig';
import { createIngestRun, type IngestRunDeps } from './ingestRunController.svelte';

function mkFile(id: string, size = 10): IngestFile {
  return {
    id,
    relPath: id,
    file: new File(['x'.repeat(size)], id.split('/').pop()!),
    size,
  };
}

function successResponse(identifiers: string[]): BatchIngestResponse {
  return {
    status: 'success',
    summary: {
      successful: identifiers.length,
      duplicates: 0,
      failed: 0,
      crops_indexed: identifiers.length,
      n_embedded: 0,
      n_not_embedded: 0,
      n_filtered: 0,
    },
    results: identifiers.map((image_path) => ({
      status: 'success',
      image_id: `img-${image_path}`,
      image_path,
      imohash: 'h',
      n_crops: 1,
      n_regions: 0,
      n_embedded: 0,
      n_not_embedded: 0,
      n_filtered: 0,
      error: null,
      error_kind: null,
      source_identifier: image_path,
    })),
  };
}

function deferred<T>() {
  let resolve!: (v: T) => void;
  let reject!: (e: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

const config = resolveIngestConfig(servedIngestConfig());

function baseDeps(overrides: Partial<IngestRunDeps> = {}): IngestRunDeps {
  return {
    lookup: vi.fn(async (): Promise<IngestPathLookupResponse> => ({ known_paths: {} })),
    upload: vi.fn(async (req) => successResponse(req.identifiers)),
    config,
    concurrency: 2,
    ...overrides,
  };
}

describe('createIngestRun — prefilter', () => {
  it('skips files whose identifier the server already knows', async () => {
    const lookup = vi.fn(async (): Promise<IngestPathLookupResponse> => ({
      known_paths: { 'src/a.jpg': 'img-1' },
    }));
    const upload = vi.fn(async (req) => successResponse(req.identifiers));
    const run = createIngestRun(baseDeps({ lookup, upload }));
    await run.start([mkFile('a.jpg'), mkFile('b.jpg')], {
      source: 'src',
      identifierPrefix: 'src/',
      skipLookup: false,
    });
    expect(run.totals.skipped_known).toBe(1);
    expect(run.results.get('a.jpg')?.kind).toBe('skipped');
    // Only the non-skipped file was ever uploaded.
    const sentIds = (
      upload.mock.calls as unknown as { identifiers: string[] }[][]
    ).flatMap((c) => c[0]!.identifiers);
    expect(sentIds).toEqual(['src/b.jpg']);
  });

  it('continues (relying on server dedup) when the lookup call fails', async () => {
    const lookup = vi.fn().mockRejectedValue(new Error('network down'));
    const upload = vi.fn(async (req) => successResponse(req.identifiers));
    const run = createIngestRun(baseDeps({ lookup, upload }));
    await run.start([mkFile('a.jpg')], {
      source: 'src',
      identifierPrefix: 'src/',
      skipLookup: false,
    });
    expect(run.state).toBe('done');
    expect(run.totals.successful).toBe(1);
  });

  it('skipLookup bypasses the lookup call entirely', async () => {
    const lookup = vi.fn();
    const upload = vi.fn(async (req) => successResponse(req.identifiers));
    const run = createIngestRun(baseDeps({ lookup, upload }));
    await run.start([mkFile('a.jpg')], {
      source: 'src',
      identifierPrefix: 'src/',
      skipLookup: true,
    });
    expect(lookup).not.toHaveBeenCalled();
    expect(run.totals.successful).toBe(1);
  });
});

describe('createIngestRun — concurrency', () => {
  it('never runs more than `concurrency` uploads at once', async () => {
    let inFlight = 0;
    let maxInFlight = 0;
    const gates = [
      deferred<void>(),
      deferred<void>(),
      deferred<void>(),
      deferred<void>(),
    ];
    let call = 0;
    const upload = vi.fn(async (req) => {
      inFlight++;
      maxInFlight = Math.max(maxInFlight, inFlight);
      const gate = gates[call++]!;
      await gate.promise;
      inFlight--;
      return successResponse(req.identifiers);
    });
    const run = createIngestRun(
      baseDeps({
        lookup: vi.fn(async () => ({ known_paths: {} })),
        upload,
        concurrency: 2,
        config: { ...config, uploadMaxImages: 1 },
      }),
    );
    const files = Array.from({ length: 4 }, (_, i) => mkFile(`f${i}.jpg`));
    const startP = run.start(files, {
      source: 'src',
      identifierPrefix: 'src/',
      skipLookup: true,
    });
    // Let the two allowed workers reach `upload`.
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();
    expect(inFlight).toBeLessThanOrEqual(2);
    gates[0]!.resolve();
    gates[1]!.resolve();
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();
    gates[2]!.resolve();
    gates[3]!.resolve();
    await startP;
    expect(maxInFlight).toBeLessThanOrEqual(2);
    expect(run.totals.successful).toBe(4);
  });
});

describe('createIngestRun — pause/resume/cancel', () => {
  it('pause stops new dispatch while an in-flight chunk finishes; resume continues', async () => {
    const gate = deferred<void>();
    let uploadCalls = 0;
    const upload = vi.fn(async (req) => {
      uploadCalls++;
      if (uploadCalls === 1) await gate.promise;
      return successResponse(req.identifiers);
    });
    const run = createIngestRun(
      baseDeps({
        lookup: vi.fn(async () => ({ known_paths: {} })),
        upload,
        concurrency: 1,
        config: { ...config, uploadMaxImages: 1 },
      }),
    );
    const files = [mkFile('a.jpg'), mkFile('b.jpg')];
    const startP = run.start(files, {
      source: 'src',
      identifierPrefix: 'src/',
      skipLookup: true,
    });
    await vi.waitFor(() => expect(run.state).toBe('uploading'));
    run.pause();
    expect(run.state).toBe('paused');
    gate.resolve();
    // Give the in-flight chunk a chance to finish; it must NOT start
    // chunk 2 while paused.
    await new Promise((r) => setTimeout(r, 10));
    expect(uploadCalls).toBe(1);
    run.resume();
    await startP;
    expect(uploadCalls).toBe(2);
    expect(run.state).toBe('done');
    expect(run.totals.successful).toBe(2);
  });

  it('cancel aborts and marks unsent files not_sent', async () => {
    const gate = deferred<void>();
    gate.promise.catch(() => {});
    const upload = vi.fn(async (req) => {
      await gate.promise.catch(() => {});
      return successResponse(req.identifiers);
    });
    const run = createIngestRun(
      baseDeps({
        lookup: vi.fn(async () => ({ known_paths: {} })),
        upload,
        concurrency: 1,
        config: { ...config, uploadMaxImages: 1 },
      }),
    );
    const files = [mkFile('a.jpg'), mkFile('b.jpg')];
    const startP = run.start(files, {
      source: 'src',
      identifierPrefix: 'src/',
      skipLookup: true,
    });
    await vi.waitFor(() => expect(upload).toHaveBeenCalled());
    run.cancel();
    gate.reject(new DOMException('Aborted', 'AbortError'));
    await startP;
    expect(run.state).toBe('cancelled');
    expect(run.results.get('b.jpg')?.kind).toBe('not_sent');
  });
});

describe('createIngestRun — response handling', () => {
  it('maps success/duplicate/failed results back to their files', async () => {
    const upload = vi.fn(async (): Promise<BatchIngestResponse> => ({
      status: 'partial',
      summary: {
        successful: 1,
        duplicates: 1,
        failed: 1,
        crops_indexed: 1,
        n_embedded: 0,
        n_not_embedded: 0,
        n_filtered: 0,
      },
      results: [
        {
          status: 'success',
          image_id: 'img-1',
          image_path: 'src/a.jpg',
          imohash: 'h',
          n_crops: 3,
          n_regions: 0,
          n_embedded: 0,
          n_not_embedded: 0,
          n_filtered: 0,
          error: null,
          error_kind: null,
          source_identifier: 'src/a.jpg',
        },
        {
          status: 'duplicate',
          image_id: 'img-2',
          image_path: 'src/b.jpg',
          imohash: 'h2',
          n_crops: 0,
          n_regions: 0,
          n_embedded: 0,
          n_not_embedded: 0,
          n_filtered: 0,
          error: null,
          error_kind: null,
          source_identifier: 'src/b.jpg',
        },
        {
          status: 'failed',
          image_id: '',
          image_path: 'src/c.jpg',
          imohash: '',
          n_crops: 0,
          n_regions: 0,
          n_embedded: 0,
          n_not_embedded: 0,
          n_filtered: 0,
          error: 'decode error',
          error_kind: 'decode_failed',
          source_identifier: 'src/c.jpg',
        },
      ],
    }));
    const run = createIngestRun(
      baseDeps({
        lookup: vi.fn(async () => ({ known_paths: {} })),
        upload,
        concurrency: 3,
      }),
    );
    await run.start([mkFile('a.jpg'), mkFile('b.jpg'), mkFile('c.jpg')], {
      source: 'src',
      identifierPrefix: 'src/',
      skipLookup: true,
    });
    expect(run.results.get('a.jpg')).toMatchObject({ kind: 'ingested', n_crops: 3 });
    expect(run.results.get('b.jpg')).toMatchObject({ kind: 'duplicate' });
    expect(run.results.get('c.jpg')).toMatchObject({
      kind: 'failed',
      error: 'decode error',
      error_kind: 'decode_failed',
    });
    expect(run.totals).toMatchObject({
      successful: 1,
      duplicates: 1,
      failed: 1,
      crops_indexed: 3,
    });
  });

  it('maps a byte-upload result by source_identifier, not the server-persisted image_path (BA-1)', async () => {
    const upload = vi.fn(
      async (req: IngestUploadRequest): Promise<BatchIngestResponse> => ({
        status: 'success',
        summary: {
          successful: req.identifiers.length,
          duplicates: 0,
          failed: 0,
          crops_indexed: req.identifiers.length,
          n_embedded: 0,
          n_not_embedded: 0,
          n_filtered: 0,
        },
        results: req.identifiers.map((identifier, i) => ({
          status: 'success' as const,
          image_id: `img-${i}`,
          // The server-persisted, content-addressed path -- deliberately
          // NOT equal to the identifier the client sent.
          image_path: `/data/uploads/ab/ab12${i}.jpg`,
          imohash: `ab12${i}`,
          n_crops: 1,
          n_regions: 0,
          n_embedded: 0,
          n_not_embedded: 0,
          n_filtered: 0,
          error: null,
          error_kind: null,
          source_identifier: identifier,
        })),
      }),
    );
    const run = createIngestRun(
      baseDeps({
        lookup: vi.fn(async () => ({ known_paths: {} })),
        upload,
        concurrency: 1,
      }),
    );
    await run.start([mkFile('a.jpg')], {
      source: 'src',
      identifierPrefix: 'src/',
      skipLookup: true,
    });
    expect(run.results.get('a.jpg')).toMatchObject({
      kind: 'ingested',
      image_id: 'img-0',
    });
  });

  it('falls back to request order for an in-batch byte-identical duplicate served with source_identifier: null', async () => {
    // A live backend bug: the second copy of a byte-identical pair
    // uploaded in the same chunk comes back with source_identifier: null
    // (and no rewritten image_path either, since it's a duplicate — the
    // server never persisted new bytes for it). Without the request-order
    // fallback this row can't be matched to either file by identifier and
    // both would misreport as "no result returned" for the second file.
    const upload = vi.fn(
      async (req: IngestUploadRequest): Promise<BatchIngestResponse> => ({
        status: 'partial',
        summary: {
          successful: 1,
          duplicates: 1,
          failed: 0,
          crops_indexed: 1,
          n_embedded: 0,
          n_not_embedded: 0,
          n_filtered: 0,
        },
        results: [
          {
            status: 'success',
            image_id: 'img-a',
            image_path: '/data/uploads/ab/ab120.jpg',
            imohash: 'dupe-hash',
            n_crops: 3,
            n_regions: 0,
            n_embedded: 0,
            n_not_embedded: 0,
            n_filtered: 0,
            error: null,
            error_kind: null,
            source_identifier: req.identifiers[0]!,
          },
          {
            status: 'duplicate',
            image_id: 'img-a',
            image_path: '/data/uploads/ab/ab120.jpg',
            imohash: 'dupe-hash',
            n_crops: 0,
            n_regions: 0,
            n_embedded: 0,
            n_not_embedded: 0,
            n_filtered: 0,
            error: null,
            error_kind: null,
            // The live-backend bug this fallback defends against.
            source_identifier: null,
          },
        ],
      }),
    );
    const run = createIngestRun(
      baseDeps({
        lookup: vi.fn(async () => ({ known_paths: {} })),
        upload,
        concurrency: 1,
      }),
    );
    await run.start([mkFile('a.jpg'), mkFile('b.jpg')], {
      source: 'src',
      identifierPrefix: 'src/',
      skipLookup: true,
    });
    expect(run.results.get('a.jpg')).toMatchObject({ kind: 'ingested', n_crops: 3 });
    expect(run.results.get('b.jpg')).toMatchObject({
      kind: 'duplicate',
      image_id: 'img-a',
    });
    expect(run.totals).toMatchObject({ successful: 1, duplicates: 1, failed: 0 });
  });

  it('auto-pauses on a served 503 and shows the served detail', async () => {
    let calls = 0;
    const upload = vi.fn(async (req) => {
      calls++;
      if (calls === 1) {
        throw new ApiError(503, 'x', { detail: 'detector not configured' });
      }
      return successResponse(req.identifiers);
    });
    const run = createIngestRun(
      baseDeps({
        lookup: vi.fn(async () => ({ known_paths: {} })),
        upload,
        concurrency: 1,
      }),
    );
    const files = [mkFile('a.jpg')];
    const startP = run.start(files, {
      source: 'src',
      identifierPrefix: 'src/',
      skipLookup: true,
    });
    await vi.waitFor(() => expect(run.state).toBe('paused'));
    expect(run.pauseReason).toBe('detector not configured');
    run.resume();
    await startP;
    expect(run.state).toBe('done');
    expect(run.totals.successful).toBe(1);
  });

  it('halves the chunk once on a backend-style (JSON) 413', async () => {
    let calls = 0;
    const upload = vi.fn(async (req) => {
      calls++;
      if (req.files.length > 1) {
        throw new ApiError(413, 'x', { detail: 'too many images' });
      }
      return successResponse(req.identifiers);
    });
    const run = createIngestRun(
      baseDeps({
        lookup: vi.fn(async () => ({ known_paths: {} })),
        upload,
        concurrency: 1,
      }),
    );
    const files = [mkFile('a.jpg'), mkFile('b.jpg')];
    await run.start(files, { source: 'src', identifierPrefix: 'src/', skipLookup: true });
    expect(run.state).toBe('done');
    expect(run.totals.successful).toBe(2);
    expect(calls).toBeGreaterThan(1);
  });

  it('marks a single-file chunk failed with error_kind too_large on a persistent backend-style 413', async () => {
    const upload = vi.fn(async () => {
      throw new ApiError(413, 'x', { detail: 'single image exceeds the byte limit' });
    });
    const run = createIngestRun(
      baseDeps({
        lookup: vi.fn(async () => ({ known_paths: {} })),
        upload,
        concurrency: 1,
      }),
    );
    await run.start([mkFile('a.jpg')], {
      source: 'src',
      identifierPrefix: 'src/',
      skipLookup: true,
    });
    expect(run.results.get('a.jpg')).toMatchObject({
      kind: 'failed',
      error: 'single image exceeds the byte limit',
      error_kind: 'too_large',
    });
  });

  it('stops the run on an nginx-style (HTML) 413, never retrying', async () => {
    const upload = vi.fn(async () => {
      throw new ApiError(413, 'x', '<html>413 Request Entity Too Large</html>');
    });
    const run = createIngestRun(
      baseDeps({
        lookup: vi.fn(async () => ({ known_paths: {} })),
        upload,
        concurrency: 1,
      }),
    );
    await run.start([mkFile('a.jpg')], {
      source: 'src',
      identifierPrefix: 'src/',
      skipLookup: true,
    });
    expect(run.state).toBe('error');
    expect(run.errorReason).toMatch(/CROPWRIGHT_INGEST_MAX_REQUEST_MB/);
    expect(upload).toHaveBeenCalledTimes(1);
  });

  it('marks a 422 chunk failed with the served detail', async () => {
    const upload = vi.fn(async () => {
      throw new ApiError(422, 'x', { detail: 'image_paths length mismatch' });
    });
    const run = createIngestRun(
      baseDeps({
        lookup: vi.fn(async () => ({ known_paths: {} })),
        upload,
        concurrency: 1,
      }),
    );
    await run.start([mkFile('a.jpg')], {
      source: 'src',
      identifierPrefix: 'src/',
      skipLookup: true,
    });
    expect(run.results.get('a.jpg')).toMatchObject({
      kind: 'failed',
      error: 'image_paths length mismatch',
    });
  });
});

describe('createIngestRun — retry failed', () => {
  it('resends only failed and not-sent files', async () => {
    let calls = 0;
    const upload = vi.fn(async (req) => {
      calls++;
      if (calls === 1) {
        return {
          status: 'partial' as const,
          summary: {
            successful: 1,
            duplicates: 0,
            failed: 1,
            crops_indexed: 1,
            n_embedded: 0,
            n_not_embedded: 0,
            n_filtered: 0,
          },
          results: [
            {
              status: 'success' as const,
              image_id: 'img-1',
              image_path: 'src/a.jpg',
              imohash: 'h',
              n_crops: 1,
              n_regions: 0,
              n_embedded: 0,
              n_not_embedded: 0,
              n_filtered: 0,
              error: null,
              error_kind: null,
              source_identifier: 'src/a.jpg',
            },
            {
              status: 'failed' as const,
              image_id: '',
              image_path: 'src/b.jpg',
              imohash: '',
              n_crops: 0,
              n_regions: 0,
              n_embedded: 0,
              n_not_embedded: 0,
              n_filtered: 0,
              error: 'transient',
              error_kind: 'detector_infer',
              source_identifier: 'src/b.jpg',
            },
          ],
        };
      }
      return successResponse(req.identifiers);
    });
    const run = createIngestRun(
      baseDeps({
        lookup: vi.fn(async () => ({ known_paths: {} })),
        upload,
        concurrency: 2,
      }),
    );
    await run.start([mkFile('a.jpg'), mkFile('b.jpg')], {
      source: 'src',
      identifierPrefix: 'src/',
      skipLookup: true,
    });
    expect(run.results.get('b.jpg')?.kind).toBe('failed');
    await run.retryFailed();
    const secondCallIds = (
      upload.mock.calls[1] as unknown as [{ identifiers: string[] }]
    )[0].identifiers;
    expect(secondCallIds).toEqual(['src/b.jpg']);
    expect(run.results.get('b.jpg')?.kind).toBe('ingested');
  });
});
