import { describe, expect, it } from 'vitest';
import { servedIngestConfig as servedConfig } from '$lib/test/fixtures/ingestConfig';
import { resolveIngestConfig, serverPathIngestAvailable } from './ingestConfig';

describe('resolveIngestConfig', () => {
  it('resolves every served field', () => {
    const resolved = resolveIngestConfig(
      servedConfig({
        upload: {
          enabled: false,
          max_images_per_request: 64,
          max_bytes_per_request: 10 * 1024 * 1024,
          accepted_extensions: ['.tif'],
          persists_bytes: true,
        },
        batch: { enabled: true, max_items: 500, source_roots: ['/data'] },
        region_drain: { poll_interval_s: 5, stable_polls: 4 },
      }),
    );
    expect(resolved.uploadEnabled).toBe(false);
    expect(resolved.uploadMaxImages).toBe(64);
    expect(resolved.acceptedExtensions).toEqual(['.tif']);
    expect(resolved.uploadPersistsBytes).toBe(true);
    expect(resolved.batchEnabled).toBe(true);
    expect(resolved.batchMaxItems).toBe(500);
    expect(resolved.batchSourceRoots).toEqual(['/data']);
    expect(resolved.regionDrainPollIntervalS).toBe(5);
    expect(resolved.regionDrainStablePolls).toBe(4);
  });

  it('uploadMaxBytes is the tighter of the nginx proxy cap and the served backend limit', () => {
    // The served backend limit (10MB) is far under the default 256MB
    // nginx cap's 90%-headroom value — the served limit must win, or a
    // chunk sized only against the proxy cap would 413 against the
    // backend's own enforced upload.max_bytes_per_request.
    const tight = resolveIngestConfig(
      servedConfig({
        upload: {
          enabled: true,
          max_images_per_request: 128,
          max_bytes_per_request: 10 * 1024 * 1024,
          accepted_extensions: ['.jpg'],
          persists_bytes: true,
        },
      }),
    );
    expect(tight.uploadMaxBytes).toBe(10 * 1024 * 1024);

    // A served limit larger than the nginx cap must not widen the chunk
    // planner past what the proxy itself will actually let through.
    const loose = resolveIngestConfig(
      servedConfig({
        upload: {
          enabled: true,
          max_images_per_request: 128,
          max_bytes_per_request: 10 * 1024 * 1024 * 1024,
          accepted_extensions: ['.jpg'],
          persists_bytes: true,
        },
      }),
    );
    expect(loose.uploadMaxBytes).toBeLessThan(10 * 1024 * 1024 * 1024);
  });
});

describe('serverPathIngestAvailable', () => {
  const withBatch = (enabled: boolean, source_roots: string[]) =>
    resolveIngestConfig(
      servedConfig({ batch: { enabled, max_items: 256, source_roots } }),
    );

  it('needs both the served batch.enabled and at least one source root', () => {
    expect(serverPathIngestAvailable(withBatch(true, ['/data']))).toBe(true);
    expect(serverPathIngestAvailable(withBatch(false, ['/data']))).toBe(false);
    expect(serverPathIngestAvailable(withBatch(true, []))).toBe(false);
  });
});
