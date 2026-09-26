import { describe, expect, it } from 'vitest';
import type { IngestConfig } from '$lib/types';
import {
  DOCUMENTED_ACCEPTED_EXTENSIONS,
  DOCUMENTED_UPLOAD_MAX_IMAGES,
  resolveIngestConfig,
} from './ingestConfig';

function servedConfig(overrides: Partial<IngestConfig> = {}): IngestConfig {
  return {
    upload: {
      enabled: true,
      max_images_per_request: 128,
      max_bytes_per_request: 500 * 1024 * 1024,
      accepted_extensions: ['.jpg', '.jpeg', '.png'],
      persists_bytes: true,
    },
    batch: { enabled: true, max_items: 256, source_roots: [] },
    region_drain: { poll_interval_s: 10, stable_polls: 3 },
    ...overrides,
  };
}

describe('resolveIngestConfig', () => {
  it('falls back to the documented interim constants when nothing is served', () => {
    const resolved = resolveIngestConfig(null);
    expect(resolved.uploadMaxImages).toBe(DOCUMENTED_UPLOAD_MAX_IMAGES);
    expect(resolved.acceptedExtensions).toEqual(DOCUMENTED_ACCEPTED_EXTENSIONS);
    expect(resolved.uploadEnabled).toBe(true);
    expect(resolved.batchEnabled).toBe(false);
    expect(resolved.uploadPersistsBytes).toBeNull();
    expect(resolved.batchMaxItems).toBeNull();
    expect(resolved.regionDrainStablePolls).toBeNull();
  });

  it('prefers every served field over its interim constant (BA-2, c5c606f)', () => {
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
