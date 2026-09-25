import { describe, expect, it } from 'vitest';
import {
  DOCUMENTED_ACCEPTED_EXTENSIONS,
  DOCUMENTED_UPLOAD_MAX_IMAGES,
  resolveIngestConfig,
} from './ingestConfig';

describe('resolveIngestConfig', () => {
  it('falls back to the documented interim constants when nothing is served', () => {
    const resolved = resolveIngestConfig(null);
    expect(resolved.uploadMaxImages).toBe(DOCUMENTED_UPLOAD_MAX_IMAGES);
    expect(resolved.acceptedExtensions).toEqual(DOCUMENTED_ACCEPTED_EXTENSIONS);
    expect(resolved.uploadEnabled).toBe(true);
    expect(resolved.batchEnabled).toBe(false);
    expect(resolved.uploadPersistsBytes).toBeNull();
  });

  it('prefers every served field over its interim constant', () => {
    const resolved = resolveIngestConfig({
      upload: {
        enabled: false,
        max_images_per_request: 64,
        accepted_extensions: ['.tif'],
        persists_bytes: true,
      },
      batch: { enabled: true, max_items_per_request: 500, source_roots: ['/data'] },
      region_drain: { poll_interval_s: 5 },
    });
    expect(resolved.uploadEnabled).toBe(false);
    expect(resolved.uploadMaxImages).toBe(64);
    expect(resolved.acceptedExtensions).toEqual(['.tif']);
    expect(resolved.uploadPersistsBytes).toBe(true);
    expect(resolved.batchEnabled).toBe(true);
    expect(resolved.batchMaxItems).toBe(500);
    expect(resolved.batchSourceRoots).toEqual(['/data']);
    expect(resolved.regionDrainPollIntervalS).toBe(5);
  });
});
