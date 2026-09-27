import type { IngestConfig } from '$lib/types';

/** A served `GET {API_PREFIX}/ingest/config` body, with per-test overrides. */
export function servedIngestConfig(overrides: Partial<IngestConfig> = {}): IngestConfig {
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
