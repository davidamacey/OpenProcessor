/**
 * The archive-upload size the client allows: the lower of the served
 * `GET /datasets/formats` `upload.max_bytes` and this deployment's own
 * nginx `client_max_body_size` for `/datasets/uploads`
 * (`CROPWRIGHT_DATASET_UPLOAD_MAX_MB`, substituted into
 * `window.__CROPWRIGHT_DATASET_UPLOAD_MAX_MB__` by docker-entrypoint.sh).
 * A file above either one would be refused anyway, after a long upload.
 */
export const DEFAULT_DATASET_UPLOAD_MAX_MB = 2048;

export function proxyUploadMaxBytes(): number {
  let mb = DEFAULT_DATASET_UPLOAD_MAX_MB;
  if (typeof window !== 'undefined') {
    const raw = (window as unknown as Record<string, unknown>)
      .__CROPWRIGHT_DATASET_UPLOAD_MAX_MB__;
    const n = typeof raw === 'string' || typeof raw === 'number' ? Number(raw) : NaN;
    if (Number.isFinite(n) && n > 0) mb = n;
  }
  return mb * 1024 * 1024;
}

export function datasetUploadMaxBytes(servedMaxBytes: number): number {
  return Math.min(servedMaxBytes, proxyUploadMaxBytes());
}
