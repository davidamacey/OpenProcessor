/**
 * The archive-upload size the client allows: the lower of the served
 * `GET /datasets/formats` `upload.max_bytes` and this deployment's own
 * nginx `client_max_body_size` for `/datasets/uploads`
 * (`CROPWRIGHT_DATASET_UPLOAD_MAX_MB`, substituted into
 * `window.__CROPWRIGHT_DATASET_UPLOAD_MAX_MB__` by docker-entrypoint.sh).
 * A file above either one would be refused anyway, after a long upload.
 */
export const DEFAULT_DATASET_UPLOAD_MAX_MB = 2048;

/** nginx counts the whole multipart body against `client_max_body_size`, not
 *  just the archive: about 200 bytes of framing for one file. A small fixed
 *  allowance keeps an archive within that of the limit from passing the
 *  client check and then failing on the proxy. */
export const MULTIPART_ALLOWANCE_BYTES = 64 * 1024;

export function proxyUploadMaxBytes(): number {
  let mb = DEFAULT_DATASET_UPLOAD_MAX_MB;
  if (typeof window !== 'undefined') {
    const raw = (window as unknown as Record<string, unknown>)
      .__CROPWRIGHT_DATASET_UPLOAD_MAX_MB__;
    const n = typeof raw === 'string' || typeof raw === 'number' ? Number(raw) : NaN;
    if (Number.isFinite(n) && n > 0) mb = n;
  }
  return mb * 1024 * 1024 - MULTIPART_ALLOWANCE_BYTES;
}

export function datasetUploadMaxBytes(servedMaxBytes: number): number {
  return Math.min(servedMaxBytes, proxyUploadMaxBytes());
}
