/**
 * The backend's typed "source images are not reachable" answer for an image
 * route (`detail.error === 'source_missing'`). An `<img>` error event carries
 * no body, so on failure we re-request the same URL and read the answer the
 * server decided; the frontend only renders it.
 */
export interface SourceMissing {
  alias: string;
  root_mounted: boolean;
  message: string;
}

export function parseSourceMissing(body: unknown): SourceMissing | null {
  const d = (body as { detail?: unknown } | null)?.detail;
  if (!d || typeof d !== 'object') return null;
  const r = d as Record<string, unknown>;
  if (r.error !== 'source_missing' || typeof r.alias !== 'string') return null;
  return {
    alias: r.alias,
    root_mounted: r.root_mounted === true,
    message: typeof r.message === 'string' ? r.message : 'source image not found',
  };
}

export async function fetchSourceMissing(
  url: string,
  fetchFn: typeof fetch = fetch,
): Promise<SourceMissing | null> {
  try {
    const res = await fetchFn(url);
    if (res.status !== 404) return null;
    return parseSourceMissing(await res.json());
  } catch {
    return null;
  }
}
