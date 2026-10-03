/**
 * v0.4.0 generic-detector wrappers: seed classes from the detector, the
 * project ingest policy (read, write, cost preview) and the detections
 * summary. Import from `$lib/api_detector` directly; never re-exported from
 * `api.ts` (that would make the two modules circular). Types live in
 * `$lib/types_detector`.
 */
import { ApiError, apiFetch, qs, scoped } from '$lib/api';
import type {
  DetectionsSummary,
  IngestPolicy,
  IngestPolicyBody,
  IngestPolicyPreview,
  IngestPolicyPutResponse,
  IngestPolicyUpdate,
  SeedFromDetectorRequest,
  SeedFromDetectorResponse,
} from '$lib/types_detector';
import type { ItemFilterQuery } from '$lib/types_itemFilter';

/** `POST /classes/seed_from_detector`. `dry_run` is always sent explicitly;
 *  `names` / `group` only when set. */
export function seedFromDetector(
  req: SeedFromDetectorRequest & { dry_run: boolean },
  signal?: AbortSignal,
): Promise<SeedFromDetectorResponse> {
  const body: SeedFromDetectorRequest = { dry_run: req.dry_run };
  if (req.names != null) body.names = req.names;
  if (req.group != null && req.group !== '') body.group = req.group;
  return apiFetch<SeedFromDetectorResponse>(
    `${scoped()}/classes/seed_from_detector`,
    { method: 'POST', body: JSON.stringify(body) },
    signal,
  );
}

/** `GET /ingest/policy`. */
export function getIngestPolicy(signal?: AbortSignal): Promise<IngestPolicy> {
  return apiFetch<IngestPolicy>(`${scoped()}/ingest/policy`, {}, signal);
}

/** `PUT /ingest/policy` (409 `revision_conflict`, 422 `detector_not_servable`,
 *  503 `detector_unavailable`). */
export function putIngestPolicy(
  req: IngestPolicyUpdate,
  signal?: AbortSignal,
): Promise<IngestPolicyPutResponse> {
  return apiFetch<IngestPolicyPutResponse>(
    `${scoped()}/ingest/policy`,
    { method: 'PUT', body: JSON.stringify(req) },
    signal,
  );
}

/** `POST /ingest/policy/preview`: what the draft would embed among the
 *  detections already stored. Never writes. */
export function previewIngestPolicy(
  body: IngestPolicyBody,
  signal?: AbortSignal,
): Promise<IngestPolicyPreview> {
  return apiFetch<IngestPolicyPreview>(
    `${scoped()}/ingest/policy/preview`,
    { method: 'POST', body: JSON.stringify(body) },
    signal,
  );
}

/** `GET /detections/summary` with the optional shared item filter. */
export function getDetectionsSummary(
  filter: ItemFilterQuery = {},
  signal?: AbortSignal,
): Promise<DetectionsSummary> {
  return apiFetch<DetectionsSummary>(
    `${scoped()}/detections/summary${qs({ ...filter })}`,
    {},
    signal,
  );
}

/** A served `{detail: {error, message, ...}}` refusal of the routes above. */
export interface DetectorErrorDetail {
  error: string;
  message: string;
  /** 422 `detector_not_servable`. */
  reasons?: string[];
  /** 422 `unknown_detector_names`. */
  unknown_names?: string[];
  /** 409 `revision_conflict`. */
  current_revision?: number;
}

export function detectorErrorDetail(e: unknown): DetectorErrorDetail | null {
  if (!(e instanceof ApiError)) return null;
  const body = e.body;
  if (!body || typeof body !== 'object') return null;
  const detail = (body as { detail?: unknown }).detail;
  if (!detail || typeof detail !== 'object' || Array.isArray(detail)) return null;
  const d = detail as Record<string, unknown>;
  if (typeof d.error !== 'string') return null;
  return {
    ...d,
    message: typeof d.message === 'string' ? d.message : d.error,
  } as DetectorErrorDetail;
}

/** The served refusal as lines to show verbatim: the message, then each
 *  served reason / unknown name. */
export function detectorErrorLines(e: unknown): string[] {
  const d = detectorErrorDetail(e);
  if (d) {
    return [
      d.message,
      ...(d.reasons ?? []),
      ...(d.unknown_names?.length ? [`Unknown: ${d.unknown_names.join(', ')}`] : []),
    ];
  }
  if (e instanceof ApiError && e.detail) return [e.detail];
  return [(e as Error)?.message ?? String(e)];
}
