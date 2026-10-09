/**
 * Label-confirmation wrappers (#119): the per-project VLM scope policy and
 * the detector/VLM accuracy audit. Import from `$lib/api_labelConfirmation`
 * directly; never re-exported from `api.ts` (that would make the two
 * modules circular). Types live in `$lib/types_labelConfirmation`.
 */
import { apiFetch, mapRawCrop, qs, scoped, type RawCrop } from '$lib/api';
import type { Crop } from '$lib/types';
import type {
  AuditReport,
  AuditStartRequest,
  AuditStartResponse,
  VlmPolicy,
  VlmPolicyUpdate,
} from '$lib/types_labelConfirmation';

/** `GET /vlm/policy`. */
export function getVlmPolicy(signal?: AbortSignal): Promise<VlmPolicy> {
  return apiFetch<VlmPolicy>(`${scoped()}/vlm/policy`, {}, signal);
}

/** `PUT /vlm/policy` (409 `revision_conflict`). Takes effect on the VLM
 *  worker's next poll and on the next auto-label run. */
export function putVlmPolicy(
  req: VlmPolicyUpdate,
  signal?: AbortSignal,
): Promise<VlmPolicy> {
  return apiFetch<VlmPolicy>(
    `${scoped()}/vlm/policy`,
    { method: 'PUT', body: JSON.stringify(req) },
    signal,
  );
}

/** `POST /audit/start` (409 `audit_no_candidates`). Only the fields the
 *  operator set are sent; the server's defaults apply to the rest. */
export function startAudit(
  req: AuditStartRequest,
  signal?: AbortSignal,
): Promise<AuditStartResponse> {
  const body: AuditStartRequest = {};
  if (req.min_per_class != null) body.min_per_class = req.min_per_class;
  if (req.sample_size != null) body.sample_size = req.sample_size;
  return apiFetch<AuditStartResponse>(
    `${scoped()}/audit/start`,
    { method: 'POST', body: JSON.stringify(body) },
    signal,
  );
}

/** `GET /audit/report`; `min_per_class` is sent only when set. */
export function getAuditReport(
  minPerClass?: number | null,
  signal?: AbortSignal,
): Promise<AuditReport> {
  return apiFetch<AuditReport>(
    `${scoped()}/audit/report${qs({ min_per_class: minPerClass ?? undefined })}`,
    {},
    signal,
  );
}

export interface AuditQueuePage {
  items: Crop[];
  total: number;
  page: number;
  pageSize: number;
}

/** `GET /audit/queue`: drawn crops still waiting for a human label. */
export async function getAuditQueue(
  page = 1,
  pageSize = 30,
  batchId?: string | null,
  signal?: AbortSignal,
): Promise<AuditQueuePage> {
  const res = await apiFetch<{
    items: RawCrop[];
    total: number;
    page: number;
    page_size: number;
  }>(
    `${scoped()}/audit/queue${qs({ page, page_size: pageSize, batch_id: batchId ?? undefined })}`,
    {},
    signal,
  );
  return {
    items: res.items.map(mapRawCrop),
    total: res.total,
    page: res.page,
    pageSize: res.page_size,
  };
}
