/**
 * Item writes: label, discard, dismiss, the undo routes, slot metadata,
 * exclude and the needs-new-class flag.
 */
import { apiFetch, assertNonEmptyBatch, scoped } from '$lib/api';
import { mapRawCrop, type RawCrop } from '$lib/api_cropMap';
import type {
  BulkLabelConflict,
  BulkLabelResult,
  Crop,
  CropRegionUndoBatchResult,
  CropUndoBatchResult,
} from '$lib/types';
import type { SlotSpec } from '$lib/annotations/types';
import type { SelectionExcludeResult } from '$lib/types_itemFilter';

export function putCropLabel(
  cropId: string,
  classId: number,
  signal?: AbortSignal,
): Promise<Crop> {
  return apiFetch<Crop>(
    `${scoped()}/crops/${encodeURIComponent(cropId)}/label`,
    {
      method: 'PUT',
      body: JSON.stringify({ class_id: classId, validated: true }),
    },
    signal,
  );
}

export function bulkLabel(
  cropIds: string[],
  classId: number,
  signal?: AbortSignal,
): Promise<BulkLabelResult> {
  // Backend route is PUT (matches the single-crop /label PUT shape).
  return apiFetch<BulkLabelResult>(
    `${scoped()}/crops/batch_label`,
    {
      method: 'PUT',
      body: JSON.stringify({ crop_ids: cropIds, class_id: classId, validated: true }),
    },
    signal,
  );
}

/**
 * Undo the crop's most recent human class write. The backend restores
 * its own snapshot (earlier human label, VLM suggestion, ingest proposal
 * or unlabeled) and returns the restored item. `409` means nothing is
 * left to undo.
 */
export async function undoCropLabel(cropId: string, signal?: AbortSignal): Promise<Crop> {
  const raw = await apiFetch<RawCrop>(
    `${scoped()}/crops/${encodeURIComponent(cropId)}/label/undo`,
    { method: 'POST' },
    signal,
  );
  return mapRawCrop(raw);
}

/**
 * `POST {API_PREFIX}/crops/label/undo_batch` — batch form of
 * `undoCropLabel`: each crop is restored independently to its own state
 * before its most recent human class write, so undoing a `bulkLabel` or
 * `resolveNewClassProposal` means passing the same ids that write
 * reported in `updated_ids`. `409` when no crop in the batch had
 * anything to undo.
 */
export async function undoLabelBatch(
  cropIds: string[],
  signal?: AbortSignal,
): Promise<CropUndoBatchResult> {
  assertNonEmptyBatch('undo', cropIds);
  type Raw = {
    items?: RawCrop[];
    undone?: number;
    nothing_to_undo?: string[];
    conflicts?: string[];
    not_found?: string[];
  };
  const raw = await apiFetch<Raw>(
    `${scoped()}/crops/label/undo_batch`,
    { method: 'POST', body: JSON.stringify({ crop_ids: cropIds }) },
    signal,
  );
  return {
    items: (raw.items ?? []).map(mapRawCrop),
    undone: raw.undone ?? 0,
    nothing_to_undo: raw.nothing_to_undo ?? [],
    conflicts: raw.conflicts ?? [],
    not_found: raw.not_found ?? [],
  };
}

/** Options shared by the single and batch discard endpoints. */
export interface DiscardOptions {
  /** Clear the crop's class (it doesn't belong where it is). Default true. */
  clear_class?: boolean;
  /** Also stamp review_dismissed_at, so it never resurfaces in /review. Default false. */
  dismiss_from_review?: boolean;
}

/**
 * Discard a single crop. Recorded like a label write, so
 * `POST /crops/{id}/label/undo` reverses it — callers should push an
 * undo entry for the crop on success.
 */
export async function discardCrop(
  cropId: string,
  opts: DiscardOptions = {},
  signal?: AbortSignal,
): Promise<Crop> {
  const raw = await apiFetch<RawCrop>(
    `${scoped()}/crops/${encodeURIComponent(cropId)}/discard`,
    { method: 'POST', body: JSON.stringify(opts) },
    signal,
  );
  return mapRawCrop(raw);
}

export interface DiscardBatchResult {
  items: Crop[];
  discarded: number;
  conflicts: BulkLabelConflict[];
  not_found: string[];
}

/** Bulk variant of `discardCrop`. `items` carries only the crops actually
 *  discarded — conflicted/not-found ids are reported separately. */
export async function discardCropsBatch(
  cropIds: string[],
  opts: DiscardOptions = {},
  signal?: AbortSignal,
): Promise<DiscardBatchResult> {
  assertNonEmptyBatch('discard', cropIds);
  const raw = await apiFetch<{
    items: RawCrop[];
    discarded: number;
    conflicts: BulkLabelConflict[];
    not_found: string[];
  }>(
    `${scoped()}/crops/discard_batch`,
    { method: 'POST', body: JSON.stringify({ crop_ids: cropIds, ...opts }) },
    signal,
  );
  return { ...raw, items: raw.items.map(mapRawCrop) };
}

/**
 * Permanently dismiss a crop from every /review queue.
 *
 * Stamps ``review_dismissed_at`` + ``review_dismissed_by='human'`` on
 * the crop. The server's review_queue handler must_not's any crop with
 * ``review_dismissed_at``, so dismissed crops never reappear until
 * `reviewUndismissCrop` clears it. The crop's class / region state is
 * left intact — only review visibility changes.
 */
export function reviewDismissCrop(cropId: string, signal?: AbortSignal): Promise<void> {
  return apiFetch<void>(
    `${scoped()}/crops/${encodeURIComponent(cropId)}/review_dismiss`,
    { method: 'POST' },
    signal,
  );
}

/** Reverses `reviewDismissCrop`, returning the restored item. */
export async function reviewUndismissCrop(
  cropId: string,
  signal?: AbortSignal,
): Promise<Crop> {
  const raw = await apiFetch<RawCrop>(
    `${scoped()}/crops/${encodeURIComponent(cropId)}/review_undismiss`,
    { method: 'POST' },
    signal,
  );
  return mapRawCrop(raw);
}

/**
 * Dismiss the VLM's proposed class on a crop ("reject VLM suggestion").
 * The backend clears `vlm_proposed_class_*` and returns the updated
 * item. `409` means the crop had no suggestion to dismiss.
 */
export async function vlmDismissCrop(
  cropId: string,
  signal?: AbortSignal,
): Promise<Crop> {
  const raw = await apiFetch<RawCrop>(
    `${scoped()}/crops/${encodeURIComponent(cropId)}/vlm_dismiss`,
    { method: 'POST' },
    signal,
  );
  return mapRawCrop(raw);
}

/**
 * `POST {API_PREFIX}/crops/{id}/vlm_dismiss/undo` (M6, backend b654da5):
 * put `vlm_dismissed_*` back to its state before the latest
 * `vlm_dismiss`, so the dismissed suggestion is live again. `409` means
 * there was no dismissal to undo.
 */
export async function undoVlmDismiss(
  cropId: string,
  signal?: AbortSignal,
): Promise<Crop> {
  const raw = await apiFetch<RawCrop>(
    `${scoped()}/crops/${encodeURIComponent(cropId)}/vlm_dismiss/undo`,
    { method: 'POST' },
    signal,
  );
  return mapRawCrop(raw);
}

/**
 * `POST {API_PREFIX}/crops/{id}/region/undo` (M6, backend b654da5):
 * restore the region to its state before the most recent not-yet-undone
 * human region write (confirm, reject, false positive, box edit, status
 * or text change) — repeated calls step back further. Class fields are
 * untouched. `409` means nothing is left to undo.
 */
export async function undoCropRegion(
  cropId: string,
  signal?: AbortSignal,
): Promise<Crop> {
  const raw = await apiFetch<RawCrop>(
    `${scoped()}/crops/${encodeURIComponent(cropId)}/region/undo`,
    { method: 'POST' },
    signal,
  );
  return mapRawCrop(raw);
}

/**
 * `POST {API_PREFIX}/crops/region/undo_batch` — batch form of
 * `undoCropRegion`: each crop is restored independently, so undoing a
 * `PUT /crops/batch_region` or `POST /regions/batch_status` means passing
 * the same ids that write touched. `409` when no crop in the batch had
 * anything to undo.
 */
export async function undoCropRegionBatch(
  cropIds: string[],
  signal?: AbortSignal,
): Promise<CropRegionUndoBatchResult> {
  assertNonEmptyBatch('undo', cropIds);
  type Raw = {
    items?: RawCrop[];
    undone?: number;
    nothing_to_undo?: string[];
    conflicts?: string[];
    not_found?: string[];
  };
  const raw = await apiFetch<Raw>(
    `${scoped()}/crops/region/undo_batch`,
    { method: 'POST', body: JSON.stringify({ crop_ids: cropIds }) },
    signal,
  );
  return {
    items: (raw.items ?? []).map(mapRawCrop),
    undone: raw.undone ?? 0,
    nothing_to_undo: raw.nothing_to_undo ?? [],
    conflicts: raw.conflicts ?? [],
    not_found: raw.not_found ?? [],
  };
}

/**
 * PATCH a slot's item-level metadata (status / text / rejection reason)
 * without touching any box. The BODY KEYS are the spec's own wire field
 * names — for the region slot `region_status` / `region_rejection_reason`
 * only: its text is per box (no `text.valueField`), written with
 * `patchRegionBox`, and the backend rejects a `region_text` key. Keys
 * whose capability is absent, or whose value is `undefined` (as opposed
 * to `null`, which clears), are omitted.
 */
export async function patchSlotMeta(
  spec: SlotSpec,
  cropId: string,
  patch: {
    status?: string | null;
    text?: string | null;
    rejectionReason?: string | null;
  },
  signal?: AbortSignal,
): Promise<{ crop_id: string; updated_fields: string[]; item: Crop }> {
  const cap = spec.capabilities;
  const body: Record<string, unknown> = {};
  if (patch.status !== undefined && cap.lifecycle?.statusField) {
    body[cap.lifecycle.statusField] = patch.status;
  }
  if (patch.text !== undefined && cap.text?.valueField) {
    body[cap.text.valueField] = patch.text;
  }
  if (patch.rejectionReason !== undefined && cap.lifecycle?.rejectionReasonField) {
    body[cap.lifecycle.rejectionReasonField] = patch.rejectionReason;
  }
  const path = spec.endpoints.patchMeta?.(cropId);
  if (!path) {
    return Promise.reject(new Error(`slot "${spec.key}" has no patchMeta endpoint`));
  }
  const res = await apiFetch<{
    crop_id: string;
    updated_fields: string[];
    item: RawCrop;
  }>(`${scoped()}${path}`, { method: 'PATCH', body: JSON.stringify(body) }, signal);
  return { ...res, item: mapRawCrop(res.item) };
}

// -- exclude / ignore crops ----------------------------------------------

/** Reasons the operator can attach when ignoring crops. 'ignore' is the
 *  default one-click value; the others record *why* for later analysis. */
export type ExcludeReason =
  'ignore' | 'blurry' | 'unidentifiable' | 'not_the_subject' | 'partial_crop';

export function excludeCrops(
  cropIds: string[],
  reason: ExcludeReason = 'ignore',
  signal?: AbortSignal,
): Promise<SelectionExcludeResult & { excluded: number }> {
  return apiFetch<SelectionExcludeResult & { excluded: number }>(
    `${scoped()}/crops/batch_exclude`,
    {
      method: 'POST',
      body: JSON.stringify({ crop_ids: cropIds, reason }),
    },
    signal,
  );
}

export function unexcludeCrops(
  cropIds: string[],
  signal?: AbortSignal,
): Promise<SelectionExcludeResult & { unexcluded: number }> {
  return apiFetch<SelectionExcludeResult & { unexcluded: number }>(
    `${scoped()}/crops/batch_unexclude`,
    {
      method: 'POST',
      body: JSON.stringify({ crop_ids: cropIds }),
    },
    signal,
  );
}

// -- needs new class flag ------------------------------------------------

export function flagNeedsNewClass(
  cropIds: string[],
  note: string = '',
  signal?: AbortSignal,
): Promise<{ flagged: number; errors: number }> {
  return apiFetch<{ flagged: number; errors: number }>(
    `${scoped()}/crops/flag_new_class`,
    {
      method: 'POST',
      body: JSON.stringify({ crop_ids: cropIds, note }),
    },
    signal,
  );
}
