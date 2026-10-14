/**
 * Region-box writes: per-item box lists, batch box state and batch region
 * status.
 */
import { apiFetch, assertNonEmptyBatch, scoped } from '$lib/api';
import { mapRawCrop, type RawCrop } from '$lib/api_cropMap';
import { ApiError } from '$lib/api_errors';
import { REGION_BASE, type RegionBrowseItem } from '$lib/api_regions';
import type { Crop } from '$lib/types';
import type { VectorRefresh } from '$lib/types_itemFilter';
import type { RegionBoxInput } from '$lib/annotations/multiBox';
import type { SlotSpec } from '$lib/annotations/types';

/* ------------------------------------------------------------------ */
/* Multi-box region writes. Element shapes are RegionBoxInput from      */
/* annotations/multiBox.ts.                                             */
/* ------------------------------------------------------------------ */

/** A per-item region write: the served post-write item and, when served,
 *  how many of its boxes got a vector (`pending` > 0: embed them later). */
export interface RegionItemWrite {
  crop: Crop;
  vectorRefresh: VectorRefresh | null;
}

/** `PUT /crops/{crop_id}/regions` — replaces the box list on one
 *  crop. `regionStatus` optionally applies a whole-set status to the
 *  built list in the same write (Enter-after-edit: `'detected'`). */
export async function putRegionBoxes(
  cropId: string,
  boxes: RegionBoxInput[],
  opts: { regionStatus?: string; expectedRegionRevision?: number } = {},
  signal?: AbortSignal,
): Promise<RegionItemWrite> {
  const body: Record<string, unknown> = {
    boxes,
    frame: 'parent',
    region_label_source: 'human',
  };
  if (opts.regionStatus != null) body.region_status = opts.regionStatus;
  if (opts.expectedRegionRevision != null) {
    body.expected_region_revision = opts.expectedRegionRevision;
  }
  const raw = await apiFetch<{ item: RawCrop; vector_refresh?: VectorRefresh | null }>(
    `${scoped()}/crops/${encodeURIComponent(cropId)}/regions`,
    { method: 'PUT', body: JSON.stringify(body) },
    signal,
  );
  return { crop: mapRawCrop(raw.item), vectorRefresh: raw.vector_refresh ?? null };
}

/** `PUT /crops/batch_regions` — replaces every listed crop's box
 *  list with the SAME new boxes (every element must be `box_id: null`;
 *  typically `boxes: []`, "none visible"). */
export async function putBatchRegions(
  cropIds: string[],
  boxes: Array<{ bbox_norm: [number, number, number, number]; state?: string }>,
  opts: { regionStatus?: string } = {},
  signal?: AbortSignal,
): Promise<{ vectorRefresh: VectorRefresh | null }> {
  assertNonEmptyBatch('batch region replace', cropIds);
  const body: Record<string, unknown> = { crop_ids: cropIds, boxes };
  if (opts.regionStatus != null) body.region_status = opts.regionStatus;
  const raw = await apiFetch<{ vector_refresh?: VectorRefresh | null }>(
    `${scoped()}/crops/batch_regions`,
    { method: 'PUT', body: JSON.stringify(body) },
    signal,
  );
  return { vectorRefresh: raw.vector_refresh ?? null };
}

/** `PATCH /crops/{crop_id}/regions/{box_id}` — per-box state/text
 *  flip. Used by the selected-box accept/reject keymap actions. */
export async function patchRegionBox(
  cropId: string,
  boxId: string,
  patch: { state?: string; text?: string | null; expectedRegionRevision?: number },
  signal?: AbortSignal,
): Promise<RegionItemWrite> {
  const body: Record<string, unknown> = {};
  if (patch.state != null) body.state = patch.state;
  if (patch.text !== undefined) body.text = patch.text;
  if (patch.expectedRegionRevision != null) {
    body.expected_region_revision = patch.expectedRegionRevision;
  }
  const raw = await apiFetch<{ item: RawCrop; vector_refresh?: VectorRefresh | null }>(
    `${scoped()}/crops/${encodeURIComponent(cropId)}/regions/${encodeURIComponent(boxId)}`,
    { method: 'PATCH', body: JSON.stringify(body) },
    signal,
  );
  return { crop: mapRawCrop(raw.item), vectorRefresh: raw.vector_refresh ?? null };
}

/** A stale `expected_region_revision` is `409 region_conflict` on the
 *  per-item box writes: the body carries the current revision, the
 *  current box ids and the current `item`, which the caller adopts
 *  instead of re-deriving anything. */
export interface RegionConflictDetail {
  currentRegionRevision: number;
  currentBoxIds: string[];
  item: Crop;
}

export function regionConflictDetail(e: unknown): RegionConflictDetail | null {
  if (!(e instanceof ApiError) || e.status !== 409) return null;
  const detail = (e.body as { detail?: unknown } | null)?.detail;
  if (!detail || typeof detail !== 'object') return null;
  const d = detail as Record<string, unknown>;
  if (d.error !== 'region_conflict' || !d.item || typeof d.item !== 'object') return null;
  return {
    currentRegionRevision:
      typeof d.current_region_revision === 'number' ? d.current_region_revision : 0,
    currentBoxIds: Array.isArray(d.current_box_ids)
      ? d.current_box_ids.filter((x): x is string => typeof x === 'string')
      : [],
    item: mapRawCrop(d.item as RawCrop),
  };
}

export interface RegionBatchConflict {
  crop_id: string;
  error: string;
  message: string;
  current_source: string | null;
  current_region_revision: number;
  current_box_ids: string[];
  item: RawCrop;
}

export interface RegionBatchBoxStateResult {
  updated: number;
  invalid: Array<{ crop_id: string; box_id: string; error: string; message: string }>;
  conflicts: RegionBatchConflict[];
  items: Crop[];
  vectorRefresh: VectorRefresh | null;
}

/** `POST /regions/batch_box_state` — one state on many boxes
 *  across items (region-gallery triage / a region cluster = a set of
 *  boxes). Never flips a whole item's other boxes — use `batchRegionStatus`
 *  for that. */
export async function postBatchBoxState(
  targets: Array<{ cropId: string; boxId: string }>,
  state: string,
  signal?: AbortSignal,
): Promise<RegionBatchBoxStateResult> {
  if (targets.length === 0)
    throw new Error('postBatchBoxState requires at least one target');
  type Raw = {
    updated: number;
    invalid?: Array<{ crop_id: string; box_id: string; error: string; message: string }>;
    conflicts?: RegionBatchConflict[];
    items?: RawCrop[];
    vector_refresh?: VectorRefresh | null;
  };
  const raw = await apiFetch<Raw>(
    `${scoped()}/regions/batch_box_state`,
    {
      method: 'POST',
      body: JSON.stringify({
        targets: targets.map((t) => ({ crop_id: t.cropId, box_id: t.boxId })),
        state,
        region_label_source: 'human',
      }),
    },
    signal,
  );
  return {
    updated: raw.updated,
    invalid: raw.invalid ?? [],
    conflicts: raw.conflicts ?? [],
    items: (raw.items ?? []).map(mapRawCrop),
    vectorRefresh: raw.vector_refresh ?? null,
  };
}

/**
 * Bulk-set region_status over many crops. Backend: `POST {API_PREFIX}/regions/batch_status`.
 * The cluster-view triage op: select outlier regions → mark all false_positive,
 * or bulk-confirm good regions (status='detected' + verified=true).
 *
 * Reads its path from `spec.endpoints.batchStatus`
 * rather than hardcoding `${REGION_BASE}/batch_status` a second time —
 * closes the "declared but dead" gap without importing a specific
 * profile into this generic module (falls back to the REGION_BASE path
 * if a spec declares no batchStatus endpoint, matching today's only caller).
 */
export interface BatchStatusInvalidEntry {
  crop_id: string;
  detail: string;
}

export async function batchRegionStatus(
  spec: SlotSpec,
  cropIds: string[],
  status: string,
  opts: { verified?: boolean; labelSource?: string } = {},
  signal?: AbortSignal,
): Promise<{
  updated: number;
  // One RegionBatchConflict shape across every region batch route: an older
  // backend serves only {crop_id, current_source}; a current backend
  // serves the full RegionBatchConflict. Typed as a superset
  // (every RegionBatchConflict field optional here) so both eras read
  // safely without a second type.
  conflicts: Array<{
    crop_id: string;
    current_source: string | null;
    error?: string;
    message?: string;
    current_region_revision?: number;
    current_box_ids?: string[];
    item?: RawCrop;
  }>;
  invalid: BatchStatusInvalidEntry[];
  items: RegionBrowseItem[];
}> {
  const path = spec.endpoints.batchStatus?.() ?? `${REGION_BASE}/batch_status`;
  const lc = spec.capabilities.lifecycle;
  if (!lc) {
    return Promise.reject(new Error(`slot "${spec.key}" has no lifecycle capability`));
  }
  const body: Record<string, unknown> = {
    crop_ids: cropIds,
    [lc.statusField]: status,
  };
  // p5 (2026-09-24 interactive pass): used to always send
  // `[verifiedField]: opts.verified ?? null`, so a bulk call that
  // never passed `verified` (e.g. reject/mark-false-positive) sent
  // an explicit `null` the server ignores. Omit the key entirely unless
  // the caller actually asked to set it.
  if (lc.verifiedField && opts.verified !== undefined) {
    body[lc.verifiedField] = opts.verified;
  }
  if (lc.labelSourceField) body[lc.labelSourceField] = opts.labelSource ?? 'human';
  return apiFetch(
    `${scoped()}${path}`,
    { method: 'POST', body: JSON.stringify(body) },
    signal,
  );
}
