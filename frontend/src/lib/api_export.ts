/**
 * Dataset export, single-class export, the test holdout and the download URLs.
 */
import { apiBase, apiFetch, qs, scoped } from '$lib/api';
import type { ItemFilter } from '$lib/types_itemFilter';
import type {
  ExportDatasetList,
  ExportResult,
  ExportStatus,
  SingleClassExportResult,
  SingleClassExportStatus,
  TestHoldoutFreezeResult,
  TestHoldoutStats,
} from '$lib/types_export';
import type { DatasetExportSpec } from '$lib/annotations/datasetExport';

/**
 * Kick off a YOLO export job. Optional `version_tag` is included in the
 * manifest (Section L14). Server returns 202 + job id while it runs.
 */
export function exportYolo(
  opts: {
    version_tag?: string;
    require_fully_labeled_images?: boolean;
    /** The shared item filter; sent only when it names something. */
    item_filter?: ItemFilter;
  } = {},
  signal?: AbortSignal,
): Promise<ExportResult> {
  const body: Record<string, unknown> = {};
  if (opts.item_filter && Object.keys(opts.item_filter).length > 0) {
    body.item_filter = opts.item_filter;
  }
  if (opts.version_tag) body.version_tag = opts.version_tag;
  // Default is false server-side — only send it when the operator opted
  // in, so the payload stays minimal for the common case.
  if (opts.require_fully_labeled_images) {
    body.require_fully_labeled_images = true;
  }
  return apiFetch<ExportResult>(
    `${scoped()}/export/yolo`,
    { method: 'POST', body: JSON.stringify(body) },
    signal,
  );
}

/** Poll current export state. */
export function exportStatus(signal?: AbortSignal): Promise<ExportStatus> {
  return apiFetch<ExportStatus>(`${scoped()}/export/status`, {}, signal);
}

/** Options an operator picks per single-class export build. */
export interface SingleClassExportOptions {
  version_tag?: string;
  empty_bg_ratio?: number;
  max_positive_images?: number;
  skip_test_split?: boolean;
  dedup_threshold?: number | null;
  image_mode?: 'whole_frame' | 'item_crop';
  img_max_side?: 640 | 1280;
}

/**
 * Build a narrowed single-class dataset from a slot's declared export
 * (`DatasetExportSpec`). Synchronous on the server; returns the export
 * dir + counts when done. Only ever called behind
 * `isDatasetExportAvailable(…, spec.kind)` — see `/train`'s gate.
 */
export function exportSingleClass(
  spec: DatasetExportSpec,
  opts: SingleClassExportOptions = {},
  signal?: AbortSignal,
): Promise<SingleClassExportResult> {
  const body: Record<string, unknown> = {
    profile_name: spec.profileName,
    box_source: spec.boxSource,
    class_ids: spec.classIds,
  };
  if (spec.regionClassName) body.region_class_name = spec.regionClassName;
  for (const [k, v] of Object.entries(opts)) {
    if (v !== undefined) body[k] = v;
  }
  return apiFetch<SingleClassExportResult>(
    `${scoped()}${spec.buildPath}`,
    { method: 'POST', body: JSON.stringify(body) },
    signal,
  );
}

/** Last build of a slot's single-class export (its own `current` symlink). */
export function exportSingleClassStatus(
  spec: DatasetExportSpec,
  signal?: AbortSignal,
): Promise<SingleClassExportStatus> {
  return apiFetch<SingleClassExportStatus>(
    `${scoped()}${spec.statusPath}${qs({ profile_name: spec.profileName })}`,
    {},
    signal,
  );
}

/**
 * List every materialized dataset version on disk (newest first) so the
 * operator can train on any past export — a small sample, a larger subset, or
 * the full set — for consistent data re-use across model-size upgrades.
 */
export function listDatasets(
  filter: { kind?: string; profile_name?: string } = {},
  signal?: AbortSignal,
): Promise<ExportDatasetList> {
  return apiFetch<ExportDatasetList>(
    `${scoped()}/export/datasets${qs(filter)}`,
    {},
    signal,
  );
}

// -- test holdout --------------------------------------------------------

/**
 * `POST {API_PREFIX}/test_holdout/freeze` — deterministic (SHA1-of-
 * `crop_id` per class), so the body is `{percent}` only as of
 * OpenProcessor df01309; an extra field such as `seed` is a 422
 * (`additionalProperties: false`). `force` re-runs an existing freeze
 * and is a query param, not a body field.
 */
export function freezeTestHoldout(
  payload: { percent: number; force?: boolean },
  signal?: AbortSignal,
): Promise<TestHoldoutFreezeResult> {
  return apiFetch<TestHoldoutFreezeResult>(
    `${scoped()}/test_holdout/freeze${qs({ force: payload.force ? true : undefined })}`,
    { method: 'POST', body: JSON.stringify({ percent: payload.percent }) },
    signal,
  );
}

export function getTestHoldoutStats(signal?: AbortSignal): Promise<TestHoldoutStats> {
  return apiFetch<TestHoldoutStats>(`${scoped()}/test_holdout/stats`, {}, signal);
}

// -- registry/manifest downloads (used as anchor `download` URLs) --------

export function getClassRegistryUrl(): string {
  return `${apiBase}${scoped()}/export/registry/class_registry.json`;
}

export function getDataYamlUrl(): string {
  return `${apiBase}${scoped()}/export/registry/data.yaml`;
}

export function getManifestUrl(): string {
  return `${apiBase}${scoped()}/export/registry/manifest.json`;
}
