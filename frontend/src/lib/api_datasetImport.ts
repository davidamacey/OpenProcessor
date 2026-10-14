/**
 * Labeled-dataset import: formats, upload, preview, jobs, issues, entries,
 * cancel, resume and undo (`/datasets/*`).
 */
import { apiFetch, qs, scoped } from '$lib/api';
import { structuredDetail } from '$lib/api_errors';
import type {
  DatasetErrorDetail,
  DatasetFormatsResponse,
  DatasetImportEntryPage,
  DatasetImportJob,
  DatasetImportList,
  DatasetImportRequest,
  DatasetIssuePage,
  DatasetPreview,
  DatasetPreviewRequest,
  DatasetUndoReport,
  DatasetUndoRequest,
  DatasetUploadResponse,
} from '$lib/types_import';

// -- labeled-dataset import and Reprocess --------------------------------
//
// Every route is scoped to the
// active project (the import's target is the path's project; no body
// carries `project`). A backend without dataset import 404s `GET /datasets/formats`,
// which `datasetsAvailability` treats as "not deployed yet".

export function getDatasetFormats(signal?: AbortSignal): Promise<DatasetFormatsResponse> {
  return apiFetch<DatasetFormatsResponse>(`${scoped()}/datasets/formats`, {}, signal);
}

/** `POST /datasets/uploads` — one multipart `file` (a .zip/.tar/.tar.gz),
 *  streamed server-side; the response's `dataset_path` is what the
 *  preview then reads. */
export function uploadDatasetArchive(
  file: File,
  signal?: AbortSignal,
): Promise<DatasetUploadResponse> {
  const fd = new FormData();
  fd.append('file', file, file.name);
  return apiFetch<DatasetUploadResponse>(
    `${scoped()}/datasets/uploads`,
    { method: 'POST', body: fd },
    signal,
  );
}

/** Dry run: writes nothing. Dataset problems come back as `issues`,
 *  never as a 4xx (only a malformed body or a disallowed root 422s). */
export function previewDataset(
  body: DatasetPreviewRequest,
  signal?: AbortSignal,
): Promise<DatasetPreview> {
  return apiFetch<DatasetPreview>(
    `${scoped()}/datasets/preview`,
    { method: 'POST', body: JSON.stringify(body) },
    signal,
  );
}

/** 202 with a new job, or 200 with the existing one (`reused: true`). */
export function startDatasetImport(
  body: DatasetImportRequest,
  signal?: AbortSignal,
): Promise<DatasetImportJob> {
  return apiFetch<DatasetImportJob>(
    `${scoped()}/datasets/imports`,
    { method: 'POST', body: JSON.stringify(body) },
    signal,
  );
}

export function listDatasetImports(
  params: { page?: number; page_size?: number; status?: string | null } = {},
  signal?: AbortSignal,
): Promise<DatasetImportList> {
  return apiFetch<DatasetImportList>(
    `${scoped()}/datasets/imports${qs({
      page: params.page,
      page_size: params.page_size,
      status: params.status,
    })}`,
    {},
    signal,
  );
}

export function getDatasetImport(
  importId: string,
  signal?: AbortSignal,
): Promise<DatasetImportJob> {
  return apiFetch<DatasetImportJob>(
    `${scoped()}/datasets/imports/${encodeURIComponent(importId)}`,
    {},
    signal,
  );
}

export function getDatasetImportIssues(
  importId: string,
  params: { code?: string | null; page?: number; page_size?: number } = {},
  signal?: AbortSignal,
): Promise<DatasetIssuePage> {
  return apiFetch<DatasetIssuePage>(
    `${scoped()}/datasets/imports/${encodeURIComponent(importId)}/issues${qs({
      code: params.code,
      page: params.page,
      page_size: params.page_size,
    })}`,
    {},
    signal,
  );
}

export function getDatasetImportEntries(
  importId: string,
  params: {
    split?: string | null;
    label_state?: string | null;
    status?: string | null;
    page?: number;
    page_size?: number;
  } = {},
  signal?: AbortSignal,
): Promise<DatasetImportEntryPage> {
  return apiFetch<DatasetImportEntryPage>(
    `${scoped()}/datasets/imports/${encodeURIComponent(importId)}/entries${qs({
      split: params.split,
      label_state: params.label_state,
      status: params.status,
      page: params.page,
      page_size: params.page_size,
    })}`,
    {},
    signal,
  );
}

export function cancelDatasetImport(
  importId: string,
  signal?: AbortSignal,
): Promise<DatasetImportJob> {
  return apiFetch<DatasetImportJob>(
    `${scoped()}/datasets/imports/${encodeURIComponent(importId)}/cancel`,
    { method: 'POST' },
    signal,
  );
}

export function resumeDatasetImport(
  importId: string,
  signal?: AbortSignal,
): Promise<DatasetImportJob> {
  return apiFetch<DatasetImportJob>(
    `${scoped()}/datasets/imports/${encodeURIComponent(importId)}/resume`,
    { method: 'POST' },
    signal,
  );
}

/** Dry run → `DatasetUndoReport`; apply → 202 `DatasetImportJob`
 *  (`status: undoing`). */
export function undoDatasetImport(
  importId: string,
  body: DatasetUndoRequest,
  signal?: AbortSignal,
): Promise<DatasetUndoReport | DatasetImportJob> {
  return apiFetch<DatasetUndoReport | DatasetImportJob>(
    `${scoped()}/datasets/imports/${encodeURIComponent(importId)}/undo`,
    { method: 'POST', body: JSON.stringify(body) },
    signal,
  );
}

/**
 * The structured dataset-import refusal (`{detail: ConfigErrorDetail}` with the
 * optional `issues`/`unmapped`/`import_id`), or `null` when the error
 * isn't one. The UI shows `message` and branches only on `error`.
 */
export const datasetErrorDetail = (e: unknown): DatasetErrorDetail | null =>
  structuredDetail<DatasetErrorDetail>(e);
