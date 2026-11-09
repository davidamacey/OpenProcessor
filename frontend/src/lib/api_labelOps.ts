/**
 * Label-operation wrappers: the project's recorded bulk label and move
 * operations and their undo. Import from `$lib/api_labelOps` directly.
 * Types live in `$lib/types_labelOps`.
 */
import { apiFetch, qs, scoped } from '$lib/api';
import type {
  LabelOpAccepted,
  LabelOpList,
  LabelOpUndoResponse,
} from '$lib/types_labelOps';

/** `GET /label_ops`, newest first; `after` is the previous page's `next_after`. */
export function getLabelOps(
  params: { limit: number; after?: string | null },
  signal?: AbortSignal,
): Promise<LabelOpList> {
  return apiFetch<LabelOpList>(
    `${scoped()}/label_ops${qs({ limit: params.limit, after: params.after })}`,
    {},
    signal,
  );
}

/** `POST /label_ops/{op_id}/undo` with `dry_run: true`: counts what an undo
 *  would restore and keep, writes nothing (409 `already_undone`,
 *  `label_op_busy`, `label_op_in_progress`; 404 `label_op_not_found`). */
export function previewLabelOpUndo(
  opId: string,
  signal?: AbortSignal,
): Promise<LabelOpUndoResponse> {
  return apiFetch<LabelOpUndoResponse>(
    `${scoped()}/label_ops/${encodeURIComponent(opId)}/undo`,
    { method: 'POST', body: JSON.stringify({ dry_run: true }) },
    signal,
  );
}

/** `POST /label_ops/{op_id}/undo` with `dry_run: false`: queues the pass as a
 *  `label_undo` job (202); follow `job_id` with `undoLabelOpToEnd` or
 *  `runLabelOp`. */
export function queueLabelOpUndo(
  opId: string,
  signal?: AbortSignal,
): Promise<LabelOpAccepted> {
  return apiFetch<LabelOpAccepted>(
    `${scoped()}/label_ops/${encodeURIComponent(opId)}/undo`,
    { method: 'POST', body: JSON.stringify({ dry_run: false }) },
    signal,
  );
}
