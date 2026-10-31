/**
 * Label-operation wrappers: the project's recorded bulk label and move
 * operations and their undo. Import from `$lib/api_labelOps` directly.
 * Types live in `$lib/types_labelOps`.
 */
import { apiFetch, qs, scoped } from '$lib/api';
import type { LabelOpList, LabelOpUndoResponse } from '$lib/types_labelOps';

/** `GET /label_ops`, newest first. */
export function getLabelOps(limit?: number, signal?: AbortSignal): Promise<LabelOpList> {
  return apiFetch<LabelOpList>(`${scoped()}/label_ops${qs({ limit })}`, {}, signal);
}

/** `POST /label_ops/{op_id}/undo`. The body always names `dry_run`: true
 *  counts what would happen and writes nothing, false writes (409
 *  `already_undone`, `label_op_busy`, `label_op_in_progress`; 404
 *  `label_op_not_found`). */
export function undoLabelOp(
  opId: string,
  dryRun: boolean,
  signal?: AbortSignal,
): Promise<LabelOpUndoResponse> {
  return apiFetch<LabelOpUndoResponse>(
    `${scoped()}/label_ops/${encodeURIComponent(opId)}/undo`,
    { method: 'POST', body: JSON.stringify({ dry_run: dryRun }) },
    signal,
  );
}
