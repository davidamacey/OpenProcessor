// Label operations: wire types for the `LabelOp*` schemas of the contract
// (`contracts/openapi/curation.json`), pinned key for key by
// `contract/labelOpsContract.test.ts`.

export const LABEL_OP_STATUSES = [
  'writing',
  'interrupted',
  'written',
  'undoing',
  'undo_interrupted',
  'undone',
] as const;
export type LabelOpStatus = (typeof LABEL_OP_STATUSES)[number];

/** `LabelOpUndoPass`: one completed undo pass. */
export interface LabelOpUndoPass {
  at: string;
  restored: number;
  already_undone: number;
  kept_changed: number;
  kept_holdout: number;
  kept_stale_class: number;
  conflicts: number;
}

/** `LabelOp`: one recorded bulk label or move. Class identity is the name. */
export interface LabelOp {
  op_id: string;
  kind:
    'label' | 'move' | 'assist_loop_propagate' | 'assist_loop_review' | 'adopt_detector';
  at: string;
  finished_at: string | null;
  class_name: string | null;
  target_cluster_uid: string | null;
  label_source: string | null;
  target: 'crop_ids' | 'selection';
  selected: number;
  count: number | null;
  conflicts: number | null;
  current: number;
  status: LabelOpStatus;
  undone: boolean;
  undoable: boolean;
  last_undo: LabelOpUndoPass | null;
}

/** `LabelOpAccepted`: the 202 of a queued relabel, move or undo. Follow
 *  `job_id` at `GET /jobs/{job_id}`; its `result` is `{op_id, updated, conflicts,
 *  complete}` for a relabel or move. An empty selection is answered with all
 *  three fields null (nothing was queued). */
export interface LabelOpAccepted {
  op_id: string | null;
  job_id: string | null;
  status: string | null;
}

/** `CropLabelResponse`: `PUT /crops/{id}/label`. The single write is recorded as a label
 *  operation in the same request; Z undoes it by `op_id` like a bulk write. */
export interface CropLabelWritten {
  crop_id: string;
  class_id: number;
  class_name: string;
  op_id: string;
}

/** `LabelOpList`. */
export interface LabelOpList {
  items: LabelOp[];
  limit: number;
  /** Bulk operations kept. */
  retention: number;
  /** Single label writes kept, counted apart from the bulk ones. */
  single_retention: number;
  next_after: string | null;
}

/** `LabelOpUndoRequest`. */
export interface LabelOpUndoRequest {
  dry_run: boolean;
}

/** `LabelOpUndoResponse`: what an undo pass did, or for a dry run would do. */
export interface LabelOpUndoResponse {
  op_id: string;
  dry_run: boolean;
  status: string;
  matched: number;
  restored: number;
  already_undone: number;
  kept_changed: number;
  kept_holdout: number;
  kept_stale_class: number;
  conflicts: number;
  conflict_ids: string[];
}

/** Operations per `GET /label_ops` page (within the served `limit` range, pinned by the contract test). */
export const LABEL_OPS_PAGE_SIZE = 20;
