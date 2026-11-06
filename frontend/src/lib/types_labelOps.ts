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
  kind: 'label' | 'move' | 'assist_loop_propagate' | 'assist_loop_review';
  at: string;
  finished_at: string | null;
  class_name: string | null;
  target_cluster_id: number | null;
  target_kind: 'class' | 'candidate' | null;
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

/** `LabelOpList`. */
export interface LabelOpList {
  items: LabelOp[];
  limit: number;
  retention: number;
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
