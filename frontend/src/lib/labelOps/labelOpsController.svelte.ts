/**
 * `/label-ops`: the project's recorded bulk label and move operations, and
 * undo with a served dry run first (`GET /label_ops`, `POST
 * /label_ops/{id}/undo`).
 *
 * Thin: whether an operation is undoable, what an undo would restore and
 * what it would keep (changed, holdout, stale class, conflicts) are all
 * served. Nothing here counts or decides; it sequences dry run -> confirm.
 */
import { apiErrorText } from '$lib/api_errors';
import { getLabelOps, undoLabelOp } from '$lib/api_labelOps';
import {
  LABEL_OPS_MAX_LIMIT,
  type LabelOp,
  type LabelOpList,
  type LabelOpUndoResponse,
} from '$lib/types_labelOps';

export interface LabelOpsDeps {
  list: (limit?: number, signal?: AbortSignal) => Promise<LabelOpList>;
  undo: (
    opId: string,
    dryRun: boolean,
    signal?: AbortSignal,
  ) => Promise<LabelOpUndoResponse>;
}

const DEFAULT_DEPS: LabelOpsDeps = { list: getLabelOps, undo: undoLabelOp };

export const LABEL_OPS_PAGE_SIZE = 20;

/** idle: detail shown; dry: dry run in flight; ready: dry run shown, awaiting
 *  Confirm; writing: the real undo in flight; done: the real undo's result. */
export type UndoPhase = 'idle' | 'dry' | 'ready' | 'writing' | 'done';

export class LabelOpsController {
  loading = $state(true);
  loadError = $state<string | null>(null);
  items = $state<LabelOp[]>([]);
  limit = $state(LABEL_OPS_PAGE_SIZE);
  retention = $state<number | null>(null);

  /** The operation whose detail dialog is open. */
  selectedId = $state<string | null>(null);
  phase = $state<UndoPhase>('idle');
  undoError = $state<string | null>(null);
  /** The dry run (phase `ready`) or the real result (phase `done`). */
  result = $state<LabelOpUndoResponse | null>(null);

  #deps: LabelOpsDeps;

  constructor(deps: LabelOpsDeps = DEFAULT_DEPS) {
    this.#deps = deps;
  }

  selected = $derived(this.items.find((o) => o.op_id === this.selectedId) ?? null);

  /** More operations may exist beyond what the list shows. */
  canShowMore = $derived(
    this.items.length >= this.limit && this.limit < LABEL_OPS_MAX_LIMIT,
  );

  async load(signal?: AbortSignal): Promise<void> {
    this.loading = true;
    this.loadError = null;
    try {
      const res = await this.#deps.list(this.limit, signal);
      this.items = res.items;
      this.retention = res.retention;
    } catch (e) {
      if (signal?.aborted) return;
      this.loadError = apiErrorText(e);
    } finally {
      if (!signal?.aborted) this.loading = false;
    }
  }

  async showMore(): Promise<void> {
    this.limit = Math.min(this.limit + LABEL_OPS_PAGE_SIZE, LABEL_OPS_MAX_LIMIT);
    await this.load();
  }

  open(opId: string): void {
    this.selectedId = opId;
    this.#resetUndo();
  }

  close(): void {
    if (this.phase === 'dry' || this.phase === 'writing') return;
    this.selectedId = null;
    this.#resetUndo();
  }

  /** Step 1: ask the server what an undo would do. Writes nothing. */
  async startUndo(): Promise<void> {
    const id = this.selectedId;
    if (!id || this.phase === 'dry' || this.phase === 'writing') return;
    this.undoError = null;
    this.result = null;
    this.phase = 'dry';
    try {
      this.result = await this.#deps.undo(id, true);
      this.phase = 'ready';
    } catch (e) {
      this.undoError = apiErrorText(e);
      this.phase = 'idle';
    }
  }

  /** Step 2: only after a dry run was shown. Refreshes the list afterwards. */
  async confirmUndo(): Promise<void> {
    const id = this.selectedId;
    if (!id || this.phase !== 'ready') return;
    this.undoError = null;
    this.phase = 'writing';
    try {
      this.result = await this.#deps.undo(id, false);
      this.phase = 'done';
    } catch (e) {
      this.undoError = apiErrorText(e);
      this.phase = 'idle';
      this.result = null;
    }
    await this.load();
  }

  /** Back from the dry run without writing. */
  cancelUndo(): void {
    if (this.phase === 'ready') this.#resetUndo();
  }

  #resetUndo(): void {
    this.phase = 'idle';
    this.result = null;
    this.undoError = null;
  }
}
