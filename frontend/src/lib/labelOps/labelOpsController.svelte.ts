/**
 * `/label-ops`: the project's recorded bulk label and move operations, and
 * undo with a served dry run first (`GET /label_ops`, `POST
 * /label_ops/{id}/undo`).
 *
 * Thin: whether an operation is undoable, what an undo would restore and
 * what it would keep (changed, holdout, stale class, conflicts) are all
 * served. Nothing here counts or decides; it sequences dry run -> confirm.
 * The list pages by the served `next_after` (Load older), newest first.
 */
import { apiErrorText } from '$lib/api_errors';
import { getLabelOps, undoLabelOp } from '$lib/api_labelOps';
import {
  LABEL_OPS_PAGE_SIZE,
  type LabelOp,
  type LabelOpList,
  type LabelOpUndoResponse,
} from '$lib/types_labelOps';

export interface LabelOpsDeps {
  list: (
    params: { limit: number; after?: string | null },
    signal?: AbortSignal,
  ) => Promise<LabelOpList>;
  undo: (
    opId: string,
    dryRun: boolean,
    signal?: AbortSignal,
  ) => Promise<LabelOpUndoResponse>;
}

const DEFAULT_DEPS: LabelOpsDeps = { list: getLabelOps, undo: undoLabelOp };

/** idle: detail shown; dry: dry run in flight; ready: dry run shown, awaiting
 *  Confirm; writing: the real undo in flight; done: the real undo's result. */
export type UndoPhase = 'idle' | 'dry' | 'ready' | 'writing' | 'done';

export class LabelOpsController {
  loading = $state(true);
  loadError = $state<string | null>(null);
  items = $state<LabelOp[]>([]);
  retention = $state<number | null>(null);
  /** Pass as `after` for the next older page; null at the end of the kept history. */
  nextAfter = $state<string | null>(null);
  loadingOlder = $state(false);
  olderError = $state<string | null>(null);

  /** The operation whose detail dialog is open. */
  selectedId = $state<string | null>(null);
  phase = $state<UndoPhase>('idle');
  undoError = $state<string | null>(null);
  /** The dry run (phase `ready`) or the real result (phase `done`). */
  result = $state<LabelOpUndoResponse | null>(null);

  #deps: LabelOpsDeps;
  /** Bumped by every reload so an older page that lands afterwards is dropped. */
  #gen = 0;

  constructor(deps: LabelOpsDeps = DEFAULT_DEPS) {
    this.#deps = deps;
  }

  selected = $derived(this.items.find((o) => o.op_id === this.selectedId) ?? null);

  canLoadOlder = $derived(this.nextAfter !== null);

  async load(signal?: AbortSignal): Promise<void> {
    this.#gen++;
    this.loading = true;
    this.loadError = null;
    try {
      const res = await this.#deps.list({ limit: LABEL_OPS_PAGE_SIZE }, signal);
      this.items = res.items;
      this.retention = res.retention;
      this.nextAfter = res.next_after;
      this.olderError = null;
    } catch (e) {
      if (signal?.aborted) return;
      this.loadError = apiErrorText(e);
    } finally {
      if (!signal?.aborted) this.loading = false;
    }
  }

  /** Append the next older page. A failure keeps the list and the cursor so the button retries. */
  async loadOlder(): Promise<void> {
    const after = this.nextAfter;
    if (!after || this.loadingOlder || this.loading) return;
    const gen = this.#gen;
    this.loadingOlder = true;
    this.olderError = null;
    try {
      const res = await this.#deps.list({ limit: LABEL_OPS_PAGE_SIZE, after });
      if (gen !== this.#gen) return;
      this.items = [...this.items, ...res.items];
      this.nextAfter = res.next_after;
    } catch (e) {
      if (gen === this.#gen) this.olderError = apiErrorText(e);
    } finally {
      this.loadingOlder = false;
    }
  }

  /** Re-serve every shown operation (an undo can change other operations' `current`). */
  async #reloadShown(): Promise<void> {
    const shown = this.items.length;
    const gen = ++this.#gen;
    this.loading = true;
    this.loadError = null;
    try {
      const items: LabelOp[] = [];
      let after: string | null = null;
      let retention = this.retention;
      do {
        const res: LabelOpList = await this.#deps.list(
          after === null
            ? { limit: LABEL_OPS_PAGE_SIZE }
            : { limit: LABEL_OPS_PAGE_SIZE, after },
        );
        items.push(...res.items);
        retention = res.retention;
        after = res.next_after;
      } while (after !== null && items.length < shown);
      if (gen !== this.#gen) return;
      this.items = items;
      this.retention = retention;
      this.nextAfter = after;
      this.olderError = null;
    } catch (e) {
      if (gen === this.#gen) this.loadError = apiErrorText(e);
    } finally {
      if (gen === this.#gen) this.loading = false;
    }
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
    await this.#reloadShown();
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
