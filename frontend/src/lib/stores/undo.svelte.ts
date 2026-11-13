/**
 * UndoStore — ring buffer of the last 50 human write *actions*, spanning
 * kinds (`UndoEntry.kind`): single and bulk label writes and moves, all recorded by the
 * server as label operations (`'label_op'`, undone by operation id, so the same undo is in
 * the served Label history after a reload), discards and new-class resolves (`'label'`,
 * default, undone per crop through the class history), region writes (`'region'`, M6), VLM-suggestion dismissals
 * (`'vlm_dismiss'`, M6/V1 undo) and ignore / restore writes (`'exclude'`
 * / `'unexclude'`, the selection actions). One entry per confirmed write, however
 * many crops it touched — a bulk label, a move, a batch region status
 * change, or a new-class-proposal resolve over N crops is ONE entry, so
 * one Z reverses the whole action. Calling code records what a confirmed
 * write touched via `recordLabelOp`/`recordWrites`/`recordRegionWrites`/
 * `recordVlmDismiss`; Z pops the newest entry (across all three kinds)
 * and asks the backend to undo it on the route matching its kind
 * (single-crop or batch, depending on how many ids it holds). The
 * backend owns what "undo" restores and returns the restored item(s) for
 * the page to render.
 *
 * Kept as ONE stack with a `kind` tag rather than a separate stack per
 * kind: on `/clusters/[id]` a label write and a Reject-VLM dismiss can
 * happen back to back in the same session, and Z must reverse whichever
 * one actually happened last — a per-kind stack can't express that
 * ordering without the caller tracking a second, parallel "what kind was
 * most recent" fact itself, which is exactly the bug class this ring
 * buffer exists to avoid. (The page-scoped ignore/un-ignore history in
 * `clusterController` stays a genuinely separate, local stack — it's
 * bound to its own `X`/`U` keys, never `Z`, so there's no cross-kind
 * ordering to get right by sharing this one.)
 */

import { activeProjectKey } from '$lib/api';
import { getCropsByIds } from '$lib/api_cropRead';
import { undoLabelOpToEnd } from '$lib/labelOps/runLabelOp';
import {
  undoCropLabel,
  undoCropRegion,
  undoCropRegionBatch,
  undoLabelBatch,
  undoProbeDismiss,
  undoVlmDismiss,
  excludeCrops,
  unexcludeCrops,
} from '$lib/api_cropWrites';
import { ApiError, apiErrorText } from '$lib/api_errors';
import { onProjectChange } from '$lib/projectChange';
import { toastStore } from '$stores/toast.svelte';
import type { Crop, UndoEntry } from '$lib/types';

const MAX = 50;

async function revertLabelOp(entry: UndoEntry): Promise<Crop[]> {
  const out = await undoLabelOpToEnd(entry.op_id!, false);
  toastStore.success(
    `Reverted ${out.restored}${out.kept_changed > 0 ? ` (${out.kept_changed} changed since, kept)` : ''}.`,
  );
  return entry.crop_ids.length > 0 ? await getCropsByIds(entry.crop_ids) : [];
}

async function revertExclusion(entry: UndoEntry, kind: 'exclude' | 'unexclude') {
  if (kind === 'exclude') await unexcludeCrops(entry.crop_ids);
  else await excludeCrops(entry.crop_ids, 'ignore');
  toastStore.success(
    kind === 'exclude'
      ? `Restored ${entry.crop_ids.length}.`
      : `Ignored ${entry.crop_ids.length} again.`,
  );
  return [];
}

async function revertBatch(ids: string[], region: boolean): Promise<Crop[]> {
  const res = region ? await undoCropRegionBatch(ids) : await undoLabelBatch(ids);
  const parts = [`Reverted ${res.undone}.`];
  if (res.nothing_to_undo.length > 0) {
    parts.push(`${res.nothing_to_undo.length} nothing to undo.`);
  }
  if (res.conflicts.length > 0) {
    parts.push(`${res.conflicts.length} conflict(s).`);
  }
  toastStore.success(parts.join(' '));
  return res.items;
}

/** Undo one entry on the route matching its kind; throws on failure. */
async function revertEntry(entry: UndoEntry): Promise<Crop[]> {
  const kind = entry.kind ?? 'label';
  if (kind === 'label_op') return revertLabelOp(entry);
  if (kind === 'exclude' || kind === 'unexclude') return revertExclusion(entry, kind);
  if (kind === 'vlm_dismiss') {
    const crop = await undoVlmDismiss(entry.crop_ids[0]!);
    toastStore.success('VLM suggestion restored.');
    return [crop];
  }
  if (kind === 'probe_dismiss') {
    const crop = await undoProbeDismiss(entry.crop_ids[0]!);
    toastStore.success('Back in Disagreements.');
    return [crop];
  }
  const region = kind === 'region';
  if (entry.crop_ids.length === 1) {
    const id = entry.crop_ids[0]!;
    const crop = region ? await undoCropRegion(id) : await undoCropLabel(id);
    toastStore.success('Reverted.');
    return [crop];
  }
  return revertBatch(entry.crop_ids, region);
}

/** `UndoEntry` tagged with the project it was recorded under. Crop ids
 *  are content-derived (the same image gets the same `crop_id` in every
 *  project — projects_plan §2.1), so a bare crop id isn't enough to know
 *  which project's write an entry reverts. Internal to this store: every
 *  push/pop below adds/drops the tag; `UndoEntry`'s own shape (what the
 *  rest of the app reads/constructs) is unchanged. */
type ScopedUndoEntry = UndoEntry & { project: string };

class UndoStore {
  // $state.raw, not $state: deep reactivity would wrap every pushed entry
  // in a Proxy, so `remove()` could never match the raw object the caller
  // still holds. Every mutation below reassigns the array, so raw state is
  // just as reactive for readers.
  stack = $state.raw<ScopedUndoEntry[]>([]);

  push(entry: UndoEntry): void {
    // Mutate-and-tag rather than spread into a new object: `remove()`
    // matches by identity against whatever a caller passed to `push`, so
    // a fresh copy here would silently break that contract for any
    // caller holding onto the entry it pushed.
    const scoped = Object.assign(entry, { project: activeProjectKey() });
    const next = [...this.stack, scoped];
    if (next.length > MAX) next.shift();
    this.stack = next;
  }

  pop(): ScopedUndoEntry | undefined {
    if (this.stack.length === 0) return undefined;
    const next = [...this.stack];
    const entry = next.pop();
    this.stack = next;
    return entry;
  }

  /**
   * Drop specific entries by identity.
   *
   * Revert paths must use this rather than a bare `pop()`: the global
   * stack can change during an in-flight request (the operator can press
   * Z mid-flight and pop *your* entry), so a blind pop would remove an
   * unrelated action's history instead.
   */
  remove(entries: UndoEntry[]): void {
    if (entries.length === 0) return;
    const s = new Set(entries);
    this.stack = this.stack.filter((e) => !s.has(e));
  }

  clear(): void {
    this.stack = [];
  }

  /**
   * Registered with the project-change registry (bottom of this file):
   * runs on every project switch.
   * The stack is cleared outright rather than filtered down to the new
   * project's own entries: a project's undo history is session-scoped
   * curation UI state, not something worth carrying across a switch, and
   * clearing is what guarantees Z can never revert a different
   * project's write. See `resetForProjectChange()` in
   * `SourceImageOverlay.svelte` for the sibling reset on its crop-context
   * cache.
   */
  resetForProjectChange(): void {
    this.clear();
  }

  /**
   * Record the crops a discard or a new-class resolve just landed on (writes that are not
   * label operations; Z undoes them through the class history), as ONE undo
   * entry for the whole write. Call with the server's own `updated_ids`
   * (never the request ids minus conflicts computed locally) — the
   * served list is the only authoritative record of which crops the
   * write actually reached. Skipped entirely when the write reached no
   * crop (every id conflicted), so Z never pops a no-op entry.
   */
  recordWrites(updatedIds: string[]): void {
    if (updatedIds.length === 0) return;
    this.push({ crop_ids: [...updatedIds], at: Date.now(), kind: 'label' });
  }

  /**
   * A single or bulk relabel, or a move, recorded by the server as a label operation: Z
   * undoes it by `opId` (queued as a job). `cropIds` are the items the request named, used only
   * to show the restored items afterwards; a selection write has none and the page
   * reloads instead. Skipped when the server queued nothing.
   */
  recordLabelOp(opId: string | null, cropIds: string[]): void {
    if (opId === null) return;
    this.push({ crop_ids: [...cropIds], at: Date.now(), kind: 'label_op', op_id: opId });
  }

  /**
   * M6: record a confirmed human region write (confirm, reject, false
   * positive, box edit, status/text change — single or batch) as one
   * undo entry. Same "server's own updated-ids list, never the request
   * ids" rule as `recordWrites`.
   */
  recordRegionWrites(updatedIds: string[]): void {
    if (updatedIds.length === 0) return;
    this.push({ crop_ids: [...updatedIds], at: Date.now(), kind: 'region' });
  }

  /**
   * An "ignore" write (selection exclude): Z restores the served ids with
   * `batch_unexclude`. Same served-ids rule as `recordWrites`.
   */
  recordExclusion(updatedIds: string[]): void {
    if (updatedIds.length === 0) return;
    this.push({ crop_ids: [...updatedIds], at: Date.now(), kind: 'exclude' });
  }

  /** A "restore" write (selection unexclude): Z ignores the ids again. */
  recordUnexclusion(updatedIds: string[]): void {
    if (updatedIds.length === 0) return;
    this.push({ crop_ids: [...updatedIds], at: Date.now(), kind: 'unexclude' });
  }

  /** M6/V1: record a Reject-VLM (`vlm_dismiss`) as one undo entry. */
  recordVlmDismiss(cropId: string): void {
    this.push({ crop_ids: [cropId], at: Date.now(), kind: 'vlm_dismiss' });
  }

  /** Record a "Keep my label" (`probe_dismiss`) as one undo entry. */
  recordProbeDismiss(cropId: string): void {
    this.push({ crop_ids: [cropId], at: Date.now(), kind: 'probe_dismiss' });
  }

  /**
   * Z: undo the newest entry (any kind) on the server and return the
   * restored crop(s). Route selection is `kind` × batch-vs-single:
   *  - `'label_op'`: `POST /label_ops/{op_id}/undo` (a queued job, followed).
   *  - `'label'` (default, unset on any entry pushed before this field
   *    existed): `POST /crops/{id}/label/undo` / `.../label/undo_batch`.
   *  - `'region'`: `POST /crops/{id}/region/undo` / `.../region/undo_batch`.
   *  - `'vlm_dismiss'`: `POST /crops/{id}/vlm_dismiss/undo` (no batch
   *    route — a dismiss is always recorded one crop at a time).
   * Returns `[]` when there was nothing to undo or the call failed (both
   * toasted here). A failed call re-pushes the entry so Z stays
   * retryable; a 409 (nothing left to undo, on any route) does not,
   * since the server has nothing left for it.
   */
  async undoLast(): Promise<Crop[]> {
    const entry = this.pop();
    if (!entry) {
      toastStore.info('Nothing to undo.');
      return [];
    }
    if (entry.kind !== 'label_op' && entry.crop_ids.length === 0) {
      // A strict batch body 422s on an empty list; nothing to revert.
      toastStore.info('Nothing to undo.');
      return [];
    }
    try {
      return await revertEntry(entry);
    } catch (e) {
      if (e instanceof ApiError && e.status === 409) {
        toastStore.info('Nothing left to undo.');
      } else {
        toastStore.error(`Undo failed: ${apiErrorText(e)}`);
        this.push(entry);
      }
      return [];
    }
  }
}

export const undoStore = new UndoStore();

// Z must never revert a different project's write.
onProjectChange(() => undoStore.resetForProjectChange());
