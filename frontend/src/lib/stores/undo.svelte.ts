/**
 * UndoStore — ring buffer of the last 50 human write *actions*, spanning
 * three kinds (`UndoEntry.kind`): class-label writes (`'label'`,
 * default), region writes (`'region'`, M6) and VLM-suggestion dismissals
 * (`'vlm_dismiss'`, M6/V1 undo). One entry per confirmed write, however
 * many crops it touched — a bulk label, a move, a batch region status
 * change, or a new-class-proposal resolve over N crops is ONE entry, so
 * one Z reverses the whole action. Calling code records what a confirmed
 * write touched via `recordWrites`/`recordRegionWrites`/
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

import {
  ApiError,
  undoCropLabel,
  undoCropRegion,
  undoCropRegionBatch,
  undoLabelBatch,
  undoVlmDismiss,
} from '$lib/api';
import { toastStore } from '$stores/toast.svelte';
import type { Crop, UndoEntry } from '$lib/types';

const MAX = 50;

class UndoStore {
  // $state.raw, not $state: deep reactivity would wrap every pushed entry
  // in a Proxy, so `remove()` could never match the raw object the caller
  // still holds. Every mutation below reassigns the array, so raw state is
  // just as reactive for readers.
  stack = $state.raw<UndoEntry[]>([]);

  push(entry: UndoEntry): void {
    const next = [...this.stack, entry];
    if (next.length > MAX) next.shift();
    this.stack = next;
  }

  pop(): UndoEntry | undefined {
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
   * Record the crops a human class write just landed on, as ONE undo
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
   * M6: record a confirmed human region write (confirm, reject, false
   * positive, box edit, status/text change — single or batch) as one
   * undo entry. Same "server's own updated-ids list, never the request
   * ids" rule as `recordWrites`.
   */
  recordRegionWrites(updatedIds: string[]): void {
    if (updatedIds.length === 0) return;
    this.push({ crop_ids: [...updatedIds], at: Date.now(), kind: 'region' });
  }

  /** M6/V1: record a Reject-VLM (`vlm_dismiss`) as one undo entry. */
  recordVlmDismiss(cropId: string): void {
    this.push({ crop_ids: [cropId], at: Date.now(), kind: 'vlm_dismiss' });
  }

  /**
   * Z: undo the newest entry (any kind) on the server and return the
   * restored crop(s). Route selection is `kind` × batch-vs-single:
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
    const kind = entry.kind ?? 'label';
    try {
      if (kind === 'vlm_dismiss') {
        const crop = await undoVlmDismiss(entry.crop_ids[0]!);
        toastStore.success('VLM suggestion restored.');
        return [crop];
      }
      if (entry.crop_ids.length === 1) {
        const crop =
          kind === 'region'
            ? await undoCropRegion(entry.crop_ids[0]!)
            : await undoCropLabel(entry.crop_ids[0]!);
        toastStore.success('Reverted.');
        return [crop];
      }
      const res =
        kind === 'region'
          ? await undoCropRegionBatch(entry.crop_ids)
          : await undoLabelBatch(entry.crop_ids);
      const parts = [`Reverted ${res.undone}.`];
      if (res.nothing_to_undo.length > 0) {
        parts.push(`${res.nothing_to_undo.length} nothing to undo.`);
      }
      if (res.conflicts.length > 0) {
        parts.push(`${res.conflicts.length} conflict(s).`);
      }
      toastStore.success(parts.join(' '));
      return res.items;
    } catch (e) {
      if (e instanceof ApiError && e.status === 409) {
        toastStore.info('Nothing left to undo.');
      } else {
        toastStore.error(`Undo failed: ${(e as Error).message}`);
        this.push(entry);
      }
      return [];
    }
  }
}

export const undoStore = new UndoStore();
