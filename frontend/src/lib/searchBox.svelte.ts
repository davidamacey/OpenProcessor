/**
 * State/logic for `<SemanticSearchBox>` (P2-14 — free-text search over
 * vehicle crops, `GET /curation/search/text`). Same "logic extracted from the
 * component so it's unit-testable" pattern as `pager.svelte.ts` /
 * `strategyBar.svelte.ts` / `selection.svelte.ts`.
 *
 * Owns exactly: the debounced-vs-immediate-submit trigger, in-flight
 * request cancellation (a fresh query aborts whatever request is still
 * outstanding — the classic type-ahead race), and loading/error/empty
 * state. It does NOT own where results land — `onResults` hands the raw
 * `{items, total}` back to the caller, which is free to feed them
 * straight into an existing `Pager`'s settable `items`/`total` (see
 * `SemanticSearchBox.svelte`'s doc comment) so the host page's existing
 * CropCard grid, selection, DnD, and label hotkeys keep working
 * completely unchanged — this module never renders anything and never
 * touches labeling.
 */

import type { SearchCrop } from './types';

export interface SemanticSearchResult {
  items: SearchCrop[];
  total: number;
}

export interface SemanticSearchOptions {
  /** Performs one page-1 search. Must honor `signal` (abort). */
  search: (q: string, signal: AbortSignal) => Promise<SemanticSearchResult>;
  /** Debounce window for typed input, ms. Enter bypasses this entirely. */
  debounceMs?: number;
  onResults?: (res: SemanticSearchResult) => void;
  /** Fired when the box returns to the empty/no-search state (clear(),
   *  or the query is emptied by hand). Lets the host page restore its
   *  normal (non-search) pager contents. */
  onClear?: () => void;
}

export interface SemanticSearchBox {
  /** Bound to the text input. */
  query: string;
  readonly loading: boolean;
  readonly error: string | null;
  /** True once a non-empty search has actually been submitted (debounce
   *  fired, or Enter/submit() was called) and not yet cleared. */
  readonly active: boolean;
  /** Call on every keystroke; debounces before firing a search. A blank
   *  value clears immediately (no debounce needed to know "nothing"). */
  oninput(value: string): void;
  /** Fire immediately, bypassing the debounce (Enter key). No-op on a
   *  blank query — same as clear(). */
  submit(): void;
  /** Reset to the no-search state and abort any in-flight request. */
  clear(): void;
}

export function createSemanticSearchBox(opts: SemanticSearchOptions): SemanticSearchBox {
  const debounceMs = opts.debounceMs ?? 300;

  let query = $state<string>('');
  let loading = $state<boolean>(false);
  let error = $state<string | null>(null);
  let active = $state<boolean>(false);

  let timer: ReturnType<typeof setTimeout> | null = null;
  let controller: AbortController | null = null;

  function cancelPending(): void {
    if (timer != null) {
      clearTimeout(timer);
      timer = null;
    }
    controller?.abort();
    controller = null;
  }

  async function run(q: string): Promise<void> {
    cancelPending();
    const myController = new AbortController();
    controller = myController;
    loading = true;
    error = null;
    try {
      const res = await opts.search(q, myController.signal);
      if (myController.signal.aborted) return;
      active = true;
      opts.onResults?.(res);
    } catch (e) {
      if (myController.signal.aborted) return;
      if (e instanceof DOMException && e.name === 'AbortError') return;
      error = (e as Error).message || 'Search failed';
    } finally {
      if (controller === myController) loading = false;
    }
  }

  return {
    get query() {
      return query;
    },
    set query(next: string) {
      query = next;
    },
    get loading() {
      return loading;
    },
    get error() {
      return error;
    },
    get active() {
      return active;
    },

    oninput(value: string): void {
      query = value;
      cancelPending();
      const trimmed = value.trim();
      if (!trimmed) {
        // Blank input reverts to the no-search state immediately — no
        // reason to wait out a debounce window to learn "nothing typed."
        error = null;
        loading = false;
        if (active) {
          active = false;
          opts.onClear?.();
        }
        return;
      }
      timer = setTimeout(() => {
        timer = null;
        void run(trimmed);
      }, debounceMs);
    },

    submit(): void {
      cancelPending();
      const trimmed = query.trim();
      if (!trimmed) {
        if (active) {
          active = false;
          opts.onClear?.();
        }
        return;
      }
      void run(trimmed);
    },

    clear(): void {
      cancelPending();
      query = '';
      loading = false;
      error = null;
      const wasActive = active;
      active = false;
      if (wasActive) opts.onClear?.();
    },
  };
}
