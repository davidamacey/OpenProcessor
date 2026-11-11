/**
 * The project's pinned pairs, newest first, as served by
 * `GET /near_duplicates/decisions`. Any pin can be taken back from here; the
 * list is read again after an undo (and when a group dialog changes a pin).
 */
import { apiErrorText } from '$lib/api_errors';
import { listNearDupDecisions, undecideNearDup } from '$lib/api_nearDup';
import type { NearDupDecisionView } from '$lib/types_nearDup';

const PAGE = 20;

export function createNearDupDecisionsController(onchanged: () => void = () => {}) {
  let items = $state<NearDupDecisionView[]>([]);
  let nextCursor = $state<string | null>(null);
  let loaded = $state(false);
  let loading = $state(false);
  let busy = $state(false);
  let error = $state<string | null>(null);
  const abort = new AbortController();

  async function load(append = false): Promise<void> {
    if (append && !nextCursor) return;
    loading = true;
    error = null;
    try {
      const page = await listNearDupDecisions(
        { limit: PAGE, ...(append && nextCursor ? { cursor: nextCursor } : {}) },
        abort.signal,
      );
      items = append ? [...items, ...page.items] : page.items;
      nextCursor = page.next_cursor;
      loaded = true;
    } catch (e) {
      if (!abort.signal.aborted) error = apiErrorText(e);
    } finally {
      loading = false;
    }
  }

  async function undo(decisionId: string): Promise<void> {
    if (busy) return;
    busy = true;
    error = null;
    try {
      await undecideNearDup(decisionId, abort.signal);
      await load();
      onchanged();
    } catch (e) {
      if (!abort.signal.aborted) error = apiErrorText(e);
    } finally {
      busy = false;
    }
  }

  function dispose(): void {
    abort.abort();
  }

  return {
    get items() {
      return items;
    },
    get hasMore() {
      return nextCursor !== null;
    },
    get loaded() {
      return loaded;
    },
    get loading() {
      return loading;
    },
    get busy() {
      return busy;
    },
    get error() {
      return error;
    },
    load,
    undo,
    dispose,
  };
}

export type NearDupDecisionsController = ReturnType<
  typeof createNearDupDecisionsController
>;
