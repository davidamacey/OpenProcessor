/**
 * One open group: its members and the pin each has to the representative. A
 * decision pins a pair together (`same`) or apart (`different`) for every
 * later run and is applied by the next run, so this controller never regroups
 * client-side. The pins are served on each member (`decision`), so any pin can
 * be undone, including one made in an earlier visit; after a decision or an
 * undo the group is read again.
 */
import { apiErrorText } from '$lib/api_errors';
import { decideNearDup, getNearDupGroup, undecideNearDup } from '$lib/api_nearDup';
import type {
  NearDupDecision,
  NearDupGroupDetail,
  NearDupLevel,
} from '$lib/types_nearDup';

export function createNearDupGroupController(level: NearDupLevel, groupId: string) {
  let detail = $state<NearDupGroupDetail | null>(null);
  let loading = $state(true);
  let error = $state<string | null>(null);
  let busy = $state(false);
  let actionError = $state<string | null>(null);
  const abort = new AbortController();

  async function load(): Promise<void> {
    loading = true;
    error = null;
    try {
      detail = await getNearDupGroup(level, groupId, abort.signal);
    } catch (e) {
      if (!abort.signal.aborted) error = apiErrorText(e);
    } finally {
      loading = false;
    }
  }

  async function decide(
    aId: string,
    bId: string,
    decision: NearDupDecision,
  ): Promise<void> {
    if (busy) return;
    busy = true;
    actionError = null;
    try {
      await decideNearDup({ level, a_id: aId, b_id: bId, decision });
      detail = await getNearDupGroup(level, groupId, abort.signal);
    } catch (e) {
      actionError = apiErrorText(e);
    } finally {
      busy = false;
    }
  }

  async function undo(decisionId: string): Promise<void> {
    if (busy) return;
    busy = true;
    actionError = null;
    try {
      await undecideNearDup(decisionId);
      detail = await getNearDupGroup(level, groupId, abort.signal);
    } catch (e) {
      actionError = apiErrorText(e);
    } finally {
      busy = false;
    }
  }

  function dispose(): void {
    abort.abort();
  }

  return {
    get detail() {
      return detail;
    },
    get loading() {
      return loading;
    },
    get error() {
      return error;
    },
    get pinned() {
      return detail?.members.filter((m) => m.decision !== null).length ?? 0;
    },
    get busy() {
      return busy;
    },
    get actionError() {
      return actionError;
    },
    load,
    decide,
    undo,
    dispose,
  };
}

export type NearDupGroupController = ReturnType<typeof createNearDupGroupController>;
