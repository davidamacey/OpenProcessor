/**
 * One open group: its members and the decisions made on it. A decision pins a
 * pair together (`same`) or apart (`different`) for every later run and is
 * applied by the next run, so this controller never regroups client-side.
 * Undo is offered for the decisions made in this view only: the API serves no
 * decision list, only the id returned when one is made.
 */
import { apiErrorText } from '$lib/api_errors';
import {
  decideNearDup,
  getNearDupGroup,
  previewItemsForImages,
  undecideNearDup,
} from '$lib/api_nearDup';
import type {
  NearDupDecision,
  NearDupDecisionView,
  NearDupGroupDetail,
  NearDupLevel,
} from '$lib/types_nearDup';

export function createNearDupGroupController(level: NearDupLevel, groupId: string) {
  let detail = $state<NearDupGroupDetail | null>(null);
  let previews = $state<Record<string, string>>({});
  let loading = $state(true);
  let error = $state<string | null>(null);
  let decisions = $state<NearDupDecisionView[]>([]);
  let busy = $state(false);
  let actionError = $state<string | null>(null);
  const abort = new AbortController();

  async function load(): Promise<void> {
    loading = true;
    error = null;
    try {
      detail = await getNearDupGroup(level, groupId, abort.signal);
      if (level === 'image') {
        try {
          previews = await previewItemsForImages(
            detail.members.map((m) => m.member_id),
            abort.signal,
          );
        } catch {
          // A missing thumbnail is cosmetic; the member rows still render.
        }
      }
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
      const made = await decideNearDup({ level, a_id: aId, b_id: bId, decision });
      decisions = [...decisions.filter((d) => d.id !== made.id), made];
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
      decisions = decisions.filter((d) => d.id !== decisionId);
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
    get previews() {
      return previews;
    },
    get loading() {
      return loading;
    },
    get error() {
      return error;
    },
    get decisions() {
      return decisions;
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
