/**
 * The served summary of the current selection (`POST {P}/selection/summary`):
 * read again whenever the selection changes, with the last answer kept until
 * the next one lands. Must be created inside an effect scope.
 */
import { apiErrorText } from '$lib/api_errors';
import { getSelectionSummary } from '$lib/api_select';
import type { ItemSelection, SelectionSummary } from '$lib/types_itemFilter';

const DEBOUNCE_MS = 150;

export interface SelectionSummaryState {
  readonly summary: SelectionSummary | null;
  readonly error: string | null;
  readonly loading: boolean;
}

export function createSelectionSummary(
  selection: () => ItemSelection | null,
): SelectionSummaryState {
  let summary = $state<SelectionSummary | null>(null);
  let error = $state<string | null>(null);
  let loading = $state(false);

  $effect(() => {
    const sel = selection();
    if (!sel) {
      summary = null;
      error = null;
      loading = false;
      return;
    }
    const ctrl = new AbortController();
    loading = true;
    const timer = setTimeout(() => {
      getSelectionSummary($state.snapshot(sel) as ItemSelection, ctrl.signal)
        .then((res) => {
          summary = res;
          error = null;
        })
        .catch((e: unknown) => {
          if (ctrl.signal.aborted) return;
          summary = null;
          error = apiErrorText(e);
        })
        .finally(() => {
          if (!ctrl.signal.aborted) loading = false;
        });
    }, DEBOUNCE_MS);
    return () => {
      clearTimeout(timer);
      ctrl.abort();
    };
  });

  return {
    get summary() {
      return summary;
    },
    get error() {
      return error;
    },
    get loading() {
      return loading;
    },
  };
}
