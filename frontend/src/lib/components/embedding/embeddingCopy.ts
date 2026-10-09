/**
 * Frontend copy for the contract's `embedding_state` values (the backend
 * serves no labels for them). `embedded` and `null` (written before the
 * field existed) get no badge; only the three states that mean "no vector"
 * are named here.
 */
import type { EmbeddingState } from '$lib/types_itemFilter';

export type NoVectorState = Exclude<EmbeddingState, 'embedded'>;

export const EMBEDDING_STATE_COPY: Record<
  NoVectorState,
  { label: string; compactLabel: string; tooltip: string; warning: boolean }
> = {
  failed: {
    label: 'No vector: encoder failed',
    compactLabel: 'Embed failed',
    tooltip: 'The encoder failed for this item at ingest; run Embed to retry.',
    warning: true,
  },
  deferred: {
    label: 'No vector yet',
    compactLabel: 'No vector',
    tooltip: 'Embedding was deferred (lazy policy or a combine that dropped the vector).',
    warning: false,
  },
  not_selected: {
    label: 'Not embedded',
    compactLabel: 'Not embedded',
    tooltip: 'The ingest policy did not select this item for a vector.',
    warning: false,
  },
};

export function noVectorCopy(
  state: EmbeddingState | null | undefined,
): (typeof EMBEDDING_STATE_COPY)[NoVectorState] | null {
  if (state == null || state === 'embedded') return null;
  return EMBEDDING_STATE_COPY[state] ?? null;
}

export interface EmbeddingStateChip {
  state: string;
  label: string;
  title: string;
  count: number;
}

const CHIP_ORDER = ['failed', 'deferred', 'not_selected', 'unknown'];

/** The served `by_state` as chips: states with a count only, `embedded`
 *  left out (the totals already say it), the three no-vector states named
 *  by the badge copy, any other key (legacy `unknown`) verbatim. */
export function embeddingStateChips(
  byState: Record<string, number> | null | undefined,
): EmbeddingStateChip[] {
  const entries = Object.entries(byState ?? {}).filter(
    ([k, n]) => k !== 'embedded' && n > 0,
  );
  const rank = (k: string) => {
    const i = CHIP_ORDER.indexOf(k);
    return i === -1 ? CHIP_ORDER.length : i;
  };
  return entries
    .sort((a, b) => rank(a[0]) - rank(b[0]) || a[0].localeCompare(b[0]))
    .map(([state, count]) => {
      const copy = noVectorCopy(state as EmbeddingState);
      return {
        state,
        label: copy?.label ?? state,
        title: copy?.tooltip ?? '',
        count,
      };
    });
}
