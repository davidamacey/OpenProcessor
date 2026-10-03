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
  { label: string; tooltip: string; warning: boolean }
> = {
  failed: {
    label: 'No vector: encoder failed',
    tooltip: 'The encoder failed for this item at ingest; run Embed to retry.',
    warning: true,
  },
  deferred: {
    label: 'No vector yet',
    tooltip: 'Embedding was deferred (lazy policy or a combine that dropped the vector).',
    warning: false,
  },
  not_selected: {
    label: 'Not embedded',
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
