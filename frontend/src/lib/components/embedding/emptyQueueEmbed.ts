import type { ReprocessRequest } from '$lib/types_import';

/** The served `empty_state` facts this needs (`GET /review/tabs`). */
export interface EmptyQueueEmbedState {
  has_unembedded_items?: boolean;
  suggested_reprocess?: ReprocessRequest | null;
}

/**
 * The request behind an empty queue's "Embed them" action: the served
 * `suggested_reprocess`, only while the deployment serves
 * `has_unembedded_items: true` and the queue's own served reason is about
 * vectors (the same reason-text gate the probe and score links use). Never
 * composed here.
 */
export function emptyQueueEmbedRequest(
  emptyState: EmptyQueueEmbedState | null | undefined,
  reasonText: string | null | undefined,
): ReprocessRequest | null {
  if (!emptyState?.has_unembedded_items) return null;
  if (!emptyState.suggested_reprocess) return null;
  if (!reasonText || !/embed|vector/i.test(reasonText)) return null;
  return emptyState.suggested_reprocess;
}
