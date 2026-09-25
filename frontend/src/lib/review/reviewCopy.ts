/**
 * Display copy for `/review` states the backend explains with an id or a
 * served sentence (visual audit 2026-09-24, R3 and R6). Pure so it can be
 * tested without mounting the page. Nothing here decides anything: every
 * input is served (`/review/tabs` description, the queue response's
 * `sort_fallback_reason`, `/locate`'s `reason`) or is the operator's own
 * filter state.
 */

import { humanizeId } from '$lib/humanizeId';

/** Toast text for a `?crop_id=` deep link `/locate` says isn't in the
 *  queue. `/locate` documents two reasons (`not_found`, `filtered_out`);
 *  anything else is shown as served. */
export function locateMissMessage(reason: string | null | undefined): string {
  if (reason === 'not_found') {
    return 'That crop is not in this review queue — no crop with that id exists.';
  }
  if (reason === 'filtered_out') {
    return "That crop is not in this review queue — it doesn't match this tab's filters, or it was already reviewed.";
  }
  if (reason) return `That crop is not in this review queue: ${reason}`;
  return 'That crop is not in this review queue (it may already be reviewed).';
}

/** Shown where an item has no class at all (R6) — never a blank value. */
export const NO_CLASS_YET = 'no class yet';

/** The VLM's served `vlm_class_empty_reason` id, as prose (R6): "no_answer"
 *  → "VLM gave no class — No answer". */
export function vlmEmptyReasonText(reason: string): string {
  return `VLM gave no class — ${humanizeId(reason)}`;
}

export interface EmptyQueueInput {
  /** Served (or static fallback) label of the active tab or preset. */
  label: string;
  /** Served `/review/tabs` description of the active tab or preset. */
  description: string | null;
  /** Served `sort_fallback_reason` from the empty response, if any. */
  sortFallbackReason: string | null;
  /** Whether the operator has any of their own filters narrowing the queue. */
  filtersActive: boolean;
}

export interface EmptyQueueMessage {
  title: string;
  lines: string[];
}

/**
 * What an empty queue says instead of a bare "Queue empty." (R3). Lists,
 * in order: what the queue is meant to hold (served description), why its
 * ordering has nothing to work with yet (served fallback reason — the
 * signal that the data this tab depends on isn't there), and whether the
 * operator's own filters could be the cause.
 */
export function emptyQueueMessage(input: EmptyQueueInput): EmptyQueueMessage {
  const lines: string[] = [];
  if (input.description) lines.push(`This queue holds: ${input.description}.`);
  if (input.sortFallbackReason) {
    lines.push(
      `The data this queue is ordered by is not there yet: ${input.sortFallbackReason}`,
    );
  }
  if (input.filtersActive) {
    lines.push('Your filters may be hiding items — clear them to see the whole queue.');
  } else if (!input.sortFallbackReason) {
    lines.push('Nothing currently needs review here.');
  }
  return { title: `The ${input.label} queue is empty.`, lines };
}
