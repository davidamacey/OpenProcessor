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
  /** Served `empty_reason` from the empty response — the backend's own,
   *  more direct explanation (#36 item 9), e.g. "no probe predictions —
   *  run a probe". Takes priority over `sortFallbackReason` when both are
   *  set, since it's the backend saying outright why, not just what
   *  ordering fell back. */
  emptyReason?: string | null;
  /** Whether the operator has any of their own filters narrowing the queue. */
  filtersActive: boolean;
  /** Served `/review/tabs` `empty_state` (#36 item 9) — when the deployment
   *  has never run a probe or computed item scores at all, the empty
   *  message can point straight at the control that would populate this
   *  queue, rather than leaving the operator to guess. `null` until the
   *  tabs vocabulary has loaded. */
  emptyState?: { has_probe_predictions: boolean; has_item_scores: boolean } | null;
}

export interface EmptyQueueMessage {
  title: string;
  lines: string[];
  /** Set when the served `emptyState` says the prerequisite this queue
   *  needs (probe predictions or item scores) has never been computed —
   *  a link target the page renders as an anchor. */
  /** A project section path; the page builds the full link with
   *  `projectHref()`. */
  link?: { href: '/train' | '/settings'; text: string };
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
  if (input.emptyReason) {
    // #36 item 9: the server's own direct reason — e.g. "no probe
    // predictions — run a probe" — takes priority over the ordering
    // fallback note, since it says outright why, not just what the
    // queue's sort degraded to.
    lines.push(input.emptyReason);
  } else if (input.sortFallbackReason) {
    lines.push(
      `The data this queue is ordered by is not there yet: ${input.sortFallbackReason}`,
    );
  }
  if (input.filtersActive) {
    lines.push('Your filters may be hiding items — clear them to see the whole queue.');
  } else if (!input.emptyReason && !input.sortFallbackReason) {
    lines.push('Nothing currently needs review here.');
  }
  const reasonText = `${input.emptyReason ?? ''} ${input.sortFallbackReason ?? ''}`;
  let link: EmptyQueueMessage['link'];
  if (input.emptyState?.has_probe_predictions === false && /probe/.test(reasonText)) {
    link = { href: '/train', text: 'Run a probe on /train' };
  } else if (input.emptyState?.has_item_scores === false && /score/.test(reasonText)) {
    link = { href: '/settings', text: 'Compute scores on /settings' };
  }
  return { title: `The ${input.label} queue is empty.`, lines, link };
}

/**
 * F8 D6: the item's position in the whole served queue, 1-based. A deep
 * link loads only the located page (`firstPage`), so the cursor's index
 * within the loaded buffer (e.g. 8) is not the queue position (the served
 * rank 67 → 68th). The served `/locate` places a crop at
 * `rank = (page - 1) * page_size + index`, so this is `rank + 1` for it.
 */
export function queuePosition(
  firstPage: number,
  pageSize: number,
  cursor: number,
): number {
  return (Math.max(1, firstPage) - 1) * pageSize + cursor + 1;
}
