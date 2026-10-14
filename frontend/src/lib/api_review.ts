/**
 * The review queues: served tabs and filters, the queue page, locate, and the
 * new-class proposals.
 */
import { apiFetch, qs, scoped } from '$lib/api';
import { mapRawCrop, type RawCrop } from '$lib/api_cropMap';
import type { ReprocessRequest } from '$lib/types_import';
import type { PaginatedResponse } from '$lib/types';
import type {
  ResolveNewClassRequest,
  ResolveNewClassResponse,
  ReviewItem,
  ReviewTab,
} from '$lib/types_review';

/** One `value`/`label` option of a `ReviewFilterSpec` (3f1a11e adoption). */
export interface ReviewFilterOption {
  value: string;
  label: string;
}

/** Self-describing spec for one of a review tab's filters with a fixed
 *  value set (`GET {API_PREFIX}/review/tabs`, 3f1a11e adoption) — e.g. the
 *  region tab's `region_status` (all / detected only / verifier-rejected
 *  candidates only). `param` is the query parameter to send on both
 *  `GET {API_PREFIX}/review/{tab}` and its `/locate` route; an unknown value 400s.
 *  The frontend renders one `<select>` per entry generically — no
 *  tab-specific code reads `param` by name. */
export const REVIEW_FILTER_KINDS = [
  'enum',
  'multi_enum',
  'class_names',
  'bool',
  'number',
  'integer',
  'text',
] as const;
export type ReviewFilterKind = (typeof REVIEW_FILTER_KINDS)[number];

export interface ReviewFilterSpec {
  param: string;
  kind: ReviewFilterKind;
  label: string;
  /** The fixed values of an `enum` / `multi_enum`; empty for every other kind. */
  options: ReviewFilterOption[];
  /** Bounds of a `number` / `integer` filter. */
  min: number | null;
  max: number | null;
  /** How to fill the filter; empty when the label says it. */
  description: string;
  /** Value applied when the param is omitted: `null` = no filter, `[]` = none
   *  selected, `false` = off. */
  default: unknown;
  /** True when `default` is null; an enum then lists an explicit
   *  `{value: '', label: 'Any'}` option, and choosing it omits the param. */
  allows_unset: boolean;
}

/** One entry of `GET {API_PREFIX}/review/tabs` — the served
 *  label/description for a review tab or preset id. */
export interface ReviewTabVocabularyEntry {
  id: string;
  label: string;
  description: string;
  /** Query parameters this tab honours — a parameter not listed is
   *  accepted and ignored server-side. Drives which filter-bar controls
   *  render for the active tab. */
  filters: string[];
  /** Values the tab applies when a filter is omitted, e.g.
   *  `{max_rank: 2}` for the two primary-subject tabs. */
  filter_defaults: Record<string, unknown>;
  /** Self-describing enum filters this tab honours (empty for none). */
  filter_specs: ReviewFilterSpec[];
}

/** `empty_state` on `GET {API_PREFIX}/review/tabs` — whether the
 *  deployment has ANY probe predictions or item scores at all, so an
 *  empty Uncertainty/Model-disagreements/score-sorted queue can point at
 *  the missing prerequisite (run a probe, compute scores) instead of just
 *  saying "empty". */
export interface ReviewEmptyState {
  has_probe_predictions: boolean;
  has_item_scores: boolean;
  /** Whether any labeled-dataset import has written labels. */
  has_imported_labels: boolean;
  /** v0.4.0: whether any item has no vector, and the request that would
   *  embed them (served only then). */
  has_unembedded_items: boolean;
  suggested_reprocess: ReprocessRequest | null;
}

/** `GET {API_PREFIX}/review/tabs`. */
export interface ReviewTabsResponse {
  tabs: ReviewTabVocabularyEntry[];
  empty_state: ReviewEmptyState;
}

/** Every review tab's served vocabulary, in served order, plus the
 *  deployment-wide `empty_state`. The frontend keeps its own tab
 *  structure/ids (`reviewTabs.ts`) and only overlays the served
 *  label/description/filters on top. */
export function getReviewTabs(signal?: AbortSignal): Promise<ReviewTabsResponse> {
  return apiFetch<ReviewTabsResponse>(`${scoped()}/review/tabs`, {}, signal);
}

export async function getReviewQueue(
  tab: ReviewTab,
  page = 1,
  pageSize = 30,
  filter: Record<string, unknown> = {},
  signal?: AbortSignal,
): Promise<PaginatedResponse<ReviewItem>> {
  // The {API_PREFIX}/review API ships bbox_norm + region_bbox_norm as
  // [x1,y1,x2,y2] arrays. The labeler's ReviewItem extends Crop where
  // bboxes are {cx,cy,w,h} objects. Normalize each item through
  // mapRawCrop so SlotBboxEditor + getThumbUrl + confirmSlot all see the
  // same shape regardless of the endpoint that produced the item.
  // proposed_class_id/name come straight off RawCrop/mapRawCrop now —
  // they're served on every crop-shaped item (item 11, 2026-09-24
  // logic-moves), not a review-only field — so only the genuinely
  // review-specific extras are declared here.
  type RawReviewItem = RawCrop & {
    reason?: string;
    probe_pred_class?: string | null;
    probe_pred_class_id?: number | null;
    probe_pred_entropy?: number | null;
    needs_new_class?: boolean;
    needs_new_class_note?: string | null;
  };
  type RawPage = {
    total: number;
    page: number;
    page_size: number;
    items: RawReviewItem[];
    /** Set when the requested `?sort=` fell back to the default — see
     *  PaginatedResponse.sort_fallback_reason in types.ts. */
    sort_fallback_reason?: string | null;
    /** The sort id actually applied — see PaginatedResponse.sort_applied. */
    sort_applied?: string | null;
    /** Set when total === 0 — see PaginatedResponse.empty_reason (#36 item 9). */
    empty_reason?: string | null;
  };
  const raw = await apiFetch<RawPage>(
    `${scoped()}/review/${tab}${qs({ page, page_size: pageSize, ...filter })}`,
    {},
    signal,
  );
  const items: ReviewItem[] = (raw.items ?? []).map((it) => {
    const base = mapRawCrop(it);
    return {
      ...base,
      reason: it.reason ?? '',
      probe_pred_class: it.probe_pred_class ?? null,
      probe_pred_class_id: it.probe_pred_class_id ?? null,
      probe_pred_entropy: it.probe_pred_entropy ?? null,
      needs_new_class: !!it.needs_new_class,
      needs_new_class_note: it.needs_new_class_note ?? null,
    };
  });
  return {
    items,
    total: raw.total ?? items.length,
    page: raw.page ?? page,
    page_size: raw.page_size ?? pageSize,
    sort_fallback_reason: raw.sort_fallback_reason ?? null,
    sort_applied: raw.sort_applied ?? null,
    empty_reason: raw.empty_reason ?? null,
  };
}

/** `GET {API_PREFIX}/review/{tab}/locate`— where a specific crop sits in a
 *  review queue under the given filters/sort, without paging through it
 *  by hand. Powers `/review?crop_id=` deep links:
 *  `in_queue: false` means the crop doesn't match this tab's
 *  filters (or is already handled) — `reason` explains why when the
 *  backend sends one. */
export interface ReviewLocateResult {
  crop_id: string;
  in_queue: boolean;
  rank: number | null;
  page: number | null;
  page_size: number;
  total: number;
  reason: string | null;
  sort_applied: string | null;
  /** M11: mirrors `PaginatedResponse.sort_fallback_reason` — set when the
   *  sort `/locate` resolved (tab default, or the caller's own `sort`)
   *  fell back because its field has 0% coverage. */
  sort_fallback_reason: string | null;
}

export async function locateInReviewQueue(
  tab: string,
  cropId: string,
  pageSize: number,
  filter: Record<string, unknown> = {},
  signal?: AbortSignal,
): Promise<ReviewLocateResult> {
  const raw = await apiFetch<Partial<ReviewLocateResult>>(
    `${scoped()}/review/${tab}/locate${qs({ crop_id: cropId, page_size: pageSize, ...filter })}`,
    {},
    signal,
  );
  return {
    crop_id: raw.crop_id ?? cropId,
    in_queue: !!raw.in_queue,
    rank: raw.rank ?? null,
    page: raw.page ?? null,
    page_size: raw.page_size ?? pageSize,
    total: raw.total ?? 0,
    reason: raw.reason ?? null,
    sort_applied: raw.sort_applied ?? null,
    sort_fallback_reason: raw.sort_fallback_reason ?? null,
  };
}

/** `GET {API_PREFIX}/review/new_class_proposals/summary` — the aggregate
 *  `/classes`'s Proposals section renders (top VLM-proposed-but-unmatched
 *  terms, with counts and a handful of sample crop ids each), distinct
 *  from paging the `new_class_proposals` review tab item-by-item. */
/** DQ-M11 fix (dq-queues cutover, 2026-09-24): `flag` is `null` for a
 *  term worth one-click creating (the only kind `top_terms` holds now),
 *  or the served reason it isn't: `existing_class` (map to `class_id`
 *  instead of creating), `generic_parent` (a super-category, e.g.
 *  a super-category term), or `non_object` (junk, e.g. "abstract_blur"). */
export interface NewClassProposalTerm {
  label: string;
  count: number;
  sample_crop_ids: string[];
  flag: 'existing_class' | 'generic_parent' | 'non_object' | null;
  class_id: number | null;
}

/** Server-side term-classification rules, rendered as help text —
 *  `generic_terms`/`non_object_terms` name the deployment's configured
 *  vocab (each overridable via its `_env` var). */
export interface NewClassTermRules {
  generic_terms: string[];
  non_object_terms: string[];
  registry_groups_are_generic: boolean;
  existing_classes_flagged: boolean;
  generic_terms_env: string;
  non_object_terms_env: string;
}

export interface NewClassProposalsSummary {
  total_pending: number;
  /** Queue items with no proposed term at all (a human "needs new
   *  class" flag with no name) — not represented in top_terms/
   *  flagged_terms, since there's no label to key a row on. */
  without_term: number;
  /** Only `flag: null` terms — every one is one-click "Create class &
   *  assign". */
  top_terms: NewClassProposalTerm[];
  /** The rest: `existing_class` / `generic_parent` / `non_object`. No
   *  create action is offered for these; `existing_class` offers
   *  map-to-`class_id` instead. */
  flagged_terms: NewClassProposalTerm[];
  term_rules: NewClassTermRules | null;
}

function asProposalTerms(v: unknown): NewClassProposalTerm[] {
  if (!Array.isArray(v)) return [];
  const out: NewClassProposalTerm[] = [];
  for (const raw of v) {
    if (typeof raw !== 'object' || raw === null) continue;
    const r = raw as Record<string, unknown>;
    if (typeof r.label !== 'string') continue;
    out.push({
      label: r.label,
      count: typeof r.count === 'number' ? r.count : 0,
      sample_crop_ids: Array.isArray(r.sample_crop_ids)
        ? (r.sample_crop_ids as string[])
        : [],
      flag:
        r.flag === 'existing_class' ||
        r.flag === 'generic_parent' ||
        r.flag === 'non_object'
          ? r.flag
          : null,
      class_id: typeof r.class_id === 'number' ? r.class_id : null,
    });
  }
  return out;
}

export async function getNewClassProposalsSummary(
  signal?: AbortSignal,
): Promise<NewClassProposalsSummary> {
  const raw = await apiFetch<Record<string, unknown>>(
    `${scoped()}/review/new_class_proposals/summary`,
    {},
    signal,
  );
  return {
    total_pending: typeof raw.total_pending === 'number' ? raw.total_pending : 0,
    without_term: typeof raw.without_term === 'number' ? raw.without_term : 0,
    top_terms: asProposalTerms(raw.top_terms),
    flagged_terms: asProposalTerms(raw.flagged_terms),
    term_rules:
      raw.term_rules && typeof raw.term_rules === 'object'
        ? (raw.term_rules as NewClassTermRules)
        : null,
  };
}

/**
 * `POST {API_PREFIX}/review/new_class_proposals/resolve` — bulk-resolve
 * every pending `vlm_new_class_pending` item proposing `body.label`, not
 * just the summary's capped `sample_crop_ids` preview. Exactly one of
 * `body.class_id` (map to an existing registry class) / `body.create`
 * (register a new one first) must be set — the backend 422s otherwise;
 * an unknown `class_id` is 400, a duplicate `create.class_name` is 409
 * (both surface as `ApiError.detail`).
 *
 * `dryRun: true` (`?dry_run=true`) reports `matched`/`matched_ids` and
 * writes/creates nothing — callers use it to show "this will assign N
 * crops" before the real resolve. Every write is a restorable
 * `class_id_history` snapshot server-side, so `undoLabelBatch` with the
 * response's `updated_ids` puts each item back exactly as a pending
 * proposal.
 */
export function resolveNewClassProposal(
  body: ResolveNewClassRequest,
  opts: { dryRun?: boolean } = {},
  signal?: AbortSignal,
): Promise<ResolveNewClassResponse> {
  return apiFetch<ResolveNewClassResponse>(
    `${scoped()}/review/new_class_proposals/resolve${qs({ dry_run: opts.dryRun || undefined })}`,
    { method: 'POST', body: JSON.stringify(body) },
    signal,
  );
}
