/**
 * State/logic for the shared `<StrategyBar>` control (curation-strategy
 * plan Phase 3/4, docs/curation-strategy-plan-2026-09.md §5.4/§5.1).
 *
 * Holds the operator's current sort selection, any active score-filter
 * thresholds, and (Phase 4) the `k` count for a pool-scale overlay
 * selection, and exposes `toQueryParams()` — the single
 * place that turns "what the operator picked" into the query-param
 * object `getReviewQueue`/`getCluster` forward to the backend. Same
 * "logic extracted from the component so it's unit-testable" pattern as
 * `pager.svelte.ts` / `selection.svelte.ts`: a factory function
 * returning a plain object with getters/setters over `$state`, not a
 * class, so callers can destructure/bind (`bind:value={bar.sort}`)
 * exactly like the pager/selection helpers.
 *
 * Deliberately dumb: this module doesn't know about `{API_PREFIX}/methods`,
 * `stable`/`experimental`/`shadow` status, or field coverage — that's
 * capability discovery, owned by `strategiesStore` and rendered by
 * `StrategyBar.svelte`. This module only knows "what is currently
 * selected" and "how to serialize it," so it stays trivially testable
 * without mocking the store or the network.
 *
 * `sort` doubles as `/clusters/[id]`'s `orderMode` — both routes pick
 * one id out of whatever `{API_PREFIX}/methods` currently reports; the query key
 * it's serialized under differs per route (`sort` for `/review`,
 * `order` for `/clusters/[id]`), which is why `toQueryParams()` only
 * covers the `/review` shape and the cluster page reads `.sort` directly
 * (see `src/routes/clusters/[id]/+page.svelte`'s `cropQuery()`).
 *
 * `formatAppliedSort`/`formatPinnedSortFallback` below are the one
 * exception to "deliberately dumb" above — both are pure functions the
 * component calls to turn `{API_PREFIX}/methods` state it already looked
 * up (a served label, a served `field_coverage`) into display text, kept
 * here rather than inline in the `.svelte` file so they're unit-testable
 * without mounting anything (`strategyBar.svelte.test.ts`).
 */

import { hasFieldCoverage } from './strategies';

export interface StrategyBarOptions {
  /** The id treated as "no explicit sort requested" — omitted from
   *  toQueryParams() and restored by reset(). Defaults to 'default'. */
  defaultId?: string;
  /** Sort ids that are actually pool-scale overlays (Phase 4's `'diverse'`),
   *  not a real `{API_PREFIX}/review/{tab}` or `{API_PREFIX}/crops` sort param — that
   *  endpoint 400s if `sort=diverse` is ever forwarded to it (P2-10).
   *  `toQueryParams()` omits `sort` entirely whenever the current
   *  selection is one of these; the caller drives the overlay through its
   *  own separate call (`selectDiverse`), not through the sort query
   *  param. Defaults to `[]` so every existing caller (today: only
   *  `/clusters/[id]`, which reads `.sort` directly rather than going
   *  through `toQueryParams()`) is unaffected. */
  overlayIds?: string[];
}

export interface StrategyBar {
  /** Currently selected sort/order id. */
  sort: string;
  /** Minimum mistakenness threshold, or null = no filter. */
  minMistakenness: number | null;
  /** Hide near-duplicate crops (only meaningful once the backend reports
   *  non-zero coverage for a dup-group field — StrategyBar.svelte gates
   *  whether this is even offered). */
  hideNearDuplicates: boolean;
  /** "How many diverse crops?" count for a pool-scale overlay selection
   *  (currently only `sort === 'diverse'` on `/clusters/[id]` — the
   *  curation-strategy plan's Phase 4 `k` stepper). `null` = not set by
   *  the operator yet; the caller seeds/reads a route-specific default
   *  (e.g. the page size) at the point of use, since "what's a sane
   *  default" is a per-route concept this shared module deliberately
   *  doesn't know about (same reasoning as the module-level doc comment
   *  above re: `{API_PREFIX}/methods`/status). Not part of `toQueryParams()` —
   *  only `/clusters/[id]` forwards it today, reading `.k` directly the
   *  same way it already reads `.sort` directly. */
  k: number | null;
  /** True when every field is at its default (nothing to reset). */
  readonly isDefault: boolean;
  /** Query-param object for `getReviewQueue`'s `filter` argument. Only
   *  includes keys that differ from their default — `qs()` would drop
   *  `null`/`undefined` anyway, but omitting them here keeps a request
   *  with nothing selected byte-identical to today's pre-Phase-3 calls. */
  toQueryParams(): Record<string, unknown>;
  /** Back to defaults — same shape as pager's implicit reset-on-loadFirst. */
  reset(): void;
}

export function createStrategyBar(opts: StrategyBarOptions = {}): StrategyBar {
  const defaultId = opts.defaultId ?? 'default';
  // eslint-disable-next-line svelte/prefer-svelte-reactivity -- immutable lookup set built once from options, only ever read via .has()
  const overlayIds = new Set(opts.overlayIds ?? []);

  let sort = $state<string>(defaultId);
  let minMistakenness = $state<number | null>(null);
  let hideNearDuplicates = $state<boolean>(false);
  let k = $state<number | null>(null);

  return {
    get sort() {
      return sort;
    },
    set sort(next: string) {
      sort = next;
    },
    get minMistakenness() {
      return minMistakenness;
    },
    set minMistakenness(next: number | null) {
      minMistakenness = next;
    },
    get hideNearDuplicates() {
      return hideNearDuplicates;
    },
    set hideNearDuplicates(next: boolean) {
      hideNearDuplicates = next;
    },
    get k() {
      return k;
    },
    set k(next: number | null) {
      k = next;
    },
    get isDefault() {
      return (
        sort === defaultId && minMistakenness == null && !hideNearDuplicates && k == null
      );
    },

    toQueryParams(): Record<string, unknown> {
      const params: Record<string, unknown> = {};
      // An overlay id (e.g. 'diverse') is never a real backend sort param
      // — the caller drives it through a separate call, so omit `sort`
      // entirely rather than forwarding an id the review-queue endpoint
      // would 400 on (P2-10).
      if (sort !== defaultId && !overlayIds.has(sort)) params.sort = sort;
      if (minMistakenness != null) params.min_mistakenness = minMistakenness;
      if (hideNearDuplicates) params.hide_near_duplicates = true;
      return params;
    },

    reset(): void {
      sort = defaultId;
      minMistakenness = null;
      hideNearDuplicates = false;
      k = null;
    },
  };
}

/**
 * What the StrategyBar summary chip should show for the server's
 * `sort_applied` (2026-09-24 logic-moves W5, item 10's `sort_applied`).
 * Pure so it's testable without mounting the component: returns `null`
 * when there's nothing worth surfacing (no applied value yet, or it
 * matches what the operator already selected — a tab's own default id
 * `requested` never carries, so `applied` alone communicates that case)
 * and a short "→ applied" string otherwise, e.g. when the operator asked
 * for `'mistakenness'` but the backend fell back to
 * `'atypicality_default'` because the field isn't backfilled yet.
 */
export function formatAppliedSort(
  requested: string,
  applied: string | null | undefined,
  sorts: ReadonlyArray<{ id: string; label: string }> = [],
): string | null {
  if (!applied) return null;
  if (applied === requested) return null;
  // R4 (visual audit 2026-09-24): show the served `/methods` label, not the
  // raw id ("atypicality" → "Atypicality (outlier-first)"); the id only when
  // the backend serves no label for it.
  return sorts.find((s) => s.id === applied)?.label ?? applied;
}

/**
 * What the StrategyBar summary chip should show when the deployment's
 * *pinned* review-sort default (`GET {API_PREFIX}/settings`'s
 * `defaults.sort`) has zero field coverage — visual-audit S1's last
 * bullet (`docs/design/visual-audit-2026-09-24.md`). That doc's
 * "deferred, BACKEND" status note was wrong: `GET {API_PREFIX}/methods`
 * already serves real `field_coverage`/`field_coverage_total` on every
 * `sort` entry with a `requires_field`, and `hasFieldCoverage`
 * (`strategies.ts`) is already the single gate that turns that into
 * "safe to offer" — this reuses it rather than re-deriving the
 * null-vs-zero rule a second time.
 *
 * Returns `null` unless ALL of:
 *  - a pinned sort id was resolved for the active tab (the caller has
 *    already gated this on `tabHonorsPinnedSortDefault`, `reviewTabs.ts`
 *    — this function has no tab awareness of its own);
 *  - the operator hasn't picked their own override yet (`requestedSort`
 *    is still the "no override" sentinel — an explicit pick, even one
 *    that happens to equal the pinned id, means the operator asked for
 *    it, not the deployment default silently applying);
 *  - the pinned entry's `field_coverage` is confirmed zero (not merely
 *    unknown — see `hasFieldCoverage`'s own doc comment);
 *  - the backend actually reported a `sort_applied` (never guessed
 *    client-side — `appliedLabel` is `null` whenever `appliedSort` is
 *    absent).
 *
 * Deliberately returns ONE merged string, not a second chip alongside
 * the plain "requested → applied" mismatch text `formatAppliedSort`
 * already produces: in this exact scenario `formatAppliedSort` would
 * also fire (the pinned default failing coverage is *why* `applied`
 * differs from the sentinel `requested`), so a caller must render only
 * one of the two — this one, when it returns non-null — never both.
 */
export function formatPinnedSortFallback(params: {
  /** `bar.sort` — the operator's current selection (or the sentinel). */
  requestedSort: string;
  /** `strategyBar.svelte.ts`'s "no override" sentinel id for this bar. */
  sentinelSortId: string;
  /** The deployment's pinned `sort` axis id for the active tab, or
   *  `null` when unpinned or the active tab doesn't honor it. */
  pinnedSortId: string | null | undefined;
  /** The `/methods` `review_sorts` entry for `pinnedSortId`, or `null`
   *  when it isn't (yet) in the served list. */
  pinnedSortEntry: { label: string; field_coverage?: number | null } | null | undefined;
  /** The server's `sort_applied`, verbatim — never computed here. */
  appliedSort: string | null | undefined;
  /** `formatAppliedSort(requestedSort, appliedSort, …)`'s own result —
   *  already resolves the served `/methods` label for `appliedSort`. */
  appliedLabel: string | null;
}): string | null {
  const {
    requestedSort,
    sentinelSortId,
    pinnedSortId,
    pinnedSortEntry,
    appliedSort,
    appliedLabel,
  } = params;
  if (!pinnedSortId || !pinnedSortEntry) return null;
  if (requestedSort !== sentinelSortId) return null;
  if (hasFieldCoverage(pinnedSortEntry)) return null;
  if (!appliedSort || !appliedLabel) return null;
  return `pinned default ${pinnedSortEntry.label} has no coverage yet — using ${appliedLabel}`;
}
