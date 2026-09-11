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
 * Deliberately dumb: this module doesn't know about `/curation/methods`,
 * `stable`/`experimental`/`shadow` status, or field coverage — that's
 * capability discovery, owned by `strategiesStore` and rendered by
 * `StrategyBar.svelte`. This module only knows "what is currently
 * selected" and "how to serialize it," so it stays trivially testable
 * without mocking the store or the network.
 *
 * `sort` doubles as `/clusters/[id]`'s `orderMode` — both routes pick
 * one id out of whatever `/curation/methods` currently reports; the query key
 * it's serialized under differs per route (`sort` for `/review`,
 * `order` for `/clusters/[id]`), which is why `toQueryParams()` only
 * covers the `/review` shape and the cluster page reads `.sort` directly
 * (see `src/routes/clusters/[id]/+page.svelte`'s `cropQuery()`).
 */

export interface StrategyBarOptions {
  /** The id treated as "no explicit sort requested" — omitted from
   *  toQueryParams() and restored by reset(). Defaults to 'default'. */
  defaultId?: string;
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
   *  above re: `/curation/methods`/status). Not part of `toQueryParams()` —
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
      if (sort !== defaultId) params.sort = sort;
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
