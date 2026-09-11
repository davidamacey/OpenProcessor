<script lang="ts">
  /**
   * Shared sort/filter strategy control (curation-strategy plan Phase 3/4 —
   * docs/curation-strategy-plan-2026-09.md §5.1/§5.4/§7). Used on both
   * `/review` (below the tab strip) and `/clusters/[id]` (alongside the
   * existing "Outliers first" button, which stays as a one-click shortcut
   * for muscle memory — this is the fuller, additive selector).
   *
   * Hard constraints from the plan + CLAUDE.md's Keyboard shortcuts
   * section: pointer-first only, zero new global keybindings, and
   * collapsed-by-default so it costs no vertical space until an operator
   * opts in — a one-line summary chip that expands into the actual
   * controls on click, never a fixed overlay that could cover the grid.
   *
   * Sort options come straight from `strategiesStore` (`/curation/methods`),
   * filtered to `stable`/`experimental` only — `shadow`/`disabled` entries
   * must never be selectable (they're either mid-validation or explicitly
   * killed). The currently-selected sort gets a small amber "beta" badge
   * when its status is `experimental`.
   *
   * Filter chips (min-mistakenness threshold, hide-near-duplicates) only
   * render when the backend actually reports non-zero `field_coverage`
   * for a matching `scores`/`overlays` entry — an un-backfilled or
   * not-yet-shipped scorer means the filter would silently do nothing,
   * so it degrades to hidden rather than showing a control that can't
   * work yet.
   *
   * Phase 4 adds the pool-scale `'diverse'` overlay (core-set /
   * k-center-greedy selection) as an option `/clusters/[id]` can fold into
   * this same sort selector, plus a `k` stepper ("how many diverse
   * crops?") that only appears once `'diverse'` is both selected and
   * actually offered — see `isDiverseOverlayAvailable` in `strategies.ts`,
   * the single gate `/clusters/[id]` and this component both use.
   */

  import { isDiverseOverlayAvailable, type MethodStatus } from '$lib/strategies';
  import type { StrategyBar } from '$lib/strategyBar.svelte';
  import { strategiesStore } from '$stores/strategies.svelte';

  /** Provenance echoed back by a pool-scale overlay ordering (Phase 4 —
   *  see `PaginatedResponse.order_method`/`.order_version`/`.n_pool` in
   *  types.ts). All optional: absent entirely until a backend actually
   *  ships `order=diverse` returning these fields. */
  interface DiverseMeta {
    method?: string | null;
    version?: string | null;
    n_pool?: number | null;
  }

  interface Props {
    bar: StrategyBar;
    /** Restrict selectable sort/overlay ids beyond the stable/experimental
     *  status filter. `/clusters/[id]`'s crop query forwards its selection
     *  as `/curation/crops?order=`, which only special-cases `'outliers'` and
     *  (Phase 4) `'diverse'` today (docs/curation-strategy-plan-2026-09.md
     *  §1/§7) — so that route passes `['default', 'outliers', 'diverse']`
     *  to avoid offering an id the endpoint would silently ignore.
     *  `'diverse'` only actually appears in the rendered options when
     *  `/curation/methods` reports it (see `sortOptions` below) — passing the id
     *  here is necessary but not sufficient. Omit for `/review`, where
     *  every stable/experimental `review_sorts` entry is fair game. */
    allowedIds?: string[] | null;
    /** Whether to offer the score-filter chips at all. `/clusters/[id]`
     *  passes `false` — its crop query doesn't forward these params. */
    showFilters?: boolean;
    /** Seed value for the `k` stepper the first time the operator selects
     *  `'diverse'` and `bar.k` is still unset. Route-specific ("current
     *  page size" on `/clusters/[id]`) — omit on routes that never offer
     *  `'diverse'`. Falls back to `diverseKMin` if omitted. */
    diverseKDefault?: number;
    /** `k` stepper bounds. Min 4: below that "diversity" over a handful of
     *  items isn't a meaningful selection criterion — a plain click-through
     *  is just as fast. Max 500: matches `/curation/crops`'s own `page_size`
     *  ceiling (`Query(..., le=500)`, op_crops.py) — the same number the
     *  backend already treats as "a single request's worth," vs. the
     *  thousands-scale pool selection the plan reserves for a backend job
     *  (`OP_SELECT_MAX_N`, `POST /curation/select/diverse`), not a page control. */
    diverseKMin?: number;
    diverseKMax?: number;
    /** Provenance for the current diverse-mode response, if the backend
     *  sent any (see `DiverseMeta`). Surfaced as a small inline note/tooltip
     *  next to the stepper — not a new prominent UI element. */
    diverseMeta?: DiverseMeta | null;
  }

  let {
    bar,
    allowedIds = null,
    showFilters = true,
    diverseKDefault,
    diverseKMin = 4,
    diverseKMax = 500,
    diverseMeta = null,
  }: Props = $props();

  // getMethods()/init() never throws (404 or any error degrades to
  // FALLBACK_METHODS) and init() itself is idempotent — safe to call on
  // every mount without a guard. No listener is installed by this call.
  $effect(() => {
    void strategiesStore.init();
  });

  const sortOptions = $derived.by(() => {
    // Only a caller that restricts allowedIds (today: `/clusters/[id]`)
    // also considers the `overlays` registry — `'diverse'` lives there,
    // not in `review_sorts` (plan §3: overlays never write cluster_id, a
    // different axis from review-queue sort/filter, even though this page
    // folds both into one `orderMode` selection for UX simplicity). Plain
    // `/review` (allowedIds=null) is unaffected: only `review_sorts` is
    // considered there, exactly as before Phase 4.
    const source = allowedIds
      ? [...strategiesStore.methods.review_sorts, ...strategiesStore.methods.overlays]
      : strategiesStore.methods.review_sorts;
    const seen = new Set<string>();
    const deduped = source.filter((s) => {
      if (seen.has(s.id)) return false;
      seen.add(s.id);
      return true;
    });
    const stableOrExperimental = deduped.filter(
      (s) => s.status === 'stable' || s.status === 'experimental',
    );
    return allowedIds
      ? stableOrExperimental.filter((s) => allowedIds!.includes(s.id))
      : stableOrExperimental;
  });

  const currentSort = $derived(sortOptions.find((s) => s.id === bar.sort) ?? null);
  const currentIsBeta = $derived(currentSort?.status === 'experimental');
  const currentLabel = $derived(currentSort?.label ?? bar.sort);

  // The k stepper only renders once 'diverse' is both selected AND
  // actually offered. isDiverseOverlayAvailable is the same predicate
  // `/clusters/[id]` uses to decide whether to widen allowedIds in the
  // first place — one gate, checked at both call sites, so a backend
  // that never shipped (or disabled via OP_SELECT_DIVERSE_ENABLED) the
  // overlay can never leave a stray bar.sort === 'diverse' rendering a
  // control the endpoint doesn't support.
  const diverseSelected = $derived(
    bar.sort === 'diverse' && isDiverseOverlayAvailable(strategiesStore.methods.overlays),
  );

  function hasCoverage(entry: {
    status: MethodStatus;
    field_coverage?: number | null;
  }): boolean {
    return (
      (entry.status === 'stable' || entry.status === 'experimental') &&
      (entry.field_coverage ?? 0) > 0
    );
  }

  const mistakenessInfo = $derived(
    showFilters
      ? (strategiesStore.methods.scores.find(
          (s) => s.id === 'mistakenness' && hasCoverage(s),
        ) ?? null)
      : null,
  );
  // Any dup-related overlay/score with reported coverage — id naming
  // isn't pinned down yet (§3 lists `dup_group_id` as the field, not a
  // fixed overlay id), so match loosely rather than guessing a name.
  const nearDupInfo = $derived(
    showFilters
      ? ([...strategiesStore.methods.overlays, ...strategiesStore.methods.scores].find(
          (e) => /dup/i.test(e.id) && hasCoverage(e),
        ) ?? null)
      : null,
  );

  let expanded = $state(false);
</script>

<div class="inline-flex flex-wrap items-center gap-1.5 text-xs">
  {#if !expanded}
    <button
      type="button"
      class="flex items-center gap-1 rounded border px-2 py-1 {bar.isDefault
        ? 'border-zinc-700 bg-zinc-900 text-zinc-300 hover:bg-zinc-800'
        : 'border-blue-500/60 bg-blue-500/15 text-blue-100 hover:bg-blue-500/25'}"
      onclick={() => (expanded = true)}
      title="Sort / filter strategy"
    >
      <span class="text-zinc-500">sort:</span>
      <span>{currentLabel}</span>
      {#if currentIsBeta}
        <span
          class="rounded border border-amber-500/60 bg-amber-500/15 px-1 text-[9px] uppercase tracking-wide text-amber-200"
        >
          beta
        </span>
      {/if}
      <span class="text-zinc-500">▾</span>
    </button>
  {:else}
    {#if sortOptions.length > 1}
      <label class="flex items-center gap-1.5">
        <span class="text-zinc-500">sort</span>
        <select
          bind:value={bar.sort}
          class="rounded border border-zinc-700 bg-zinc-900 px-1.5 py-1 text-zinc-100 focus:border-blue-500 focus:outline-none"
        >
          {#each sortOptions as opt (opt.id)}
            <option value={opt.id}>
              {opt.label}{opt.status === 'experimental' ? ' · beta' : ''}
            </option>
          {/each}
        </select>
      </label>
    {:else}
      <!-- No alternate sorts reported yet (backend not deployed, or
           /curation/methods reports a registry shape this build doesn't
           recognize — both degrade to an empty review_sorts list here
           rather than a broken/empty <select>). -->
      <span class="text-zinc-500">no alternate sorts available yet</span>
    {/if}
    {#if currentIsBeta}
      <span
        class="rounded border border-amber-500/60 bg-amber-500/15 px-1.5 py-0.5 text-[9px] uppercase tracking-wide text-amber-200"
        title="Experimental — not yet fully validated (openprocessor/docs/design/curation_scores.md)"
      >
        beta
      </span>
    {/if}

    {#if mistakenessInfo}
      <label class="flex items-center gap-1">
        <span class="text-zinc-500">min mistakenness</span>
        <input
          type="number"
          min="0"
          max="1"
          step="0.05"
          value={bar.minMistakenness ?? ''}
          oninput={(e) => {
            const v = (e.currentTarget as HTMLInputElement).value;
            bar.minMistakenness = v === '' ? null : Number(v);
          }}
          class="w-14 rounded border border-zinc-700 bg-zinc-900 px-1.5 py-1 text-zinc-100 focus:border-blue-500 focus:outline-none"
        />
      </label>
    {/if}

    {#if nearDupInfo}
      <button
        type="button"
        class="rounded border px-2 py-1 {bar.hideNearDuplicates
          ? 'border-blue-500/60 bg-blue-500/15 text-blue-100'
          : 'border-zinc-700 bg-zinc-900 text-zinc-300 hover:bg-zinc-800'}"
        onclick={() => (bar.hideNearDuplicates = !bar.hideNearDuplicates)}
      >
        hide near-dupes
      </button>
    {/if}

    <!-- Pool-scale overlay: 'diverse' (Phase 4, core-set / k-center-greedy
         selection). Only rendered once diverseSelected is true, i.e. the
         operator picked it AND /curation/methods actually reports it — never
         shown for a backend that hasn't shipped this yet. -->
    {#if diverseSelected}
      <label class="flex items-center gap-1">
        <span class="text-zinc-500">how many diverse crops?</span>
        <input
          type="number"
          min={diverseKMin}
          max={diverseKMax}
          step="1"
          value={bar.k ?? diverseKDefault ?? diverseKMin}
          oninput={(e) => {
            const raw = Number((e.currentTarget as HTMLInputElement).value);
            if (!Number.isFinite(raw)) return;
            bar.k = Math.min(diverseKMax, Math.max(diverseKMin, Math.round(raw)));
          }}
          class="w-16 rounded border border-zinc-700 bg-zinc-900 px-1.5 py-1 text-zinc-100 focus:border-blue-500 focus:outline-none"
          title="Pool-scale diverse selection (core-set / k-center-greedy). Capped {diverseKMin}-{diverseKMax} per request — larger cohorts are a backend job (POST /curation/select/diverse), not a page-size control."
        />
      </label>
      {#if diverseMeta?.method || diverseMeta?.version || diverseMeta?.n_pool != null}
        <span
          class="text-zinc-500"
          title={[
            diverseMeta?.method ? `method: ${diverseMeta.method}` : null,
            diverseMeta?.version ? `version: ${diverseMeta.version}` : null,
          ]
            .filter(Boolean)
            .join(' · ') || undefined}
        >
          {#if diverseMeta?.n_pool != null}
            (from {diverseMeta.n_pool.toLocaleString()} in scope{diverseMeta.n_pool >
            diverseKMax
              ? ', sampled'
              : ''})
          {/if}
        </span>
      {/if}
    {/if}

    {#if !bar.isDefault}
      <button
        type="button"
        class="rounded bg-zinc-800 px-2 py-1 text-zinc-300 hover:bg-zinc-700"
        onclick={() => bar.reset()}
      >
        reset
      </button>
    {/if}

    <button
      type="button"
      class="rounded border border-zinc-700 bg-zinc-900 px-1.5 py-1 text-zinc-400 hover:bg-zinc-800"
      onclick={() => (expanded = false)}
      title="Collapse"
    >
      ×
    </button>
  {/if}
</div>
