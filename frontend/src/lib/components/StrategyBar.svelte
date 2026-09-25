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
   * Sort options come straight from `strategiesStore` (`{API_PREFIX}/methods`),
   * filtered to `stable`/`experimental` only — `shadow`/`disabled` entries
   * must never be selectable (they're either mid-validation or explicitly
   * killed) — and, since audit-remediation plan Phase 6 (P1-2), further
   * filtered by `hasFieldCoverage` to drop any entry the backend reports
   * as genuinely 0% covered (e.g. `mistakenness` today — no probe
   * checkpoint has ever run, so offering it would be a control with no
   * effect). An entry whose coverage is merely *unknown* (`null`/absent —
   * a transient backend failure, or `requires_field: null`) still shows;
   * only a confirmed `field_coverage === 0` hides it. The currently-selected
   * sort gets a small amber "beta" badge when its status is `experimental`.
   *
   * Filter chips (min-mistakenness threshold, hide-near-duplicates) render
   * whenever the matching `scores`/`overlays` entry passes the same
   * `hasFieldCoverage` gate (P1-3) — hidden only on a confirmed-zero
   * `field_coverage`, not on unknown/absent coverage, so a transient
   * `{API_PREFIX}/methods` hiccup can never make a chip that already works vanish.
   *
   * Phase 4 adds the pool-scale `'diverse'` overlay (core-set /
   * k-center-greedy selection) as an option `/clusters/[id]` can fold into
   * this same sort selector, plus a `k` stepper ("how many diverse
   * crops?") that only appears once `'diverse'` is both selected and
   * actually offered — see `isDiverseOverlayAvailable` in `strategies.ts`,
   * the single gate `/clusters/[id]` and this component both use.
   */

  import { untrack } from 'svelte';
  import ChevronDownIcon from './ChevronDownIcon.svelte';
  import {
    hasFieldCoverage,
    isDiverseOverlayAvailable,
    type MethodStatus,
  } from '$lib/strategies';
  import {
    formatAppliedSort,
    formatPinnedSortFallback,
    type StrategyBar,
  } from '$lib/strategyBar.svelte';
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
     *  as `{API_PREFIX}/crops?order=`, which only special-cases `'outliers'` and
     *  (Phase 4) `'diverse'` today (docs/curation-strategy-plan-2026-09.md
     *  §1/§7) — so that route passes `['default', 'outliers', 'diverse']`
     *  to avoid offering an id the endpoint would silently ignore.
     *  `'diverse'` only actually appears in the rendered options when
     *  `{API_PREFIX}/methods` reports it (see `sortOptions` below) — passing the id
     *  here is necessary but not sufficient. Omit for `/review`, where
     *  every stable/experimental `review_sorts` entry is fair game. */
    allowedIds?: string[] | null;
    /** Whether to offer the score-filter chips at all. `/clusters/[id]`
     *  passes `false` — its crop query doesn't forward these params. */
    showFilters?: boolean;
    /** Offer the pool-scale `'diverse'` overlay in the sort dropdown
     *  without inheriting `allowedIds`'s other restrictions (P2-10,
     *  `/review`). `/clusters/[id]` continues to get `'diverse'` via
     *  `allowedIds` instead — this prop exists so a route that wants
     *  every `review_sorts` entry AND `'diverse'` doesn't have to
     *  enumerate every sort id by hand. Still gated by
     *  `isDiverseOverlayAvailable` like every other `'diverse'` path. */
    offerDiverse?: boolean;
    /** Seed value for the `k` stepper the first time the operator selects
     *  `'diverse'` and `bar.k` is still unset. Route-specific ("current
     *  page size" on `/clusters/[id]`) — omit on routes that never offer
     *  `'diverse'`. Falls back to `diverseKMin` if omitted. */
    diverseKDefault?: number;
    /** `k` stepper bounds. Min 4: below that "diversity" over a handful of
     *  items isn't a meaningful selection criterion — a plain click-through
     *  is just as fast. Max 500: matches `{API_PREFIX}/crops`'s own `page_size`
     *  ceiling (`Query(..., le=500)`, crops.py) — the same number the
     *  backend already treats as "a single request's worth," vs. the
     *  thousands-scale pool selection the plan reserves for a backend job
     *  (`OP_SELECT_MAX_N`, `POST {API_PREFIX}/select/diverse`), not a page control. */
    diverseKMin?: number;
    diverseKMax?: number;
    /** Provenance for the current diverse-mode response, if the backend
     *  sent any (see `DiverseMeta`). Surfaced as a small inline note/tooltip
     *  next to the stepper — not a new prominent UI element. */
    diverseMeta?: DiverseMeta | null;
    /** The sort id the backend actually applied (`sort_applied` on the
     *  review-queue response, 2026-09-24 logic-moves W5 item 10) — may
     *  differ from `bar.sort` (a tab's own default beats the deployment
     *  default when the operator hasn't picked one, or a requested sort
     *  fell back). Omit on a route that doesn't have it yet. */
    appliedSort?: string | null;
    /** `sort_fallback_reason` from `GET {API_PREFIX}/review/{tab}` (or its
     *  `/locate`) — a human-readable string when the resolved sort (a
     *  tab default or an explicit request) fell back because its field
     *  has 0% coverage (M11, docs/design/interactive-pass-2026-09-24.md).
     *  Shown right next to the `sort_applied` chip, not as a separate
     *  banner, so the "what did the backend actually order by, and why"
     *  story lives in one place. `null`/omitted renders nothing. */
    fallbackReason?: string | null;
    /** The deployment's pinned `sort` default (`GET {API_PREFIX}/settings`'s
     *  `defaults.sort`), when it's resolved and applies to the caller's
     *  active tab — the caller decides applicability (see
     *  `tabHonorsPinnedSortDefault`, `reviewTabs.ts`) and passes `null`
     *  otherwise. Visual-audit S1's last bullet: when this pinned entry's
     *  `/methods` `field_coverage` is confirmed zero and the operator
     *  hasn't picked their own override, the summary chip says so and
     *  names the backend's real `sort_applied` instead of the plain
     *  requested-vs-applied mismatch text. Omit on a route that has no
     *  concept of a pinned deployment default (`/clusters/[id]`). */
    pinnedSortId?: string | null;
  }

  let {
    bar,
    allowedIds = null,
    showFilters = true,
    offerDiverse = false,
    diverseKDefault,
    diverseKMin = 4,
    diverseKMax = 500,
    diverseMeta = null,
    appliedSort = null,
    fallbackReason = null,
    pinnedSortId = null,
  }: Props = $props();

  // getMethods()/init() never throws (404 or any error degrades to
  // FALLBACK_METHODS) and init() itself is idempotent — safe to call on
  // every mount without a guard. No listener is installed by this call.
  $effect(() => {
    void strategiesStore.init();
  });

  // `bar.sort`'s value the moment this component mounts — the "no
  // override" sentinel (`strategyBar.svelte.ts`'s `defaultId`, 'default'
  // at both current call sites). Captured once (plain `let`, not
  // `$derived`) rather than read live: sortOptions must keep offering a
  // way back to "no override" even after the operator picks a real
  // strategy and `bar.sort` moves away from this value — reading
  // `bar.sort` live for this would make the synthetic option vanish the
  // moment it stopped being selected (see the fix note below).
  const sentinelSortId = untrack(() => bar.sort);

  const sortOptions = $derived.by(() => {
    // Only a caller that restricts allowedIds (today: `/clusters/[id]`) —
    // or one that opts in via offerDiverse (today: `/review`, P2-10) —
    // also considers the `overlays` registry — `'diverse'` lives there,
    // not in `review_sorts` (plan §3: overlays never write cluster_id, a
    // different axis from review-queue sort/filter, even though these
    // pages fold both into one `orderMode`/sort selection for UX
    // simplicity). Plain `/review` pre-P2-10 (allowedIds=null,
    // offerDiverse=false) is unaffected: only `review_sorts` is
    // considered there, exactly as before Phase 4.
    // Only 'diverse' is ever folded in from the overlays registry — other
    // overlay entries (e.g. 'viz_projection') aren't sort-shaped and must
    // never leak into this dropdown.
    const diverseOverlayOnly = strategiesStore.methods.overlays.filter(
      (o) => o.id === 'diverse',
    );
    const source =
      allowedIds || offerDiverse
        ? [...strategiesStore.methods.review_sorts, ...diverseOverlayOnly]
        : strategiesStore.methods.review_sorts;
    const seen = new Set<string>();
    const deduped = source.filter((s) => {
      if (seen.has(s.id)) return false;
      seen.add(s.id);
      return true;
    });
    // Audit-remediation plan Phase 6 (P1-2): a sort backed by a field with
    // real, confirmed-zero coverage (e.g. `mistakenness` today -- no probe
    // checkpoint has ever run) is genuinely inert, so it's excluded here,
    // not just left selectable-but-useless. `hasFieldCoverage` is what
    // keeps this from also excluding a sort whose coverage is merely
    // unknown (null/undefined) -- see that function's doc comment.
    const stableOrExperimental = deduped.filter(
      (s) =>
        (s.status === 'stable' || s.status === 'experimental') && hasFieldCoverage(s),
    );
    // `allowedIds` still restricts to an explicit id list when given
    // (`/clusters/[id]`'s `{API_PREFIX}/crops?order=` only special-cases a few
    // ids). `offerDiverse` with no `allowedIds` (today: `/review`) needs
    // no further filtering here — `source` above already limited the
    // overlays half to exactly `'diverse'`, so every review_sorts entry
    // plus that one overlay entry is exactly what should render.
    const filtered = allowedIds
      ? stableOrExperimental.filter((s) => allowedIds!.includes(s.id))
      : stableOrExperimental;
    // `sentinelSortId` ("no override") is never a real `{API_PREFIX}/methods`
    // entry. Without a matching <option> the <select> either silently
    // falls back to displaying its first real option while `bar.sort`
    // stays on the sentinel (a DOM/state mismatch), or — on
    // `/clusters/[id]`, where `allowedIds` restricts to ['default',
    // 'outliers', 'diverse'] and neither 'default' nor 'outliers' is
    // ever a registry entry — `filtered` can be at most length 1
    // ('diverse'), so the old `length > 1` gate could never render the
    // control at all. Always include a synthetic "back to default"
    // option, unconditionally (not just when `bar.sort` currently
    // doesn't match anything) — gating on the *live* `bar.sort` would
    // make this entry vanish the instant the operator picked the one
    // real option, collapsing the list back below the render threshold
    // and leaving no way to switch back.
    if (!filtered.some((s) => s.id === sentinelSortId)) {
      return [
        { id: sentinelSortId, label: 'Default order', status: 'stable' as MethodStatus },
        ...filtered,
      ];
    }
    return filtered;
  });

  const currentSort = $derived(sortOptions.find((s) => s.id === bar.sort) ?? null);
  const currentIsBeta = $derived(currentSort?.status === 'experimental');
  const currentLabel = $derived(currentSort?.label ?? bar.sort);
  // What the backend actually ordered by, when it differs from what the
  // operator picked (a tab default beating an unset selection, or a
  // requested sort falling back) — null when there's nothing to add.
  const appliedLabel = $derived(
    formatAppliedSort(bar.sort, appliedSort, strategiesStore.methods.review_sorts),
  );

  // Visual-audit S1's last bullet: the pinned deployment default (when the
  // caller resolved one for this tab) failing field coverage. Computed
  // from the same `strategiesStore.methods.review_sorts` list every other
  // sort-label lookup on this component already reads — no second fetch.
  const pinnedSortEntry = $derived(
    pinnedSortId
      ? (strategiesStore.methods.review_sorts.find((s) => s.id === pinnedSortId) ?? null)
      : null,
  );
  const pinnedFallbackLabel = $derived(
    formatPinnedSortFallback({
      requestedSort: bar.sort,
      sentinelSortId,
      pinnedSortId,
      pinnedSortEntry,
      appliedSort,
      appliedLabel,
    }),
  );

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

  // Audit-remediation plan Phase 6 (P1-3): delegates the null-vs-zero
  // distinction to the shared, unit-tested `hasFieldCoverage` (strategies.ts)
  // instead of the old local nullish-coalesce-to-zero-then-compare gate,
  // which was the bug -- it silently treated "coverage unknown" (undefined,
  // on every backend before this phase) the same as "coverage confirmed
  // zero," hiding these chips permanently regardless of real data.
  function hasCoverage(entry: {
    status: MethodStatus;
    field_coverage?: number | null;
  }): boolean {
    return (
      (entry.status === 'stable' || entry.status === 'experimental') &&
      hasFieldCoverage(entry)
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

<div class="inline-flex min-w-0 max-w-full flex-wrap items-center gap-1.5 text-xs">
  {#if !expanded}
    <button
      type="button"
      class="btn-sm min-w-0 max-w-full {bar.isDefault
        ? 'border-zinc-700 bg-zinc-900 text-zinc-300 hover:bg-zinc-800'
        : 'border-blue-500/60 bg-blue-500/15 text-blue-100 hover:bg-blue-500/25'}"
      onclick={() => (expanded = true)}
      title="Sort / filter strategy"
    >
      <span class="text-zinc-500">sort:</span>
      <span>{currentLabel}</span>
      {#if pinnedFallbackLabel}
        <!-- S1 (visual audit 2026-09-24): merges with, never doubles up
             alongside, the plain "→ applied" mismatch chip below — a
             zero-coverage pinned default is exactly why applied differs
             from requested here, so only this more specific message
             renders. -->
        <span
          data-testid="pinned-sort-fallback-chip"
          class="inline-block max-w-[20rem] truncate rounded border border-amber-500/60 bg-amber-500/15 px-1 align-middle text-[10px] text-amber-200 md:max-w-[32rem]"
          title={`The deployment's pinned default review-queue sort has no data yet, so this tab is using the backend's own fallback order instead. ${pinnedFallbackLabel}`}
        >
          {pinnedFallbackLabel}
        </span>
      {:else if appliedLabel}
        <span
          class="text-zinc-500"
          title="The backend applied this sort — a tab default, or a fallback from what was requested."
        >
          → {appliedLabel}
        </span>
      {/if}
      {#if fallbackReason}
        <!-- R3 (visual audit 2026-09-24): the served reason is a long
             sentence — truncated here (full text in the tooltip) so it can
             never push the bar past a narrow viewport. -->
        <span
          data-testid="sort-fallback-chip"
          class="inline-block max-w-[16rem] truncate rounded border border-amber-500/60 bg-amber-500/15 px-1 align-middle text-[10px] text-amber-200 md:max-w-[28rem]"
          title={`The requested sort couldn't be honored server-side — showing the fallback order instead. ${fallbackReason}`}
        >
          fallback: {fallbackReason}
        </span>
      {/if}
      {#if currentIsBeta}
        <span
          class="rounded border border-amber-500/60 bg-amber-500/15 px-1 text-[9px] uppercase tracking-wide text-amber-200"
        >
          beta
        </span>
      {/if}
      <span class="text-zinc-500"><ChevronDownIcon size={15} /></span>
    </button>
  {:else}
    {#if sortOptions.length > 1}
      <label class="flex items-center gap-1.5">
        <span class="text-zinc-500">sort</span>
        <select bind:value={bar.sort} class="select-sm">
          {#each sortOptions as opt (opt.id)}
            <option value={opt.id}>
              {opt.label}{opt.status === 'experimental' ? ' · beta' : ''}
            </option>
          {/each}
        </select>
      </label>
      {#if pinnedFallbackLabel}
        <span data-testid="pinned-sort-fallback-chip" class="text-amber-300"
          >{pinnedFallbackLabel}</span
        >
      {:else if appliedLabel}
        <span class="text-zinc-500">applied: {appliedLabel}</span>
      {/if}
      {#if fallbackReason}
        <span class="text-amber-300">fallback: {fallbackReason}</span>
      {/if}
    {:else}
      <!-- No alternate sorts reported yet (backend not deployed, or
           {API_PREFIX}/methods reports a registry shape this build doesn't
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
          class="input-sm w-14"
        />
      </label>
    {/if}

    {#if nearDupInfo}
      <button
        type="button"
        class="btn-sm {bar.hideNearDuplicates
          ? 'border-blue-500/60 bg-blue-500/15 text-blue-100'
          : 'border-zinc-700 bg-zinc-900 text-zinc-300 hover:bg-zinc-800'}"
        onclick={() => (bar.hideNearDuplicates = !bar.hideNearDuplicates)}
      >
        hide near-dupes
      </button>
    {/if}

    <!-- Pool-scale overlay: 'diverse' (Phase 4, core-set / k-center-greedy
         selection). Only rendered once diverseSelected is true, i.e. the
         operator picked it AND {API_PREFIX}/methods actually reports it — never
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
          class="input-sm w-16"
          title="Pool-scale diverse selection (core-set / k-center-greedy). Capped {diverseKMin}-{diverseKMax} per request — larger cohorts are a backend job (POST select/diverse), not a page-size control."
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
        class="btn-sm bg-zinc-800 text-zinc-300 hover:bg-zinc-700"
        onclick={() => bar.reset()}
      >
        reset
      </button>
    {/if}

    <button
      type="button"
      class="btn-sm btn-icon border-zinc-700 bg-zinc-900 text-zinc-400 hover:bg-zinc-800"
      onclick={() => (expanded = false)}
      title="Collapse"
    >
      ×
    </button>
  {/if}
</div>
