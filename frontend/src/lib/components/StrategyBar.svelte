<script lang="ts">
  /**
   * Shared sort/filter strategy control (curation-strategy plan Phase 3 —
   * docs/curation-strategy-plan-2026-09.md §5.1/§5.4). Used on both
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
   */

  import type { MethodStatus } from '$lib/strategies';
  import type { StrategyBar } from '$lib/strategyBar.svelte';
  import { strategiesStore } from '$stores/strategies.svelte';

  interface Props {
    bar: StrategyBar;
    /** Restrict selectable sort ids beyond the stable/experimental status
     *  filter. `/clusters/[id]`'s crop query forwards its selection as
     *  `/curation/crops?order=`, which only special-cases `'outliers'` today
     *  (docs/curation-strategy-plan-2026-09.md §1) — so that route passes
     *  `['default', 'outliers']` to avoid offering an id the endpoint
     *  would silently ignore. Omit for `/review`, where every
     *  stable/experimental `review_sorts` entry is fair game. */
    allowedIds?: string[] | null;
    /** Whether to offer the score-filter chips at all. `/clusters/[id]`
     *  passes `false` — its crop query doesn't forward these params. */
    showFilters?: boolean;
  }

  let { bar, allowedIds = null, showFilters = true }: Props = $props();

  // getMethods()/init() never throws (404 or any error degrades to
  // FALLBACK_METHODS) and init() itself is idempotent — safe to call on
  // every mount without a guard. No listener is installed by this call.
  $effect(() => {
    void strategiesStore.init();
  });

  const sortOptions = $derived.by(() => {
    const stableOrExperimental = strategiesStore.methods.review_sorts.filter(
      (s) => s.status === 'stable' || s.status === 'experimental',
    );
    return allowedIds
      ? stableOrExperimental.filter((s) => allowedIds!.includes(s.id))
      : stableOrExperimental;
  });

  const currentSort = $derived(sortOptions.find((s) => s.id === bar.sort) ?? null);
  const currentIsBeta = $derived(currentSort?.status === 'experimental');
  const currentLabel = $derived(currentSort?.label ?? bar.sort);

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
