<script lang="ts">
  /**
   * Free-text semantic search over vehicle crops (P2-14 — `GET
   * /curation/search/text`). Renders alongside `<StrategyBar>` on `/review` and
   * `/clusters`/`/clusters/[id]`, gated by `isSemanticSearchAvailable`
   * (`$lib/strategies`) exactly like every other overlay control in this
   * repo — a caller must not mount this against a backend/flag that
   * hasn't shipped `semantic_search`.
   *
   * This component owns exactly the query box: debounced-vs-immediate
   * submit, in-flight request cancellation, loading/error state (see
   * `$lib/searchBox.svelte.ts`, unit-tested independently). It never
   * renders results itself — `onResults`/`onClear` hand raw
   * `{items, total}` / "back to normal" back to the host page, which
   * feeds an existing `Pager`'s settable `items`/`total` (see
   * `pager.svelte.ts`) so the host's existing CropCard grid, selection,
   * DnD, and label hotkeys keep working completely unchanged — this is
   * deliberately not a parallel rendering path.
   */
  import { searchCrops } from '$lib/api';
  import { createSemanticSearchBox } from '$lib/searchBox.svelte';
  import type { SemanticSearchResult } from '$lib/searchBox.svelte';
  import { onMount } from 'svelte';

  interface Props {
    /** Extra query params threaded to `GET /curation/search/text` — e.g.
     *  `{cluster_id}` on `/clusters/[id]`, or the effective tab + live
     *  filter object on `/review`. Omitting any scoping key (e.g. no
     *  `cluster_id`) means "search everything" — the global `/clusters`
     *  search deliberately passes no cluster/tab scope. */
    filter?: Record<string, unknown>;
    pageSize?: number;
    placeholder?: string;
    /** Seed the box with a query (e.g. from `?q=` on mount) and fire it
     *  immediately, bypassing the debounce — same as pressing Enter. */
    initialQuery?: string | null;
    onResults?: (res: SemanticSearchResult) => void;
    onClear?: () => void;
    /** Fires with the live query text on every keystroke — lets a host
     *  (e.g. /clusters' search-mode header bar + `?q=` URL sync) track
     *  what's currently typed without this component owning navigation. */
    onQueryChange?: (q: string) => void;
  }

  let {
    filter = {},
    pageSize = 30,
    placeholder = 'Search crops (e.g. "red sedan", "pickup at night")…',
    initialQuery = null,
    onResults,
    onClear,
    onQueryChange,
  }: Props = $props();

  // Last-result bookkeeping for the "Showing X of Y results for '…'" line —
  // `createSemanticSearchBox` only ever hands `{items, total}` up to the
  // host via `onResults`/`onClear`; it doesn't retain them itself, so this
  // component tracks just enough of its own last search to render the
  // count without duplicating the host's actual result state.
  let lastShown = $state(0);
  let lastTotal = $state(0);
  let lastQuery = $state('');

  const box = createSemanticSearchBox({
    debounceMs: 300,
    search: (q, signal) => searchCrops(q, 1, pageSize, filter, signal),
    onResults: (res) => {
      lastShown = res.items.length;
      lastTotal = res.total;
      lastQuery = box.query.trim();
      onResults?.(res);
    },
    onClear: () => onClear?.(),
  });

  onMount(() => {
    if (initialQuery) {
      box.query = initialQuery;
      onQueryChange?.(initialQuery);
      box.submit();
    }
  });
</script>

<div class="inline-flex items-center gap-2 text-xs">
  <div class="relative w-72 shrink-0">
    <span
      class="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-zinc-500"
      aria-hidden="true"
    >
      <svg
        width="13"
        height="13"
        viewBox="0 0 16 16"
        fill="none"
        stroke="currentColor"
        stroke-width="1.6"
        stroke-linecap="round"
      >
        <circle cx="6.75" cy="6.75" r="5" />
        <line x1="10.5" y1="10.5" x2="14.5" y2="14.5" />
      </svg>
    </span>
    <input
      type="text"
      value={box.query}
      oninput={(e) => {
        const v = (e.currentTarget as HTMLInputElement).value;
        box.oninput(v);
        onQueryChange?.(v);
      }}
      onkeydown={(e) => {
        if (e.key === 'Enter') {
          e.preventDefault();
          box.submit();
        }
      }}
      {placeholder}
      spellcheck="false"
      class="w-full rounded-full border border-zinc-700 bg-zinc-900 py-1.5 pl-8 pr-7 text-zinc-100 placeholder:text-zinc-500 focus:border-blue-500 focus:outline-none"
    />
    {#if box.query}
      <button
        type="button"
        class="absolute right-1.5 top-1/2 -translate-y-1/2 rounded-full px-1.5 py-0.5 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200"
        onclick={() => {
          box.clear();
          onQueryChange?.('');
        }}
        title="Clear search"
      >
        ×
      </button>
    {/if}
  </div>

  {#if box.loading}
    <span class="text-zinc-500">searching…</span>
  {:else if box.error}
    <span class="rounded border border-red-500/50 bg-red-500/10 px-1.5 py-0.5 text-red-200">
      {box.error}
    </span>
  {:else if box.active}
    <span class="text-zinc-400">
      Showing {lastShown} of {lastTotal} result{lastTotal === 1 ? '' : 's'} for “{lastQuery}”
    </span>
  {/if}
</div>
