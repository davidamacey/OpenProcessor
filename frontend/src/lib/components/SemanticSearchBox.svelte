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

  const box = createSemanticSearchBox({
    debounceMs: 300,
    search: (q, signal) => searchCrops(q, 1, pageSize, filter, signal),
    onResults: (res) => onResults?.(res),
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

<div class="inline-flex flex-1 items-center gap-1.5 text-xs">
  <div class="relative flex-1">
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
      class="w-full rounded border border-zinc-700 bg-zinc-900 px-2 py-1 pr-7 text-zinc-100 placeholder:text-zinc-500 focus:border-blue-500 focus:outline-none"
    />
    {#if box.query}
      <button
        type="button"
        class="absolute right-1 top-1/2 -translate-y-1/2 rounded px-1 text-zinc-500 hover:text-zinc-200"
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
  {/if}
  {#if box.error}
    <span class="rounded border border-red-500/50 bg-red-500/10 px-1.5 py-0.5 text-red-200">
      {box.error}
    </span>
  {/if}
</div>
