<script lang="ts">
  import { apiErrorText } from '$lib/api';
  /**
   * `/export`: limit the export to the items a filter matches. Collapsed by
   * default. While the filter names something, the line "Matching items: N"
   * is the server's `total_crops` for the same filter
   * (`GET /stats/dataset?<filter>`), never a client count; an empty filter
   * shows no count and the export sends no `item_filter`.
   */
  import { untrack } from 'svelte';
  import ItemFilterBar from './ItemFilterBar.svelte';
  import { getMatchingItemCount } from '$lib/api';
  import { withoutOpenVocab } from '$lib/itemFilter/itemFilterState.svelte';
  import type { ItemFilterState } from '$lib/itemFilter/itemFilterState.svelte';

  interface Props {
    state: ItemFilterState;
  }

  let { state: filter }: Props = $props();

  let count = $state<number | null>(null);
  let error = $state<string | null>(null);
  let loading = $state(false);

  const query = $derived(filter.toQuery(withoutOpenVocab));
  const queryKey = $derived(JSON.stringify(query));
  const active = $derived(Object.keys(query).length > 0);
  let seq = 0;

  $effect(() => {
    void queryKey;
    untrack(() => {
      const mine = ++seq;
      count = null;
      error = null;
      if (!active) {
        loading = false;
        return;
      }
      loading = true;
      getMatchingItemCount(query)
        .then((n) => {
          if (mine === seq) count = n;
        })
        .catch((e: unknown) => {
          if (mine !== seq) return;
          error = apiErrorText(e);
        })
        .finally(() => {
          if (mine === seq) loading = false;
        });
    });
  });
</script>

<details class="mt-3 text-xs" open={active} data-testid="export-item-filter">
  <summary class="cursor-pointer text-zinc-400">Only items matching a filter</summary>
  <div class="mt-2 space-y-2">
    <ItemFilterBar state={filter} visible={withoutOpenVocab} {error} />
    {#if active}
      <p class="text-zinc-300" data-testid="export-matching-count">
        {#if loading}
          Counting…
        {:else if count != null}
          Matching items: <strong class="text-zinc-100">{count.toLocaleString()}</strong>
        {/if}
      </p>
    {/if}
  </div>
</details>
