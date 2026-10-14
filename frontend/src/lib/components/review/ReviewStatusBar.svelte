<!--
  The /review status bar: queue position, loaded count and paging state.
-->
<script lang="ts">
  import type { Pager } from '$lib/pager.svelte';
  import type { ReviewItem } from '$lib/types_review';

  interface Props {
    queue: Pager<ReviewItem>;
    currentPosition: number;
  }

  let { queue, currentPosition }: Props = $props();
</script>

<!-- Status bar — review is one-at-a-time so there's no "scroll to load more"
     affordance; the queue auto-fetches the next page in the background as
     the cursor advances toward the end of the loaded items. -->
<div
  class="flex items-center justify-between gap-3 border-t border-zinc-800 px-4 py-2 text-sm"
>
  <span class="font-mono text-xs text-zinc-500">
    {queue.items.length > 0 ? currentPosition : 0} / {queue.total}
    {#if queue.items.length < queue.total}
      <span class="ml-1 text-zinc-600">(loaded {queue.items.length})</span>
    {/if}
  </span>
  <span class="font-mono text-xs text-zinc-400">
    {#if queue.loadingMore}loading more…{:else if !queue.hasMore && queue.items.length > 0}all
      loaded{:else if queue.hasMore}auto-fetching{/if}
  </span>
</div>
