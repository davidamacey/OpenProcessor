<!--
  "N items in scope have no vector": shown under ordered views (outliers,
  diverse, core-first) and semantic search results when the served count is
  above zero. "Embed them" opens the Reprocess dialog with the served
  `suggested_reprocess`, as served; without one there is no action.
-->
<script lang="ts">
  import ReprocessControl from '$components/datasets/ReprocessControl.svelte';
  import type { ReprocessRequest } from '$lib/types_import';

  let {
    count,
    context,
    suggestedReprocess = null,
    onapplied,
  }: {
    count: number | null | undefined;
    context: 'ordered' | 'search';
    suggestedReprocess?: ReprocessRequest | null;
    onapplied?: () => void;
  } = $props();

  const text = $derived(
    `${(count ?? 0).toLocaleString()} items in scope have no vector and ${
      context === 'ordered' ? 'are not ranked' : 'cannot match a text search'
    }`,
  );
</script>

{#if count != null && count > 0}
  <span
    class="inline-flex flex-wrap items-center gap-2 rounded border border-amber-500/40 bg-amber-500/10 px-2 py-0.5 text-xs text-amber-200"
    data-testid="unembedded-banner"
  >
    {text}
    {#if suggestedReprocess}
      <ReprocessControl
        target={{ kind: 'request', request: suggestedReprocess }}
        buttonLabel="Embed them"
        {onapplied}
      />
    {/if}
  </span>
{/if}
