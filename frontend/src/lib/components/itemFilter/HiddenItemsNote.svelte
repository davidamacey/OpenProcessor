<script lang="ts">
  /**
   * Says how many items a list leaves out, so its total can differ from the
   * dashboard's. Both counts are the server's (`hidden` on the item list pages).
   */
  import { formatCount } from '$lib/format/count';
  import type { HiddenCounts } from '$lib/types';

  let { hidden }: { hidden: HiddenCounts | null } = $props();

  const parts = $derived(
    hidden
      ? [
          hidden.test_holdout > 0
            ? `${formatCount(hidden.test_holdout)} test holdout`
            : null,
          hidden.excluded > 0 ? `${formatCount(hidden.excluded)} ignored` : null,
        ].filter((p): p is string => p !== null)
      : [],
  );
</script>

{#if parts.length > 0}
  <span
    class="text-fg-3"
    data-testid="hidden-items-note"
    title="These items are kept out of every list unless you ask for them."
  >
    · {parts.join(' and ')} not listed
  </span>
{/if}
