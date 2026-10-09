<!--
  A badge for an item's served `embedding_state`: shown only for the states
  that mean "no vector" (`failed`, `deferred`, `not_selected`). `embedded`
  and `null` (written before the field) render nothing. `compact` is the
  `CropCard` chip form.
-->
<script lang="ts">
  import type { EmbeddingState } from '$lib/types_itemFilter';
  import { noVectorCopy } from './embeddingCopy';

  let {
    state: embeddingState,
    compact = false,
  }: { state: EmbeddingState | null; compact?: boolean } = $props();

  const copy = $derived(noVectorCopy(embeddingState));
</script>

{#if copy}
  <span
    class="shrink-0 rounded-sm border px-1 py-0.5 {compact
      ? 'text-[10px]'
      : 'text-xs'} {copy.warning
      ? 'border-amber-500/40 bg-amber-500/15 text-amber-200'
      : 'border-zinc-600 bg-zinc-800 text-zinc-300'}"
    title={`${copy.label}. ${copy.tooltip}`}
    data-testid="embedding-state-badge"
    data-embedding-state={embeddingState}>{compact ? copy.compactLabel : copy.label}</span
  >
{/if}
