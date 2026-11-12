<!--
  The "Embedding" row of `CropMetaPanel`'s main <dl>: the served
  `embedding_states` (a `<dt>/<dd>` pair), one line per embedding space. A space
  with no key has no state for the item.
-->
<script lang="ts">
  import type { Crop } from '$lib/types';
  import EmbeddingStateBadge from '$lib/components/embedding/EmbeddingStateBadge.svelte';

  let { crop }: { crop: Crop } = $props();

  const states = $derived(Object.entries(crop.embedding_states ?? {}));
</script>

<dt class="text-fg-3">Embedding</dt>
<dd class="text-fg-1" data-testid="embedding-row">
  {#if states.length === 0}
    no state served
  {:else}
    {#each states as [space, state] (space)}
      <div data-testid="embedding-row-space" data-space={space}>
        <span class="font-mono text-fg-2">{space}</span>
        {#if state === 'embedded'}
          embedded
        {:else}
          <EmbeddingStateBadge {state} {space} />
        {/if}
      </div>
    {/each}
  {/if}
</dd>
