<!--
  The "Embedding" row of `CropMetaPanel`'s main <dl>: the served
  `embedding_state` (a `<dt>/<dd>` pair). `null` is an item written before
  the field existed.
-->
<script lang="ts">
  import type { Crop } from '$lib/types';
  import EmbeddingStateBadge from '$lib/components/embedding/EmbeddingStateBadge.svelte';

  let { crop }: { crop: Crop } = $props();

  const state = $derived(crop.embedding_state ?? null);
</script>

<dt class="text-zinc-500">Embedding</dt>
<dd class="text-zinc-200" data-testid="embedding-row">
  {#if state === null}
    unknown (written before v0.4.0)
  {:else if state === 'embedded'}
    embedded
  {:else}
    <EmbeddingStateBadge {state} />
  {/if}
</dd>
