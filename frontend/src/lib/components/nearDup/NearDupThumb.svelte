<!--
  One group member's picture. An item-level member is an item id and takes its
  crop thumbnail; an image-level member is an image id, shown as the source
  frame of an item standing on it (`previews`, served by the items list).
-->
<script lang="ts">
  import { getSourceImageScaled, getThumbUrl } from '$lib/api_cropRead';
  import type { NearDupLevel } from '$lib/types_nearDup';

  let {
    level,
    memberId,
    previews,
    size = 'md',
  }: {
    level: NearDupLevel;
    memberId: string;
    previews: Record<string, string>;
    size?: 'sm' | 'md';
  } = $props();

  const src = $derived.by(() => {
    if (level === 'item') return getThumbUrl(memberId, 256);
    const itemId = previews[memberId];
    return itemId ? getSourceImageScaled(itemId, 360) : null;
  });
  const box = $derived(size === 'sm' ? 'h-16 w-16' : 'h-28 w-28');
</script>

{#if src}
  <img
    {src}
    alt={`${level} ${memberId}`}
    loading="lazy"
    class="{box} shrink-0 rounded border border-zinc-800 bg-zinc-900 object-cover"
    data-testid="near-dup-thumb"
  />
{:else}
  <div
    class="{box} flex shrink-0 items-center justify-center rounded border border-zinc-800 bg-zinc-900 text-center text-[10px] text-zinc-500"
    data-testid="near-dup-thumb-missing"
  >
    no preview
  </div>
{/if}
