<!--
  One group member's picture. An item-level member is an item id and takes its
  crop thumbnail; an image-level member is an image id and takes the image's
  own thumbnail.
-->
<script lang="ts">
  import { getImageThumbUrl, getThumbUrl } from '$lib/api_cropRead';
  import type { NearDupLevel } from '$lib/types_nearDup';

  let {
    level,
    memberId,
    size = 'md',
  }: {
    level: NearDupLevel;
    memberId: string;
    size?: 'sm' | 'md';
  } = $props();

  const src = $derived(
    level === 'item' ? getThumbUrl(memberId, 256) : getImageThumbUrl(memberId, 360),
  );
  let failed = $state(false);
  const box = $derived(size === 'sm' ? 'h-16 w-16' : 'h-28 w-28');
</script>

{#if !failed}
  <img
    {src}
    onerror={() => (failed = true)}
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
