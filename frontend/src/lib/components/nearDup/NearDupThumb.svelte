<!--
  One group member's picture: the served `thumbnail_url` (an image thumbnail
  at the image level, a crop thumbnail at the item level).
-->
<script lang="ts">
  import { resolveApiUrl } from '$lib/api';

  let {
    url,
    label,
    size = 'md',
  }: {
    url: string;
    label: string;
    size?: 'sm' | 'md';
  } = $props();

  let failed = $state(false);
  const box = $derived(size === 'sm' ? 'h-16 w-16' : 'h-28 w-28');
</script>

{#if !failed}
  <img
    src={resolveApiUrl(url)}
    alt={label}
    loading="lazy"
    onerror={() => (failed = true)}
    class="{box} shrink-0 rounded border border-line bg-surface-1 object-cover"
    data-testid="near-dup-thumb"
  />
{:else}
  <div
    class="{box} flex shrink-0 items-center justify-center rounded border border-line bg-surface-1 text-center text-caption text-fg-3"
    data-testid="near-dup-thumb-missing"
  >
    no preview
  </div>
{/if}
