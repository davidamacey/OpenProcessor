<script lang="ts">
  import type { SourceMissing } from '$lib/sourceMissing';

  /** `interactive` is a strip along the bottom that takes the pointer (its tooltip works); the
   *  default covers the picture and ignores the pointer so the card under it stays clickable. */
  let {
    missing,
    compact = false,
    interactive = false,
  }: { missing: SourceMissing; compact?: boolean; interactive?: boolean } = $props();
</script>

<div
  role="status"
  title={missing.message}
  class="absolute flex flex-col items-center justify-center gap-0.5 bg-surface-1/80 p-1 text-center text-fg-1 {interactive
    ? 'inset-x-1 bottom-1 rounded'
    : 'pointer-events-none inset-0'}"
>
  <span class={compact ? 'text-caption font-semibold' : 'text-body font-semibold'}>
    {missing.root_mounted ? 'Image file missing' : 'Source not mounted'}
  </span>
  <code class="max-w-full truncate {compact ? 'text-caption' : 'text-meta'}">
    {missing.alias}
  </code>
  {#if !compact}
    <span class="text-meta text-fg-2">{missing.message}</span>
  {/if}
</div>
