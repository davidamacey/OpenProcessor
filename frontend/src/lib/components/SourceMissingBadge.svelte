<script lang="ts">
  import type { SourceMissing } from '$lib/sourceMissing';

  let { missing, compact = false }: { missing: SourceMissing; compact?: boolean } = $props();
</script>

<div
  role="status"
  title={missing.message}
  class="pointer-events-none absolute inset-0 flex flex-col items-center justify-center gap-0.5 bg-zinc-900/80 p-1 text-center text-zinc-200"
>
  <span class={compact ? 'text-[10px] font-semibold' : 'text-sm font-semibold'}>
    {missing.root_mounted ? 'Image file missing' : 'Source not mounted'}
  </span>
  <span class="max-w-full truncate font-mono {compact ? 'text-[9px]' : 'text-xs'}">
    {missing.alias}
  </span>
  {#if !compact}
    <span class="text-xs text-zinc-400">{missing.message}</span>
  {/if}
</div>
