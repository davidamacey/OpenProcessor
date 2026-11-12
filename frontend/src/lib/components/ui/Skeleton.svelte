<script lang="ts">
  /**
   * A loading placeholder: `lines` grey bars (the last one shorter), or one block when `block`
   * is set (give it a height class). Announced once as "Loading" through the status role;
   * the bars are hidden from assistive tech. The pulse stops under prefers-reduced-motion
   * (app.css).
   */
  interface Props {
    lines?: number;
    block?: boolean;
    class?: string;
  }
  let { lines = 3, block = false, class: className = '' }: Props = $props();
</script>

<div role="status" aria-label="Loading" class="animate-pulse {className}">
  {#if block}
    <div
      class="h-full min-h-8 w-full rounded-control bg-surface-2"
      aria-hidden="true"
    ></div>
  {:else}
    <div class="flex flex-col gap-2" aria-hidden="true">
      {#each Array.from({ length: lines }, (_, i) => i) as i (i)}
        <div
          class="h-3.5 rounded-chip bg-surface-2 {i === lines - 1 && lines > 1
            ? 'w-2/3'
            : 'w-full'}"
        ></div>
      {/each}
    </div>
  {/if}
</div>
