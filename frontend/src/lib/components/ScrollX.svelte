<script lang="ts">
  /**
   * A horizontally scrolling box that shows it can scroll (F8 D5: the
   * `/bakeoff` ranked table cut off after "Precision" at 800px with nothing
   * on screen saying more columns existed). While content is hidden to the
   * right, a solid edge marker and a "scroll →" hint sit on the right edge
   * (no gradient, per the style rules); both go away at the end.
   */
  import type { Snippet } from 'svelte';

  interface Props {
    children: Snippet;
    class?: string;
    testId?: string;
  }
  let { children, class: className = '', testId }: Props = $props();

  let el = $state<HTMLDivElement | null>(null);
  let moreRight = $state(false);

  function measure(): void {
    if (!el) return;
    moreRight = el.scrollLeft + el.clientWidth < el.scrollWidth - 1;
  }

  $effect(() => {
    if (!el) return;
    measure();
    const ro = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(measure) : null;
    ro?.observe(el);
    for (const child of Array.from(el.children)) ro?.observe(child);
    return () => ro?.disconnect();
  });
</script>

<div class="relative">
  <div
    bind:this={el}
    onscroll={measure}
    class="overflow-x-auto {className}"
    data-testid={testId}
  >
    {@render children()}
  </div>
  {#if moreRight}
    <div
      class="pointer-events-none absolute inset-y-0 right-0 w-1 rounded-r-lg bg-blue-500/50"
      aria-hidden="true"
    ></div>
    <span
      class="pointer-events-none absolute right-1 top-1 rounded bg-zinc-900/90 px-1.5 py-0.5 text-[10px] text-zinc-300"
      data-testid="scroll-hint">scroll →</span
    >
  {/if}
</div>
