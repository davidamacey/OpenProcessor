<script lang="ts">
  /**
   * A single-row, horizontally scrolling strip that SHOWS when it has more
   * content off-screen (visual audit 2026-09-24, R2 and the narrow top nav):
   * a plain `overflow-x-auto` row gave no hint that "New class proposals" /
   * "Regions" or "Settings" sat past the edge at 800px, and an active item
   * past the edge was simply invisible.
   *
   * - A chevron button appears on each side that has hidden content;
   *   clicking it scrolls by most of a strip width. Solid background, not a
   *   fade (the app's no-gradient rule).
   * - Whenever `activeKey` changes (and on mount), the child marked
   *   `data-active="true"` or `aria-current="page"` is scrolled into view.
   */
  import type { Snippet } from 'svelte';

  interface Props {
    children: Snippet;
    /** Renders the strip as a `<nav>` with this label when set. */
    navLabel?: string;
    /** Changing this re-centres the active child. */
    activeKey?: unknown;
    /** Extra classes for the wrapping `<nav>` (flex-item placement); only used with `navLabel`. */
    navClass?: string;
    /** Extra classes for the scrolling row (gap, text size, …). */
    class?: string;
    testId?: string;
  }

  let {
    children,
    navLabel,
    navClass = '',
    activeKey,
    class: klass = '',
    testId,
  }: Props = $props();

  let scroller = $state<HTMLDivElement | null>(null);
  let canLeft = $state(false);
  let canRight = $state(false);

  function measure(): void {
    const el = scroller;
    if (!el) return;
    canLeft = el.scrollLeft > 1;
    canRight = el.scrollLeft + el.clientWidth < el.scrollWidth - 1;
  }

  function scrollByPage(dir: -1 | 1): void {
    const el = scroller;
    if (!el) return;
    el.scrollBy({ left: dir * Math.max(80, el.clientWidth * 0.7), behavior: 'smooth' });
  }

  // Room left for the chevron buttons that overlay each edge, so an
  // "in view" active item is never hidden under one.
  const EDGE_PAD = 28;

  function revealActive(): void {
    const el = scroller;
    if (!el) return;
    const active = el.querySelector<HTMLElement>(
      '[data-active="true"], [aria-current="page"]',
    );
    if (!active) return;
    const er = el.getBoundingClientRect();
    const ar = active.getBoundingClientRect();
    if (ar.left < er.left + EDGE_PAD) {
      el.scrollLeft -= er.left + EDGE_PAD - ar.left;
    } else if (ar.right > er.right - EDGE_PAD) {
      el.scrollLeft += ar.right - (er.right - EDGE_PAD);
    }
  }

  $effect(() => {
    const el = scroller;
    if (!el) return;
    measure();
    // A size change (served labels arriving, a window resize) can push the
    // active item back off-screen — re-reveal it, then re-measure.
    const onResize = () => {
      revealActive();
      measure();
    };
    const ro =
      typeof ResizeObserver !== 'undefined' ? new ResizeObserver(onResize) : null;
    ro?.observe(el);
    for (const child of Array.from(el.children)) ro?.observe(child);
    el.addEventListener('scroll', measure, { passive: true });
    window.addEventListener('resize', onResize);
    return () => {
      ro?.disconnect();
      el.removeEventListener('scroll', measure);
      window.removeEventListener('resize', onResize);
    };
  });

  $effect(() => {
    void activeKey;
    if (!scroller) return;
    revealActive();
    measure();
  });
</script>

{#snippet strip()}
  <div class="relative flex min-w-0 grow items-center">
    {#if canLeft}
      <button
        type="button"
        class="absolute left-0 z-10 flex h-full items-center border-r border-zinc-800 bg-zinc-950 px-1 text-zinc-400 hover:text-white"
        aria-label="Scroll left — more items"
        title="More items to the left"
        data-testid={testId ? `${testId}-more-left` : undefined}
        onclick={() => scrollByPage(-1)}
      >
        ‹
      </button>
    {/if}
    <div
      bind:this={scroller}
      class="scrollbar-none flex min-w-0 grow items-center overflow-x-auto whitespace-nowrap {klass}"
      data-testid={testId}
    >
      {@render children()}
    </div>
    {#if canRight}
      <button
        type="button"
        class="absolute right-0 z-10 flex h-full items-center border-l border-zinc-800 bg-zinc-950 px-1 text-zinc-400 hover:text-white"
        aria-label="Scroll right — more items"
        title="More items to the right"
        data-testid={testId ? `${testId}-more-right` : undefined}
        onclick={() => scrollByPage(1)}
      >
        ›
      </button>
    {/if}
  </div>
{/snippet}

{#if navLabel}
  <nav class="flex min-w-0 shrink items-center {navClass}" aria-label={navLabel}>
    {@render strip()}
  </nav>
{:else}
  {@render strip()}
{/if}
