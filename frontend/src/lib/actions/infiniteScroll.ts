/**
 * Svelte action — fires `onload` when the sentinel element scrolls into view.
 *
 * Usage:
 *   <div use:infiniteScroll={{ onload: loadMore, disabled: !hasMore || loading }}></div>
 *
 * Place the sentinel after your grid items so it becomes visible only when
 * the user has scrolled to (or near) the bottom. Tune ``rootMargin`` to
 * pre-fetch earlier — '400px' loads the next page when the bottom is 400px
 * from the viewport, hiding network latency for fast scrollers.
 *
 * The root is auto-detected: walks up the DOM until it finds an ancestor
 * with overflow-auto / overflow-scroll / overflow-y-* — that's the actual
 * scroll container. Falls back to the viewport if no such ancestor exists.
 * This is critical because most labeler pages have an internal scroll
 * container (overflow-auto on a flex child); a viewport-rooted observer
 * never fires inside one of those, so the sentinel sits "below the fold"
 * but never intersects the viewport.
 */

import type { Action } from 'svelte/action';

export interface InfiniteScrollOptions {
  /** Callback fired when the sentinel enters the viewport. */
  onload: () => void | Promise<void>;
  /** Skip firing while true (e.g. while a fetch is in flight, or no more pages). */
  disabled?: boolean;
  /** Pre-fetch margin. Default '400px' loads next page slightly before user reaches bottom. */
  rootMargin?: string;
}

function findScrollRoot(node: HTMLElement): HTMLElement | null {
  let cur: HTMLElement | null = node.parentElement;
  while (cur && cur !== document.body) {
    const style = getComputedStyle(cur);
    const oy = style.overflowY;
    const ox = style.overflowX;
    if (oy === 'auto' || oy === 'scroll' || ox === 'auto' || ox === 'scroll') {
      return cur;
    }
    cur = cur.parentElement;
  }
  return null;
}

export const infiniteScroll: Action<HTMLElement, InfiniteScrollOptions> = (
  node,
  initial,
) => {
  let opts = initial;
  let observer: IntersectionObserver | null = null;
  let firing = false;

  function setup(): void {
    teardown();
    observer = new IntersectionObserver(
      async (entries) => {
        if (!entries.some((e) => e.isIntersecting)) return;
        if (opts.disabled || firing) return;
        firing = true;
        try {
          await opts.onload();
        } finally {
          firing = false;
        }
      },
      {
        root: findScrollRoot(node),
        rootMargin: opts.rootMargin ?? '400px',
      },
    );
    observer.observe(node);
  }

  function teardown(): void {
    observer?.disconnect();
    observer = null;
  }

  setup();

  return {
    update(next: InfiniteScrollOptions): void {
      const marginChanged = (opts.rootMargin ?? '400px') !== (next.rootMargin ?? '400px');
      opts = next;
      if (marginChanged) setup();
    },
    destroy: teardown,
  };
};
