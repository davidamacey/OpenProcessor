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
 * Re-runs ``onload`` if the sentinel becomes visible again after disabled
 * was true (e.g. after the new page settles and ``loading`` flips back).
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
      { rootMargin: opts.rootMargin ?? '400px' },
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
