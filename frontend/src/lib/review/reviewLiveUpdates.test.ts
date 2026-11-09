/**
 * The review live pill counts crop events and, on a server `reset` (events
 * were missed), re-reads the queue once instead of trusting the count.
 */
import { describe, expect, it, vi } from 'vitest';

const subscribe = vi.hoisted(() => vi.fn());
vi.mock('$lib/sse', () => ({ subscribeCurationEvents: subscribe }));
vi.mock('svelte', () => ({ onMount: (fn: () => unknown) => fn() }));

import { createReviewLiveUpdates } from './reviewLiveUpdates.svelte';

describe('createReviewLiveUpdates', () => {
  it('counts crop events and refetches once on a server reset', () => {
    const reload = vi.fn();
    const live = createReviewLiveUpdates(reload);
    const opts = subscribe.mock.calls[0][0];
    opts.onEvent({ type: 'crop.created' });
    opts.onEvent({ type: 'something.else' });
    expect(live.liveNewCount).toBe(1);
    expect(reload).not.toHaveBeenCalled();
    opts.onResync();
    expect(reload).toHaveBeenCalledTimes(1);
    expect(live.liveNewCount).toBe(0);
  });
});
