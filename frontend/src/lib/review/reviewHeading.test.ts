/** The Review view names the queue it reads in a heading, which is also its page title. */
import { expect, it } from 'vitest';
import { reviewSource } from '$lib/testing/reviewPageSources';

it('the header carries the queue label as the (screen-reader) page heading, shown once as text', () => {
  expect(reviewSource('lib/components/review/ReviewHeader.svelte')).toMatch(
    /<h1 class="sr-only" data-testid="review-queue-label">\{tabs\.activeQueueLabel\}<\/h1>/,
  );
});
