/** The Review view names the queue it reads in a heading, which is also its page title. */
import { expect, it } from 'vitest';
import { reviewSource } from '$lib/testing/reviewPageSources';

it('the header carries the queue label as the heading', () => {
  expect(reviewSource('lib/components/review/ReviewHeader.svelte')).toMatch(
    /<h1[^>]*data-testid="review-queue-label">\s*\{tabs\.activeQueueLabel\}\s*<\/h1>/,
  );
});
