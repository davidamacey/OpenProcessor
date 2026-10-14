/**
 * The un-dismiss panel reverses discard()'s reviewDismissCrop via
 * `POST {API_PREFIX}/crops/{id}/review_undismiss`, and the panel requests a
 * sort the backend accepts and reports a failed load. Static source scan (no
 * component-mount harness; see slotReviewCharacterization.test.ts's doc
 * comment).
 */
import { describe, expect, it } from 'vitest';
import { reviewSource } from '$lib/testing/reviewPageSources';

const src = reviewSource('lib/review/reviewDismissedController.svelte.ts');
const panelSrc = reviewSource('lib/components/review/DismissedPanel.svelte');

describe("un-dismiss panel reverses discard()'s reviewDismissCrop", () => {
  it('toggleDismissedPanel loads {API_PREFIX}/crops?review_dismissed=true', () => {
    const fn = src.match(/async function toggleDismissedPanel\([\s\S]*?\n {2}\}/)?.[0];
    expect(fn).not.toBeUndefined();
    expect(fn).toMatch(/getCrops\(\{\s*review_dismissed: true/);
  });

  it('undismiss calls reviewUndismissCrop and drops the item from the local list', () => {
    const fn = src.match(/async function undismiss\([\s\S]*?\n {2}\}/)?.[0];
    expect(fn).not.toBeUndefined();
    expect(fn).toMatch(/await reviewUndismissCrop\(crop\.id\)/);
  });
});

describe('Dismissed panel sends a sort the backend accepts and reports failure', () => {
  it('toggleDismissedPanel requests updated_at:desc, not the 400-ing "recent"', () => {
    const fn = src.match(/async function toggleDismissedPanel\([\s\S]*?\n {2}\}/)?.[0];
    expect(fn).not.toBeUndefined();
    expect(fn).toMatch(/sort:\s*'updated_at:desc'/);
    expect(fn).not.toMatch(/sort:\s*'recent'/);
  });

  it('a failed load sets dismissedError instead of silently emptying the list', () => {
    const fn = src.match(/async function toggleDismissedPanel\([\s\S]*?\n {2}\}/)?.[0];
    expect(fn).toMatch(/dismissedError = apiErrorText\(e\);/);
    expect(fn).toMatch(/dismissedItems = \[\];/);
  });

  it('the panel renders an error state before falling back to the empty state', () => {
    const markup = panelSrc.match(/\{#if dismissedLoading\}[\s\S]*?\{\/if\}/)?.[0];
    expect(markup).not.toBeUndefined();
    expect(markup).toMatch(/\{:else if dismissedError\}/);
    // The error branch must come before the empty-state branch so a
    // failed request never renders "No dismissed crops."
    expect(markup!.indexOf('dismissedError')).toBeLessThan(
      markup!.indexOf('No dismissed crops.'),
    );
  });
});
