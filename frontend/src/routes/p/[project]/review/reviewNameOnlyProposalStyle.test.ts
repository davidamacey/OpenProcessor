/**
 * A name-only class proposal (no `proposed_class_id`) is styled as a hint,
 * not as a confirmable proposal. Static source scan (no component-mount
 * harness; see slotReviewCharacterization.test.ts's doc comment).
 */
import { describe, expect, it } from 'vitest';
import { reviewSource } from '$lib/testing/reviewPageSources';

const src = reviewSource('lib/components/review/ReviewItemMeta.svelte');

describe('a name-only proposal is styled as a hint, not a confirmable proposal', () => {
  it('the Proposed row branches on proposed_class_id == null before using the yellow confirmable style', () => {
    expect(src).toMatch(
      /\{#if current\.proposed_class_name && current\.proposed_class_id == null\}/,
    );
  });

  it('the name-only branch uses a neutral color, not text-yellow-200', () => {
    const block = src.match(
      /\{#if current\.proposed_class_name && current\.proposed_class_id == null\}[\s\S]*?\{:else\}/,
    )?.[0];
    expect(block).not.toBeUndefined();
    expect(block).not.toMatch(/text-yellow-200/);
    expect(block).toMatch(/hint only/);
  });
});
