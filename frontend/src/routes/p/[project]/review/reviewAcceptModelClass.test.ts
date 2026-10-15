/**
 * "Accept model's class" assigns `probe_pred_class_id` directly, and no
 * client-side proposed_class_* fill-ins remain. Static source scan (no
 * component-mount harness; see slotReviewCharacterization.test.ts's doc
 * comment).
 */
import { describe, expect, it } from 'vitest';
import { reviewPageSource, reviewSource } from '$lib/testing/reviewPageSources';

const actionsSrc = reviewSource('lib/review/reviewItemActions.svelte.ts');
const metaSrc = reviewSource('lib/components/review/ReviewItemMeta.svelte');
const pageSrc = reviewPageSource();

describe('the "Accept model\'s class" button assigns probe_pred_class_id directly', () => {
  it('acceptModelClass assigns current.probe_pred_class_id with no name lookup', () => {
    const fn = actionsSrc.match(/async function acceptModelClass\([\s\S]*?\n {2}\}/)?.[0];
    expect(fn).not.toBeUndefined();
    expect(fn).toMatch(/await assign\(current\.probe_pred_class_id\)/);
  });

  it('the accept button renders only when the served probe opinion allows it (probeOpinion)', () => {
    // F8 D1: the class-id check plus the served probe_disagreement rule
    // live in $lib/review/probeOpinion.ts (probeOpinion.test.ts).
    // The function doc comment above also says "Accept model's class" —
    // skip past it to the actual <button> label in the template.
    const idx = metaSrc.indexOf("Accept model's class", metaSrc.indexOf('</script>'));
    expect(idx).toBeGreaterThan(-1);
    const preceding = metaSrc.slice(0, idx);
    const lastIf = [...preceding.matchAll(/\{#if [^}]*\}/g)].pop();
    expect(lastIf?.[0]).toBe('{#if opinion.showAccept}');
  });
});

describe('no client-side proposed_class_* fill-ins remain', () => {
  it('never sets a literal proposed_class_id/_name (diverse hydration, undo-restore, search results all read the served value)', () => {
    expect(pageSrc).not.toMatch(/proposed_class_id:\s*null/);
    expect(pageSrc).not.toMatch(/proposed_class_name:\s*null/);
    expect(pageSrc).not.toMatch(/proposed_class_id:\s*crop\.class_id/);
  });
});
