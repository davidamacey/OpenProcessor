/**
 * `<ClusterBadge>` is a small presentational component (no
 * `@testing-library/svelte` in this repo — see StrategyBar.test.ts's
 * header comment for why every component test here is a static source
 * scan rather than a mount harness). This asserts the two contractual
 * behaviors the global-search feature (docs: /clusters search-mode plan)
 * depends on:
 *  - the amber "Unlabeled #N" fallback when there's no dominant class
 *  - navigation to `/clusters/{uid}` on click, via `goto`, not a raw <a>
 *    (so it composes inside the absolutely-positioned overlay without a
 *    full page reload)
 */

import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));

function read(rel: string): string {
  return readFileSync(path.resolve(here, rel), 'utf-8');
}

describe('ClusterBadge.svelte', () => {
  const src = read('./ClusterBadge.svelte');

  it('renders an amber "Unlabeled #N" variant when dominantClassName is null', () => {
    expect(src).toMatch(/Unlabeled \{numberText\}/);
    expect(src).toMatch(/amber/);
  });

  it('renders "{short uid} · {dominant_class_name}" in the labeled case', () => {
    expect(src).toMatch(/\{numberText\} · \{dominantClassName\}/);
  });

  it('shows the served display_no, not the uid, and keeps the uid in the title', () => {
    expect(src).toMatch(/displayNoText\(\{ display_no: displayNo \}\)/);
    expect(src).not.toMatch(/slice\(0, 8\)/);
    expect(src).toMatch(/title="Open cluster \{clusterUid\}"/);
  });

  it('navigates via goto, not a raw anchor tag', () => {
    expect(src).toMatch(
      /goto\(resolve\(projectHref\(`\/clusters\/\$\{encodeURIComponent\(clusterUid\)\}`\)\)\)/,
    );
    expect(src).not.toMatch(/<a\s/);
  });

  it('stops click propagation so the badge click never also selects/opens the underlying crop card', () => {
    expect(src).toMatch(/stopPropagation/);
  });
});
