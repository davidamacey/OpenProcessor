/**
 * `<ClusterBadge>` is a small presentational component (no
 * `@testing-library/svelte` in this repo — see StrategyBar.test.ts's
 * header comment for why every component test here is a static source
 * scan rather than a mount harness). This asserts the two contractual
 * behaviors the global-search feature (docs: /clusters search-mode plan)
 * depends on:
 *  - the amber "Unlabeled #{id}" fallback when there's no dominant class
 *  - navigation to `/clusters/{id}` on click, via `goto`, not a raw <a>
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

  it('renders an amber "Unlabeled #{id}" variant when dominantClassName is null', () => {
    expect(src).toMatch(/Unlabeled #\{clusterId\}/);
    expect(src).toMatch(/amber/);
  });

  it('renders "#{id} · {dominant_class_name}" in the labeled case', () => {
    expect(src).toMatch(/#\{clusterId\} · \{dominantClassName\}/);
  });

  it('navigates via goto, not a raw anchor tag', () => {
    expect(src).toMatch(/goto\(`\/clusters\/\$\{clusterId\}`\)/);
    expect(src).not.toMatch(/<a\s/);
  });

  it('stops click propagation so the badge click never also selects/opens the underlying crop card', () => {
    expect(src).toMatch(/stopPropagation/);
  });
});
