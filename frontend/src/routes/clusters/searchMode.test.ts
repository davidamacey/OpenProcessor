/**
 * Global dataset-wide search mode on `/clusters` (see CLAUDE.md's
 * "Curation-strategy selector bar" section + the implementation plan
 * this shipped from). No `@testing-library/svelte` in this repo (see
 * `StrategyBar.test.ts`'s header comment), so this is a static source
 * scan of the load-bearing behaviors the plan called out explicitly:
 *
 *  - unregistering dropOnClassStore when search mode ends (flagged in
 *    the plan as an easy-to-miss regression: forgetting this leaves
 *    /clusters eating drops after the user backs out of search)
 *  - forcing the embedding-plot toggle off when search activates,
 *    mirroring the existing isLicensePlateFilter guard
 *  - the mode-switch branch rendering before the existing card-grid
 *    `{:else}`, matching the showEmbeddingViz precedent
 *  - M (move) and AHC grouping are NOT wired for search-mode label
 *    actions (out of scope per the plan — search results aren't a
 *    single cluster)
 */

import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));

function read(rel: string): string {
  return readFileSync(path.resolve(here, rel), 'utf-8');
}

/**
 * Extract the full `$effect(() => { if (!searchModeActive) return; ... });`
 * block by counting braces from its opening `{`, rather than a regex
 * anchored to a specific indentation depth — indentation shifts on every
 * prettier reformat and previously broke this test without any real
 * behavior change.
 */
function extractSearchModeEffect(src: string): string {
  const anchor = '$effect(() => {\n    if (!searchModeActive) return;';
  const start = src.indexOf(anchor);
  if (start === -1) throw new Error('searchModeActive effect not found');
  const braceStart = src.indexOf('{', start);
  let depth = 0;
  for (let i = braceStart; i < src.length; i++) {
    if (src[i] === '{') depth++;
    else if (src[i] === '}') {
      depth--;
      if (depth === 0) return src.slice(start, i + 1);
    }
  }
  throw new Error('unbalanced braces in searchModeActive effect');
}

describe('/clusters search mode', () => {
  const src = read('./+page.svelte');

  it('registers dropOnClassStore only inside the searchModeActive-gated effect, and unregisters on cleanup', () => {
    const body = extractSearchModeEffect(src);
    expect(body).toMatch(/dropOnClassStore\.register/);
    expect(body).toMatch(/return \(\) => \{\s*offDrop\(\);/);
  });

  it('forces showEmbeddingViz off when search mode activates, same guard idiom as isLicensePlateFilter', () => {
    expect(src).toMatch(/if \(showEmbeddingViz\) showEmbeddingViz = false;/);
  });

  it('only wires A / Z / X / Escape in search mode — no M (move) or sub-cluster grouping', () => {
    const body = extractSearchModeEffect(src);
    expect(body).toMatch(/reg\(\s*'a',/);
    expect(body).toMatch(/reg\(\s*'z',/);
    expect(body).toMatch(/reg\(\s*'x',/);
    expect(body).toMatch(/reg\(\s*\n?\s*'escape',/);
    expect(body).not.toMatch(/reg\('m',/);
    expect(body).not.toMatch(/openMovePicker/);
  });

  it('renders the search-results branch before the existing card-grid {:else} (mirrors showEmbeddingViz)', () => {
    const gridIdx = src.indexOf('<!-- Grid -->');
    const searchIdx = src.indexOf('{#if searchModeActive}');
    const embedIdx = src.indexOf('{:else if showEmbeddingViz}');
    expect(gridIdx).toBeGreaterThan(-1);
    expect(searchIdx).toBeGreaterThan(gridIdx);
    expect(embedIdx).toBeGreaterThan(searchIdx);
  });

  it('does not pass a cluster_id/tab scope into the SemanticSearchBox filter — global search is the point', () => {
    const filterMatch = src.match(/filter=\{[^}]*\?\s*\{[^}]*\}\s*:\s*\{\}\}/);
    expect(filterMatch).not.toBeNull();
    expect(filterMatch![0]).not.toMatch(/cluster_id/);
    expect(filterMatch![0]).not.toMatch(/\btab\b/);
  });

  it('syncs the query to the URL as ?q= via replaceState + keepFocus, not a normal navigation', () => {
    expect(src).toMatch(/replaceState:\s*true,\s*keepFocus:\s*true/);
  });

  it('batches cluster metadata via getClusters() rather than recomputing dominant class client-side (D-4: no representatives needed for badge lookup)', () => {
    expect(src).toMatch(/await getClusters\(\{ representatives_limit: 0 \}\)/);
    expect(src).not.toMatch(/dominant_class_name\s*=.*\.filter\(/);
  });

  it('exits search mode by clearing the ?q= param and restoring the normal grid', () => {
    expect(src).toMatch(/function exitSearchMode/);
    expect(src).toMatch(/syncSearchUrl\(null\)/);
  });
});
