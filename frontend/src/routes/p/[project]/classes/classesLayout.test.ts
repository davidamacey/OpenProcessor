/**
 * F-53 / F-58 (fresh-start findings 2026-09-25). Source scan, matching
 * this directory's convention: the route page isn't mountable here.
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(path.join(here, '+page.svelte'), 'utf-8');

describe('/classes layout', () => {
  it('F-53: the class registry renders before the proposals list, which is collapsed', () => {
    const table = src.indexOf('data-testid="class-row-{cls.id}"');
    const proposals = src.indexOf('data-testid="proposals-section"');
    expect(table).toBeGreaterThan(-1);
    expect(proposals).toBeGreaterThan(table);
    // A <details> without `open` starts collapsed.
    const tag = src.slice(src.lastIndexOf('<details', proposals), proposals);
    expect(tag).not.toMatch(/\bopen\b/);
  });

  it('F-58: the per-term hide button says it is not saved', () => {
    expect(src).toContain('Not saved: the crops stay pending.');
    expect(src).not.toContain(
      "Dismiss this term from the list (doesn't touch the crops)",
    );
  });
});
