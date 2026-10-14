/**
 * Source scan, matching
 * this directory's convention: the route page isn't mountable here.
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const LIB = path.resolve(here, '../../../../lib');
const pageSrc = readFileSync(path.join(here, '+page.svelte'), 'utf-8');
const tableSrc = readFileSync(
  path.join(LIB, 'components/classes/ClassRegistryTable.svelte'),
  'utf-8',
);
const proposalsSrc = readFileSync(
  path.join(LIB, 'components/classes/NewClassProposalsPanel.svelte'),
  'utf-8',
);

describe('/classes layout', () => {
  it('F-53: the class registry renders before the proposals list, which is collapsed', () => {
    expect(tableSrc).toContain('data-testid="class-row-{cls.id}"');
    expect(proposalsSrc).toContain('data-testid="proposals-section"');
    const table = pageSrc.indexOf('<ClassRegistryTable');
    expect(table).toBeGreaterThan(-1);
    expect(pageSrc.indexOf('<NewClassProposalsPanel')).toBeGreaterThan(table);
    const proposals = proposalsSrc.indexOf('data-testid="proposals-section"');
    // A <details> without `open` starts collapsed.
    const tag = proposalsSrc.slice(
      proposalsSrc.lastIndexOf('<details', proposals),
      proposals,
    );
    expect(tag).toMatch(/<details/);
    expect(tag).not.toMatch(/\bopen\b/);
  });

  it('F-58: the per-term hide button says it is not saved', () => {
    expect(proposalsSrc).toContain('Not saved: the crops stay pending.');
    expect(proposalsSrc).not.toContain(
      "Dismiss this term from the list (doesn't touch the crops)",
    );
  });
});
