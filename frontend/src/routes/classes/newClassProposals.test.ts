/**
 * W5 (docs/design/logic-moves-adoption-plan-2026-09-24.md) — the
 * `/classes` Proposals section: GET {API_PREFIX}/review/new_class_proposals/
 * summary, "Create class & assign" (addClass then bulkLabel on the served
 * sample_crop_ids), and "Map to existing" (bulkLabel onto a picked class).
 *
 * This repo has no component-mount harness (see
 * slotReviewCharacterization.test.ts's doc comment); pinned via the same
 * static source-scan convention.
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(path.join(here, '+page.svelte'), 'utf-8');

describe('/classes Proposals section', () => {
  it('loadProposals fetches getNewClassProposalsSummary and degrades to an inline error, not a crash', () => {
    const fn = src.match(/async function loadProposals\([\s\S]*?\n {2}\}/)?.[0];
    expect(fn).toBeDefined();
    expect(fn).toMatch(/proposalsSummary = await getNewClassProposalsSummary\(\)/);
    expect(fn).toMatch(/catch \(e\) \{[\s\S]*proposalsError = \(e as Error\)\.message;/);
  });

  it('createClassAndAssign creates the class then bulk-assigns the served sample_crop_ids to it', () => {
    const start = src.indexOf('async function createClassAndAssign(');
    const end = src.indexOf('\n  async function mapToExisting(', start);
    const fn = src.slice(start, end);
    expect(start).toBeGreaterThan(-1);
    expect(fn).toMatch(/const created = await addClass\(/);
    expect(fn).toMatch(/await bulkLabel\(term\.sample_crop_ids, created\.class_id\)/);
  });

  it('mapToExisting bulk-assigns the served sample_crop_ids onto the picked existing class', () => {
    const start = src.indexOf('async function mapToExisting(');
    const end = src.indexOf('</script>', start);
    const fn = src.slice(start, end);
    expect(start).toBeGreaterThan(-1);
    expect(fn).toMatch(/await bulkLabel\(term\.sample_crop_ids, targetId\)/);
  });
});
