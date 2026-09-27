/**
 * W5 (docs/design/logic-moves-adoption-plan-2026-09-24.md) plus the
 * 2026-09-24 bulk-resolve adoption (OpenProcessor 2f5cda2) — the
 * `/classes` Proposals section: GET {API_PREFIX}/review/new_class_proposals/
 * summary, and "Create class & assign" / "Map to existing", both now
 * dry-running `POST {API_PREFIX}/review/new_class_proposals/resolve`
 * for a confirm count before the real resolve. This fully replaces the
 * old sample_crop_ids-only bulkLabel path — there is no bulkLabel call
 * left in this file at all.
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
  it('does not import or call the retired sample-only bulkLabel path', () => {
    expect(src).not.toMatch(/\bbulkLabel\(/);
    expect(src).not.toMatch(/[,{]\s*bulkLabel\s*[,}]/);
  });

  it('loadProposals fetches getNewClassProposalsSummary and degrades to an inline error, not a crash', () => {
    const fn = src.match(/async function loadProposals\([\s\S]*?\n {2}\}/)?.[0];
    expect(fn).toBeDefined();
    expect(fn).toMatch(/proposalsSummary = await getNewClassProposalsSummary\(\)/);
    expect(fn).toMatch(/catch \(e\) \{[\s\S]*proposalsError = \(e as Error\)\.message;/);
  });

  it('createClassAndAssign dry-runs the resolve with a create payload, confirms the served matched count, then resolves for real', () => {
    const start = src.indexOf('async function createClassAndAssign(');
    const end = src.indexOf('\n  async function mapToExisting(', start);
    const fn = src.slice(start, end);
    expect(start).toBeGreaterThan(-1);
    expect(fn).toMatch(/create:\s*\{\s*class_name:\s*name,\s*group:\s*''\s*\}/);
    expect(fn).toMatch(
      /const preview = await resolveNewClassProposal\(body, \{ dryRun: true \}\)/,
    );
    expect(fn).toMatch(/window\.confirm\(/);
    expect(fn).toMatch(/preview\.matched/);
    expect(fn).toMatch(/const res = await resolveNewClassProposal\(body\);/);
    expect(fn).toMatch(/reportResolve\(res,/);
  });

  it('mapToExisting dry-runs the resolve with the picked class_id, confirms the served matched count, then resolves for real', () => {
    const start = src.indexOf('async function mapToExisting(');
    const end = src.indexOf('</script>', start);
    const fn = src.slice(start, end);
    expect(start).toBeGreaterThan(-1);
    expect(fn).toMatch(/label: term\.label, class_id: targetId/);
    expect(fn).toMatch(
      /const preview = await resolveNewClassProposal\(body, \{ dryRun: true \}\)/,
    );
    expect(fn).toMatch(/window\.confirm\(/);
    expect(fn).toMatch(/preview\.matched/);
    expect(fn).toMatch(/const res = await resolveNewClassProposal\(body\);/);
    expect(fn).toMatch(/reportResolve\(res,/);
  });

  it('reportResolve records undo via undoStore.recordWrites(updated_ids) and toasts served counts', () => {
    const start = src.indexOf('function reportResolve(');
    const end = src.indexOf('\n  async function createClassAndAssign(', start);
    const fn = src.slice(start, end);
    expect(start).toBeGreaterThan(-1);
    expect(fn).toMatch(/undoStore\.recordWrites\(res\.updated_ids\)/);
    expect(fn).toMatch(/res\.updated/);
    expect(fn).toMatch(/res\.conflicts\.length/);
    expect(fn).toMatch(/res\.skipped\.length/);
  });

  it('both actions refresh the proposals summary after a successful resolve; create also refreshes the classes list', () => {
    const createStart = src.indexOf('async function createClassAndAssign(');
    const createEnd = src.indexOf('\n  async function mapToExisting(', createStart);
    const createFn = src.slice(createStart, createEnd);
    expect(createFn).toMatch(/loadProposals\(\)/);
    expect(createFn).toMatch(/classesStore\.clearAndRefetch\(\)/);

    const mapStart = src.indexOf('async function mapToExisting(');
    const mapEnd = src.indexOf('</script>', mapStart);
    const mapFn = src.slice(mapStart, mapEnd);
    expect(mapFn).toMatch(/await loadProposals\(\);/);
  });
});

/**
 * DQ-M11 fix (dq-queues cutover, 2026-09-24): the backend now flags
 * super-category/junk/already-registered terms (`flagged_terms`) instead
 * of offering them through the same one-click "Create class & assign"
 * path as `top_terms` — the original DQ-M11 bug was exactly a one-click
 * create over 89 "widget_c" crops making a super-class.
 */
describe('/classes Proposals section — flagged_terms (DQ-M11)', () => {
  // Rendering (one list sorted by count, no Create for a flagged term, the
  // one-click map for existing_class, map-to-existing for the rest) is
  // mount-tested in visualAudit.test.ts (visual audit 2026-09-24, L2) —
  // the old <details> source scans here pinned the layout L2 replaced.

  it('flagReason renders the served flag as a human reason: generic parent / not an object / existing class → map to X', () => {
    const fn = src.match(/function flagReason\([\s\S]*?\n {2}\}/)?.[0];
    expect(fn).toBeDefined();
    expect(fn).toMatch(/generic_parent.*generic parent/s);
    expect(fn).toMatch(/non_object.*not an object/s);
    expect(fn).toMatch(/existing_class/);
    expect(fn).toMatch(/map to/);
  });

  it('shows without_term and term_rules as help text, not silently dropped', () => {
    expect(src).toMatch(/proposalsSummary\.without_term/);
    expect(src).toMatch(/proposalsSummary\.term_rules/);
  });
});
