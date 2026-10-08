/**
 * Source-scan coverage for the /clusters/[id] frontend fixes in
 * docs/design/interactive-pass-2026-09-24.md §6 FRONTEND (M7 half, and
 * the "invalid move target" bullet / m17). No component-mount harness
 * exists for this route (see clusterMoveRace.test.ts's doc comment), so
 * this follows the same static source-scan convention as
 * logicMovesW3W6.test.ts in this directory.
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(path.join(here, '+page.svelte'), 'utf-8');

describe('M7 (FE half): Run VLM polls only the job it just started, and confirms first', () => {
  it('runVlm asks for confirmation before starting the job', () => {
    const fn = src.match(/async function runVlm\(\)[\s\S]*?\n {2}\}/)?.[0];
    expect(fn).not.toBeUndefined();
    expect(fn).toMatch(/window\.confirm\(/);
  });

  it('passes the just-started job’s own job_id into pollAutoLabelJob, not a bare call', () => {
    const fn = src.match(/async function runVlm\(\)[\s\S]*?\n {2}\}/)?.[0];
    expect(fn).not.toBeUndefined();
    // Must read the id off the response of runVlmOnCluster (`vlmJob =
    // await runVlmOnCluster(...)`) and thread it into pollAutoLabelJob —
    // a call with no 4th arg would silently resume the M7 bug (reading
    // whatever job happens to be in the status slot).
    expect(fn).toMatch(/vlmJob = await runVlmOnCluster\(/);
    expect(fn).toMatch(/pollAutoLabelJob\(\s*\([^)]*\)\s*=>[\s\S]*?vlmJob\.job_id/);
  });
});

describe('invalid move target / m17: the negative-id client rule is gone', () => {
  it('confirmMovePicker no longer rejects a negative id itself — only NaN', () => {
    const fn = src.match(/async function confirmMovePicker\(\)[\s\S]*?\n {2}\}/)?.[0];
    expect(fn).not.toBeUndefined();
    expect(fn).not.toMatch(/id < 0/);
    expect(fn).not.toMatch(/non-negative/);
    // Still guards against a genuinely non-numeric entry (NaN can't be
    // sent as a JSON int at all).
    expect(fn).toMatch(/Number\.isFinite\(id\)/);
  });
});
