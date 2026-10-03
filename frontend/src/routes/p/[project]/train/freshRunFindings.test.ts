/**
 * F-63(b) / F-64 (fresh-start findings 2026-09-25). Source scan: the
 * route page isn't mountable in this harness (matching this directory's
 * other tests).
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(path.join(here, '+page.svelte'), 'utf-8');

describe('/train fresh-run findings', () => {
  it('F-63(b): TrainForm stays mounted during a run (not the else-branch of isActive)', () => {
    const form = src.indexOf('<TrainForm');
    const chain = src.slice(src.lastIndexOf('{#if', form), form);
    expect(chain).toMatch(/\{#if/);
    expect(chain).not.toMatch(/isActive/);
    expect(src.slice(form, src.indexOf('/>', form))).toMatch(/disabled=\{isActive\}/);
  });

  it('F-63(b): past runs render above the training-cohorts section', () => {
    const past = src.indexOf('<!-- Past runs -->');
    const cohorts = src.indexOf('<!-- Training cohorts');
    expect(past).toBeGreaterThan(-1);
    expect(cohorts).toBeGreaterThan(past);
  });

  it('F-64: the promote default name has no hardcoded version suffix', () => {
    expect(src).not.toMatch(/_v7/);
    expect(src).toMatch(/promoteDefaultName = defaultTritonName\(r\.job_id\)/);
  });
});
