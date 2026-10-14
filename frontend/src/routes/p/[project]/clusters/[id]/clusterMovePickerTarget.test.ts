/**
 * The cluster move picker leaves id validation to the server: the old
 * client-side negative-id rule is gone. Static source scan (no
 * component-mount harness; see clusterMoveRace.test.ts's doc comment).
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(path.join(here, '+page.svelte'), 'utf-8');

describe('the negative-id client rule for the move target is gone', () => {
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
