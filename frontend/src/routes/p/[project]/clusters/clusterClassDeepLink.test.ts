/**
 * A name-form `?class=` deep link on /clusters resolves against the loaded
 * class registry. Static source scan (no component-mount harness; see
 * clusters/[id]/clusterMoveRace.test.ts's doc comment).
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(path.join(here, '+page.svelte'), 'utf-8');

describe('a name-form ?class= deep link resolves against the loaded registry', () => {
  it('classFilter falls back to the normalized active-class name lookup when the param is not numeric', () => {
    const fn = src.match(
      /const classFilter = \$derived\.by\(\(\) => \{[\s\S]*?\n {2}\}\);/,
    )?.[0];
    expect(fn).not.toBeUndefined();
    expect(fn).toMatch(/findActiveClassByName\(classesStore\.classes, v\)/);
  });
});
