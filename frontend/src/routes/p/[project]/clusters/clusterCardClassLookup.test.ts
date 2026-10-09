/**
 * A cluster card's class link uses the served `dominant_class_id`, and a
 * name lookup (deep link, slot inventory card) resolves to the ACTIVE class
 * when a deprecated class carries the same name. Static source scan, like
 * the sibling /clusters page tests.
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(path.join(here, '+page.svelte'), 'utf-8');

describe('/clusters class lookups', () => {
  it('a slot-bound card click routes by the served dominant_class_id, not a name lookup', () => {
    const fn = src.slice(src.indexOf('P2.11 fix'), src.indexOf('Infinite scroll owns'));
    expect(fn).toMatch(/c\.dominant_class_id/);
    expect(fn).not.toMatch(/toLowerCase/);
  });

  it('every name-to-class lookup goes through the active-first normalized lookup', () => {
    expect(src).toMatch(/findActiveClassByName\(classesStore\.classes, v\)/);
    expect(src).toMatch(/findActiveClassByName\(classesStore\.classes, className\)/);
    expect(src).not.toMatch(/c\.name\.toLowerCase\(\)/);
  });
});
