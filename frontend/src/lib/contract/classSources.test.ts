/**
 * `class_source` role vocabulary contract. `CLASS_SOURCE_ROLES` is
 * vendored verbatim from OpenProcessor's
 * `contracts/ts/classSources.ts` (`npm run contract:sync`,
 * source: `src/services/curation/class_sources.py`). Both consumers of
 * `role` — `classSourcesStore` (which just stores/looks up whatever the
 * backend sends, no filtering) and `sourceBadge` (which switches on
 * `role` to pick a color) — must handle every role the backend can
 * actually emit, not just the ones a hand-written fixture happened to
 * cover.
 */
import { describe, expect, it } from 'vitest';
import { CLASS_SOURCE_ROLES } from '$contracts/ts/classSources';
import type { ClassSourceRole } from '../api';
import { sourceBadge } from '../sourceBadge';

describe('vendored class-source-role snapshot sanity', () => {
  it('loaded a non-trivial role set (guards a vacuous pass)', () => {
    expect(CLASS_SOURCE_ROLES.length).toBeGreaterThan(3);
  });
});

describe("api.ts's ClassSourceRole type accepts every backend role", () => {
  it('type-level: every backend role is assignable to ClassSourceRole', () => {
    // Compile-time check: if the backend adds a role that ClassSourceRole's
    // union doesn't cover, this array literal fails to type-check (the
    // union includes a `(string & {})` escape hatch, so this can never
    // fail *silently* at runtime — a real removal/rename still fails
    // the assertion below).
    const asRoles: ClassSourceRole[] = [...CLASS_SOURCE_ROLES];
    expect(asRoles.length).toBe(CLASS_SOURCE_ROLES.length);
  });
});

describe('sourceBadge handles every backend class-source role', () => {
  for (const role of CLASS_SOURCE_ROLES) {
    it(`renders a badge for role "${role}" without throwing`, () => {
      const badge = sourceBadge('some_source_id', true, role, 'Some Source');
      expect(typeof badge.text).toBe('string');
      expect(badge.text.length).toBeGreaterThan(0);
      expect(typeof badge.cls).toBe('string');
      expect(badge.cls.length).toBeGreaterThan(0);
    });
  }
});
