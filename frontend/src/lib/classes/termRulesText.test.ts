/**
 * F-52 (fresh-start findings 2026-09-25): the `/classes` proposals help
 * line interpolated empty served lists raw ("(, plus …)" / "()").
 */
import { describe, expect, it } from 'vitest';
import { termRulesText } from './proposalRows';
import type { NewClassTermRules } from '$lib/api';

function rules(over: Partial<NewClassTermRules> = {}): NewClassTermRules {
  return {
    generic_terms: [],
    non_object_terms: [],
    registry_groups_are_generic: false,
    existing_classes_flagged: false,
    generic_terms_env: '',
    non_object_terms_env: '',
    ...over,
  };
}

describe('termRulesText', () => {
  it('empty lists never render as "()" or a leading ", plus"', () => {
    const text = termRulesText(
      rules({ registry_groups_are_generic: true, existing_classes_flagged: true }),
    );
    expect(text).not.toBeNull();
    expect(text).not.toContain('()');
    expect(text).not.toContain('(, plus');
    expect(text).not.toContain('non-object');
    expect(text).toContain('generic-parent term (any class-registry group name)');
    expect(text).toContain('already name a registered class');
  });

  it('lists the served terms when present', () => {
    const text = termRulesText(
      rules({
        generic_terms: ['animal', 'object'],
        non_object_terms: ['background'],
        registry_groups_are_generic: true,
      }),
    );
    expect(text).toBe(
      'Terms are auto-flagged, not offered a one-click create, when they match a ' +
        'generic-parent term (animal, object, plus any class-registry group name), or ' +
        'match a non-object term (background).',
    );
  });

  it('returns null when nothing is configured', () => {
    expect(termRulesText(rules())).toBeNull();
  });
});
