import { describe, expect, it } from 'vitest';
import { validatePathTemplate, renderPathTemplate } from './templatePath';

describe('validatePathTemplate', () => {
  it('accepts a plain prefix-relative path', () => {
    const r = validatePathTemplate('/crops/x/plate');
    expect(r.template).toBe('/crops/x/plate');
    expect(r.placeholders).toEqual([]);
  });

  it('accepts allow-listed placeholders', () => {
    const r = validatePathTemplate('/crops/{cropId}/region_thumbnail?size={size}');
    expect(r.template).toBeDefined();
    expect(r.placeholders).toEqual(['cropId', 'size']);
  });

  it('rejects a non-string', () => {
    expect(validatePathTemplate(42).error).toBeDefined();
  });

  it('rejects an absolute URL', () => {
    expect(validatePathTemplate('https://evil.example/x').error).toBeDefined();
  });

  it('rejects a protocol-relative path', () => {
    expect(validatePathTemplate('//evil.example/x').error).toBeDefined();
  });

  it('rejects path traversal', () => {
    expect(validatePathTemplate('/crops/../../etc/passwd').error).toBeDefined();
  });

  it('rejects a placeholder outside the allow-list, naming it', () => {
    const r = validatePathTemplate('/crops/{userId}/t');
    expect(r.error).toMatch(/userId/);
    expect(r.error).toMatch(/allow-list/);
  });

  it('rejects an expression disguised as a placeholder', () => {
    const r = validatePathTemplate('/crops/{cropId.toUpperCase()}/t');
    expect(r.error).toMatch(/cropId\.toUpperCase\(\)/);
  });

  it('rejects an overlong path', () => {
    expect(validatePathTemplate('/' + 'a'.repeat(300)).error).toBeDefined();
  });

  it('rejects a restricted allow-list not honored (e.g. cohort placeholder in a path template)', () => {
    const r = validatePathTemplate('/crops/{classId}/t', ['cropId']);
    expect(r.error).toMatch(/classId/);
  });
});

describe('renderPathTemplate', () => {
  it('substitutes and encodes values, matching hand-written closures', () => {
    const t = '/crops/{cropId}/region_thumbnail?size={size}';
    expect(renderPathTemplate(t, { cropId: 'abc', size: 160 })).toBe(
      '/crops/abc/region_thumbnail?size=160',
    );
  });

  it('encodeURIComponents a hostile crop id', () => {
    const t = '/crops/{cropId}/plate';
    expect(renderPathTemplate(t, { cropId: 'a/b?c' })).toBe('/crops/a%2Fb%3Fc/plate');
  });
});
