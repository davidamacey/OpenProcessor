import { describe, expect, it } from 'vitest';
import { filterPersistKey, parsePersistedFilter } from './persistedFilter';

describe('/clusters persisted filter', () => {
  it('is keyed by project slug', () => {
    expect(filterPersistKey('a')).not.toBe(filterPersistKey('b'));
    expect(filterPersistKey('a')).toContain('a');
  });

  it('keeps a valid sort and unlabeledOnly', () => {
    expect(
      parsePersistedFilter(JSON.stringify({ sort: 'size_desc', unlabeledOnly: true })),
    ).toEqual({ sort: 'size_desc', unlabeledOnly: true });
  });

  it('drops an unknown sort (stale or hand-edited storage)', () => {
    expect(
      parsePersistedFilter(JSON.stringify({ sort: 'bogus', unlabeledOnly: true })),
    ).toEqual({ sort: null, unlabeledOnly: true });
  });

  it('returns defaults for malformed or absent storage', () => {
    expect(parsePersistedFilter(null)).toEqual({ sort: null, unlabeledOnly: false });
    expect(parsePersistedFilter('{nope')).toEqual({ sort: null, unlabeledOnly: false });
    expect(parsePersistedFilter('"x"')).toEqual({ sort: null, unlabeledOnly: false });
    expect(parsePersistedFilter(JSON.stringify({ unlabeledOnly: 'yes' }))).toEqual({
      sort: null,
      unlabeledOnly: false,
    });
  });
});
