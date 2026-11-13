import { describe, expect, it, vi } from 'vitest';
import { applyScopeSort, scopeSortOf } from './scopeSort';

describe('the one cluster Sort', () => {
  it('reads the choice from the order and the detector sort', () => {
    expect(scopeSortOf(false, 'default')).toBe('default');
    expect(scopeSortOf(true, 'default')).toBe('outliers');
    expect(scopeSortOf(false, 'desc')).toBe('confidence_desc');
    expect(scopeSortOf(true, 'asc')).toBe('confidence_asc');
  });

  it('writes exactly one of the two, clearing the other', () => {
    const to = { setOrder: vi.fn(), setDetectorSort: vi.fn() };
    applyScopeSort('outliers', to);
    expect(to.setOrder).toHaveBeenLastCalledWith('outliers');
    expect(to.setDetectorSort).toHaveBeenLastCalledWith('default');
    applyScopeSort('confidence_asc', to);
    expect(to.setOrder).toHaveBeenLastCalledWith('default');
    expect(to.setDetectorSort).toHaveBeenLastCalledWith('asc');
  });
});
