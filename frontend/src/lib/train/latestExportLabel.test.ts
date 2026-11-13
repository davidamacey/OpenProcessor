import { describe, expect, it } from 'vitest';
import { latestExportLabel } from './trainDisplay';

describe('latestExportLabel', () => {
  it('names the newest export by its served time, never the symlink', () => {
    expect(latestExportLabel('2026-11-11T04:47:12Z')).toBe(
      'Latest export · 2026-11-11 04:47',
    );
  });
  it('is just "Latest export" when no time is served', () => {
    expect(latestExportLabel(null)).toBe('Latest export');
    expect(latestExportLabel(undefined)).toBe('Latest export');
  });
});
