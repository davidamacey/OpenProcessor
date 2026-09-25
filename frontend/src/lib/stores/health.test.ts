import { describe, expect, it } from 'vitest';
import { healthChip, HEALTH_CHIP_TEXT } from './health.svelte';

describe('healthChip', () => {
  it('reads "checking", not "down", before the first poll has settled', () => {
    expect(healthChip(false, null)).toBe('checking');
    expect(HEALTH_CHIP_TEXT[healthChip(false, null)]).not.toContain('down');
  });

  it('reports ok / down once a poll has settled', () => {
    expect(healthChip(true, 1)).toBe('ok');
    expect(healthChip(false, 1)).toBe('down');
  });
});
