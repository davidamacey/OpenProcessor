import { describe, expect, it } from 'vitest';
import { adequacyChipClass, adequacyTooltip } from './adequacy';

// Server-served tiers (`GET {API_PREFIX}/classes` / `/stats/classes`,
// verified live on d037be8: every sample class in this deployment reports
// `adequacy: "block"` against `thresholds.block_below: 20`). This module
// must render exactly the level it's handed — no recomputation from a
// validated_count.
describe('adequacyChipClass', () => {
  it('renders green for the server "ok" tier', () => {
    expect(adequacyChipClass('ok')).toContain('green');
  });

  it('renders orange for the server "warn" tier', () => {
    expect(adequacyChipClass('warn')).toContain('orange');
  });

  it('renders red for the server "block" tier', () => {
    expect(adequacyChipClass('block')).toContain('red');
  });

  it('renders neutral, not a guessed color, for an absent/unknown level', () => {
    const cls = adequacyChipClass(undefined);
    expect(cls).not.toContain('green');
    expect(cls).not.toContain('orange');
    expect(cls).not.toContain('red');
  });
});

describe('adequacyTooltip', () => {
  it('reports the served level verbatim, not a recomputed word', () => {
    expect(adequacyTooltip('warn', 412)).toBe('412 validated · warn');
  });

  it('falls back to "unknown" rather than inventing a tier', () => {
    expect(adequacyTooltip(null, 0)).toBe('0 validated · unknown');
  });
});
