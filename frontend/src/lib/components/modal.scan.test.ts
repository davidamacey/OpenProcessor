/**
 * Dialogs are `Modal.svelte` and nothing else. A hand-rolled scrim needs a
 * click handler on a non-interactive element, which is exactly what the
 * `a11y_click_events_have_key_events` suppressions papered over; the shared
 * shell dismisses from the keyboard (Escape), traps and restores focus, and
 * keeps the click on a screen-reader-hidden layer.
 */
import { describe, expect, it } from 'vitest';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const SRC = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');

const HAND_ROLLED = [
  /svelte-ignore\s+a11y_click_events_have_key_events/,
  /role="(?:alert)?dialog"/,
  /\bfixed inset-0\b/,
];

function svelteFiles(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const full = path.join(dir, name);
    if (statSync(full).isDirectory()) return svelteFiles(full);
    return name.endsWith('.svelte') ? [full] : [];
  });
}

const hit = (line: string): boolean => HAND_ROLLED.some((re) => re.test(line));

describe('modal dialogs', () => {
  it('the patterns catch each hand-rolled spelling (negative control)', () => {
    expect(hit('  <!-- svelte-ignore a11y_click_events_have_key_events -->')).toBe(true);
    expect(hit('    role="dialog"')).toBe(true);
    expect(hit('    class="fixed inset-0 z-40 flex bg-black/60"')).toBe(true);
    expect(hit('<Modal label="Add class" {onclose}>')).toBe(false);
  });

  it('are built only by Modal.svelte', () => {
    const offenders = svelteFiles(SRC).flatMap((f) => {
      const rel = path.relative(SRC, f);
      if (rel === path.join('lib', 'components', 'Modal.svelte')) return [];
      return readFileSync(f, 'utf-8')
        .split('\n')
        .filter(hit)
        .map((line) => `${rel}: ${line.trim()}`);
    });
    expect(offenders).toEqual([]);
  });
});
