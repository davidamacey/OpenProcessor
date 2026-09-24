/**
 * Security ratchet: a deployment-supplied profile can never reach a
 * code-evaluation sink. Source scan in the style of
 * `../../apiPrefixScan.test.ts`, over every non-test `.ts` under
 * `src/lib/annotations/`.
 */

import { readdirSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { stripComments } from '$lib/testing/sourceScan';

const here = path.dirname(fileURLToPath(import.meta.url));
const annotationsRoot = path.resolve(here, '..');

function walk(dir: string, out: string[] = []): string[] {
  for (const entry of readdirSync(dir)) {
    const full = path.join(dir, entry);
    if (statSync(full).isDirectory()) walk(full, out);
    else out.push(full);
  }
  return out;
}

const scanned = walk(annotationsRoot)
  .filter((f) => f.endsWith('.ts') && !f.endsWith('.test.ts'))
  .map((f) => path.relative(annotationsRoot, f))
  .sort();

// Sink names below are deliberately phrased in prose, never spelling out
// the exact code-evaluation call shapes this file scans FOR, so this
// file's own source text can never trip its sibling static ship-gate
// grep and flag itself. The RegExp patterns are the actual, unambiguous
// checks — the `name` field is only a human-readable label.
const SINKS: Array<{ name: string; pattern: RegExp }> = [
  { name: 'the eval builtin', pattern: /\beval\s*\(/ },
  { name: 'a Function constructor call', pattern: /\bnew\s+Function\s*\(/ },
  { name: 'a bare Function call', pattern: /[^.\w]Function\s*\(/ },
  {
    name: 'setTimeout/setInterval with a string',
    pattern: /set(?:Timeout|Interval)\s*\(\s*['"`]/,
  },
  {
    name: 'innerHTML/outerHTML/insertAdjacentHTML',
    pattern: /\b(innerHTML|outerHTML|insertAdjacentHTML)\b/,
  },
  { name: 'import( with a non-literal argument', pattern: /\bimport\s*\(\s*[^'"`)]/ },
];

describe('a deployment-supplied profile can never reach a code-evaluation sink', () => {
  it('scans a sane population', () => {
    expect(scanned.length).toBeGreaterThan(5);
    expect(scanned).toContain(path.join('config', 'parseSlotConfig.ts'));
  });

  it('finds zero offenders', () => {
    const offenders: string[] = [];
    for (const rel of scanned) {
      const lines = stripComments(
        readFileSync(path.join(annotationsRoot, rel), 'utf-8'),
      ).split('\n');
      lines.forEach((line, i) => {
        for (const sink of SINKS) {
          if (sink.pattern.test(line))
            offenders.push(`${rel}:${i + 1}  [${sink.name}]  ${line.trim()}`);
        }
      });
    }
    expect(offenders).toEqual([]);
  });
});
