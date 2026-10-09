/**
 * `<SemanticSearchBox>` is a pointer/text-input-only control — same
 * "no global keydown listener" constraint CLAUDE.md's Keyboard shortcuts
 * section places on every new control (StrategyBar.test.ts's header
 * comment explains why this repo does the check via a static source
 * scan instead of a component-mount harness: there's no
 * `@testing-library/svelte` here). The Enter-to-submit / clear-button
 * behavior itself is exercised at the logic layer in
 * `searchBox.svelte.test.ts`, which this component only wires up.
 */

import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));

function read(rel: string): string {
  return readFileSync(path.resolve(here, rel), 'utf-8');
}

describe('SemanticSearchBox.svelte', () => {
  const src = read('./SemanticSearchBox.svelte');

  it('never calls addEventListener — no global listeners', () => {
    expect(src).not.toMatch(/addEventListener/);
  });

  it('handles Enter via a local onkeydown on the input, not a window/document listener', () => {
    expect(src).toMatch(/onkeydown=/);
    expect(src).not.toMatch(/(window|document)\.addEventListener/);
  });

  it('renders a clear affordance only while there is an active query', () => {
    expect(src).toMatch(/\{#if box\.query\}/);
    expect(src).toMatch(/box\.clear\(\)/);
  });

  it('delegates debounce/abort/loading/error state entirely to createSemanticSearchBox', () => {
    expect(src).toMatch(/createSemanticSearchBox/);
    // Must not hand-roll its own setTimeout/AbortController — that logic
    // lives in searchBox.svelte.ts, exactly once, and is unit-tested there.
    expect(src).not.toMatch(/setTimeout|new AbortController/);
  });
});
