/**
 * Static source scan for the `/settings` deployment-defaults admin
 * page (docs/design/curation-settings-ui-plan-2026-09-21.md §6.4).
 * This repo has no `@testing-library/svelte` harness (see
 * `StrategyBar.test.ts`/`AssistScopeBar.test.ts`'s header), so this
 * asserts the wiring a reviewer would check by eye.
 */

import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(path.resolve(here, './+page.svelte'), 'utf-8');

describe('/settings deployment-defaults page', () => {
  it('never calls addEventListener — pointer-only, no third global keydown listener', () => {
    expect(src).not.toMatch(/addEventListener/);
  });

  it('imports settableAxes/advisoryAxes/axisOptions/effectiveDefaultId from $lib/curationSettings', () => {
    const importBlock = src.match(
      /import\s*\{([^}]*)\}\s*from\s*['"]\$lib\/curationSettings['"]/s,
    );
    expect(importBlock).not.toBeNull();
    const names = importBlock![1];
    expect(names).toMatch(/settableAxes/);
    expect(names).toMatch(/advisoryAxes/);
    expect(names).toMatch(/axisOptions/);
    expect(names).toMatch(/effectiveDefaultId/);
  });

  it('contains no hardcoded axis-id comparison — iterates the SETTINGS_AXES table instead', () => {
    expect(src).not.toMatch(/['"]cluster['"]\s*===/);
    expect(src).not.toMatch(/['"]sort['"]\s*===/);
    expect(src).not.toMatch(/tab === 'sort'/);
  });

  it('contains no inline status filter', () => {
    expect(src).not.toMatch(/status === 'stable'/);
  });

  it('renders the advisory section behind an availability check', () => {
    expect(src).toMatch(/\{#if advisoryVisible\}/);
  });

  it('the advisory branch contains no <select and no Save button', () => {
    const start = src.indexOf('Advertised but not yet wired');
    expect(start).toBeGreaterThan(-1);
    const sectionCloseIdx = src.indexOf('{/if}', start);
    const slice = src.slice(start, sectionCloseIdx === -1 ? undefined : sectionCloseIdx);
    expect(slice).not.toMatch(/<select/);
    expect(slice).not.toMatch(/>\s*Save\s*</);
  });

  it('the deployment-wide banner copy is present', () => {
    expect(src).toMatch(/no per-user setting/);
    expect(src).toMatch(/no undo/);
  });

  it('calls strategiesStore.reset() somewhere after a save', () => {
    expect(src).toMatch(/strategiesStore\.reset\(\)/);
  });

  it("calls keyboardStore.setScope('settings')", () => {
    expect(src).toMatch(/keyboardStore\.setScope\('settings'\)/);
  });

  it('contains no bare /curation or /curation literal', () => {
    expect(src).not.toMatch(/(?:['"`]|\})\/(?:op|curation)(?:[/'"`]|\$)/);
  });

  it('renders a Clear control wired to curationSettingsStore.clearDefault, gated on `pinned`', () => {
    expect(src).toMatch(/>\s*Clear\s*<|Clear(?:ing…)?/);
    expect(src).toMatch(/disabled=\{!pinned/);
    expect(src).toMatch(/clearDefault/);
  });

  it('the advisory branch still contains no Clear button', () => {
    const start = src.indexOf('Advertised but not yet wired');
    const sectionCloseIdx = src.indexOf('{/if}', start);
    const slice = src.slice(start, sectionCloseIdx === -1 ? undefined : sectionCloseIdx);
    expect(slice).not.toMatch(/>\s*Clear\s*</);
  });
});
