import { readdirSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

/**
 * The project binding happens in the `/p/[project]` layout `load`, and
 * `app.html` preloads on hover. A link to ANOTHER project (one that passes an
 * explicit slug to `projectHref`) would therefore bind that project, in the
 * background, on mere hover. Such links preload on tap only.
 */
const srcRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

function svelteFiles(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const full = path.join(dir, name);
    if (statSync(full).isDirectory()) return svelteFiles(full);
    return name.endsWith('.svelte') ? [full] : [];
  });
}

const CROSS_PROJECT_HREF = /projectHref\(\s*(?:'[^']*'|`[^`]*`)\s*,\s*[\w.]+\s*\)/;
const TAP = 'data-sveltekit-preload-data="tap"';

/** Anchors in `source` whose href names another project but do not preload on tap. */
export function hoverPreloadingCrossProjectAnchors(source: string): string[] {
  return (source.match(/<a\b[^>]*>/g) ?? []).filter(
    (tag) => CROSS_PROJECT_HREF.test(tag) && !tag.includes(TAP),
  );
}

describe('cross-project links', () => {
  it('flags a cross-project anchor that preloads on hover', () => {
    const bad = `<a href={resolve(projectHref('/dashboard', other))} class="btn">Open</a>`;
    expect(hoverPreloadingCrossProjectAnchors(bad)).toHaveLength(1);
  });

  it('accepts a cross-project anchor that preloads on tap, and a same-project one', () => {
    const tapped = `<a href={resolve(projectHref('/dashboard', other))} ${TAP}>Open</a>`;
    const same = `<a href={resolve(projectHref('/dashboard'))}>Open</a>`;
    expect(hoverPreloadingCrossProjectAnchors(tapped)).toEqual([]);
    expect(hoverPreloadingCrossProjectAnchors(same)).toEqual([]);
  });

  it('no page or component links to another project with hover preload', () => {
    const offenders = svelteFiles(srcRoot).filter(
      (f) => hoverPreloadingCrossProjectAnchors(readFileSync(f, 'utf-8')).length > 0,
    );
    expect(offenders.map((f) => path.relative(srcRoot, f))).toEqual([]);
  });
});
