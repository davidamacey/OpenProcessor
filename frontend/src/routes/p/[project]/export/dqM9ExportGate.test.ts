/**
 * DQ-M9 frontend half (docs/design/data-quality-pass-2026-09-24.md §7
 * FRONTEND item 6): the Export button was enabled unconditionally, over a
 * table that could be all red "block" (every class at 0 `class_validated`)
 * — `POST /export/yolo` itself has no readiness gate. Fixed to disable
 * Export (with an explanatory tooltip) when the served rows say nothing
 * is exportable — see src/lib/export/exportDatasetRows.ts's
 * `isNothingExportable()` for the served-only rule (no hardcoded
 * threshold).
 *
 * Same static source-scan convention as exportPageCleanup.test.ts — no
 * `@testing-library/svelte` mount harness for a page this size.
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(path.join(here, '+page.svelte'), 'utf-8');

describe('DQ-M9: Export is disabled when the served rows say nothing is exportable', () => {
  it('nothingExportable is derived from isNothingExportable(rows), gated on the initial loading window', () => {
    expect(src).toMatch(
      /const nothingExportable = \$derived\(!loading && isNothingExportable\(rows\)\);/,
    );
  });

  it('the Export button is disabled by exportRunning OR nothingExportable', () => {
    expect(src).toMatch(/disabled=\{exportRunning \|\| nothingExportable\}/);
  });

  it('carries an explanatory tooltip when disabled for this reason', () => {
    const idx = src.indexOf('disabled={exportRunning || nothingExportable}');
    expect(idx).toBeGreaterThan(-1);
    const slice = src.slice(idx, idx + 250);
    expect(slice).toMatch(/title=\{nothingExportable/);
    expect(slice).toMatch(/Nothing to export/);
  });
});

/**
 * dq-queues cutover (2026-09-24): POST {API_PREFIX}/export/yolo and
 * /export/single_class now 422 with a "nothing to export" detail
 * (e.g. "nothing to export: 0 items are class_validated..."). The export
 * flow's catch already renders `(e as Error).message`, and ApiError's
 * message is built from the response's `detail` field (api.ts's
 * errorDetail()) — so the served 422 text reaches the toast/inline
 * error verbatim with no special-casing needed. This proves the catch
 * path is generic, not that it silently swallows the detail.
 */
describe('dq-queues: the 422 "nothing to export" detail surfaces through the generic catch', () => {
  it('the export catch block renders (e as Error).message, not a generic fallback string', () => {
    const idx = src.indexOf('async function runExport');
    expect(idx).toBeGreaterThan(-1);
    const fn = src.slice(idx, src.indexOf('\n  }\n', idx));
    expect(fn).toMatch(/catch \(e\)/);
    expect(fn).toMatch(/\(e as Error\)\.message/);
  });
});
