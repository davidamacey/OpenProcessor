import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(path.join(here, '+page.svelte'), 'utf-8');

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
  it('the export catch block renders apiErrorText(e), not a generic fallback string', () => {
    const idx = src.indexOf('async function runExport');
    expect(idx).toBeGreaterThan(-1);
    const fn = src.slice(idx, src.indexOf('\n  }\n', idx));
    expect(fn).toMatch(/catch \(e\)/);
    expect(fn).toMatch(/apiErrorText\(e\)/);
  });
});
