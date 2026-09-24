/**
 * Endpoint catalog: every backend call the frontend makes (path template
 * + method + query params), checked against the vendored OpenAPI
 * snapshot (`contracts/openprocessor/openapi/curation.json`, synced
 * from OpenProcessor's `contracts/openapi/curation.json` via `npm run
 * contract:sync`). A path/method that disappears or renames on the
 * backend, or a query param the backend stops declaring, fails here.
 *
 * The call sites are extracted mechanically by
 * `apiCallScanner.ts::scanApiCallSites` — every `${API_PREFIX}/...`
 * template in the scanned files — not hand-copied. New files are added
 * to `SCANNED_FILES` deliberately; the "no other file references
 * API_PREFIX" guard below fails the build if a future call site lands
 * somewhere this test doesn't look, so that omission can't happen
 * silently.
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import { describe, expect, it } from 'vitest';
import openapi from '../../../contracts/openprocessor/openapi/curation.json';
import { scanApiCallSites, type ApiCallSite } from './apiCallScanner';

const here = path.dirname(fileURLToPath(import.meta.url));
const srcRoot = path.resolve(here, '..', '..');

/** Every file that composes a backend URL through `${API_PREFIX}`. */
const SCANNED_FILES = [
  'lib/api.ts',
  'lib/sse.ts',
  'routes/export/+page.svelte',
  'lib/components/SlotCard.svelte',
] as const;

function read(rel: string): string {
  return readFileSync(path.join(srcRoot, rel), 'utf-8');
}

/**
 * A handful of call sites compose the path from a runtime object
 * property (a slot's own declared `endpoints`/`extras.datasetExport`
 * paths — `spec.buildPath`, `spec.statusPath`, `spec.endpoints.setBox`/
 * `patchMeta`/`batchStatus`) rather than a literal or a module-level
 * constant, which `apiCallScanner` can't resolve statically.
 *
 * Each override is anchored to a `marker` — a unique, nearby string
 * (the enclosing function's declaration) that must appear verbatim in
 * the scanned file — and resolves to the nearest scanned call site
 * *after* that marker. `setSlotBox`/`patchSlotMeta`/`batchRegionStatus`
 * all happen to share the identical raw template text
 * (`` `${API_PREFIX}${path}` ``), so matching by raw text alone would
 * be ambiguous; matching by (marker, nearest-following-site) is not.
 *
 * `path`/`method`/`queryParams` are optional: an omitted field keeps
 * whatever the scanner itself resolved (e.g. `getRegions`'s query keys
 * ARE mechanically resolved via its `RegionsQuery` interface — only its
 * `browsePath` parameter, a runtime string, needs a path override).
 */
const MANUAL_OVERRIDES: Array<{
  file: (typeof SCANNED_FILES)[number];
  marker: string;
  path?: string;
  method?: string;
  queryParams?: string[] | null;
}> = [
  {
    file: 'lib/api.ts',
    marker: 'export function exportSingleClass(',
    // spec.buildPath: a region slot's declared extras.datasetExport.buildPath,
    // '/export/single_class' for the built-in profile today.
    path: '/export/single_class',
    method: 'POST',
    queryParams: [],
  },
  {
    file: 'lib/api.ts',
    marker: 'export function exportSingleClassStatus(',
    path: '/export/single_class/status',
    method: 'GET',
    queryParams: ['profile_name'],
  },
  {
    file: 'lib/api.ts',
    marker: 'export async function setSlotBox(',
    // spec.endpoints.setBox/clearBox — a region slot declares both as
    // '/crops/{id}/region'.
    path: '/crops/*/region',
    method: 'PUT',
    queryParams: [],
  },
  {
    file: 'lib/api.ts',
    marker: 'export async function patchSlotMeta(',
    // spec.endpoints.patchMeta — '/crops/{id}/region_meta' today.
    path: '/crops/*/region_meta',
    method: 'PATCH',
    queryParams: [],
  },
  {
    file: 'lib/api.ts',
    marker: 'export async function batchRegionStatus(',
    // spec.endpoints.batchStatus — '/regions/batch_status' today.
    path: '/regions/batch_status',
    method: 'POST',
    queryParams: [],
  },
  {
    file: 'lib/api.ts',
    marker: 'export async function getRegions(',
    // browsePath is the slot's declared capabilities.queue.browsePath —
    // '/regions' for a region slot. Query params ARE
    // resolved mechanically (RegionsQuery), so only path is overridden.
    path: '/regions',
  },
  {
    file: 'lib/components/SlotCard.svelte',
    marker: 'const thumbUrl = $derived(',
    // thumbCap.path(id, size) — a region slot declares
    // '/crops/{id}/region_thumbnail?size={size}'
    // (capabilities.subBox.thumbnail.path).
    path: '/crops/*/region_thumbnail',
    method: 'GET',
    queryParams: ['size'],
  },
];

interface ResolvedCall {
  file: string;
  path: string;
  method: string;
  queryParams: string[] | null;
  raw: string;
}

function resolveCalls(file: (typeof SCANNED_FILES)[number]): ResolvedCall[] {
  const src = read(file);
  const sites: ApiCallSite[] = scanApiCallSites(src);
  const overridesForFile = MANUAL_OVERRIDES.filter((o) => o.file === file);

  // marker -> the index of the nearest scanned site after it.
  const overrideSiteIndex = new Map<number, (typeof overridesForFile)[number]>();
  for (const o of overridesForFile) {
    const markerIdx = src.indexOf(o.marker);
    expect(
      markerIdx,
      `manual override marker not found in ${file}: ${o.marker}`,
    ).toBeGreaterThan(-1);
    const candidates = sites
      .filter((s) => s.index > markerIdx)
      .sort((a, b) => a.index - b.index);
    expect(
      candidates.length,
      `manual override marker matched no following call site in ${file}: ${o.marker}`,
    ).toBeGreaterThan(0);
    overrideSiteIndex.set(candidates[0].index, o);
  }

  return sites.map((s) => {
    const override = overrideSiteIndex.get(s.index);
    return {
      file,
      path: override?.path ?? s.pathTemplate,
      method: override?.method ?? s.method,
      queryParams:
        override?.queryParams !== undefined ? override.queryParams : s.queryParams,
      raw: s.raw,
    };
  });
}

describe('endpoint catalog: completeness', () => {
  it('scans a non-trivial number of call sites (guards a vacuous pass)', () => {
    const total = SCANNED_FILES.reduce((n, f) => n + resolveCalls(f).length, 0);
    expect(total).toBeGreaterThan(50);
  });

  it('no other src/ file references ${API_PREFIX} outside SCANNED_FILES', () => {
    // Mechanical completeness guard (§3.3's "endpoint catalog" design):
    // a new fetch call site in a file this test doesn't scan must fail
    // the build, not silently go unchecked.
    const out = execFileSync(
      'grep',
      ['-rl', '--include=*.ts', '--include=*.svelte', '${API_PREFIX}', srcRoot],
      { encoding: 'utf-8' },
    );
    const hits = out
      .split('\n')
      .filter(Boolean)
      .map((p) => path.relative(srcRoot, p))
      .filter(
        (p) => !p.endsWith('.test.ts') && !p.startsWith(path.join('lib', 'contract')),
      );
    const scannedSet = new Set<string>(SCANNED_FILES);
    const unscanned = hits.filter(
      (p) => !scannedSet.has(p as (typeof SCANNED_FILES)[number]),
    );
    expect(unscanned).toEqual([]);
  });
});

// -- OpenAPI lookup -----------------------------------------------------

interface OpenApiDoc {
  paths: Record<string, Record<string, { parameters?: Array<{ name?: string }> }>>;
}
const doc = openapi as unknown as OpenApiDoc;

/** OpenAPI paths are absolute under the backend's own prefix
 *  (`/curation/...`, the default `OP_API_PREFIX`). The frontend composes
 *  `${API_PREFIX}/...` where `API_PREFIX` defaults to the same
 *  `/curation` — mapping `${API_PREFIX}` -> `/curation` is exactly that
 *  default-prefix identification, matching this project's plan
 *  instructions. */
const OPENAPI_PREFIX = '/curation';

function normalizeSegments(p: string): string[] {
  return p
    .split('/')
    .filter((s) => s.length > 0)
    .map((seg) =>
      seg.startsWith('{') && seg.endsWith('}') ? '*' : seg === '*' ? '*' : seg,
    );
}

interface OpenApiOperation {
  openApiPath: string;
  method: string;
  params: Set<string>;
}

function findOperation(frontendPath: string, method: string): OpenApiOperation | null {
  const wanted = normalizeSegments(frontendPath);
  for (const [openApiPath, methods] of Object.entries(doc.paths)) {
    if (!openApiPath.startsWith(OPENAPI_PREFIX)) continue;
    const opSegs = normalizeSegments(openApiPath.slice(OPENAPI_PREFIX.length));
    if (opSegs.length !== wanted.length) continue;
    const matches = opSegs.every(
      (seg, i) => seg === '*' || wanted[i] === '*' || seg === wanted[i],
    );
    if (!matches) continue;
    const op = methods[method.toLowerCase()];
    if (!op) continue;
    return {
      openApiPath,
      method,
      params: new Set(
        (op.parameters ?? []).map((p) => p.name).filter((x): x is string => !!x),
      ),
    };
  }
  return null;
}

describe('endpoint catalog: every call resolves to a real OpenAPI operation', () => {
  for (const file of SCANNED_FILES) {
    const calls = resolveCalls(file);
    describe(file, () => {
      it('found at least one call site', () => {
        expect(calls.length).toBeGreaterThan(0);
      });

      for (const call of calls) {
        const label = `${call.method} ${call.path}`;
        it(`${label} exists in the OpenAPI contract`, () => {
          const op = findOperation(call.path, call.method);
          expect(
            op,
            `no OpenAPI operation matches ${label} (raw: ${call.raw})`,
          ).not.toBeNull();
        });

        it(`${label} — every sent query key is a declared OpenAPI parameter`, () => {
          if (call.queryParams == null) {
            // Untyped passthrough (e.g. a bare `Record<string, unknown>`
            // filter object) the scanner correctly declines to guess at.
            // Path+method is still checked above.
            return;
          }
          const op = findOperation(call.path, call.method);
          if (!op) return; // already failed the existence assertion above
          const undeclared = call.queryParams.filter((k) => !op.params.has(k));
          expect(
            undeclared,
            `${label} sends undeclared params (raw: ${call.raw})`,
          ).toEqual([]);
        });
      }
    });
  }
});
