/**
 * Endpoint catalog: every backend call the frontend makes (path template
 * + method + query params), checked against the vendored OpenAPI
 * snapshot (`contracts/openprocessor/openapi/curation.json`, synced
 * from OpenProcessor's `contracts/openapi/curation.json` via `npm run
 * contract:sync`). A path/method that disappears or renames on the
 * backend, or a query param the backend stops declaring, fails here.
 *
 * The call sites are extracted mechanically by
 * `apiCallScanner.ts::scanApiCallSites` — every `${scoped()}/...`
 * template in the scanned files — not hand-copied. `scoped()` is the
 * one function every scoped call builds its URL through (`api.ts`,
 * groundwork for multi-project support); `globalApi()` is a separate,
 * currently-unused builder for endpoints that will stay global once
 * projects land — no call site uses it yet, so there is nothing for
 * this catalog to scan there. New files are added
 * to `SCANNED_FILES` deliberately; the "no other file references
 * scoped()" guard below fails the build if a future call site lands
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

/** Every file that composes a backend URL through `${scoped()}`. */
const SCANNED_FILES = [
  'lib/api.ts',
  'lib/sse.ts',
  'routes/p/[project]/export/+page.svelte',
  'lib/components/SlotCard.svelte',
] as const;

/** Every file that composes a backend URL through `${globalApi()}` — the
 *  small set of routes P1 keeps global (never project-scoped): the
 *  project list itself, and the global health/events used before a
 *  project is even selected. */
const GLOBAL_SCANNED_FILES = ['lib/api.ts', 'lib/sse.ts'] as const;

/** Every file that composes a backend URL through
 *  `${projectPrefix(project)}` — a SCOPED route addressed through a
 *  specific project's own served prefix rather than the active one
 *  (`/projects` row actions such as pause/resume). Resolved against the
 *  scoped OpenAPI paths exactly like `${scoped()}`. */
const PROJECT_PREFIX_MARKER = '${projectPrefix(project)}';
const PROJECT_PREFIX_SCANNED_FILES = ['lib/api.ts'] as const;

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
 * (`` `${scoped()}${path}` ``), so matching by raw text alone would
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
    file: 'lib/api.ts',
    marker: 'export function runServedNextStep(',
    // step.method/step.path: a finished import's served `next_steps`
    // entry (W10.11). The spec's one documented example is
    // `POST /regions/cluster`; anything else it serves is the server's
    // own route.
    path: '/regions/cluster',
    method: 'POST',
    queryParams: [],
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

/**
 * Routes proposed to OpenProcessor as a binding wire contract but not yet
 * implemented server-side, so they can't appear in the vendored OpenAPI
 * snapshot yet. Each entry names the plan section that binds it — remove
 * the entry (not widen it) the moment `npm run contract:sync` picks up
 * the real operation; `keymapActions.test.ts` §5.7 is the sibling check
 * that will then start failing loudly if this allow-list is stale.
 *
 * K2 (docs/design/configurable-keyboard-shortcuts-plan-2026-09-26.md §4.1):
 * OpenProcessor W2b hasn't landed the four `/keymap*` routes yet.
 */
const PENDING_BACKEND: Array<{ path: string; method: string }> = [
  { path: '/keymap', method: 'GET' },
  { path: '/keymap', method: 'PUT' },
  { path: '/keymap/validate', method: 'POST' },
  { path: '/keymap/reset', method: 'POST' },
  // W8 multi-box regions (feat/w8-multibox-lockstep, docs/design/
  // w8-multibox-frontend-plan-2026-09-26.md): new in the backend's W8
  // wave, not yet in the vendored OpenAPI snapshot (cutover/projects-
  // lifecycle 29807534 has no W8). Delete these four entries when backend
  // W8 rebases onto projects and `npm run contract:sync` picks them up.
  { path: '/crops/*/regions', method: 'PUT' },
  { path: '/crops/batch_regions', method: 'PUT' },
  { path: '/crops/*/regions/*', method: 'PATCH' },
  { path: '/regions/batch_box_state', method: 'POST' },
];

/**
 * OpenProcessor W10 (labeled-dataset import + Reprocess,
 * any_domain_plan.md §7.12 / W10.14) — built against the frozen spec
 * before the backend implements it (docs/design/
 * w10-import-reprocess-ui-plan-2026-09-27.md). DELETE THIS LIST when
 * `npm run contract:sync` vendors W10's OpenAPI; every route then has to
 * resolve for real.
 */
const PENDING_BACKEND_W10: Array<{ path: string; method: string }> = [
  { path: '/datasets/formats', method: 'GET' },
  { path: '/datasets/uploads', method: 'POST' },
  { path: '/datasets/preview', method: 'POST' },
  { path: '/datasets/imports', method: 'POST' },
  { path: '/datasets/imports', method: 'GET' },
  { path: '/datasets/imports/*', method: 'GET' },
  { path: '/datasets/imports/*/issues', method: 'GET' },
  { path: '/datasets/imports/*/entries', method: 'GET' },
  { path: '/datasets/imports/*/cancel', method: 'POST' },
  { path: '/datasets/imports/*/resume', method: 'POST' },
  { path: '/datasets/imports/*/undo', method: 'POST' },
  { path: '/reprocess', method: 'POST' },
  { path: '/crops/*/reprocess', method: 'POST' },
  { path: '/reprocess/jobs/*', method: 'GET' },
  { path: '/reprocess/jobs/*/cancel', method: 'POST' },
];

/**
 * OpenProcessor W3 (prompt-pack CRUD, any_domain_plan.md §3.2 / §7.2) and
 * the pack half of W5 (`POST /prompt_packs/test`, §5.1 / §7.5) — built
 * against the frozen spec before the backend implements them (docs/design/
 * w3-pack-editor-ui-plan-2026-09-27.md). DELETE THIS LIST when
 * `npm run contract:sync` vendors W3 (and W5 for `/prompt_packs/test`);
 * every route then has to resolve for real.
 */
const PENDING_BACKEND_W3: Array<{ path: string; method: string }> = [
  { path: '/prompt_packs', method: 'GET' },
  { path: '/prompt_packs/schema', method: 'GET' },
  { path: '/prompt_packs/validate', method: 'POST' },
  { path: '/prompt_packs/*', method: 'GET' },
  { path: '/prompt_packs/*', method: 'PUT' },
  { path: '/prompt_packs/*', method: 'DELETE' },
  { path: '/prompt_packs/*/revisions', method: 'GET' },
  { path: '/prompt_packs/*/revisions/*', method: 'GET' },
  { path: '/prompt_packs/*/clone', method: 'POST' },
  { path: '/prompt_packs/active', method: 'GET' },
  { path: '/prompt_packs/*/activate', method: 'POST' },
  { path: '/prompt_packs/active/rollback', method: 'POST' },
  { path: '/prompt_packs/test', method: 'POST' },
];

/**
 * OpenProcessor W4 (region-profile CRUD and the config vocabulary,
 * any_domain_plan.md §4.2 / §7.3 / §7.4) — built against the frozen spec
 * before the backend implements it (docs/design/
 * w4-profile-editor-ui-plan-2026-09-27.md). DELETE THIS LIST when
 * `npm run contract:sync` vendors W4; every route then has to resolve for
 * real.
 */
const PENDING_BACKEND_W4: Array<{ path: string; method: string }> = [
  { path: '/region_profiles', method: 'GET' },
  { path: '/region_profiles/schema', method: 'GET' },
  { path: '/region_profiles/validate', method: 'POST' },
  { path: '/region_profiles/*', method: 'GET' },
  { path: '/region_profiles/*', method: 'PUT' },
  { path: '/region_profiles/*', method: 'DELETE' },
  { path: '/region_profiles/*/revisions', method: 'GET' },
  { path: '/region_profiles/*/revisions/*', method: 'GET' },
  { path: '/region_profiles/*/clone', method: 'POST' },
  { path: '/region_profiles/active', method: 'GET' },
  { path: '/region_profiles/*/activate', method: 'POST' },
  { path: '/region_profiles/active/rollback', method: 'POST' },
  { path: '/region_profiles/deactivate', method: 'POST' },
  { path: '/region_profiles/active/impact', method: 'GET' },
  { path: '/config/vocabulary', method: 'GET' },
];

function isPendingBackend(path: string, method: string): boolean {
  return [
    ...PENDING_BACKEND,
    ...PENDING_BACKEND_W10,
    ...PENDING_BACKEND_W3,
    ...PENDING_BACKEND_W4,
  ].some((p) => p.path === path && p.method === method);
}

interface ResolvedCall {
  file: string;
  path: string;
  method: string;
  queryParams: string[] | null;
  raw: string;
}

function resolveCalls(
  file: (typeof SCANNED_FILES)[number],
  marker: string = '${scoped()}',
): ResolvedCall[] {
  const src = read(file);
  const sites: ApiCallSite[] = scanApiCallSites(src, marker);
  // Every MANUAL_OVERRIDES marker anchors a ${scoped()} call site — none
  // apply to the ${globalApi()} scan.
  const overridesForFile =
    marker === '${scoped()}' ? MANUAL_OVERRIDES.filter((o) => o.file === file) : [];

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

function grepFilesWith(marker: string): string[] {
  let out = '';
  try {
    out = execFileSync(
      'grep',
      ['-rlF', '--include=*.ts', '--include=*.svelte', marker, srcRoot],
      { encoding: 'utf-8' },
    );
  } catch (e) {
    if ((e as { status?: number }).status !== 1) throw e;
  }
  return out
    .split('\n')
    .filter(Boolean)
    .map((p) => path.relative(srcRoot, p))
    .filter(
      (p) => !p.endsWith('.test.ts') && !p.startsWith(path.join('lib', 'contract')),
    );
}

describe('endpoint catalog: completeness', () => {
  it('scans a non-trivial number of call sites (guards a vacuous pass)', () => {
    const total = SCANNED_FILES.reduce((n, f) => n + resolveCalls(f).length, 0);
    expect(total).toBeGreaterThan(50);
  });

  it('no other src/ file references ${scoped()} outside SCANNED_FILES', () => {
    // Mechanical completeness guard (§3.3's "endpoint catalog" design):
    // a new fetch call site in a file this test doesn't scan must fail
    // the build, not silently go unchecked.
    const out = execFileSync(
      'grep',
      ['-rl', '--include=*.ts', '--include=*.svelte', '${scoped()}', srcRoot],
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

  it('no other src/ file references ${projectPrefix(project)} outside PROJECT_PREFIX_SCANNED_FILES', () => {
    const scannedSet = new Set<string>(PROJECT_PREFIX_SCANNED_FILES);
    expect(
      grepFilesWith(PROJECT_PREFIX_MARKER).filter((p) => !scannedSet.has(p)),
    ).toEqual([]);
  });

  it('every projectPrefix() URL in api.ts uses the scanned marker', () => {
    // A wrapper that names its parameter anything but `project` would
    // slip past the marker scan; fail instead of silently skipping it.
    const src = read('lib/api.ts');
    const uses = src.match(/\$\{projectPrefix\([^)]*\)\}/g) ?? [];
    expect(uses.length).toBeGreaterThan(0);
    expect(uses.filter((u) => u !== PROJECT_PREFIX_MARKER)).toEqual([]);
  });

  it('no other src/ file references ${globalApi()} outside GLOBAL_SCANNED_FILES', () => {
    let out = '';
    try {
      out = execFileSync(
        'grep',
        ['-rl', '--include=*.ts', '--include=*.svelte', '${globalApi()}', srcRoot],
        { encoding: 'utf-8' },
      );
    } catch (e) {
      // grep exits 1 when there are no matches at all — not an error here.
      if ((e as { status?: number }).status !== 1) throw e;
    }
    const hits = out
      .split('\n')
      .filter(Boolean)
      .map((p) => path.relative(srcRoot, p))
      .filter(
        (p) => !p.endsWith('.test.ts') && !p.startsWith(path.join('lib', 'contract')),
      );
    const scannedSet = new Set<string>(GLOBAL_SCANNED_FILES);
    const unscanned = hits.filter(
      (p) => !scannedSet.has(p as (typeof GLOBAL_SCANNED_FILES)[number]),
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
 *  (`/curation/...`, the default `OP_API_PREFIX`). P1 projects cutover:
 *  every SCOPED route additionally lives under `/projects/{project}`
 *  (`GLOBAL_ROUTES` below is the fixed, explicit list of what stays
 *  global) — `${scoped()}/foo` resolves against
 *  `/curation/projects/{project}/foo`, `${globalApi()}/foo` against
 *  `/curation/foo` directly. */
const OPENAPI_PREFIX = '/curation';
const SCOPED_OPENAPI_PREFIX = '/curation/projects/*';

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

function findOperation(
  frontendPath: string,
  method: string,
  scope: 'scoped' | 'global' = 'scoped',
): OpenApiOperation | null {
  const prefix = scope === 'scoped' ? SCOPED_OPENAPI_PREFIX : OPENAPI_PREFIX;
  const wanted = normalizeSegments(`${prefix}${frontendPath}`);
  for (const [openApiPath, methods] of Object.entries(doc.paths)) {
    if (!openApiPath.startsWith(OPENAPI_PREFIX)) continue;
    const opSegs = normalizeSegments(openApiPath);
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

function describeCalls(
  file: string,
  calls: ResolvedCall[],
  scope: 'scoped' | 'global',
): void {
  describe(file, () => {
    it('found at least one call site', () => {
      expect(calls.length).toBeGreaterThan(0);
    });

    for (const call of calls) {
      const label = `${call.method} ${call.path}`;
      const pending = isPendingBackend(call.path, call.method);
      it(`${label} exists in the OpenAPI contract${pending ? ' (skipped: pending backend)' : ''}`, (ctx) => {
        if (pending) {
          ctx.skip();
          return;
        }
        const op = findOperation(call.path, call.method, scope);
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
        const op = findOperation(call.path, call.method, scope);
        if (!op) return; // already failed the existence assertion above
        const undeclared = call.queryParams.filter((k) => !op.params.has(k));
        expect(undeclared, `${label} sends undeclared params (raw: ${call.raw})`).toEqual(
          [],
        );
      });
    }
  });
}

describe('endpoint catalog: every call resolves to a real OpenAPI operation', () => {
  for (const file of SCANNED_FILES) {
    describeCalls(file, resolveCalls(file), 'scoped');
  }
});

describe('endpoint catalog: every served-project-prefix call resolves to a real OpenAPI operation', () => {
  for (const file of PROJECT_PREFIX_SCANNED_FILES) {
    describeCalls(
      `${file} (projectPrefix)`,
      resolveCalls(file, PROJECT_PREFIX_MARKER),
      'scoped',
    );
  }
});

describe('endpoint catalog: every GLOBAL call resolves to a real OpenAPI operation', () => {
  for (const file of GLOBAL_SCANNED_FILES) {
    describeCalls(file, resolveCalls(file, '${globalApi()}'), 'global');
  }
});
