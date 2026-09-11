/**
 * TypeScript types mirroring the openprocessor `GET /curation/methods` capability-
 * discovery response (curation-strategy plan, Phase 0 —
 * docs/curation-strategy-plan-2026-09.md §3, §5.3, §7).
 *
 * The backend exposes four orthogonal registries so the frontend can
 * discover what's currently offered without hardcoding ids:
 * `cluster_methods` (assignment, writes cluster_id), `review_sorts`
 * (review-queue ordering), `overlays` (selection/projection that never
 * write cluster_id), `scores` (per-crop scored fields backing sorts /
 * overlays). Every entry carries at least `id`, `label`, `status`.
 *
 * This phase (Phase 0) is pure plumbing — types + a fetch helper + an
 * inert store. Nothing in the UI renders any of this yet; that's Phase 3.
 *
 * Forward-tolerant, matching the convention documented at the top of
 * `./types.ts`: a server that's ahead of this build (new ids, a new
 * status value) must degrade gracefully, never throw. `status` values
 * this build doesn't recognize normalize to `'disabled'` — the one
 * bucket a caller must never surface as selectable — and malformed
 * entries (missing/non-string `id` or `label`) are dropped rather than
 * crashing the whole parse.
 */

export type MethodStatus = 'stable' | 'experimental' | 'shadow' | 'disabled';

const KNOWN_STATUSES: ReadonlySet<string> = new Set([
  'stable',
  'experimental',
  'shadow',
  'disabled',
]);

/**
 * Normalize a server-sent `status` string. Unknown/garbage values fall
 * back to `'disabled'` rather than throwing — that's the bucket every
 * consumer already treats as "never offer this to the operator," so an
 * unrecognized future status degrades to inert instead of crashing or
 * (worse) silently rendering as usable.
 */
export function normalizeMethodStatus(raw: unknown): MethodStatus {
  return typeof raw === 'string' && KNOWN_STATUSES.has(raw)
    ? (raw as MethodStatus)
    : 'disabled';
}

interface MethodInfoBase {
  id: string;
  label: string;
  status: MethodStatus;
  description?: string | null;
}

export interface ClusterMethodInfo extends MethodInfoBase {
  /** True on exactly the entry matching today's DEFAULT_METHOD ('ivf'). */
  default?: boolean;
}

export interface ReviewSortInfo extends MethodInfoBase {
  default?: boolean;
  /** OpenSearch field this sort reads; used to grey out the option when
   *  `field_coverage` is 0 (nothing has been backfilled yet). */
  requires_field?: string | null;
  /** Fraction [0,1] of the pool that has this field populated. */
  field_coverage?: number | null;
}

export interface OverlayInfo extends MethodInfoBase {
  requires_field?: string | null;
  field_coverage?: number | null;
}

export interface ScoreInfo extends MethodInfoBase {
  requires_field?: string | null;
  field_coverage?: number | null;
  /** Scorer version, for surfacing mixed-version coverage. */
  version?: string | null;
}

export interface OpMethodsResponse {
  cluster_methods: ClusterMethodInfo[];
  review_sorts: ReviewSortInfo[];
  overlays: OverlayInfo[];
  scores: ScoreInfo[];
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null;
}

function optBool(v: unknown): boolean | undefined {
  return typeof v === 'boolean' ? v : undefined;
}

function optString(v: unknown): string | null | undefined {
  if (typeof v === 'string') return v;
  return v === null ? null : undefined;
}

function optNumber(v: unknown): number | null | undefined {
  if (typeof v === 'number' && Number.isFinite(v)) return v;
  return v === null ? null : undefined;
}

/**
 * Normalize one raw entry shared by all four registries. Returns `null`
 * (dropped by the caller) when `id`/`label` aren't usable strings — a
 * malformed entry from a future server build should just not render,
 * not take the whole `/curation/methods` parse down with it.
 */
function normalizeBase(raw: unknown): MethodInfoBase | null {
  if (!isRecord(raw)) return null;
  const { id, label } = raw;
  if (typeof id !== 'string' || !id) return null;
  if (typeof label !== 'string' || !label) return null;
  const base: MethodInfoBase = { id, label, status: normalizeMethodStatus(raw.status) };
  const description = optString(raw.description);
  if (description !== undefined) base.description = description;
  return base;
}

function normalizeList<T extends MethodInfoBase>(
  raw: unknown,
  extra: (base: MethodInfoBase, rawEntry: Record<string, unknown>) => T,
): T[] {
  if (!Array.isArray(raw)) return [];
  const out: T[] = [];
  for (const entry of raw) {
    const base = normalizeBase(entry);
    if (!base || !isRecord(entry)) continue;
    out.push(extra(base, entry));
  }
  return out;
}

/**
 * Parse+normalize a raw `/curation/methods` payload. Never throws — any
 * unrecognized shape (missing keys, non-array registries, garbage
 * entries) degrades to empty lists for the affected registry rather than
 * propagating an exception into the api/store layer. Unknown `id`s are
 * carried through as-is (nothing here validates ids against a fixed
 * enum, by design — that's how a new server-side method shows up without
 * a frontend redeploy); unknown `status` values are neutralized to
 * `'disabled'` by `normalizeMethodStatus`.
 */
export function parseKbMethodsResponse(raw: unknown): OpMethodsResponse {
  const rec = isRecord(raw) ? raw : {};
  return {
    cluster_methods: normalizeList<ClusterMethodInfo>(rec.cluster_methods, (base, e) => ({
      ...base,
      default: optBool(e.default),
    })),
    review_sorts: normalizeList<ReviewSortInfo>(rec.review_sorts, (base, e) => ({
      ...base,
      default: optBool(e.default),
      requires_field: optString(e.requires_field),
      field_coverage: optNumber(e.field_coverage),
    })),
    overlays: normalizeList<OverlayInfo>(rec.overlays, (base, e) => ({
      ...base,
      requires_field: optString(e.requires_field),
      field_coverage: optNumber(e.field_coverage),
    })),
    scores: normalizeList<ScoreInfo>(rec.scores, (base, e) => ({
      ...base,
      requires_field: optString(e.requires_field),
      field_coverage: optNumber(e.field_coverage),
      version: optString(e.version),
    })),
  };
}

/**
 * Whether `/curation/methods` currently reports the pool-scale `diverse` overlay
 * (curation-strategy plan Phase 4 — core-set / k-center-greedy selection,
 * `POST /curation/select/diverse` + `GET /curation/crops?order=diverse`) as safe to
 * offer in the UI.
 *
 * `diverse` lives in the `overlays` registry, not `review_sorts` — it
 * never writes `cluster_id` (plan §3), so it's a different axis from a
 * review-queue sort even though `/clusters/[id]` folds it into the same
 * `orderMode`/`strategyBar.sort` selection for UX simplicity.
 *
 * Single source of truth for the gate: both `/clusters/[id]` (deciding
 * whether to widen `allowedIds` past `['default', 'outliers']`) and
 * `StrategyBar.svelte` (deciding whether to render the `k` stepper) call
 * this instead of re-deriving the same `status` check twice. `shadow` /
 * `disabled` / simply-absent (talking to a pre-Phase-4 backend, or
 * `OP_SELECT_DIVERSE_ENABLED` off) must never surface the control — same
 * stable/experimental-only bar every other overlay/score entry clears.
 */
export function isDiverseOverlayAvailable(overlays: OverlayInfo[]): boolean {
  return overlays.some(
    (o) => o.id === 'diverse' && (o.status === 'stable' || o.status === 'experimental'),
  );
}

/**
 * Hardcoded fallback for when `/curation/methods` 404s, or the request fails
 * for any other reason (plan §5.3 — graceful degradation is required so
 * the two repos can deploy independently). This must mirror what's
 * actually implemented **today**, not the target end-state:
 *
 * - `cluster_methods`: only IVF is real as a selectable default
 *   (`cluster_methods/ivf.py`, `DEFAULT_METHOD = 'ivf'` in
 *   `cluster_methods/__init__.py`). AHC/HDBSCAN exist in the backend
 *   registry but are not operator-facing defaults, so they're
 *   deliberately left out rather than guessed at.
 * - `review_sorts`: `op_review.py` hardcodes `sort = [{updated_at:
 *   desc}]` per tab today; there is no named alternative sort yet
 *   (`review_sorts.py` is Phase 3). One `'default'` entry, marked
 *   default+stable — this is NOT `'representativeness'` /
 *   `'uncertainty'` / any of the Phase-3 sort ids from the plan, since
 *   those don't exist in the backend yet.
 * - `overlays` / `scores`: none of `crop_scores/`, `selection/`,
 *   `embedding_viz.py` exist yet (Phase 1/4/5) — both lists are empty.
 */
export const FALLBACK_METHODS: OpMethodsResponse = {
  cluster_methods: [
    {
      id: 'ivf',
      label: 'FAISS IVF-512 (production)',
      status: 'stable',
      default: true,
    },
  ],
  review_sorts: [
    {
      id: 'default',
      label: 'Recent first',
      status: 'stable',
      default: true,
    },
  ],
  overlays: [],
  scores: [],
};
