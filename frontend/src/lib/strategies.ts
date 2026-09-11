/**
 * TypeScript types mirroring the openprocessor `GET /curation/methods` capability-
 * discovery response (curation-strategy plan, Phase 0 —
 * docs/curation-strategy-plan-2026-09.md §3, §5.3, §7).
 *
 * CONFIRMED AGAINST THE REAL BACKEND (2026-09-10, live end-to-end check
 * after rebuilding/restarting op-api from `feat/op-curation-scores`):
 * the actual wire shape is a single flat `{strategies: [...], flags: {...}}`
 * — every entry carries an `axis: 'cluster' | 'sort' | 'score' | 'overlay'`
 * field (see `strategy_registry.py`'s `StrategyAxis`), NOT four separate
 * top-level arrays as this file originally assumed from the plan doc's
 * illustrative example. `parseKbMethodsResponse` below reshapes the flat
 * list into the four buckets (`cluster_methods`/`review_sorts`/`overlays`/
 * `scores`) client-side by grouping on `axis`, so every downstream consumer
 * (`isDiverseOverlayAvailable`, `StrategyBar`, etc.) keeps working against
 * the original four-array `OpMethodsResponse` shape unchanged — only this
 * parse function needed to change once the real contract was confirmed.
 *
 * The same `id` can legitimately appear in more than one axis (e.g.
 * `mistakenness` is both a `score` entry — the raw scored field/coverage —
 * and a `sort` entry — the review-queue ordering built on that field).
 * This is not a collision: each bucket is consumed in its own UI context.
 *
 * Every entry carries at least `id`, `label`, `status`.
 *
 * Forward-tolerant, matching the convention documented at the top of
 * `./types.ts`: a server that's ahead of this build (new ids, a new
 * status value, an unrecognized `axis`) must degrade gracefully, never
 * throw. `status` values this build doesn't recognize normalize to
 * `'disabled'` — the one bucket a caller must never surface as selectable
 * — and malformed entries (missing/non-string `id` or `label`, or an
 * `axis` this build doesn't route anywhere) are dropped rather than
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
  /**
   * Real count of pool docs that have `requires_field` populated
   * (`strategy_registry.py`'s `_compute_field_coverage`, audit-remediation
   * plan Phase 6 — CONFIRMED against the real backend, 2026-09-11: this is
   * a raw exists-count, e.g. `124921`, NOT a `[0,1]` fraction as an earlier
   * draft of this comment assumed before the real contract shipped).
   * `null`/`undefined` means "unknown" (coverage wasn't computed — e.g. a
   * transient OpenSearch failure, or `requires_field` is itself `null`
   * because this sort doesn't depend on a backfilled field at all) and
   * must NOT be treated the same as `0` ("genuinely empty, never
   * backfilled") — see `hasFieldCoverage` below, the single place that
   * distinction is encoded.
   */
  field_coverage?: number | null;
  /** Pool size the `field_coverage` count above is out of (same denominator
   *  for every entry in a given `/curation/methods` response). `null` whenever
   *  `field_coverage` itself is `null`. */
  field_coverage_total?: number | null;
}

export interface OverlayInfo extends MethodInfoBase {
  requires_field?: string | null;
  /** See `ReviewSortInfo.field_coverage` — same raw-count semantics. */
  field_coverage?: number | null;
  /** See `ReviewSortInfo.field_coverage_total`. */
  field_coverage_total?: number | null;
  /**
   * CONFIRMED (2026-09-10, live against the real `embedding_viz.py` /
   * `op_viz.py` / `strategy_registry.py` after the Phase 5 UMAP
   * neighborhood-purity gate finished — `requires_banner`, not the
   * earlier guessed `banner_required`). Per the plan's UMAP acceptance
   * bar (>=0.30 purity ships plain, 0.15-0.30 ships behind a persistent
   * "this projection is approximate" banner, <0.15 doesn't ship at all —
   * the third tier is already handled by `isEmbeddingVizAvailable`
   * returning false), this tells the frontend which of the first two
   * tiers the real measured purity (also sent as `purity`, a float)
   * landed in. Today it's `false` — the real run measured 0.472,
   * comfortably in the "ship plain" tier.
   */
  requires_banner?: boolean;
  /** The measured 2-d neighborhood-purity value backing `requires_banner`
   *  (see `docs/design/curation_scores.md` in openprocessor for the full
   *  measurement writeup). Informational — nothing in this file
   *  re-derives a ship-tier decision from it; that's the backend's job. */
  purity?: number | null;
}

export interface ScoreInfo extends MethodInfoBase {
  requires_field?: string | null;
  /** See `ReviewSortInfo.field_coverage` — same raw-count semantics. */
  field_coverage?: number | null;
  /** See `ReviewSortInfo.field_coverage_total`. */
  field_coverage_total?: number | null;
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

/**
 * Group the real wire shape's flat `strategies` array by its `axis` field,
 * normalizing each entry through `extra`. An entry whose `axis` isn't one
 * of the four this build knows how to route (a future server's new axis)
 * or whose `id`/`label` aren't usable strings is dropped rather than
 * taking the whole parse down — same forward-tolerant contract as every
 * other malformed-entry case in this file.
 */
function normalizeAxis<T extends MethodInfoBase>(
  raw: unknown,
  axis: string,
  extra: (base: MethodInfoBase, rawEntry: Record<string, unknown>) => T,
): T[] {
  if (!Array.isArray(raw)) return [];
  const out: T[] = [];
  for (const entry of raw) {
    if (!isRecord(entry) || entry.axis !== axis) continue;
    const base = normalizeBase(entry);
    if (!base) continue;
    out.push(extra(base, entry));
  }
  return out;
}

/**
 * Parse+normalize a raw `/curation/methods` payload. Never throws — any
 * unrecognized shape (missing `strategies` key, a non-array value, garbage
 * entries, an entry whose `axis` isn't one of the four known ones) degrades
 * to empty lists for the affected bucket rather than propagating an
 * exception into the api/store layer. Unknown `id`s are carried through
 * as-is (nothing here validates ids against a fixed enum, by design —
 * that's how a new server-side method shows up without a frontend
 * redeploy); unknown `status` values are neutralized to `'disabled'` by
 * `normalizeMethodStatus`.
 *
 * The real backend sends one flat `strategies: [...]` array with an
 * `axis` field per entry (confirmed live 2026-09-10 — see this file's
 * header comment), not four separate top-level arrays. This function is
 * the sole place that reshapes it; everything downstream still sees the
 * original four-bucket `OpMethodsResponse` shape.
 */
export function parseKbMethodsResponse(raw: unknown): OpMethodsResponse {
  const rec = isRecord(raw) ? raw : {};
  const strategies = Array.isArray(rec.strategies) ? rec.strategies : [];
  return {
    cluster_methods: normalizeAxis<ClusterMethodInfo>(
      strategies,
      'cluster',
      (base, e) => ({
        ...base,
        default: optBool(e.default),
      }),
    ),
    review_sorts: normalizeAxis<ReviewSortInfo>(strategies, 'sort', (base, e) => ({
      ...base,
      default: optBool(e.default),
      requires_field: optString(e.requires_field),
      field_coverage: optNumber(e.field_coverage),
      field_coverage_total: optNumber(e.field_coverage_total),
    })),
    overlays: normalizeAxis<OverlayInfo>(strategies, 'overlay', (base, e) => ({
      ...base,
      requires_field: optString(e.requires_field),
      field_coverage: optNumber(e.field_coverage),
      field_coverage_total: optNumber(e.field_coverage_total),
      requires_banner: optBool(e.requires_banner),
      purity: optNumber(e.purity),
    })),
    scores: normalizeAxis<ScoreInfo>(strategies, 'score', (base, e) => ({
      ...base,
      requires_field: optString(e.requires_field),
      field_coverage: optNumber(e.field_coverage),
      field_coverage_total: optNumber(e.field_coverage_total),
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
 * Whether `/curation/methods` currently reports the 2-d embedding-projection
 * overlay (curation-strategy plan Phase 5 — UMAP-as-visualization-only,
 * `embedding_viz.py` + `op_viz.py`, `GET /curation/viz/projection`) as safe to
 * offer in the UI. Mirrors `isDiverseOverlayAvailable` exactly: same
 * stable/experimental-only bar, same "absent/shadow/disabled never
 * renders" contract — this is the single gate `/clusters` uses to decide
 * whether the "Embedding plot" toggle exists at all (not greyed out —
 * fully absent) and the one `EmbeddingPlot.svelte` itself never has to
 * re-derive.
 *
 * `'viz_projection'` is the real id (confirmed live 2026-09-10 against
 * `strategy_registry.py`'s `_viz_projection_strategy` — an earlier
 * placeholder id, `'umap_viz'`, was a guess made while the sibling
 * openprocessor validation pass was still running and has been corrected
 * here and in every test fixture that used it).
 */
export function isEmbeddingVizAvailable(overlays: OverlayInfo[]): boolean {
  return overlays.some(
    (o) =>
      o.id === 'viz_projection' && (o.status === 'stable' || o.status === 'experimental'),
  );
}

/**
 * Whether the currently-available embedding-viz overlay entry requires
 * the persistent "this projection is approximate" banner (plan §6's
 * middle UMAP-purity tier — `requires_banner` on `OverlayInfo`, confirmed
 * against the real backend). Returns `false` whenever the overlay isn't
 * offered at all (mirrors `isEmbeddingVizAvailable`'s own gate, so a
 * caller never needs to check both before rendering).
 */
export function isEmbeddingVizBannerRequired(overlays: OverlayInfo[]): boolean {
  const entry = overlays.find(
    (o) =>
      o.id === 'viz_projection' && (o.status === 'stable' || o.status === 'experimental'),
  );
  return !!entry?.requires_banner;
}

/**
 * Whether an entry's `field_coverage` should be treated as "has real data,
 * safe to offer" (audit-remediation plan Phase 6, P1-2/P1-3). This is the
 * single place the null-vs-zero distinction lives — every caller (the sort
 * dropdown filter, the mistakenness/near-dup score chips in
 * `StrategyBar.svelte`) must go through this instead of re-deriving it,
 * the same "one gate, checked everywhere" convention
 * `isDiverseOverlayAvailable`/`isEmbeddingVizAvailable` already use.
 *
 * `field_coverage === 0` is the ONE case that means "hide it" — a real
 * exists-count of zero, e.g. `mistakenness` today (no probe checkpoint has
 * ever run, plan §0.1's live counts). Every other value — a positive
 * count, `null` (coverage genuinely doesn't apply, e.g.
 * `requires_field: null`, or a transient backend failure per
 * `strategy_registry.py`'s "fall back to None, not 0" contract), or
 * `undefined` (an older/pre-Phase-6 backend that never sent the field at
 * all, or the synthetic "Default order" sentinel option) — must be
 * treated as "unknown, don't hide a control that might work."
 *
 * Before this fix, `StrategyBar.svelte`'s local `hasCoverage()` used
 * `(entry.field_coverage ?? 0) > 0`, which conflates "coverage is
 * null/unknown" with "coverage is zero" — the exact bug this function
 * fixes. A backend that had never shipped `field_coverage` at all (every
 * real backend, until this phase) made `?? 0` fire on every single entry,
 * hiding every chip and filtering every entry out of the sort dropdown.
 */
export function hasFieldCoverage(entry: { field_coverage?: number | null }): boolean {
  return entry.field_coverage !== 0;
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
      // Explicit null, not omitted (audit-remediation plan Phase 6): this
      // is the 404-fallback path, so field_coverage is "unknown," not
      // "empty" -- hasFieldCoverage() must keep rendering this entry.
      field_coverage: null,
    },
  ],
  overlays: [],
  scores: [],
};
