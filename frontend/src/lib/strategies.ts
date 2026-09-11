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
  /** Fraction [0,1] of the pool that has this field populated. */
  field_coverage?: number | null;
}

export interface OverlayInfo extends MethodInfoBase {
  requires_field?: string | null;
  field_coverage?: number | null;
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
    cluster_methods: normalizeAxis<ClusterMethodInfo>(strategies, 'cluster', (base, e) => ({
      ...base,
      default: optBool(e.default),
    })),
    review_sorts: normalizeAxis<ReviewSortInfo>(strategies, 'sort', (base, e) => ({
      ...base,
      default: optBool(e.default),
      requires_field: optString(e.requires_field),
      field_coverage: optNumber(e.field_coverage),
    })),
    overlays: normalizeAxis<OverlayInfo>(strategies, 'overlay', (base, e) => ({
      ...base,
      requires_field: optString(e.requires_field),
      field_coverage: optNumber(e.field_coverage),
      requires_banner: optBool(e.requires_banner),
      purity: optNumber(e.purity),
    })),
    scores: normalizeAxis<ScoreInfo>(strategies, 'score', (base, e) => ({
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
    (o) => o.id === 'viz_projection' && (o.status === 'stable' || o.status === 'experimental'),
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
    (o) => o.id === 'viz_projection' && (o.status === 'stable' || o.status === 'experimental'),
  );
  return !!entry?.requires_banner;
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
