/**
 * TypeScript types mirroring the openprocessor `GET {API_PREFIX}/methods` capability-
 * discovery response (curation-strategy plan, Phase 0 —
 * docs/curation-strategy-plan-2026-09.md §3, §5.3, §7).
 *
 * CONFIRMED AGAINST THE REAL BACKEND (2026-09-10, live end-to-end check
 * after rebuilding/restarting the backend from its curation-scores branch):
 * the actual wire shape is a single flat `{strategies: [...], flags: {...}}`
 * — every entry carries an `axis: 'cluster' | 'sort' | 'score' | 'overlay'`
 * field (see `strategy_registry.py`'s `StrategyAxis`), NOT four separate
 * top-level arrays as this file originally assumed from the plan doc's
 * illustrative example. `parseMethodsResponse` below reshapes the flat
 * list into the seven buckets (`cluster_methods`/`review_sorts`/`overlays`/
 * `scores`/`dataset_exports`/`detection_profiles`/`prompt_packs`) client-side
 * by grouping on `axis`, so every downstream consumer
 * (`isDiverseOverlayAvailable`, `StrategyBar`, etc.) keeps working against
 * the original four-array `MethodsResponse` shape unchanged — only this
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

export interface MethodInfoBase {
  id: string;
  label: string;
  status: MethodStatus;
  description?: string | null;
  /** Whether `PUT {API_PREFIX}/settings` accepts this entry as the shared
   *  default for its axis — the server's own record of which axes a
   *  default actually changes. Absent is treated as not settable. */
  settable?: boolean;
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
   *  for every entry in a given `{API_PREFIX}/methods` response). `null` whenever
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
   * `viz.py` / `strategy_registry.py` after the Phase 5 UMAP
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

/**
 * One dataset-export *kind* that `POST {API_PREFIX}/export/{id}` can
 * actually produce on this deployment (backend
 * `strategy_registry.py`'s `_export_strategies()`, `axis: 'export'`,
 * added by T-C2 of `cropwright_backend_integration_plan.md` §4.3).
 *
 * CONFIRMED against the backend @ `d8cb9dc`: today the axis holds
 * exactly one entry, `{id: 'yolo', axis: 'export', label: 'YOLO
 * detection dataset export', status: 'stable', default: true}`.
 *
 * **Absence is the signal, not a status.** OpenProcessor omits an export
 * kind it cannot produce entirely rather than advertising it `disabled`, because a status
 * implies "not yet, but this deployment could serve it later" — untrue
 * for a proprietary overlay the repo does not contain
 * (`curation_api_contract.md`, the `export` axis section). So a consumer
 * must treat "no entry" and "entry at shadow/disabled" identically:
 * hide the UI.
 */
export interface DatasetExportInfo extends MethodInfoBase {
  /** True on the kind the backend treats as its primary export. */
  default?: boolean;
}

/**
 * One vision-detection profile (model + config) an assisted auto-label
 * run can be pointed at (`axis: 'detection_profile'`, agreed with the
 * OpenProcessor session 2026-09-20,
 * docs/design/vlm-scoped-labeling-assist-plan-2026-09-20.md §1.3).
 *
 * **Now live on OpenProcessor (corrected 2026-09-21 — see
 * docs/design/curation-settings-ui-plan-2026-09-21.md §1.3.2).** An
 * earlier version of this comment claimed the axis was "not yet live on
 * any backend" and absent from every `/methods` response; that was true
 * when it was written but is now stale. `strategy_registry.py` registers
 * `_detection_profile_strategies` unconditionally and yields exactly one
 * entry — one profile selected per backend process at startup — so
 * `isDetectionProfileAvailable`/`isScopedAssistAvailable` can return
 * `true` against a real backend today. `FALLBACK_METHODS.detection_profiles`
 * stays `[]` regardless: that list models the 404/network-failure path,
 * which this correction does not change.
 *
 * Deliberately carries no `requires_field`/`field_coverage`: a
 * detection profile is a model/config selection, not a backfilled
 * OpenSearch field, so `hasFieldCoverage` does not apply to it and
 * must not be wired in.
 */
export interface DetectionProfileInfo extends MethodInfoBase {
  /** True on the profile the backend runs when none is requested. */
  default?: boolean;
}

/**
 * One VLM prompt / vocabulary set an assisted auto-label run can use
 * (`axis: 'prompt_pack'`). Same contract, same absence rule, same
 * no-field-coverage note as `DetectionProfileInfo` above.
 */
export interface PromptPackInfo extends MethodInfoBase {
  /** True on the pack the backend uses when none is requested. */
  default?: boolean;
}

export interface MethodsResponse {
  cluster_methods: ClusterMethodInfo[];
  review_sorts: ReviewSortInfo[];
  overlays: OverlayInfo[];
  scores: ScoreInfo[];
  /**
   * `axis: 'export'` entries. Named for what they are rather than by
   * mechanically pluralizing the axis, the same way `cluster` →
   * `cluster_methods` and `sort` → `review_sorts` already are.
   */
  dataset_exports: DatasetExportInfo[];
  /**
   * `axis: 'detection_profile'` entries. Named as a descriptive plural
   * noun phrase like every other bucket (`cluster` → `cluster_methods`,
   * `sort` → `review_sorts`, `export` → `dataset_exports`), not a
   * mechanical pluralization of the axis string.
   */
  detection_profiles: DetectionProfileInfo[];
  /** `axis: 'prompt_pack'` entries. */
  prompt_packs: PromptPackInfo[];
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
 * not take the whole `{API_PREFIX}/methods` parse down with it.
 */
function normalizeBase(raw: unknown): MethodInfoBase | null {
  if (!isRecord(raw)) return null;
  const { id, label } = raw;
  if (typeof id !== 'string' || !id) return null;
  if (typeof label !== 'string' || !label) return null;
  const base: MethodInfoBase = { id, label, status: normalizeMethodStatus(raw.status) };
  const description = optString(raw.description);
  if (description !== undefined) base.description = description;
  if (typeof raw.settable === 'boolean') base.settable = raw.settable;
  return base;
}

/**
 * Group the real wire shape's flat `strategies` array by its `axis` field,
 * normalizing each entry through `extra`. An entry whose `axis` isn't one
 * of the axes this build knows how to route (a future server's new axis)
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
 * Parse+normalize a raw `{API_PREFIX}/methods` payload. Never throws — any
 * unrecognized shape (missing `strategies` key, a non-array value, garbage
 * entries, an entry whose `axis` isn't one of the axes this build knows) degrades
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
 * original bucketed `MethodsResponse` shape.
 */
export function parseMethodsResponse(raw: unknown): MethodsResponse {
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
    dataset_exports: normalizeAxis<DatasetExportInfo>(
      strategies,
      'export',
      (base, e) => ({
        ...base,
        default: optBool(e.default),
      }),
    ),
    detection_profiles: normalizeAxis<DetectionProfileInfo>(
      strategies,
      'detection_profile',
      (base, e) => ({
        ...base,
        default: optBool(e.default),
      }),
    ),
    prompt_packs: normalizeAxis<PromptPackInfo>(strategies, 'prompt_pack', (base, e) => ({
      ...base,
      default: optBool(e.default),
    })),
  };
}

/**
 * Whether `{API_PREFIX}/methods` currently reports the pool-scale `diverse` overlay
 * (curation-strategy plan Phase 4 — core-set / k-center-greedy selection,
 * `POST {API_PREFIX}/select/diverse` + `GET {API_PREFIX}/crops?order=diverse`) as safe to
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
 * Whether `{API_PREFIX}/methods` currently reports the 2-d embedding-projection
 * overlay (curation-strategy plan Phase 5 — UMAP-as-visualization-only,
 * `embedding_viz.py` + `viz.py`, `GET {API_PREFIX}/viz/projection`) as safe to
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
 * Whether `{API_PREFIX}/methods` currently reports the free-text semantic-search
 * overlay (P2-14 — `GET {API_PREFIX}/search/text`) as safe to offer in the UI.
 * Mirrors `isDiverseOverlayAvailable`/`isEmbeddingVizAvailable` exactly:
 * same stable/experimental-only bar, same "absent/shadow/disabled never
 * renders" contract. `semantic_search` lives in the `overlays` registry
 * (same axis as `diverse`/`viz_projection`), not `review_sorts` — it's an
 * alternate crop pool, not an ordering over the existing one.
 *
 * `FALLBACK_METHODS.overlays` deliberately has no `semantic_search` entry
 * (it doesn't exist in the pre-P2-14 backend this fallback models), so an
 * old/flag-off backend hides the search box entirely rather than showing
 * a control that 404s — same graceful-degradation contract as every other
 * overlay gate in this file.
 */
export function isSemanticSearchAvailable(overlays: OverlayInfo[]): boolean {
  return overlays.some(
    (o) =>
      o.id === 'semantic_search' &&
      (o.status === 'stable' || o.status === 'experimental'),
  );
}

/**
 * Whether `{API_PREFIX}/methods` currently advertises the dataset-export
 * `kind` (`'yolo'`, `'single_class'`, …) as something this deployment can
 * actually produce. Mirrors `isDiverseOverlayAvailable` /
 * `isEmbeddingVizAvailable` / `isSemanticSearchAvailable` exactly: same
 * `stable`/`experimental`-only bar, same "absent / shadow / disabled
 * never renders" contract.
 *
 * This is the single gate `/train` uses to decide whether the optional
 * single-class export panel exists at all — absent, not disabled, and
 * with no status request fired behind it. It must never be replaced by
 * probing `POST {API_PREFIX}/export/{kind}` for a 404: that endpoint is
 * a *write* that kicks off a real dataset build, and probing
 * `/export/{kind}/status` instead conflates "export unsupported" with
 * "no export has run yet" (`cropwright_backend_integration_plan.md`
 * §4.3; `curation_api_contract.md`'s `export` axis section).
 */
export function isDatasetExportAvailable(
  datasetExports: DatasetExportInfo[],
  kind: string,
): boolean {
  return datasetExports.some(
    (e) => e.id === kind && (e.status === 'stable' || e.status === 'experimental'),
  );
}

/**
 * The entries of an axis a UI may actually offer: `stable` or
 * `experimental` only. `shadow` (mid-validation) and `disabled`
 * (explicitly killed) must never be selectable — the same bar every
 * `isXAvailable` gate in this file already applies, hoisted into one
 * place because the two assist axes are rendered as *lists* of options,
 * not looked up by a single known id.
 *
 * Without this, `AssistScopeBar.svelte` would re-derive the status
 * predicate inline to build its `<option>` list — which is exactly the
 * failure `hasFieldCoverage` exists to prevent (a second, divergent copy
 * of a gate; see that function's doc comment and Phase 6's P1-2/P1-3).
 * One filter, unit-tested once, used by both the gate and the renderer.
 */
export function selectableAxisEntries<T extends MethodInfoBase>(entries: T[]): T[] {
  return entries.filter((e) => e.status === 'stable' || e.status === 'experimental');
}

/** Whether any prompt pack is selectable (`stable` or `experimental`). */
export function isPromptPackAvailable(packs: PromptPackInfo[]): boolean {
  return selectableAxisEntries(packs).length > 0;
}

/**
 * Whether this deployment supports **scoped** VLM-assisted auto-labeling
 * at all — the single gate `AutoLabelPanel` uses to decide whether the
 * scope bar exists (absent, not disabled).
 *
 * Gated on the `prompt_pack` axis: it arrived in the same backend
 * change that made `class_id` on `POST {API_PREFIX}/pipeline/auto_label/start`
 * real, and `class_id` has no capability signal of its own. An unknown
 * query param used to be silently ignored, so an un-gated class picker
 * against an older backend would start a full-pool, hours-long run while
 * the UI claimed it was scoped. Hiding the control until the server
 * advertises the axis is the safe default. (`detection_profile` is not a
 * gate: OpenProcessor rejects it per run — region detection is startup
 * config — so it is display-only, on /settings.)
 */
export function isScopedAssistAvailable(
  methods: Pick<MethodsResponse, 'prompt_packs'>,
): boolean {
  return isPromptPackAvailable(methods.prompt_packs);
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
 * Hardcoded fallback for when `{API_PREFIX}/methods` 404s, or the request fails
 * for any other reason (plan §5.3 — graceful degradation is required so
 * the two repos can deploy independently). This must mirror what's
 * actually implemented **today**, not the target end-state:
 *
 * - `cluster_methods`: only IVF is real as a selectable default
 *   (`cluster_methods/ivf.py`, `DEFAULT_METHOD = 'ivf'` in
 *   `cluster_methods/__init__.py`). AHC/HDBSCAN exist in the backend
 *   registry but are not operator-facing defaults, so they're
 *   deliberately left out rather than guessed at.
 * - `review_sorts`: `review.py` hardcodes `sort = [{updated_at:
 *   desc}]` per tab today; there is no named alternative sort yet
 *   (`review_sorts.py` is Phase 3). One `'default'` entry, marked
 *   default+stable — this is NOT `'representativeness'` /
 *   `'uncertainty'` / any of the Phase-3 sort ids from the plan, since
 *   those don't exist in the backend yet.
 * - `overlays` / `scores`: none of `crop_scores/`, `selection/`,
 *   `embedding_viz.py` exist yet (Phase 1/4/5) — both lists are empty.
 */
export const FALLBACK_METHODS: MethodsResponse = {
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
  // Empty, NOT `[{id: 'yolo', …}]`. This is the 404/network-failure path,
  // and the entire point of the export axis is that an optional export
  // panel stays hidden unless the server affirmatively says it works.
  // Guessing a kind here would re-introduce exactly the "render a button
  // that 404s" failure the gate exists to prevent, on the one code path
  // where we have no information at all. Same reasoning as
  // `overlays`/`scores` being empty above.
  dataset_exports: [],
  // Both empty, and for the same reason `dataset_exports` is: this is the
  // 404/network-failure path, neither axis exists on any backend that
  // ships today, and the whole point of an optional scoping control is
  // that it stays invisible unless the server affirmatively says it
  // works. Guessing a profile/pack id here would make the dashboard
  // offer a scope the pipeline silently ignores — the exact failure
  // isScopedAssistAvailable exists to prevent.
  detection_profiles: [],
  prompt_packs: [],
};
