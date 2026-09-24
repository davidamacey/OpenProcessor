/**
 * Types + parsing + the axis policy table for the deployment-wide
 * curation defaults record (`GET,PUT {API_PREFIX}/settings`).
 *
 * VERIFIED AGAINST THE REAL BACKEND (2026-09-21, read from
 * wt-oss-hardening `src/routers/curation/settings.py` +
 * `_common.py:595-618`): the wire shape is
 *   {"defaults": {"<axis>": "<id>"}, "updated_at": str|null,
 *    "updated_by": str|null}
 * `defaults` is an OPEN map keyed by axis id — `dict[str, str]` on the
 * model, `{'type': 'object', 'enabled': False}` in the OpenSearch
 * mapping. It is deliberately NOT a fixed four-field schema, so a
 * future axis never needs a wire-format change. This module must
 * preserve that: an axis id this build has never heard of is carried
 * through untouched, never dropped.
 *
 * Forward-tolerant, matching `strategies.ts`'s convention: a server
 * ahead of this build must degrade gracefully, never throw. Malformed
 * entries are dropped individually; a malformed document yields an
 * empty record.
 */

import type { OpMethodsResponse, MethodInfoBase } from '$lib/strategies';
import { selectableAxisEntries } from '$lib/strategies';

/** The four axes the backend's `SETTABLE_DEFAULT_AXES` accepts on PUT. */
export type SettingsAxis = 'cluster' | 'sort' | 'detection_profile' | 'prompt_pack';

export interface CurationSettings {
  /**
   * Axis id -> strategy id. OPEN map: `Record<string, string>`, not
   * `Record<SettingsAxis, string>`. An axis key this build doesn't know
   * is kept verbatim so a partial PUT never clobbers it (see
   * `SETTINGS_AXES`'s note on why partial PUT is what protects it).
   */
  defaults: Record<string, string>;
  /** ISO-8601, or null when no document has ever been written. */
  updated_at: string | null;
  /**
   * ALWAYS null today — the backend hardcodes `'updated_by': None` in
   * every write body (`curation_opensearch.py:486`); there is no
   * user-account system. Carried on the wire for when there is one.
   */
  updated_by: string | null;
}

/** The "no document has ever been written" record. A 200 with this body
 *  is the normal first-run response, NOT an error. */
export const EMPTY_CURATION_SETTINGS: CurationSettings = {
  defaults: {},
  updated_at: null,
  updated_by: null,
};

/**
 * How this UI is allowed to treat one axis.
 *
 * `'settable'`  — the backend both accepts a default AND honors it on a
 *                 real per-request code path. Gets a working control.
 * `'advisory'`  — the backend accepts and stores a default, and
 *                 `/methods` reflects it, but NO per-request call site
 *                 consults it, so setting one changes nothing an
 *                 operator would observe. Gets a read-only display, never
 *                 a control.
 *
 * THIS TABLE IS THE HONESTY BOUNDARY OF THE WHOLE FEATURE. Verified
 * 2026-09-21 by grepping every `resolve_effective_default` call site in
 * wt-oss-hardening (see docs/design/curation-settings-ui-plan-2026-09-21.md
 * §1.3.1) and cross-checked against that repo's own
 * `docs/design/curation_api_contract.md`, which documents the same two
 * gaps as "a known, deliberate gap rather than an oversight."
 *
 * When the backend grows a per-request `detection_profile` /
 * `prompt_pack` selection call site, flip that axis's `kind` to
 * `'settable'` here and NOWHERE ELSE — the page, the store guard, and
 * every test read this table.
 */
export interface SettingsAxisSpec {
  axis: SettingsAxis;
  kind: 'settable' | 'advisory';
  /** Operator-facing name. The backend sends raw ids as labels for the
   *  two advisory axes (`'label': profile.name`), so the section heading
   *  has to supply the human words. */
  label: string;
  /** Which `OpMethodsResponse` bucket holds this axis's entries. */
  bucket: keyof OpMethodsResponse;
  /** One sentence shown under the control. For a settable axis this must
   *  state the real blast radius; for an advisory axis it must state
   *  that nothing changes. */
  blurb: string;
  /**
   * Extra confirm-dialog copy for an axis whose pin cannot be undone.
   * `null` for every axis today — the backend's `PUT {defaults:
   * {[axis]: null}}` clear path (see `putCurationDefaults`'s docstring)
   * closed the one gap that used to justify this field (H-1, plan §1.4:
   * `sort` had no way to un-pin once set). Kept as a field, not deleted,
   * in case a future axis reintroduces a genuinely irreversible pin.
   */
  irreversibleWarning: string | null;
}

export const SETTINGS_AXES: readonly SettingsAxisSpec[] = [
  {
    axis: 'cluster',
    kind: 'settable',
    label: 'Clustering method',
    bucket: 'cluster_methods',
    blurb:
      'Used by every auto-label run this app starts — Cropwright never sends an ' +
      'explicit clustering_method, so residual clustering resolves this shared default.',
    // Pinning the backend's own built-in ('ivf') is behaviorally identical
    // to unset, so this one is recoverable in effect even without a
    // backend delete path.
    irreversibleWarning: null,
  },
  {
    axis: 'sort',
    kind: 'settable',
    label: 'Review queue sort',
    bucket: 'review_sorts',
    blurb:
      'Applied to every /review tab that does not request its own sort — which is ' +
      'every tab on a fresh load. This REPLACES each tab’s own tuned default ' +
      '(Uncertainty, Model Disagreements, COCO Blind Spots, Plates).',
    // Was irreversible (H-1) until the backend added a clear path; the
    // page's "Clear" button now covers this axis like any other.
    irreversibleWarning: null,
  },
  {
    axis: 'detection_profile',
    kind: 'advisory',
    label: 'Detection profile',
    bucket: 'detection_profiles',
    blurb:
      'Display only. Exactly one profile is active per backend process, selected at ' +
      'startup from config — no endpoint reads a per-request or shared selection, ' +
      'so setting a default here would change what /methods reports and nothing else.',
    irreversibleWarning: null,
  },
  {
    axis: 'prompt_pack',
    kind: 'settable',
    label: 'VLM prompt pack',
    bucket: 'prompt_packs',
    // Verified 2026-09-23 on OpenProcessor main 80dd097: auto_label's
    // resolve_run_selection falls back to this default; the always-on VLM
    // worker does not read it (it uses OP_PROMPT_PACK_PATH).
    blurb:
      'Used by the VLM labeling sweep of every auto-label run that does not pick its ' +
      'own pack on the dashboard. The always-on background VLM worker keeps using its ' +
      'startup config.',
    irreversibleWarning: null,
  },
];

export function settableAxes(): SettingsAxisSpec[] {
  return SETTINGS_AXES.filter((a) => a.kind === 'settable');
}

export function advisoryAxes(): SettingsAxisSpec[] {
  return SETTINGS_AXES.filter((a) => a.kind === 'advisory');
}

export function axisSpec(axis: string): SettingsAxisSpec | null {
  return SETTINGS_AXES.find((a) => a.axis === axis) ?? null;
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null;
}

/**
 * Parse a raw `{API_PREFIX}/settings` body. Never throws.
 *
 * Only string->string pairs survive into `defaults` (the backend model is
 * `dict[str, str]`), but the KEY is not validated against `SettingsAxis`
 * — an unknown axis from a newer server is preserved verbatim. That is
 * the whole point of the open map, and dropping it here would make the
 * page silently under-report what the deployment has configured.
 */
export function parseCurationSettings(raw: unknown): CurationSettings {
  if (!isRecord(raw)) return { ...EMPTY_CURATION_SETTINGS, defaults: {} };
  const defaults: Record<string, string> = {};
  if (isRecord(raw.defaults)) {
    for (const [k, v] of Object.entries(raw.defaults)) {
      if (typeof k === 'string' && k && typeof v === 'string' && v) defaults[k] = v;
    }
  }
  return {
    defaults,
    updated_at: typeof raw.updated_at === 'string' ? raw.updated_at : null,
    updated_by: typeof raw.updated_by === 'string' ? raw.updated_by : null,
  };
}

/**
 * The entries a settings control may offer for one axis.
 *
 * Goes through the shared `selectableAxisEntries` rather than
 * re-deriving a status filter — the same rule `AssistScopeBar.test.ts`
 * enforces on that component ("contains no inline status filter"),
 * because a second divergent copy of a gate is the `hasFieldCoverage`
 * bug class (`strategies.ts`).
 *
 * DELIBERATELY UNLIKE `StrategyBar.svelte`: no synthetic
 * `'Default order'` / sentinel option is appended. `'default'` is NOT a
 * real backend sort-registry id (it exists only in this app's
 * FALLBACK_METHODS and as StrategyBar's local "no override" sentinel),
 * so PUTting it would 422 on `_validate_defaults`'s
 * "is not a currently-advertised id" branch. Do not add one back.
 */
export function axisOptions(
  methods: OpMethodsResponse,
  spec: SettingsAxisSpec,
): MethodInfoBase[] {
  return selectableAxisEntries(methods[spec.bucket] as MethodInfoBase[]);
}

/**
 * The id currently in effect for an axis: the stored shared default when
 * there is one, else whichever `/methods` entry carries `default: true`
 * (the backend derives that flag from this same record, so the two agree
 * by construction), else null.
 */
export function effectiveDefaultId(
  settings: CurationSettings,
  methods: OpMethodsResponse,
  spec: SettingsAxisSpec,
): string | null {
  const stored = settings.defaults[spec.axis];
  if (stored) return stored;
  const flagged = (methods[spec.bucket] as MethodInfoBase[]).find(
    (e) => (e as { default?: boolean }).default === true,
  );
  return flagged?.id ?? null;
}

/** True when the axis has a value stored in the shared record (as
 *  opposed to merely inheriting the backend's built-in default). */
export function isPinned(settings: CurationSettings, spec: SettingsAxisSpec): boolean {
  return typeof settings.defaults[spec.axis] === 'string';
}
