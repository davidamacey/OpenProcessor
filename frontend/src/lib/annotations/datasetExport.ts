/**
 * Reader for a slot profile's `extras.datasetExport` escape hatch
 * (P2.13/P2.15, docs/genericization-plan-2026-09-13.md §9.6).
 *
 * `SlotSpec.extras` is deliberately typed `Record<string, unknown>` — a
 * profile-private bag, not an extension point (`./types.ts:262`). This
 * module is the single place that narrows one well-known key out of it,
 * the same way `cohortsForClass` is the single place cohort specs are
 * resolved. Every field is validated at the boundary, so a profile (or,
 * once tier-2 JSON profile loading lands, a deployment config file) that
 * declares a malformed `datasetExport` yields `undefined` and the panel
 * simply does not render — never a half-built form pointing at a
 * half-built path.
 *
 * Dataset export stays profile-private on purpose: OpenProcessor
 * classifies `legacy_lpr_export.py` Bucket B and never ported it, so
 * this is one deployment's proprietary overlay, not a generalized
 * capability. Whether the panel renders at all is a separate,
 * server-side question — see `isDatasetExportAvailable` in
 * `$lib/strategies`.
 */

import type { SlotSpec } from './types';

export interface DatasetExportSpec {
  /** Export kind. Doubles as the `{API_PREFIX}/export/{kind}` path
   *  segment and the `axis: 'export'` entry id on `/methods`, which is
   *  what makes a single string enough to gate on. */
  kind: string;
  label: string;
  /** Relative to `API_PREFIX`, never absolute (cohorts.ts §2.5's rule). */
  buildPath: string;
  statusPath: string;
  /** `OpDataset.kind` this export materializes, for the dataset picker. */
  datasetKind: string;
  /** Trains with `single_cls: true` and `include_classes: null`. */
  singleClass: boolean;
  blurb: string;
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null;
}

function nonEmptyString(v: unknown): string | undefined {
  return typeof v === 'string' && v.length > 0 ? v : undefined;
}

/**
 * The slot's declared dataset export, or `undefined` when it declares
 * none (every slot but `license_plate` today) or declares a malformed
 * one.
 *
 * `options[]` is deliberately NOT read here — `/train` still renders its
 * four LPR options as typed, bound controls. Turning that array into a
 * generic form builder is follow-up work, not part of the capability
 * gate (see the Phase C plan §3.7).
 */
export function datasetExportForSlot(slot: SlotSpec): DatasetExportSpec | undefined {
  const raw = slot.extras?.datasetExport;
  if (!isRecord(raw)) return undefined;

  const kind = nonEmptyString(raw.kind);
  const label = nonEmptyString(raw.label);
  const buildPath = nonEmptyString(raw.buildPath);
  const statusPath = nonEmptyString(raw.statusPath);
  const datasetKind = nonEmptyString(raw.datasetKind);
  const blurb = nonEmptyString(raw.blurb);
  if (!kind || !label || !buildPath || !statusPath || !datasetKind || !blurb) {
    return undefined;
  }
  // Prefix-relative only. An absolute URL here would bypass apiBase and
  // API_PREFIX both, which is the bug class plateThumbUrl.test.ts exists
  // to prevent on the image side.
  if (!buildPath.startsWith('/') || !statusPath.startsWith('/')) return undefined;

  return {
    kind,
    label,
    buildPath,
    statusPath,
    datasetKind,
    singleClass: raw.singleClass === true,
    blurb,
  };
}
