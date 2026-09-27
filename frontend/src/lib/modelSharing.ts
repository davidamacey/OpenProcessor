/**
 * Display helpers for cross-project model sharing on `/models`
 * (projects P2, §5.5). Everything here reads served fields; nothing
 * derives a mapping, a count or an owner.
 */
import type { ModelInfo } from '$lib/types';
import type { ModelClassMappingSummary } from '$lib/types_models';

/**
 * `owner`: the served `project` is the active project, so the owner-only
 * sharing toggle may render. `foreign`: another project's model (listed
 * only because its owner shared it). `none`: no owning project served
 * (a base model or an external service).
 */
export type SharingRole = 'owner' | 'foreign' | 'none';

export function sharingRole(m: ModelInfo, activeSlug: string | null): SharingRole {
  if (m.project == null) return 'none';
  return m.project === activeSlug ? 'owner' : 'foreign';
}

/** The toggle needs the served sharing revision to send back as
 *  `expected_revision`; without it (backend ask BA-P2-1) it is absent. */
export function canToggleSharing(m: ModelInfo, activeSlug: string | null): boolean {
  return sharingRole(m, activeSlug) === 'owner' && typeof m.sharing_revision === 'number';
}

export function mappingText(summary: ModelClassMappingSummary): string {
  const n = summary.mapped_count;
  return n === 1 ? '1 class maps' : `${n} classes map`;
}

export function unmappedText(summary: ModelClassMappingSummary): string {
  return `${summary.unmapped.length} not in this project`;
}

export function shareConfirmText(name: string): string {
  return `Share ${name} with other projects? They will be able to see and use it; their classes are matched to its classes by name.`;
}

/** Unsharing is never described as safe: which other projects use a
 *  model isn't served yet (`used_by` stays empty until the backend's
 *  profile wave), so the copy says one may. */
export function unshareConfirmText(name: string): string {
  return `Stop sharing ${name}? Another project may be using it, and the server can't tell yet whether one is; that project would lose access to this model.`;
}
