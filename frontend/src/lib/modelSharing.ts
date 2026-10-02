/**
 * Display helpers for cross-project model sharing on `/models`
 * (projects P2, §5.5). Everything here reads served fields; nothing
 * derives a mapping, a count or an owner.
 */
import type { ModelInfo } from '$lib/types';
import type { ModelClassMappingSummary } from '$lib/types_models';

/**
 * `owner`: the server says this project owns the model (`owned`), so the
 * owner-only sharing toggle may render. `foreign`: another project's model
 * (listed only because its owner shared it). `none`: no owning project
 * (a base model or an external service).
 */
export type SharingRole = 'owner' | 'foreign' | 'none';

export function sharingRole(m: Pick<ModelInfo, 'owned' | 'project'>): SharingRole {
  if (m.owned) return 'owner';
  return m.project == null ? 'none' : 'foreign';
}

/** The toggle needs the served sharing revision to send back as
 *  `expected_revision`, which the server serves only for an owned model. */
export function canToggleSharing(
  m: Pick<ModelInfo, 'owned' | 'project' | 'sharing_revision'>,
): boolean {
  return sharingRole(m) === 'owner' && typeof m.sharing_revision === 'number';
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
