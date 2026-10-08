/**
 * Display helpers for cross-project model sharing on `/models`
 * (projects P2, §5.5). Everything here reads served fields; nothing
 * derives a mapping, a count or an owner.
 */
import type { ModelInfo } from '$lib/types';
import type { ModelClassMappingSummary, ModelSharingUser } from '$lib/types_models';

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

/** The server checks whether another project's active detection profile
 *  uses the model and refuses (409 `in_use`, naming the projects) unless
 *  forced, so the copy promises nothing about safety. */
export function unshareConfirmText(name: string): string {
  return `Stop sharing ${name}? The server refuses if another project's active detection profile uses it, and tells you which; you can then choose to unshare anyway.`;
}

/** The served `used_by` rows as `project (profile)`, comma-joined. */
export function usedByText(users: ModelSharingUser[]): string {
  return users
    .map((u) => (u.profile ? `${u.project} (${u.profile})` : u.project))
    .join(', ');
}

/** Shown once the operator has armed the forced retry. */
export function forceUnshareText(projects: string[]): string {
  return projects.length
    ? `${projects.join(', ')} will lose access to this model. The override is logged server-side.`
    : 'Projects using this model will lose access to it. The override is logged server-side.';
}

/** Warning before a forced unshare when the server could not read every project. */
export function forceUnshareUnreadableText(): string {
  return 'The server could not check every project, so a project that uses this model may lose access to it. The override is logged server-side.';
}
