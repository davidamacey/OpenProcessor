import type { AssistTarget } from '$lib/types_assist';

/** Names what a run's target covers, from the target's own ids. */
export function scopeLabel(target: AssistTarget): string {
  const scope =
    target.kind === 'cluster'
      ? target
      : target.kind === 'crops' && target.context
        ? target.context
        : null;
  if (scope) {
    return scope.cluster_subid
      ? `sub-cluster ${scope.cluster_subid}`
      : `cluster #${scope.cluster_id}`;
  }
  if (target.kind === 'crops') {
    return target.crop_ids.length === 1 ? '1 crop' : `${target.crop_ids.length} crops`;
  }
  if (target.kind === 'boxes') {
    return target.boxes.length === 1 ? '1 box' : `${target.boxes.length} boxes`;
  }
  return 'the current filter';
}
