import type { ItemFilterState } from '$lib/itemFilter/itemFilterState.svelte';
import type { ItemFilterQuery } from '$lib/types_itemFilter';
import type { WorkspaceUrlState } from './workspaceUrl';

/** The applied filter, the queue as `review_tab` and the scope, as query parameters. The one
 *  place the three combine: the Workspace views and the image editor's list both call it. */
export function buildItemQuery(
  applied: ItemFilterState,
  url: WorkspaceUrlState,
  include?: (param: string) => boolean,
): ItemFilterQuery & { review_tab?: string } {
  const q: ItemFilterQuery & { review_tab?: string } = applied.toQuery(include);
  const { queue, scope } = url;
  if (queue) q.review_tab = queue;
  if (scope?.kind === 'cluster') q.cluster_uid = [scope.id];
  if (scope?.kind === 'near_dup') q.near_dup_group = scope.id;
  if (scope?.kind === 'image') q.image_id = [scope.id];
  return q;
}
