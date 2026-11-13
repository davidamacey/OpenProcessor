/**
 * Where a cluster card leads. A cluster whose dominant class is bound to a registered slot
 * opens that slot's gallery (its regions are sub-boxes on other items, so the cluster itself
 * holds only the rare crops labelled with the class as their primary one); every other cluster
 * opens in the Workspace. The slot is resolved from the served `dominant_class_name`: class
 * identity is the name.
 */
import { slotForClassName } from '$lib/annotations/registeredSlots';
import { projectHref } from '$lib/projectPaths';
import { clusterScopeSearch } from '$lib/workspace/workspaceUrl';

export function clusterHref(
  uid: string,
  dominantClassName: string | null | undefined,
): string {
  if (dominantClassName && slotForClassName(dominantClassName) != null) {
    return projectHref(
      `/workspace?view=clusters&class_name=${encodeURIComponent(dominantClassName)}`,
    );
  }
  return projectHref(`/workspace${clusterScopeSearch(uid)}`);
}
