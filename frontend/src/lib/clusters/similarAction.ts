/**
 * The "More like this" action a crop grid hands to its cards: open the /clusters results for
 * the crop. `undefined` where the deployment does not serve semantic search (the control is
 * absent, not disabled).
 */
import { goto } from '$app/navigation';
import { resolve } from '$app/paths';
import { similarHref } from '$lib/projectPaths';
import { isSemanticSearchAvailable } from '$lib/strategies';
import type { Crop } from '$lib/types';
import { strategiesStore } from '$stores/strategies.svelte';

export function similarAction(): ((crop: Crop) => void) | undefined {
  if (!isSemanticSearchAvailable(strategiesStore.methods.overlays)) return undefined;
  return (crop) => void goto(resolve(similarHref(crop.id)));
}
