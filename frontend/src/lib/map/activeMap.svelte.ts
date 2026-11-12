/** The Map view's controller while that view is mounted, so the details panel (a sibling in the
 *  Workspace shell) can show the focused cell's served summary. */
import type { MapController } from './mapController.svelte';

export const activeMap = $state<{ ctl: MapController | null }>({ ctl: null });
