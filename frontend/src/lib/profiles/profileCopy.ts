/**
 * The words the region-profile surfaces put on the shared config
 * components (structure is shared with the prompt-pack editor).
 */
import type { ActivePanelCopy } from '$components/config/ConfigActivePanel.svelte';

export const PROFILE_ACTIVE_COPY: ActivePanelCopy = {
  title: 'Active region profile',
  noneText: 'off: region detection is off',
  noneEnvText: 'None: no region profile configured (region detection off)',
  appliedColumn: 'Profile',
  appliedRef: 'profile',
  appliedNoneText: 'none',
  rollbackTitle: 'Roll back the active region profile',
  rollbackBlurb:
    'Items still waiting are detected with the profile you roll back to. Items already processed keep their results. Workers switch at their next quiet point.',
  deactivate: {
    label: 'Turn off',
    title: 'Turn off region detection',
    blurb:
      'No new regions are detected until a profile is activated again. Items already processed keep their results.',
  },
};

/** "name" or "name rN". */
export function refLabel(name: string, revision: number | null | undefined): string {
  return revision == null ? name : `${name} r${revision}`;
}
