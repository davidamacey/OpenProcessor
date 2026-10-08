/**
 * The words the VLM surfaces put on the shared config components
 * (structure is shared with the prompt-pack and region-profile editors).
 */
import type { ActivePanelCopy } from '$components/config/ConfigActivePanel.svelte';

export const VLM_ACTIVE_COPY: ActivePanelCopy = {
  title: 'VLM for this project',
  noneText: 'off: no VLM runs for this project',
  appliedColumn: 'VLM',
  appliedRef: 'vlm',
  appliedNoneText: 'no VLM',
  rollbackTitle: 'Roll back the VLM endpoint',
  rollbackBlurb:
    'Runs that start after this use the endpoint you roll back to. Labels the VLM already wrote keep their provenance. Workers switch at their next quiet point.',
  deactivate: {
    label: 'Turn off',
    title: 'Turn off the VLM for this project',
    blurb:
      'No VLM labeling runs for this project until an endpoint is activated again. Labels already written are kept.',
  },
};
