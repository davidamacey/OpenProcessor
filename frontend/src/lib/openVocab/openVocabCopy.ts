/**
 * The words the open-vocabulary surfaces put on the shared config
 * components (structure is shared with the other config editors).
 */
import type { ActivePanelCopy } from '$components/config/ConfigActivePanel.svelte';

export const OPEN_VOCAB_ACTIVE_COPY: ActivePanelCopy = {
  title: 'Active open-vocabulary set',
  noneText: 'off: no open-vocabulary set is active',
  noneEnvText: 'None: no open-vocabulary set is active',
  appliedColumn: 'Set',
  // The served `applied[]` carries pack, profile and vlm refs only.
  appliedRef: 'profile',
  appliedNoneText: 'none',
  rollbackTitle: 'Roll back the active open-vocabulary set',
  rollbackBlurb:
    'Passes that run from now on use the set you roll back to. Items already found keep their results.',
  deactivate: {
    label: 'Turn off',
    title: 'Turn off open-vocabulary passes',
    blurb:
      'Turn off: no new open-vocabulary passes run; stored items stay. Activate a set again to resume.',
  },
};
