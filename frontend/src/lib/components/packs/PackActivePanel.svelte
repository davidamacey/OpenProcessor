<script lang="ts" module>
  import type { ActivePanelCopy } from '$components/config/ConfigActivePanel.svelte';

  export const PACK_ACTIVE_COPY: ActivePanelCopy = {
    title: 'Active pack',
    noneText: 'none: the deployment default applies',
    appliedColumn: 'Pack',
    appliedRef: 'pack',
    rollbackTitle: 'Roll back the active pack',
    rollbackBlurb:
      "Every VLM step that doesn't pick its own pack uses the active pack. Workers switch at their next quiet point.",
  };
</script>

<script lang="ts">
  /**
   * The project's active prompt pack (§7.2, §7.6 item 4): the shared
   * `ConfigActivePanel` with the pack's words. Packs have no deactivate.
   */
  import ConfigActivePanel from '$components/config/ConfigActivePanel.svelte';
  import type { PackActive } from '$lib/packs/packActive.svelte';

  interface Props {
    ctl: PackActive;
    /** Runs the rollback (the list re-reads after it; the editor doesn't). */
    onrollback: () => Promise<boolean>;
  }

  let { ctl, onrollback }: Props = $props();
</script>

<ConfigActivePanel {ctl} copy={PACK_ACTIVE_COPY} {onrollback} />
