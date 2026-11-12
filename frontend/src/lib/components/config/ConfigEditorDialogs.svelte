<script lang="ts">
  /**
   * The activate, restore and clone-to-edit dialogs of a config editor page, driven by its
   * `createConfigEditorFlow`.
   */
  import ConfigActivateDialog from '$components/config/ConfigActivateDialog.svelte';
  import ConfigCloneDialog from '$components/config/ConfigCloneDialog.svelte';
  import ConfigRestoreDialog from '$components/config/ConfigRestoreDialog.svelte';
  import type { ConfigEditorView } from '$lib/config/configEditor.svelte';
  import type { ConfigEditorFlow } from '$lib/config/editorFlow.svelte';
  import type { ValidationReport } from '$lib/types_config';
  import { toastStore } from '$stores/toast.svelte';

  interface Props {
    flow: ConfigEditorFlow;
    ed: ConfigEditorView;
    cloner: {
      busy: boolean;
      cloneError: string | null;
      cloneReport: ValidationReport | null;
    };
    /** The name of the document on screen. */
    name: string;
    /** "pack", "profile", "set". */
    noun: string;
    /** What activating means for this kind of document. */
    blurb: string;
  }

  let { flow, ed, cloner, name, noun, blurb }: Props = $props();
</script>

{#if flow.activating}
  <ConfigActivateDialog
    {ed}
    target={flow.activating}
    {blurb}
    onclose={() => (flow.activating = null)}
    onactivated={(t) =>
      toastStore.success(
        `Activated ${t.name}${t.revision == null ? '' : ` r${t.revision}`}`,
      )}
  />
{/if}

{#if flow.restoring}
  <ConfigRestoreDialog
    {ed}
    {noun}
    onclose={() => (flow.restoring = false)}
    onrestored={(from, to) =>
      toastStore.success(`Restored revision ${from} as revision ${to}`)}
  />
{/if}

{#if flow.cloning}
  <ConfigCloneDialog
    title="Clone {name}"
    nameLabel="New {noun} name"
    busy={cloner.busy}
    error={cloner.cloneError}
    report={cloner.cloneReport}
    onconfirm={(newName) => void flow.doClone(newName)}
    oncancel={() => (flow.cloning = false)}
  />
{/if}
