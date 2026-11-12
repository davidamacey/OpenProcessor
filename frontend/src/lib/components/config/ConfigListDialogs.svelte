<script lang="ts" generics="Row extends { name: string }">
  /**
   * The Clone and Delete dialogs of a config list page, driven by its
   * `createCloneDeleteFlow` and list controller.
   */
  import type { Snippet } from 'svelte';
  import ConfirmDialog from '$components/ConfirmDialog.svelte';
  import ConfigCloneDialog from '$components/config/ConfigCloneDialog.svelte';
  import type { CloneDeleteFlow } from '$lib/config/cloneDeleteFlow.svelte';
  import type { ValidationReport } from '$lib/types_config';

  interface Props {
    flow: CloneDeleteFlow<Row>;
    list: {
      busy: boolean;
      cloneError: string | null;
      cloneReport: ValidationReport | null;
      deleteError: string | null;
    };
    /** "New pack name", "New set name", ... */
    nameLabel: string;
    /** What the delete does, said under the title. */
    deleteNote: Snippet<[Row]>;
    /** Offer "copy from another project"; off for a kind with no such clone (the presets). */
    offerProjects?: boolean;
  }

  let { flow, list, nameLabel, deleteNote, offerProjects = true }: Props = $props();
</script>

{#if flow.cloneFrom}
  {@const from = flow.cloneFrom}
  <ConfigCloneDialog
    title="Clone {from.name}{from.source === 'template' ? ' (template)' : ''}"
    {nameLabel}
    withDescription
    busy={list.busy}
    error={list.cloneError}
    report={list.cloneReport}
    offerProjects={offerProjects && from.source !== 'template'}
    onconfirm={(name, description, fromProject) =>
      void flow.doClone(name, description, fromProject)}
    oncancel={() => (flow.cloneFrom = null)}
  />
{/if}

{#if flow.deleting}
  {@const row = flow.deleting}
  <ConfirmDialog
    title="Delete {row.name}"
    confirmLabel="Delete"
    danger
    busy={list.busy}
    onconfirm={() => void flow.doDelete()}
    oncancel={() => (flow.deleting = null)}
  >
    <p>{@render deleteNote(row)}</p>
    {#if list.deleteError}<p class="text-danger-text" data-testid="delete-error">
        {list.deleteError}
      </p>{/if}
  </ConfirmDialog>
{/if}
