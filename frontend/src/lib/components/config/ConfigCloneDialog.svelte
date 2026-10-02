<script lang="ts">
  /**
   * The clone dialog shared by the config list pages and the editors'
   * "Clone to edit": a new name (and optionally a description), then the
   * served refusal (`name_conflict`, `validation_failed` with its report)
   * shown verbatim.
   */
  import ConfirmDialog from '$components/ConfirmDialog.svelte';
  import type { ValidationReport } from '$lib/types_config';
  import ConfigIssueList from './ConfigIssueList.svelte';

  interface Props {
    title: string;
    /** "New pack name" / "New profile name". */
    nameLabel: string;
    busy: boolean;
    error: string | null;
    report: ValidationReport | null;
    withDescription?: boolean;
    onconfirm: (name: string, description: string) => void;
    oncancel: () => void;
  }

  let {
    title,
    nameLabel,
    busy,
    error,
    report,
    withDescription = false,
    onconfirm,
    oncancel,
  }: Props = $props();

  let name = $state('');
  let description = $state('');
</script>

<ConfirmDialog
  {title}
  confirmLabel="Clone"
  {busy}
  confirmDisabled={name.trim() === ''}
  onconfirm={() => onconfirm(name, description)}
  {oncancel}
>
  <label class="flex flex-col gap-1 text-xs text-zinc-400">
    {nameLabel}
    <input class="input input-sm font-mono" bind:value={name} data-testid="clone-name" />
  </label>
  {#if withDescription}
    <label class="flex flex-col gap-1 text-xs text-zinc-400">
      Description (optional)
      <input class="input input-sm" bind:value={description} />
    </label>
  {/if}
  {#if error}
    <p class="text-red-300" data-testid="clone-error">{error}</p>
  {/if}
  {#if report}
    <ConfigIssueList issues={[...report.errors, ...report.warnings]} showField />
  {/if}
</ConfirmDialog>
