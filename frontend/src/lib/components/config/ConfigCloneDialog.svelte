<script lang="ts">
  /**
   * The clone dialog shared by the config list pages and the editors'
   * "Clone to edit": a new name (and optionally a description), then the
   * served refusal (`name_conflict`, `validation_failed` with its report)
   * shown verbatim. With `offerProjects`, a picker over the other served
   * selectable projects chooses a project to copy the doc from
   * (`from_project`; the doc of that name in that project); it is handed on
   * only when chosen.
   */
  import { projectsStore } from '$stores/projects.svelte';
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
    /** Offer the "copy from another project" picker. */
    offerProjects?: boolean;
    onconfirm: (name: string, description: string, fromProject: string | null) => void;
    oncancel: () => void;
  }

  let {
    title,
    nameLabel,
    busy,
    error,
    report,
    withDescription = false,
    offerProjects = false,
    onconfirm,
    oncancel,
  }: Props = $props();

  let name = $state('');
  let description = $state('');
  let fromProject = $state('');

  const otherProjects = $derived(
    projectsStore.selectable.filter((p) => p.slug !== projectsStore.current?.slug),
  );
</script>

<ConfirmDialog
  {title}
  confirmLabel="Clone"
  {busy}
  confirmDisabled={name.trim() === ''}
  onconfirm={() => onconfirm(name, description, fromProject || null)}
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
  {#if offerProjects && otherProjects.length > 0}
    <label class="flex flex-col gap-1 text-xs text-zinc-400">
      Copy from another project (optional)
      <select
        class="input input-sm"
        bind:value={fromProject}
        data-testid="clone-from-project"
      >
        <option value="">This project</option>
        {#each otherProjects as p (p.slug)}
          <option value={p.slug}>{p.display_name} ({p.slug})</option>
        {/each}
      </select>
    </label>
  {/if}
  {#if error}
    <p class="text-red-300" data-testid="clone-error">{error}</p>
  {/if}
  {#if report}
    <ConfigIssueList issues={[...report.errors, ...report.warnings]} showField />
  {/if}
</ConfirmDialog>
