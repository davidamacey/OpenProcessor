<script lang="ts">
  /**
   * `/p/<slug>` for a slug that isn't a project (unknown, deleted), or one
   * the server marks not `selectable` (e.g. `building`, `failed`,
   * `deleting`). A clear page with a way out — the project list, and the
   * default project when there is one — instead of a wall of scoped
   * errors. Everything shown is served: the status label, the default.
   */
  import { resolve } from '$app/paths';
  import { projectHref } from '$lib/projectPaths';
  import { projectsStore, type ProjectResolution } from '$stores/projects.svelte';

  interface Props {
    resolution: Exclude<ProjectResolution, { kind: 'ok' }>;
  }
  let { resolution }: Props = $props();

  const fallback = $derived(projectsStore.defaultProject);
</script>

<div
  class="flex h-screen flex-col items-center justify-center gap-3 bg-zinc-950 px-6 text-center text-zinc-100"
  data-testid="project-unavailable"
>
  {#if resolution.kind === 'not_found'}
    <p class="text-lg font-semibold">Project not found</p>
    <p class="max-w-md text-sm text-zinc-400">
      There is no project named <span class="font-mono text-zinc-200"
        >{resolution.slug}</span
      >.
    </p>
  {:else}
    <p class="text-lg font-semibold">Project not available</p>
    <p class="max-w-md text-sm text-zinc-400">
      <span class="text-zinc-200">{resolution.project.display_name}</span>
      (<span class="font-mono">{resolution.project.slug}</span>) is
      <span data-testid="project-unavailable-status"
        >{projectsStore.statusLabel(resolution.project.status).toLowerCase()}</span
      > and can't be opened right now.
    </p>
  {/if}
  <div class="flex gap-2">
    <a
      href={resolve('/projects')}
      class="rounded border border-zinc-700 px-3 py-1.5 text-sm hover:border-zinc-500"
      >All projects</a
    >
    {#if fallback}
      <a
        href={resolve(projectHref('/dashboard', fallback.slug))}
        class="rounded border border-zinc-700 px-3 py-1.5 text-sm hover:border-zinc-500"
        >Open {fallback.display_name}</a
      >
    {/if}
  </div>
</div>
