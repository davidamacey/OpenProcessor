<script lang="ts">
  /**
   * `/projects` — the global project management page (not
   * project-scoped). Lists the served projects (the server decides which
   * statuses are listed; archived ones only with "Show archived"), the
   * served shard `capacity`, and the P3 lifecycle actions: create, edit,
   * archive / unarchive, copy settings, delete. Every action is offered
   * from served flags alone:
   *
   * - Open, Edit: `selectable`
   * - Copy settings: `writable`
   * - Archive: `archivable`; Unarchive: `unarchivable`
   * - Delete: `deletable`
   * - Pause / Resume pipeline: `writable`, once the row's served pause
   *   state (`GET {prefix}/pause`, read for every `selectable` row) has
   *   loaded; the button offered is the opposite of the served `paused`
   * - Create: disabled only by a served `capacity.status === 'blocked'`
   * - Combine projects: only when `combineAvailability` says the backend
   *   mounts the combine router (one-shot probe)
   * - A project the server says came from a combine (`origin.kind ===
   *   'combine'` with a `job_id`) links to that job, and its delete is
   *   labelled "Undo combine" (undoing a combine is deleting the target)
   *
   * State and writes live in `projectsAdminController.svelte.ts`.
   */
  import { onMount } from 'svelte';
  import { resolve } from '$app/paths';
  import CloneSettingsDialog from '$components/projects/CloneSettingsDialog.svelte';
  import CreateProjectDialog from '$components/projects/CreateProjectDialog.svelte';
  import DeleteProjectDialog from '$components/projects/DeleteProjectDialog.svelte';
  import EditProjectDialog from '$components/projects/EditProjectDialog.svelte';
  import PauseProjectDialog from '$components/projects/PauseProjectDialog.svelte';
  import { combineAvailability } from '$lib/combine/combineAvailability.svelte';
  import { formatCount } from '$lib/formatCount';
  import { projectHref } from '$lib/projectPaths';
  import {
    createProjectsAdmin,
    type ActionResult,
  } from '$lib/projects/projectsAdminController.svelte';
  import { subscribeGlobalEvents } from '$lib/sse';
  import type { ProjectSummary } from '$lib/types_projects';
  import { projectsStore } from '$stores/projects.svelte';
  import { toastStore } from '$stores/toast.svelte';

  const appName =
    (import.meta.env?.PUBLIC_APP_NAME as string | undefined) || 'Cropwright';

  const admin = createProjectsAdmin();

  let createOpen = $state(false);
  let editing = $state<ProjectSummary | null>(null);
  let cloning = $state<ProjectSummary | null>(null);
  let deleting = $state<ProjectSummary | null>(null);
  let pending = $state<string | null>(null);
  let pausing = $state<{ project: ProjectSummary; pause: boolean } | null>(null);

  onMount(() => {
    admin.start();
    void admin.load();
    void combineAvailability.init();
    // A served project.* event (deleted, created, ...) re-reads the list
    // at once; the controller's poll covers a missed event.
    const sub = subscribeGlobalEvents({ onEvent: () => void admin.load() });
    return () => {
      sub.close();
      admin.stop();
    };
  });

  /** The combine job a project was made by, when the server says so. */
  function combineJobOf(p: ProjectSummary): string | null {
    return p.origin?.kind === 'combine' && p.origin.job_id ? p.origin.job_id : null;
  }

  /** "Back to a project": the active one when there is one, else the
   *  served default. */
  const home = $derived(projectsStore.current ?? projectsStore.defaultProject);
  const capacity = $derived(admin.capacity);
  const createBlocked = $derived(capacity?.status === 'blocked');

  async function lifecycle(
    p: ProjectSummary,
    verb: string,
    act: (p: ProjectSummary) => Promise<ActionResult>,
  ): Promise<void> {
    pending = p.slug;
    const res = await act(p);
    pending = null;
    if (res.ok) {
      toastStore.success(
        `${verb} "${res.project.display_name}" — now ${admin.statusLabel(res.project.status).toLowerCase()}.`,
      );
    } else {
      toastStore.error(res.message);
    }
  }

  function fmtBytes(n: number): string {
    return `${(n / 1024 ** 3).toFixed(1)} GB`;
  }
</script>

<div class="flex min-h-screen flex-col bg-zinc-950 text-zinc-100">
  <header class="flex h-12 shrink-0 items-center gap-4 border-b border-zinc-800 px-4">
    <span class="text-sm font-semibold tracking-tight">{appName}</span>
    <span class="text-zinc-600">/</span>
    <h1 class="text-sm text-zinc-200">Projects</h1>
    <span class="grow"></span>
    {#if home}
      <a
        href={resolve(projectHref('/dashboard', home.slug))}
        class="shrink-0 text-sm text-zinc-300 hover:text-white"
        data-testid="projects-back">Open {home.display_name}</a
      >
    {/if}
  </header>

  <main class="mx-auto w-full max-w-5xl flex-1 p-4">
    {#if capacity}
      <section
        class="mb-4 rounded border px-3 py-2 text-sm {capacity.status === 'blocked'
          ? 'border-red-900 bg-red-950/30'
          : capacity.status === 'warn'
            ? 'border-amber-900 bg-amber-950/30'
            : 'border-zinc-800 bg-zinc-900/40'}"
        data-testid="projects-capacity"
        data-status={capacity.status}
      >
        <p class="font-semibold">{capacity.labels[capacity.status] ?? capacity.status}</p>
        <p class="text-xs text-zinc-400">{capacity.message}</p>
        <p class="mt-1 text-xs text-zinc-500">
          {capacity.active_shards} of {capacity.soft_limit} recommended shards in use (hard
          limit
          {capacity.hard_limit}) · {capacity.per_project_shards} per project ·
          {capacity.projects_until_soft_limit} more project(s) before the recommended limit
          ·
          {fmtBytes(capacity.heap_max_bytes)} heap
        </p>
      </section>
    {:else if !admin.loading && !admin.loadError}
      <p class="mb-4 text-xs text-zinc-500" data-testid="projects-capacity-unknown">
        Capacity is unavailable right now; the server checks it when a project is created.
      </p>
    {/if}

    <div class="mb-3 flex items-center gap-3">
      <label class="flex items-center gap-2 text-sm text-zinc-300">
        <input
          type="checkbox"
          checked={admin.includeArchived}
          data-testid="projects-include-archived"
          onchange={(e) =>
            void admin.setIncludeArchived((e.currentTarget as HTMLInputElement).checked)}
        />
        Show archived
      </label>
      <span class="grow"></span>
      {#if combineAvailability.available === true}
        <a href={resolve('/projects/combine')} class="btn" data-testid="projects-combine"
          >Combine projects…</a
        >
      {/if}
      <button
        type="button"
        class="btn btn-primary"
        data-testid="projects-create"
        disabled={createBlocked}
        title={createBlocked ? capacity?.message : undefined}
        onclick={() => (createOpen = true)}>New project</button
      >
    </div>

    {#if admin.loadError}
      <p class="mb-3 text-sm text-red-300" data-testid="projects-load-error">
        {admin.loadError}
      </p>
    {/if}

    <div class="relative overflow-x-auto rounded border border-zinc-800">
      <table class="w-full text-sm" data-testid="projects-table">
        <thead class="bg-zinc-900 text-left text-xs text-zinc-400">
          <tr>
            <th class="px-3 py-2 font-normal">Project</th>
            <th class="px-3 py-2 font-normal">Status</th>
            <th class="px-3 py-2 text-right font-normal">Images</th>
            <th class="px-3 py-2 text-right font-normal">Items</th>
            <th class="px-3 py-2 text-right font-normal">Validated</th>
            <th class="px-3 py-2 text-right font-normal">Embedded</th>
            <th class="px-3 py-2 font-normal"><span class="sr-only">Actions</span></th>
          </tr>
        </thead>
        <tbody>
          {#each admin.list as p (p.slug)}
            <tr
              class="border-t border-zinc-800 align-top"
              data-testid="project-row-{p.slug}"
            >
              <td class="px-3 py-2">
                <div class="text-zinc-100">
                  {p.display_name}
                  {#if p.is_default}
                    <span class="ml-1 rounded bg-zinc-800 px-1 text-[10px] text-zinc-400"
                      >default</span
                    >
                  {/if}
                </div>
                <div class="font-mono text-[11px] text-zinc-500">{p.slug}</div>
                {#if p.description}
                  <div class="mt-0.5 text-xs text-zinc-400">{p.description}</div>
                {/if}
                {#if combineJobOf(p)}
                  <a
                    class="mt-0.5 inline-block text-xs text-blue-300 hover:underline"
                    href={resolve('/projects/combine/[job_id]', {
                      job_id: encodeURIComponent(combineJobOf(p) ?? ''),
                    })}
                    data-testid="project-combine-job-{p.slug}">Combine job</a
                  >
                {/if}
              </td>
              <td class="px-3 py-2">
                <div class="flex flex-wrap items-center gap-1">
                  <span
                    class="rounded px-1.5 py-0.5 text-xs {p.status === 'active'
                      ? 'bg-emerald-950/60 text-emerald-300'
                      : 'bg-amber-950/60 text-amber-300'}"
                    data-testid="project-status-{p.slug}"
                    >{admin.statusLabel(p.status)}</span
                  >
                  {#if p.paused}
                    <span
                      class="rounded bg-amber-950/60 px-1.5 py-0.5 text-xs text-amber-300"
                      title="Pipeline paused: workers skip this project until it's resumed"
                      data-testid="project-paused-{p.slug}">paused</span
                    >
                  {/if}
                </div>
              </td>
              <td class="px-3 py-2 text-right tabular-nums"
                >{formatCount(p.counts.images)}</td
              >
              <td class="px-3 py-2 text-right tabular-nums"
                >{formatCount(p.counts.items)}</td
              >
              <td class="px-3 py-2 text-right tabular-nums"
                >{formatCount(p.counts.validated)}</td
              >
              <td
                class="px-3 py-2 text-right tabular-nums"
                data-testid="project-embedded-{p.slug}"
                >{formatCount(p.counts.items_embedded)}</td
              >
              <td class="px-3 py-2">
                <div class="flex flex-wrap justify-end gap-1.5">
                  {#if p.selectable}
                    <a
                      href={resolve(projectHref('/dashboard', p.slug))}
                      class="btn btn-sm"
                      data-testid="project-open-{p.slug}">Open</a
                    >
                    <button
                      type="button"
                      class="btn btn-sm"
                      data-testid="project-edit-{p.slug}"
                      onclick={() => (editing = p)}>Edit</button
                    >
                  {/if}
                  {#if p.writable}
                    <button
                      type="button"
                      class="btn btn-sm"
                      data-testid="project-clone-{p.slug}"
                      onclick={() => (cloning = p)}>Copy settings</button
                    >
                  {/if}
                  {#if p.writable}
                    <button
                      type="button"
                      class="btn btn-sm"
                      data-testid="project-{p.paused ? 'resume' : 'pause'}-{p.slug}"
                      onclick={() => (pausing = { project: p, pause: !p.paused })}
                      >{p.paused ? 'Resume pipeline' : 'Pause pipeline'}</button
                    >
                  {/if}
                  {#if p.archivable}
                    <button
                      type="button"
                      class="btn btn-sm"
                      data-testid="project-archive-{p.slug}"
                      disabled={pending === p.slug}
                      onclick={() => void lifecycle(p, 'Archived', admin.archive)}
                      >Archive</button
                    >
                  {/if}
                  {#if p.unarchivable}
                    <button
                      type="button"
                      class="btn btn-sm"
                      data-testid="project-unarchive-{p.slug}"
                      disabled={pending === p.slug}
                      onclick={() => void lifecycle(p, 'Restored', admin.unarchive)}
                      >Unarchive</button
                    >
                  {/if}
                  {#if p.deletable}
                    <button
                      type="button"
                      class="btn btn-sm text-red-300"
                      data-testid="project-delete-{p.slug}"
                      onclick={() => (deleting = p)}
                      >{combineJobOf(p) ? 'Undo combine' : 'Delete'}</button
                    >
                  {/if}
                </div>
              </td>
            </tr>
          {:else}
            {#if !admin.loading}
              <tr>
                <td colspan="7" class="px-3 py-6 text-center text-sm text-zinc-500">
                  No projects.
                </td>
              </tr>
            {/if}
          {/each}
        </tbody>
      </table>
    </div>
  </main>
</div>

<CreateProjectDialog open={createOpen} {admin} onclose={() => (createOpen = false)} />
<EditProjectDialog project={editing} {admin} onclose={() => (editing = null)} />
<CloneSettingsDialog project={cloning} {admin} onclose={() => (cloning = null)} />
<DeleteProjectDialog
  project={deleting}
  {admin}
  title={deleting && combineJobOf(deleting) ? 'Undo combine' : undefined}
  onclose={() => (deleting = null)}
/>
<PauseProjectDialog
  target={pausing}
  onclose={() => (pausing = null)}
  onchanged={() => void admin.load()}
/>
