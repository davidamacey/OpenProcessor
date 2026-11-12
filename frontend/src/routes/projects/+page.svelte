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
   * - Combine projects: always offered; the backend mounts the combine router
   *   unconditionally, so there is nothing to probe
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
  import { formatBytes } from '$lib/format/bytes';
  import { formatCount } from '$lib/format/count';
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
</script>

<div class="flex min-h-screen flex-col bg-surface-0 text-fg-1">
  <header class="flex h-12 shrink-0 items-center gap-4 border-b border-line px-4">
    <span class="shrink-0 text-body font-semibold tracking-tight">{appName}</span>
    <span class="shrink-0 text-fg-3">/</span>
    <h1 class="shrink-0 text-body text-fg-1">Projects</h1>
    <span class="grow"></span>
    {#if home}
      <a
        data-sveltekit-preload-data="tap"
        href={resolve(projectHref('/dashboard', home.slug))}
        class="min-w-0 truncate text-body text-fg-2 hover:text-white"
        title="Open {home.display_name}"
        data-testid="projects-back">Open {home.display_name}</a
      >
    {/if}
  </header>

  <main class="mx-auto w-full max-w-5xl flex-1 p-4">
    {#if capacity}
      <section
        class="mb-4 rounded border px-3 py-2 text-body {capacity.status === 'blocked'
          ? 'border-danger-line bg-danger-soft'
          : capacity.status === 'warn'
            ? 'border-warn-line bg-warn-soft'
            : 'border-line bg-surface-1/40'}"
        data-testid="projects-capacity"
        data-status={capacity.status}
      >
        <p class="font-semibold">{capacity.labels[capacity.status] ?? capacity.status}</p>
        <p class="text-meta text-fg-2">{capacity.message}</p>
        <p class="mt-1 text-meta text-fg-3">
          {capacity.active_shards} of {capacity.soft_limit} recommended shards in use (hard
          limit
          {capacity.hard_limit}) · {capacity.per_project_shards} per project ·
          {capacity.projects_until_soft_limit} more project(s) before the recommended limit
          ·
          {formatBytes(capacity.heap_max_bytes)} heap
        </p>
      </section>
    {:else if !admin.loading && !admin.loadError}
      <p class="mb-4 text-meta text-fg-3" data-testid="projects-capacity-unknown">
        Capacity is unavailable right now; the server checks it when a project is created.
      </p>
    {/if}

    <div class="mb-3 flex flex-wrap items-center gap-3">
      <label class="flex items-center gap-2 text-body text-fg-2">
        <input
          type="checkbox"
          checked={admin.includeArchived}
          data-testid="projects-include-archived"
          onchange={(e) =>
            void admin.setIncludeArchived((e.currentTarget as HTMLInputElement).checked)}
        />
        Show archived
      </label>
      <input
        type="search"
        class="min-w-0 max-w-56 grow rounded border border-line-strong bg-surface-0 px-2 py-1 text-body text-fg-1 placeholder:text-fg-3"
        placeholder="Find a project…"
        aria-label="Find a project"
        data-testid="projects-search"
        value={admin.query}
        oninput={(e) => admin.type(e.currentTarget.value)}
      />
      <select
        class="rounded border border-line-strong bg-surface-0 px-2 py-1 text-body text-fg-1"
        aria-label="Status"
        data-testid="projects-status-filter"
        value={admin.statusFilter}
        onchange={(e) => void admin.setStatusFilter(e.currentTarget.value)}
      >
        <option value="">Any status</option>
        {#each admin.statusOptions as st (st.id)}
          <option value={st.id}>{st.label}</option>
        {/each}
      </select>
      <select
        class="rounded border border-line-strong bg-surface-0 px-2 py-1 text-body text-fg-1"
        aria-label="Sort"
        data-testid="projects-sort"
        value={admin.sort}
        onchange={(e) => void admin.setSort(e.currentTarget.value)}
      >
        {#each admin.sortOptions as o (o.id)}
          <option value={o.id}>{o.label}</option>
        {/each}
      </select>
      <span class="grow max-md:hidden"></span>
      <a href={resolve('/projects/combine')} class="btn" data-testid="projects-combine"
        >Combine projects…</a
      >
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
      <p class="mb-3 text-body text-danger-text" data-testid="projects-load-error">
        {admin.loadError}
      </p>
    {/if}

    <div class="relative overflow-x-auto rounded border border-line">
      <table class="w-full text-body max-md:block" data-testid="projects-table">
        <thead class="bg-surface-1 text-left text-meta text-fg-2 max-md:hidden">
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
        <tbody class="max-md:block">
          {#each admin.list as p (p.slug)}
            <tr
              class="border-t border-line align-top max-md:flex max-md:flex-wrap"
              data-testid="project-row-{p.slug}"
            >
              <td class="px-3 py-2 max-md:w-full">
                <div class="text-fg-1">
                  {p.display_name}
                  {#if p.is_default}
                    <span class="ml-1 rounded bg-surface-2 px-1 text-caption text-fg-2"
                      >default</span
                    >
                  {/if}
                </div>
                <div class="font-mono text-caption text-fg-3">{p.slug}</div>
                {#if p.description}
                  <div class="mt-0.5 text-meta text-fg-2">{p.description}</div>
                {/if}
                {#if combineJobOf(p)}
                  <a
                    class="mt-0.5 inline-block text-meta text-accent-text hover:underline"
                    href={resolve('/projects/combine/[job_id]', {
                      job_id: encodeURIComponent(combineJobOf(p) ?? ''),
                    })}
                    data-testid="project-combine-job-{p.slug}">Combine job</a
                  >
                {/if}
              </td>
              <td class="px-3 py-2 max-md:w-full max-md:pt-0">
                <div class="flex flex-wrap items-center gap-1">
                  <span
                    class="rounded px-1.5 py-0.5 text-meta {p.status === 'active'
                      ? 'bg-success-soft text-success-text'
                      : 'bg-warn-soft text-warn-text'}"
                    data-testid="project-status-{p.slug}"
                    >{admin.statusLabel(p.status)}</span
                  >
                  {#if p.paused}
                    <span
                      class="rounded bg-warn-soft px-1.5 py-0.5 text-meta text-warn-text"
                      title="Pipeline paused: workers skip this project until it's resumed"
                      data-testid="project-paused-{p.slug}">paused</span
                    >
                  {/if}
                </div>
              </td>
              <td class="px-3 py-2 text-right tabular-nums max-md:w-1/2 max-md:text-left"
                ><span class="mr-1 text-meta text-fg-3 md:hidden">Images&#32;</span
                >{formatCount(p.counts.images)}</td
              >
              <td class="px-3 py-2 text-right tabular-nums max-md:w-1/2 max-md:text-left"
                ><span class="mr-1 text-meta text-fg-3 md:hidden">Items&#32;</span
                >{formatCount(p.counts.items)}</td
              >
              <td class="px-3 py-2 text-right tabular-nums max-md:w-1/2 max-md:text-left"
                ><span class="mr-1 text-meta text-fg-3 md:hidden">Validated&#32;</span
                >{formatCount(p.counts.validated)}</td
              >
              <td
                class="px-3 py-2 text-right tabular-nums max-md:w-1/2 max-md:text-left"
                data-testid="project-embedded-{p.slug}"
                ><span class="mr-1 text-meta text-fg-3 md:hidden">Embedded&#32;</span
                >{formatCount(p.counts.items_embedded)}</td
              >
              <td class="px-3 py-2 max-md:w-full">
                <div class="flex flex-wrap justify-end gap-1.5 max-md:justify-start">
                  {#if p.selectable}
                    <a
                      data-sveltekit-preload-data="tap"
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
                      class="btn btn-sm text-danger-text"
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
                <td colspan="7" class="px-3 py-6 text-center text-body text-fg-3">
                  No projects.
                </td>
              </tr>
            {/if}
          {/each}
        </tbody>
      </table>
    </div>
    <div
      class="mt-2 flex items-center gap-3 text-meta text-fg-3"
      data-testid="projects-pager"
    >
      <span>Showing {admin.list.length} of {admin.total}</span>
      {#if admin.sortCap !== null}
        <span data-testid="projects-sort-cap"
          >Ranked among the {admin.sortCap} most recently updated.</span
        >
      {/if}
      {#if admin.hasMore}
        <button
          type="button"
          class="btn"
          disabled={admin.loading}
          data-testid="projects-more"
          onclick={() => void admin.more()}>Show more</button
        >
      {/if}
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
