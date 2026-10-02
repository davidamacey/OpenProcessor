<script lang="ts">
  /**
   * `/projects/combine/[job_id]` — one combine job (OpenProcessor P4).
   * Everything shown is the served job; `CombineJob` follows it (2 s poll
   * while queued/running, `combine.progress` wake-ups). A job is found
   * again after a reload through the target project's served
   * `origin.job_id` on `/projects`.
   */
  import { page } from '$app/state';
  import { resolve } from '$app/paths';
  import ConfirmDialog from '$components/ConfirmDialog.svelte';
  import DeleteProjectDialog from '$components/projects/DeleteProjectDialog.svelte';
  import { combineAvailability } from '$lib/combine/combineAvailability.svelte';
  import { createCombineJob } from '$lib/combine/combineJobController.svelte';
  import { combineLabel, reportCell } from '$lib/combine/combineText';
  import { formatCount } from '$lib/formatCount';
  import { projectHref } from '$lib/projectPaths';
  import { createProjectsAdmin } from '$lib/projects/projectsAdminController.svelte';
  import type { CombineNextStep } from '$lib/types_combine';
  import type { ProjectSummary } from '$lib/types_projects';
  import CombineStepResult from '$components/combine/CombineStepResult.svelte';
  import { projectsStore } from '$stores/projects.svelte';
  import { toastStore } from '$stores/toast.svelte';

  const appName =
    (import.meta.env?.PUBLIC_APP_NAME as string | undefined) || 'Cropwright';

  const jobId = $derived(page.params.job_id ?? '');
  const job = $derived(
    createCombineJob(jobId, {
      projectOf: (slug) => projectsStore.list.find((p) => p.slug === slug) ?? null,
    }),
  );
  const admin = createProjectsAdmin();

  $effect(() => {
    void combineAvailability.init();
  });
  $effect(() => {
    if (combineAvailability.available !== true) return;
    const j = job;
    j.start();
    return () => j.stop();
  });

  let confirm = $state<'cancel' | 'resume' | null>(null);
  let pendingStep = $state<CombineNextStep | null>(null);
  let undoing = $state<ProjectSummary | null>(null);

  const served = $derived(job.job);
  const target = $derived(served?.target ?? null);
  const targetProject = $derived(
    target ? (projectsStore.list.find((p) => p.slug === target) ?? null) : null,
  );
  const reportRows = $derived(Object.entries(served?.report ?? {}));
  const pct = $derived(
    served && (served.total ?? 0) > 0
      ? Math.min(100, Math.round(((served.done ?? 0) / (served.total ?? 1)) * 100))
      : null,
  );

  async function doConfirm(): Promise<void> {
    const which = confirm;
    let ok = false;
    if (which === 'cancel') ok = await job.cancel();
    else if (which === 'resume') ok = await job.resume();
    if (ok) confirm = null;
  }

  async function doStep(): Promise<void> {
    const s = pendingStep;
    if (!s) return;
    if (await job.runNextStep(s)) {
      pendingStep = null;
      toastStore.info(`Ran ${combineLabel(s.action)}`);
    }
  }
</script>

<div class="flex min-h-screen flex-col bg-zinc-950 text-zinc-100">
  <header class="flex h-12 shrink-0 items-center gap-4 border-b border-zinc-800 px-4">
    <span class="text-sm font-semibold tracking-tight">{appName}</span>
    <span class="text-zinc-600">/</span>
    <a href={resolve('/projects')} class="text-sm text-zinc-300 hover:text-white"
      >Projects</a
    >
    <span class="text-zinc-600">/</span>
    <h1 class="text-sm text-zinc-200">Combine job</h1>
  </header>

  <main class="mx-auto w-full max-w-4xl flex-1 space-y-4 p-4">
    {#if combineAvailability.available === false}
      <p class="text-sm text-zinc-400" data-testid="combine-unavailable">
        Combining projects isn't available on this backend.
      </p>
    {:else if job.notFound}
      <p class="text-sm text-red-300" data-testid="combine-job-not-found">
        {job.notFound}
      </p>
    {:else if !served}
      {#if job.loadError}
        <p class="text-sm text-red-300" data-testid="combine-job-error">
          {job.loadError}
        </p>
      {:else}
        <p class="text-sm text-zinc-500">Loading…</p>
      {/if}
    {:else}
      <section
        class="space-y-2 rounded border border-zinc-800 p-3"
        data-testid="combine-job"
      >
        <div class="flex flex-wrap items-center gap-2">
          <span
            class="rounded px-1.5 py-0.5 text-xs {job.completed
              ? 'bg-emerald-950/60 text-emerald-300'
              : job.failed
                ? 'bg-red-950/60 text-red-300'
                : 'bg-amber-950/60 text-amber-300'}"
            data-testid="combine-job-status"
            data-status={served.status}>{combineLabel(served.status)}</span
          >
          {#if served.phase}
            <span class="text-xs text-zinc-400" data-testid="combine-job-phase"
              >Phase: {combineLabel(served.phase)}</span
            >
          {/if}
          <span class="grow"></span>
          <span class="font-mono text-[11px] text-zinc-500">{served.job_id}</span>
        </div>

        <div data-testid="combine-job-progress">
          <p class="text-xs text-zinc-300">
            {formatCount(served.done)} of {formatCount(served.total)}
            {#if pct !== null}<span class="text-zinc-500">({pct}%)</span>{/if}
          </p>
          {#if pct !== null}
            <div class="mt-1 h-1.5 w-full rounded bg-zinc-800">
              <div class="h-1.5 rounded bg-blue-500" style="width: {pct}%"></div>
            </div>
          {/if}
        </div>

        <dl class="grid grid-cols-[max-content_1fr] gap-x-3 gap-y-0.5 text-xs">
          <dt class="text-zinc-500">Target</dt>
          <dd class="font-mono text-zinc-200" data-testid="combine-job-target">
            {served.target ?? '—'}
          </dd>
          <dt class="text-zinc-500">Sources</dt>
          <dd class="font-mono text-zinc-200" data-testid="combine-job-sources">
            {(served.sources ?? []).join(', ') || '—'}
          </dd>
          <dt class="text-zinc-500">Started</dt>
          <dd class="text-zinc-300">{served.started_at ?? '—'}</dd>
          <dt class="text-zinc-500">Finished</dt>
          <dd class="text-zinc-300">{served.finished_at ?? '—'}</dd>
        </dl>

        {#if served.error}
          <p
            class="rounded border border-red-900 bg-red-950/30 px-2 py-1 text-xs text-red-200"
            data-testid="combine-job-served-error"
          >
            {served.error}
          </p>
        {/if}
        {#if job.loadError}
          <p class="text-xs text-red-300" data-testid="combine-job-error">
            {job.loadError}
          </p>
        {/if}
        {#if job.actionError}
          <p class="text-xs text-red-300" data-testid="combine-job-action-error">
            {job.actionError}
          </p>
        {/if}
      </section>

      <section
        class="flex flex-wrap items-center gap-2"
        data-testid="combine-job-actions"
      >
        {#if job.canCancel}
          <button
            type="button"
            class="btn"
            data-testid="combine-cancel"
            onclick={() => (confirm = 'cancel')}>Cancel</button
          >
        {/if}
        {#if job.canResume}
          <button
            type="button"
            class="btn"
            data-testid="combine-resume"
            onclick={() => (confirm = 'resume')}>Resume</button
          >
        {/if}
        {#if job.completed && target}
          <a
            class="btn btn-primary"
            href={resolve(projectHref('/dashboard', target))}
            data-testid="combine-open-project">Open project</a
          >
          <a
            class="btn"
            href={resolve(projectHref('/review?tab=all&combine_conflict=true', target))}
            data-testid="combine-review-conflicts">Review flagged conflicts</a
          >
          {#each served.next_steps ?? [] as step (step.action)}
            <button
              type="button"
              class="btn"
              title={step.reason}
              data-testid="combine-next-step-{step.action}"
              onclick={() => (pendingStep = step)}>{combineLabel(step.action)}</button
            >
          {/each}
        {/if}
        {#if job.failed && targetProject}
          <button
            type="button"
            class="btn text-red-300"
            data-testid="combine-undo"
            onclick={() => (undoing = targetProject)}>Undo combine</button
          >
        {/if}
      </section>

      {#if (served.next_steps ?? []).length > 0 && job.completed}
        <ul
          class="space-y-0.5 text-xs text-zinc-500"
          data-testid="combine-next-step-reasons"
        >
          {#each served.next_steps ?? [] as step (step.action)}
            {#if step.reason}<li>{combineLabel(step.action)}: {step.reason}</li>{/if}
          {/each}
        </ul>
      {/if}

      {#if job.lastStep}
        <CombineStepResult action={job.lastStep.action} result={job.lastStep.result} />
      {/if}

      {#if reportRows.length > 0}
        <section data-testid="combine-job-report">
          <h2 class="mb-1 text-sm font-semibold text-zinc-200">Report</h2>
          <div class="overflow-x-auto rounded border border-zinc-800">
            <table class="w-full text-xs">
              <tbody>
                {#each reportRows as [k, v] (k)}
                  <tr class="border-t border-zinc-800 first:border-t-0">
                    <td class="px-2 py-1 text-zinc-400">{combineLabel(k)}</td>
                    <td class="px-2 py-1 font-mono text-zinc-200" data-report-key={k}
                      >{reportCell(v)}</td
                    >
                  </tr>
                {/each}
              </tbody>
            </table>
          </div>
        </section>
      {/if}
    {/if}
  </main>
</div>

{#if confirm}
  <ConfirmDialog
    title={confirm === 'cancel' ? 'Cancel this combine?' : 'Resume this combine?'}
    confirmLabel={confirm === 'cancel' ? 'Cancel combine' : 'Resume'}
    danger={confirm === 'cancel'}
    busy={job.busy}
    onconfirm={() => void doConfirm()}
    oncancel={() => (confirm = null)}
  >
    {#if confirm === 'cancel'}
      <p>The job stops at its next step. You can resume it later.</p>
    {:else}
      <p>The job continues from where it stopped.</p>
    {/if}
    {#if job.actionError}<p class="text-red-300">{job.actionError}</p>{/if}
  </ConfirmDialog>
{/if}

{#if pendingStep}
  <ConfirmDialog
    title={combineLabel(pendingStep.action) + '?'}
    confirmLabel="Run"
    busy={job.busy}
    onconfirm={() => void doStep()}
    oncancel={() => (pendingStep = null)}
  >
    {#if pendingStep.reason}<p>{pendingStep.reason}</p>{/if}
    <p class="text-xs text-zinc-400">
      Runs <code class="font-mono"
        >{pendingStep.method.toUpperCase()} {pendingStep.path}</code
      >
      on {target}.
    </p>
    {#if job.actionError}<p class="text-red-300">{job.actionError}</p>{/if}
  </ConfirmDialog>
{/if}

<DeleteProjectDialog
  project={undoing}
  {admin}
  title="Undo combine"
  onclose={() => (undoing = null)}
/>
