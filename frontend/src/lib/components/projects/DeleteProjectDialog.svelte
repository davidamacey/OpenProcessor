<script lang="ts">
  /**
   * Guarded delete. Opening it runs the served dry run
   * (`DELETE /projects/{slug}?dry_run=true`, writes nothing) and shows
   * what would go — indexes, directories, models, running jobs — plus the
   * served `blocking` reasons (`blocking_detail`'s message, or the bare
   * code when that's all that was served). While anything blocks, there
   * is no confirm field. Otherwise the operator types the slug and the
   * real delete sends it as `confirm`; any refusal (`project_protected`,
   * `project_busy`, `confirm_mismatch`, ...) renders its served message.
   * Only offered at all when the project's served `deletable` is true.
   */
  import { focusOnMount } from '$lib/actions/focusOnMount';
  import { trapFocus } from '$lib/actions/trapFocus';
  import type { ProjectsAdmin } from '$lib/projects/projectsAdminController.svelte';
  import type { DeleteDryRunResponse, ProjectSummary } from '$lib/types_projects';
  import { toastStore } from '$stores/toast.svelte';

  interface Props {
    project: ProjectSummary | null;
    admin: ProjectsAdmin;
    onclose: () => void;
  }
  let { project, admin, onclose }: Props = $props();

  let report = $state<DeleteDryRunResponse | null>(null);
  let dryRunError = $state<string | null>(null);
  let loadingReport = $state(false);
  let typed = $state('');
  let busy = $state(false);
  let errorText = $state<string | null>(null);

  $effect(() => {
    const p = project;
    if (!p) return;
    report = null;
    dryRunError = null;
    typed = '';
    errorText = null;
    loadingReport = true;
    void admin.dryRunDelete(p).then((res) => {
      if (project?.slug !== p.slug) return;
      loadingReport = false;
      if (res.ok) report = res.report;
      else dryRunError = res.message;
    });
  });

  /** Served blocking reasons: the structured detail when served, else
   *  the bare codes. */
  const blocking = $derived.by(() => {
    if (!report) return [] as { code: string; message: string }[];
    if (report.blocking_detail && report.blocking_detail.length > 0)
      return report.blocking_detail;
    return report.blocking.map((code) => ({ code, message: code }));
  });
  const blocked = $derived(blocking.length > 0 || dryRunError !== null);

  function fmtBytes(n: number | null | undefined): string {
    if (n == null) return '—';
    if (n < 1024) return `${n} B`;
    const units = ['KB', 'MB', 'GB', 'TB'];
    let v = n;
    let i = -1;
    while (v >= 1024 && i < units.length - 1) {
      v /= 1024;
      i += 1;
    }
    return `${v.toFixed(1)} ${units[i]}`;
  }

  async function submit(): Promise<void> {
    if (!project) return;
    busy = true;
    errorText = null;
    const res = await admin.remove(project, typed.trim());
    busy = false;
    if (res.ok) {
      toastStore.info(
        `"${res.project.display_name}" is ${admin.statusLabel(res.project.status).toLowerCase()}.`,
      );
      onclose();
    } else {
      errorText = res.message;
    }
  }
</script>

{#if project}
  <!-- svelte-ignore a11y_click_events_have_key_events -->
  <div
    class="fixed inset-0 z-40 flex items-center justify-center bg-black/60 p-4"
    role="dialog"
    aria-modal="true"
    aria-label="Delete project"
    use:focusOnMount
    use:trapFocus={{ onEscape: onclose }}
    tabindex="-1"
    data-testid="delete-project-dialog"
    onclick={(e) => {
      if (e.target === e.currentTarget) onclose();
    }}
  >
    <div
      class="w-full max-w-lg rounded-lg border border-zinc-800 bg-zinc-950 p-5 shadow-2xl"
    >
      <h3 class="mb-1 text-base font-semibold">Delete {project.display_name}?</h3>
      <p class="mb-3 text-xs text-zinc-400">
        This permanently removes the project's data. Its slug
        <span class="font-mono text-zinc-200">{project.slug}</span> can never be reused.
      </p>

      {#if loadingReport}
        <p class="mb-3 text-xs text-zinc-500">Checking what would be deleted…</p>
      {:else if dryRunError}
        <p class="mb-3 text-xs text-red-300" data-testid="delete-project-dry-run-error">
          {dryRunError}
        </p>
      {:else if report}
        <div
          class="mb-3 max-h-56 overflow-auto rounded border border-zinc-800 p-2 text-xs"
        >
          <p class="mb-1 text-zinc-400">Would delete:</p>
          <ul class="space-y-0.5 text-zinc-300" data-testid="delete-project-report">
            {#each report.indexes as ix (ix.name)}
              <li>index <span class="font-mono">{ix.name}</span> · {ix.docs} docs</li>
            {/each}
            {#each report.dirs as d (d.path)}
              <li>
                directory <span class="font-mono">{d.path}</span> · {fmtBytes(d.bytes)}
              </li>
            {/each}
            {#each report.promoted_models as m (m)}
              <li>model <span class="font-mono">{m}</span></li>
            {/each}
            {#if report.mlflow_experiment}
              <li>
                experiment <span class="font-mono">{report.mlflow_experiment}</span>
              </li>
            {/if}
          </ul>
          {#if report.running_jobs.length > 0}
            <p class="mt-2 text-amber-300">{report.running_jobs.length} running job(s)</p>
          {/if}
        </div>
        {#if blocking.length > 0}
          <div
            class="mb-3 rounded border border-red-900 bg-red-950/40 px-2 py-1.5 text-xs text-red-200"
            data-testid="delete-project-blocking"
          >
            <p class="mb-1 font-semibold">Can't delete:</p>
            <ul class="list-inside list-disc">
              {#each blocking as b (b.code)}
                <li data-code={b.code}>{b.message}</li>
              {/each}
            </ul>
          </div>
        {/if}
      {/if}

      <form
        onsubmit={async (e) => {
          e.preventDefault();
          await submit();
        }}
      >
        {#if report && !blocked}
          <label class="mb-3 block text-sm">
            <span class="mb-1 block text-xs text-zinc-400"
              >Type <span class="font-mono text-zinc-200">{project.slug}</span> to confirm</span
            >
            <input
              type="text"
              bind:value={typed}
              autocomplete="off"
              data-testid="delete-project-confirm"
              class="w-full rounded-md border border-zinc-700 bg-zinc-900 px-2 py-1.5 font-mono text-sm text-zinc-100 focus:border-red-500 focus:outline-none"
            />
          </label>
        {/if}
        {#if errorText}
          <p class="mb-3 text-xs text-red-300" data-testid="delete-project-error">
            {errorText}
          </p>
        {/if}
        <div class="flex items-center justify-end gap-2">
          <button type="button" class="btn" onclick={onclose} disabled={busy}
            >Cancel</button
          >
          <button
            type="submit"
            class="btn border-red-800 text-red-200 hover:border-red-600"
            data-testid="delete-project-submit"
            disabled={busy || !report || blocked || typed.trim() === ''}
          >
            {busy ? 'Deleting…' : 'Delete'}
          </button>
        </div>
      </form>
    </div>
  </div>
{/if}
