<script lang="ts">
  /**
   * `/datasets/imports/[id]` — one import job (any_domain_plan.md §7.12
   * item 2; docs/design/w10-import-reprocess-ui-plan-2026-09-27.md §3).
   * Everything shown is the served job; `ImportJob` follows it.
   */
  import { page } from '$app/state';
  import { resolve } from '$app/paths';
  import ConfirmDialog from '$components/ConfirmDialog.svelte';
  import DatasetsGate from '$components/datasets/DatasetsGate.svelte';
  import DatasetIssueList from '$components/datasets/DatasetIssueList.svelte';
  import { datasetsAvailability } from '$lib/datasets/datasetsAvailability.svelte';
  import { createImportJob, PAGE_SIZE } from '$lib/datasets/importJobController.svelte';
  import { projectHref } from '$lib/projectPaths';
  import type { DatasetImportReport, NextStep } from '$lib/types_import';

  const importId = $derived(page.params.id ?? '');
  const job = $derived(createImportJob(importId));

  $effect(() => {
    if (datasetsAvailability.available !== true) return;
    const j = job;
    j.start();
    return () => j.stop();
  });

  let confirm = $state<'cancel' | 'resume' | 'undo' | null>(null);
  let pendingStep = $state<NextStep | null>(null);
  let undoChoices = $state({ remove_images: true, deprecate_created_classes: true });

  function openUndo(): void {
    confirm = 'undo';
    void job.undoDryRun(undoChoices);
  }

  function setUndoChoice(
    key: 'remove_images' | 'deprecate_created_classes',
    v: boolean,
  ): void {
    undoChoices = { ...undoChoices, [key]: v };
    void job.undoDryRun(undoChoices);
  }

  async function doConfirm(): Promise<void> {
    const which = confirm;
    let ok = false;
    if (which === 'cancel') ok = await job.cancel();
    else if (which === 'resume') ok = await job.resume();
    else if (which === 'undo') ok = await job.undoApply(undoChoices);
    if (ok) confirm = null;
  }

  async function doStep(): Promise<void> {
    const s = pendingStep;
    if (!s) return;
    if (await job.runNextStep(s)) pendingStep = null;
  }

  const REPORT_ROWS: Array<[keyof Omit<DatasetImportReport, 'disagreements'>, string]> = [
    ['images_created', 'Images created'],
    ['images_reused', 'Images reused'],
    ['items_created', 'Items created'],
    ['items_updated', 'Items updated'],
    ['items_noop', 'Items unchanged'],
    ['labels_written', 'Labels written'],
    ['boxes_written', 'Region boxes written'],
    ['standalone_regions', 'Standalone regions'],
    ['negatives', 'Negatives'],
    ['unlabeled', 'Unlabeled'],
    ['parents_detected', 'Parents detected'],
    ['proposals_created', 'Proposals created'],
    ['proposals_merged', 'Proposals merged'],
    ['holdout_frozen', 'Held out (test)'],
    ['label_conflicts_locked', 'Conflicts with locked labels'],
  ];

  const pct = (done: number, total: number): number =>
    total > 0 ? Math.min(100, (done / total) * 100) : 0;
</script>

<div class="mx-auto max-w-6xl space-y-6 p-6">
  <div class="flex flex-wrap items-baseline justify-between gap-2">
    <h1 class="text-lg font-semibold text-zinc-100">
      Import
      <span class="font-mono text-sm text-zinc-400">{importId}</span>
    </h1>
    <a
      class="text-xs text-blue-300 hover:underline"
      href={resolve(projectHref('/datasets/imports'))}>All imports</a
    >
  </div>

  <DatasetsGate>
    {#snippet children(formats)}
      {#if job.loadError && !job.job}
        <p class="text-sm text-red-300" data-testid="job-load-error">{job.loadError}</p>
      {:else if !job.job}
        <p class="text-sm text-zinc-500">Loading…</p>
      {:else}
        {@const j = job.job}
        <section
          class="space-y-3 rounded-lg border border-zinc-800 p-4"
          data-testid="import-job"
        >
          <div class="flex flex-wrap items-center gap-3">
            <span class="text-sm font-semibold text-zinc-100"
              >{j.name ?? j.import_id}</span
            >
            <span
              class="rounded border border-zinc-700 px-2 py-0.5 text-xs text-zinc-200"
              data-testid="job-status"
              data-status={j.status}>{job.statusLabel(formats.status_labels)}</span
            >
            <span class="text-xs text-zinc-500">
              {formats.formats.find((f) => f.format === j.source.format)?.label ??
                j.source.format}
              · <code class="font-mono">{j.source.root}</code>
            </span>
          </div>

          <div data-testid="job-progress">
            <div class="h-2 w-full overflow-hidden rounded bg-zinc-800">
              <div
                class="h-full bg-blue-500"
                style="width: {pct(j.progress.images_done, j.progress.images_total)}%"
              ></div>
            </div>
            <p class="mt-1 text-xs text-zinc-400">
              {j.progress.images_done.toLocaleString()} / {j.progress.images_total.toLocaleString()}
              images · {j.progress.chunks_done} / {j.progress.chunks_total} chunks
              {#if j.progress.images_failed > 0}
                · <span class="text-red-300"
                  >{j.progress.images_failed.toLocaleString()} failed</span
                >
              {/if}
              {#if j.progress.images_per_s != null}
                · {j.progress.images_per_s} images/s{/if}
              {#if j.progress.eta_s != null && j.poll_after_s != null}
                · about {j.progress.eta_s} s left{/if}
            </p>
          </div>

          {#if j.waiting_for}
            <p
              class="rounded border border-amber-900 bg-amber-950/30 p-2 text-xs text-amber-200"
              data-testid="job-waiting"
            >
              Waiting for: {j.waiting_for}
            </p>
          {/if}

          {#if j.error}
            <p
              class="rounded border border-red-900 bg-red-950/30 p-2 text-sm text-red-200"
              data-testid="job-error"
            >
              {j.error}
            </p>
          {/if}

          {#if job.loadError}
            <p class="text-xs text-amber-300">Could not refresh: {job.loadError}</p>
          {/if}

          <div class="flex flex-wrap gap-2">
            {#if job.canCancel}
              <button
                type="button"
                class="btn btn-sm"
                onclick={() => (confirm = 'cancel')}>Cancel import</button
              >
            {/if}
            {#if job.canResume}
              <button
                type="button"
                class="btn btn-sm"
                onclick={() => (confirm = 'resume')}>Resume</button
              >
            {/if}
            {#if job.canUndo}
              <button type="button" class="btn btn-sm" onclick={openUndo}
                >Undo import</button
              >
            {/if}
          </div>
          {#if job.actionError && !confirm && !pendingStep}
            <p class="text-sm text-red-300" data-testid="job-action-error">
              {job.actionError}
            </p>
          {/if}
        </section>

        <section class="rounded-lg border border-zinc-800 p-4">
          <h2 class="mb-2 text-sm font-semibold text-zinc-200">Report</h2>
          <dl
            class="grid grid-cols-2 gap-x-4 gap-y-1 text-xs sm:grid-cols-3 lg:grid-cols-5"
          >
            {#each REPORT_ROWS as [key, label] (key)}
              <div>
                <dt class="text-zinc-500">{label}</dt>
                <dd class="font-mono" data-testid="report-{key}">
                  {j.report[key].toLocaleString()}
                </dd>
              </div>
            {/each}
          </dl>
          {#if Object.keys(j.report.disagreements.counts).length > 0}
            <p class="mt-2 text-xs text-zinc-400">
              Disagreements:
              {#each Object.entries(j.report.disagreements.counts) as [k, n], i (k)}{k}
                {n.toLocaleString()}{i <
                Object.keys(j.report.disagreements.counts).length - 1
                  ? ', '
                  : ''}{/each}
            </p>
          {/if}
        </section>

        {#if j.next_steps.length > 0}
          <section
            class="space-y-2 rounded-lg border border-zinc-800 p-4"
            data-testid="next-steps"
          >
            <h2 class="text-sm font-semibold text-zinc-200">Next steps</h2>
            {#each j.next_steps as step (step.action + step.path)}
              <div class="flex flex-wrap items-center gap-2 text-xs">
                <button
                  type="button"
                  class="btn btn-sm"
                  onclick={() => (pendingStep = step)}>{step.action}</button
                >
                <span class="text-zinc-400">{step.reason}</span>
              </div>
            {/each}
          </section>
        {/if}

        <section class="rounded-lg border border-zinc-800 p-4">
          <h2 class="mb-2 text-sm font-semibold text-zinc-200">Class mapping</h2>
          <table class="text-xs">
            <tbody>
              {#each j.mapping as m (m.dataset_class)}
                <tr class="border-t border-zinc-800">
                  <td class="py-1 pr-4 text-zinc-300">{m.dataset_class}</td>
                  <td class="py-1 pr-4 text-zinc-500">
                    {formats.mapping_actions.find(
                      (a) => a.value === (m.kind === 'item' ? 'map' : m.kind),
                    )?.label ?? m.kind}
                  </td>
                  <td class="py-1 text-zinc-100">{m.class_name ?? '—'}</td>
                </tr>
              {/each}
            </tbody>
          </table>
        </section>

        {#if j.issues_summary.length > 0}
          <section class="rounded-lg border border-zinc-800 p-4">
            <h2 class="mb-2 text-sm font-semibold text-zinc-200">Issues</h2>
            <DatasetIssueList issues={j.issues_summary} catalog={formats.issues} />
          </section>
        {/if}

        <details
          class="rounded-lg border border-zinc-800 p-4"
          ontoggle={(e) => {
            if ((e.currentTarget as HTMLDetailsElement).open && !job.issues)
              void job.loadIssues(1);
          }}
        >
          <summary class="cursor-pointer text-sm font-semibold text-zinc-200"
            >All issues</summary
          >
          <div class="mt-2 space-y-2 text-xs">
            <select
              class="select select-sm"
              aria-label="Filter issues by code"
              value={job.issueCode}
              onchange={(e) => {
                job.issueCode = (e.currentTarget as HTMLSelectElement).value;
                void job.loadIssues(1);
              }}
            >
              <option value="">Every code</option>
              {#each formats.issues as c (c.code)}
                <option value={c.code}>{c.label}</option>
              {/each}
            </select>
            {#if job.issues}
              <DatasetIssueList issues={job.issues.items} catalog={formats.issues} />
              {@render Pager({
                page: job.issuesPage,
                total: job.issues.total,
                go: (p: number) => void job.loadIssues(p),
              })}
            {/if}
          </div>
        </details>

        <details
          class="rounded-lg border border-zinc-800 p-4"
          ontoggle={(e) => {
            if ((e.currentTarget as HTMLDetailsElement).open && !job.entries)
              void job.loadEntries(1);
          }}
        >
          <summary class="cursor-pointer text-sm font-semibold text-zinc-200"
            >Images</summary
          >
          <div class="mt-2 space-y-2 text-xs">
            <div class="flex flex-wrap gap-2">
              {#each [['split', 'Split'], ['label_state', 'Label state'], ['status', 'Status']] as [key, label] (key)}
                <input
                  class="input input-sm w-36"
                  placeholder={label}
                  aria-label="Filter images by {label.toLowerCase()}"
                  value={job.entryFilters[key as 'split' | 'label_state' | 'status']}
                  onchange={(e) => {
                    job.entryFilters = {
                      ...job.entryFilters,
                      [key]: (e.currentTarget as HTMLInputElement).value.trim(),
                    };
                    void job.loadEntries(1);
                  }}
                />
              {/each}
            </div>
            {#if job.entries}
              <table class="w-full text-left">
                <thead class="text-zinc-500">
                  <tr>
                    <th class="py-1 pr-3">File</th>
                    <th class="py-1 pr-3">Split</th>
                    <th class="py-1 pr-3">Label state</th>
                    <th class="py-1 pr-3">Status</th>
                    <th class="py-1">Error</th>
                  </tr>
                </thead>
                <tbody>
                  {#each job.entries.items as e (e.rel_path)}
                    <tr class="border-t border-zinc-800">
                      <td class="py-1 pr-3 font-mono">{e.rel_path}</td>
                      <td class="py-1 pr-3">{e.split ?? '—'}</td>
                      <td class="py-1 pr-3">{e.label_state}</td>
                      <td class="py-1 pr-3">{e.status}</td>
                      <td class="py-1 text-red-300">{e.error_kind ?? ''}</td>
                    </tr>
                  {/each}
                </tbody>
              </table>
              {@render Pager({
                page: job.entriesPage,
                total: job.entries.total,
                go: (p: number) => void job.loadEntries(p),
              })}
            {/if}
          </div>
        </details>
        {#if job.tablesError}<p class="text-xs text-red-300">{job.tablesError}</p>{/if}

        {#if confirm === 'cancel'}
          <ConfirmDialog
            title="Cancel the import"
            confirmLabel="Cancel import"
            danger
            busy={job.busy}
            onconfirm={() => void doConfirm()}
            oncancel={() => (confirm = null)}
          >
            <p>Stop this import after the chunk in progress. It can be resumed later.</p>
            {#if job.actionError}<p class="text-red-300">{job.actionError}</p>{/if}
          </ConfirmDialog>
        {:else if confirm === 'resume'}
          <ConfirmDialog
            title="Resume the import"
            confirmLabel="Resume"
            busy={job.busy}
            onconfirm={() => void doConfirm()}
            oncancel={() => (confirm = null)}
          >
            <p>Resume from the first chunk that did not finish.</p>
            {#if job.actionError}<p class="text-red-300">{job.actionError}</p>{/if}
          </ConfirmDialog>
        {:else if confirm === 'undo'}
          <ConfirmDialog
            title="Undo the import"
            confirmLabel="Undo import"
            danger
            busy={job.busy}
            confirmDisabled={!job.undoReport}
            onconfirm={() => void doConfirm()}
            oncancel={() => (confirm = null)}
          >
            <label class="flex items-center gap-2">
              <input
                type="checkbox"
                checked={undoChoices.remove_images}
                onchange={(e) =>
                  setUndoChoice(
                    'remove_images',
                    (e.currentTarget as HTMLInputElement).checked,
                  )}
              />
              Remove images this import added
            </label>
            <label class="flex items-center gap-2">
              <input
                type="checkbox"
                checked={undoChoices.deprecate_created_classes}
                onchange={(e) =>
                  setUndoChoice(
                    'deprecate_created_classes',
                    (e.currentTarget as HTMLInputElement).checked,
                  )}
              />
              Deprecate classes this import created
            </label>
            {#if job.undoReport}
              {@const r = job.undoReport}
              <dl
                class="grid grid-cols-2 gap-x-4 gap-y-0.5 text-xs"
                data-testid="undo-report"
              >
                <dt class="text-zinc-500">Items deleted</dt>
                <dd class="font-mono">{r.items_deleted}</dd>
                <dt class="text-zinc-500">Items restored</dt>
                <dd class="font-mono">{r.items_restored}</dd>
                <dt class="text-zinc-500">Labels removed</dt>
                <dd class="font-mono">{r.class_labels_removed}</dd>
                <dt class="text-zinc-500">Boxes removed</dt>
                <dd class="font-mono">{r.boxes_removed}</dd>
                <dt class="text-zinc-500">Proposals deleted</dt>
                <dd class="font-mono">{r.proposals_deleted}</dd>
                <dt class="text-zinc-500">Holdout flags cleared</dt>
                <dd class="font-mono">{r.holdout_flags_cleared}</dd>
                <dt class="text-zinc-500">Images deleted</dt>
                <dd class="font-mono">{r.images_deleted}</dd>
                <dt class="text-zinc-500">Images kept</dt>
                <dd class="font-mono">{r.images_kept}</dd>
              </dl>
              {#if r.items_kept_human_edited > 0 || r.boxes_kept_human_edited > 0}
                <p class="text-emerald-300" data-testid="undo-kept">
                  Your edits are kept: {r.items_kept_human_edited} items and
                  {r.boxes_kept_human_edited} boxes you changed stay as they are.
                </p>
              {/if}
              {#if r.classes_deprecated.length > 0}
                <p class="text-xs text-zinc-400">
                  Classes to deprecate: {r.classes_deprecated.join(', ')}
                </p>
              {/if}
            {:else if !job.actionError}
              <p class="text-xs text-zinc-500">Counting what the undo would change…</p>
            {/if}
            {#if job.actionError}<p class="text-red-300">{job.actionError}</p>{/if}
          </ConfirmDialog>
        {/if}

        {#if pendingStep}
          <ConfirmDialog
            title={pendingStep.action}
            busy={job.busy}
            onconfirm={() => void doStep()}
            oncancel={() => (pendingStep = null)}
          >
            <p>{pendingStep.reason}</p>
            <p class="font-mono text-xs text-zinc-500">
              {pendingStep.method}
              {pendingStep.path}
            </p>
            {#if job.actionError}<p class="text-red-300">{job.actionError}</p>{/if}
          </ConfirmDialog>
        {/if}
      {/if}
    {/snippet}
  </DatasetsGate>
</div>

{#snippet Pager({
  page: p,
  total,
  go,
}: {
  page: number;
  total: number;
  go: (p: number) => void;
})}
  {@const pages = Math.max(1, Math.ceil(total / PAGE_SIZE))}
  <div class="flex items-center gap-2 text-xs text-zinc-400">
    <button type="button" class="btn btn-sm" disabled={p <= 1} onclick={() => go(p - 1)}
      >Prev</button
    >
    <span>page {p} of {pages} · {total.toLocaleString()} total</span>
    <button
      type="button"
      class="btn btn-sm"
      disabled={p >= pages}
      onclick={() => go(p + 1)}>Next</button
    >
  </div>
{/snippet}
