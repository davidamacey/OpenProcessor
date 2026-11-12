<script lang="ts">
  /**
   * The tracked comparison job: served state, progress and every failed
   * piece; plus, for a job enqueued from this page, the enqueue-time class
   * mapping per model × dataset and the served warnings.
   */
  import type { BakeoffRunAccepted, BakeoffStatus } from '$lib/types_bakeoff';
  import { failureWhere, formatOverlap, hasOverlap, isTerminal } from '$lib/bakeoff/view';

  interface Props {
    jobId: string;
    status: BakeoffStatus | null;
    accepted: BakeoffRunAccepted | null;
    datasetName: (id: string) => string;
  }
  let { jobId, status, accepted, datasetName }: Props = $props();

  const progress = $derived(status?.progress ?? null);
  const pct = $derived(
    progress && progress.total > 0
      ? Math.round((progress.done / progress.total) * 100)
      : 0,
  );
  const mine = $derived(accepted?.job_id === jobId ? accepted : null);
  // Open the mapping notes when there is anything to read in them.
  const hasNotes = $derived(
    !!mine &&
      (mine.warnings.length > 0 ||
        mine.models.some(
          (m) =>
            Object.values(m.class_mapping).some(
              (cm) =>
                cm.warnings.length > 0 ||
                cm.not_covered_eval_classes.length > 0 ||
                cm.unmapped_model_classes.length > 0,
            ) || Object.values(m.train_test_overlap ?? {}).some((o) => hasOverlap(o)),
        )),
  );
</script>

<section
  class="mb-6 rounded-lg border border-line bg-surface-1/50 p-4 text-body"
  data-testid="run-status"
>
  <div class="flex flex-wrap items-center gap-2">
    <span class="text-fg-2">Job</span>
    <code class="text-fg-1">{jobId}</code>
    <span
      class="rounded px-1.5 py-0.5 text-meta {status?.state === 'error'
        ? 'bg-danger-soft text-danger-text'
        : status?.state === 'done'
          ? 'bg-success-soft text-success-text'
          : 'bg-surface-2 text-fg-2'}"
      data-testid="run-state">{status?.state ?? 'enqueued'}</span
    >
    {#if status?.profile}<span class="text-meta text-fg-3"
        >profile <span class="font-mono">{status.profile}</span></span
      >{/if}
  </div>

  {#if progress && progress.total > 0}
    <div class="mt-3">
      {#if !isTerminal(status?.state)}
        <div class="h-2 w-full overflow-hidden rounded bg-surface-2">
          <div class="h-full bg-success transition-all" style="width: {pct}%"></div>
        </div>
      {/if}
      <p class="mt-1 text-meta text-fg-3" data-testid="run-progress">
        {progress.done} / {progress.total} evaluations
      </p>
    </div>
  {/if}

  {#if status?.error}
    <p class="mt-3 text-meta text-danger-text" data-testid="run-job-error">
      Job failed: {status.error}
    </p>
  {/if}
  {#if status?.failed?.length}
    <div
      class="mt-3 rounded border border-danger-line bg-danger-soft p-2 text-meta text-danger-text"
    >
      <p class="mb-1 font-medium">
        {status.failed.length} failed piece{status.failed.length === 1 ? '' : 's'}
      </p>
      <ul class="max-h-40 space-y-0.5 overflow-auto font-mono" data-testid="run-failures">
        {#each status.failed as f, i (i)}
          <li>{failureWhere(f)} — {f.error}</li>
        {/each}
      </ul>
    </div>
  {/if}

  {#if mine}
    <details class="mt-3 text-meta" open={hasNotes} data-testid="enqueue-summary">
      <summary class="cursor-pointer text-fg-2">
        Class mapping at enqueue{mine.warnings.length
          ? ` · ${mine.warnings.length} warning${mine.warnings.length === 1 ? '' : 's'}`
          : ''}
      </summary>
      {#if mine.warnings.length}
        <ul class="mt-2 list-disc pl-5 text-warn-text">
          {#each mine.warnings as w, i (i)}<li>{w}</li>{/each}
        </ul>
      {/if}
      <table class="mt-2 w-full">
        <thead class="text-left text-caption uppercase text-fg-3">
          <tr>
            <th class="py-1 pr-2">Model</th>
            <th class="py-1 pr-2">Dataset</th>
            <th class="py-1 pr-2">Method</th>
            <th class="py-1">Notes</th>
          </tr>
        </thead>
        <tbody>
          {#each mine.models as m (m.model)}
            {#each Object.entries(m.class_mapping) as [dsId, cm] (dsId)}
              {@const overlap = m.train_test_overlap?.[dsId]}
              <tr class="border-t border-line align-top">
                <td class="py-1 pr-2 font-mono">{m.display_name}</td>
                <td class="py-1 pr-2 font-mono">{datasetName(dsId)}</td>
                <td class="py-1 pr-2">{cm.method}</td>
                <td class="py-1 text-fg-2">
                  {#if cm.not_covered_eval_classes.length}
                    <div>
                      not covered: {cm.not_covered_eval_classes
                        .map((c) => c.name)
                        .join(', ')}
                    </div>
                  {/if}
                  {#if cm.unmapped_model_classes.length}
                    <div>
                      unmapped model classes: {cm.unmapped_model_classes
                        .map((c) => c.name ?? `#${c.model_class_id}`)
                        .join(', ')}
                    </div>
                  {/if}
                  {#each cm.warnings as w, i (i)}<div class="text-warn-text">
                      {w}
                    </div>{/each}
                  {#if hasOverlap(overlap)}
                    <div class="text-danger-text">overlap: {formatOverlap(overlap)}</div>
                  {/if}
                </td>
              </tr>
            {/each}
          {/each}
        </tbody>
      </table>
    </details>
  {/if}
</section>
