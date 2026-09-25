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
  class="mb-6 rounded-lg border border-zinc-800 bg-zinc-900/50 p-4 text-sm"
  data-testid="run-status"
>
  <div class="flex flex-wrap items-center gap-2">
    <span class="text-zinc-400">Job</span>
    <code class="text-zinc-200">{jobId}</code>
    <span
      class="rounded px-1.5 py-0.5 text-xs {status?.state === 'error'
        ? 'bg-red-900/60 text-red-200'
        : status?.state === 'done'
          ? 'bg-emerald-900/60 text-emerald-200'
          : 'bg-zinc-800 text-zinc-300'}"
      data-testid="run-state">{status?.state ?? 'enqueued'}</span
    >
    {#if status?.profile}<span class="text-xs text-zinc-500"
        >profile <span class="font-mono">{status.profile}</span></span
      >{/if}
  </div>

  {#if progress && progress.total > 0}
    <div class="mt-3">
      {#if !isTerminal(status?.state)}
        <div class="h-2 w-full overflow-hidden rounded bg-zinc-800">
          <div class="h-full bg-emerald-600 transition-all" style="width: {pct}%"></div>
        </div>
      {/if}
      <p class="mt-1 text-xs text-zinc-500" data-testid="run-progress">
        {progress.done} / {progress.total} evaluations
      </p>
    </div>
  {/if}

  {#if status?.error}
    <p class="mt-3 text-xs text-red-300" data-testid="run-job-error">
      Job failed: {status.error}
    </p>
  {/if}
  {#if status?.failed?.length}
    <div
      class="mt-3 rounded border border-red-800 bg-red-950/60 p-2 text-xs text-red-200"
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
    <details class="mt-3 text-xs" open={hasNotes} data-testid="enqueue-summary">
      <summary class="cursor-pointer text-zinc-400">
        Class mapping at enqueue{mine.warnings.length
          ? ` · ${mine.warnings.length} warning${mine.warnings.length === 1 ? '' : 's'}`
          : ''}
      </summary>
      {#if mine.warnings.length}
        <ul class="mt-2 list-disc pl-5 text-amber-300">
          {#each mine.warnings as w, i (i)}<li>{w}</li>{/each}
        </ul>
      {/if}
      <table class="mt-2 w-full">
        <thead class="text-left text-[10px] uppercase text-zinc-500">
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
              <tr class="border-t border-zinc-800 align-top">
                <td class="py-1 pr-2 font-mono">{m.display_name}</td>
                <td class="py-1 pr-2 font-mono">{datasetName(dsId)}</td>
                <td class="py-1 pr-2">{cm.method}</td>
                <td class="py-1 text-zinc-400">
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
                  {#each cm.warnings as w, i (i)}<div class="text-amber-300">
                      {w}
                    </div>{/each}
                  {#if hasOverlap(overlap)}
                    <div class="text-red-300">overlap: {formatOverlap(overlap)}</div>
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
