<script lang="ts">
  /** One re-index run as served: its job status and the documents done, failed and skipped of the total. */
  import type { EmbeddingReindexRun } from '$lib/types_embeddingSpaces';

  interface Props {
    run: EmbeddingReindexRun;
  }

  let { run }: Props = $props();
</script>

<div class="flex flex-col gap-0.5 text-meta" data-testid="run-progress" data-run={run.id}>
  <div class="flex flex-wrap items-center gap-2">
    <span
      class="rounded border border-line-strong px-1.5 text-caption text-fg-1"
      data-testid="run-status">{run.job_status ?? 'waiting'}</span
    >
    <span class="text-fg-2">{run.mode}</span>
  </div>
  {#if run.docs_total != null}
    <progress
      class="h-1.5 w-40 max-w-full"
      value={run.docs_done}
      max={run.docs_total}
      aria-label="Documents embedded"
    ></progress>
  {/if}
  <span class="text-fg-2" data-testid="run-counts"
    >{run.docs_done}{run.docs_total != null ? ` of ${run.docs_total}` : ''} done · {run.docs_failed}
    failed · {run.docs_skipped} skipped</span
  >
</div>
