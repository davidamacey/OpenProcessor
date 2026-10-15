<script lang="ts">
  /**
   * A started assist run: its status and progress while it works (including
   * the wait for the region worker after a segment action), and the served
   * result counts when it ends. The proposal itself is the proposal card.
   */
  import { formatCount } from '$lib/format/count';
  import type { AssistRunState } from '$lib/types_assist';

  interface Props {
    run: AssistRunState;
  }

  let { run }: Props = $props();

  const STATUS_TEXT: Record<string, string> = {
    queued: 'Queued',
    running: 'Running',
    waiting_region: 'Waiting for the region worker',
    completed: 'Completed',
    completed_with_errors: 'Completed with errors',
    failed: 'Failed',
    cancelled: 'Cancelled',
    interrupted: 'Interrupted by an API restart',
  };

  const processed = $derived(run.progress?.processed ?? 0);
  const total = $derived(run.progress?.total ?? 0);
  const result = $derived(run.result);
  const segment = $derived(result?.segment ?? null);
</script>

<div
  class="flex flex-col gap-2 text-xs"
  data-testid="assist-progress"
  data-status={run.status}
>
  <p class="text-sm text-zinc-100" data-testid="assist-status">
    {STATUS_TEXT[run.status] ?? run.status}
    {#if total > 0}
      <span class="font-mono text-zinc-400"
        >{formatCount(processed)}/{formatCount(total)}</span
      >
    {/if}
  </p>
  {#if total > 0}
    <div class="h-1.5 rounded bg-zinc-800">
      <div
        class="h-1.5 rounded bg-sky-500/70"
        style="width: {Math.min(100, (processed / total) * 100)}%"
      ></div>
    </div>
  {/if}
  {#if run.status === 'waiting_region'}
    <p class="text-zinc-400" data-testid="assist-waiting-region">
      The items are queued for the region worker. This run continues when it has written
      their boxes{segment
        ? ` (${formatCount(segment.done)} of ${formatCount(segment.queued)} done)`
        : ''}.
    </p>
  {/if}
  {#if run.error}
    <p class="text-red-300" data-testid="assist-run-error">{run.error}</p>
  {/if}
  {#if result}
    <ul class="flex flex-col gap-0.5 text-zinc-300" data-testid="assist-result">
      {#if result.classify}
        <li>
          Class: {formatCount(result.classify.predicted)} answered, {formatCount(
            result.classify.updated,
          )} written, {formatCount(result.classify.locked_suggest_only)} kept as suggestions
          on locked items, {formatCount(result.classify.failed)} failed, {formatCount(
            result.classify.unreadable,
          )} unreadable.
        </li>
      {/if}
      {#if result.attributes}
        <li>
          Attributes: {formatCount(result.attributes.items_written)} items,
          {formatCount(result.attributes.invalid_values)} values outside the pack's list dropped.
          Suggestions only.
        </li>
      {/if}
      {#if result.verify}
        <li>
          Verify: {formatCount(result.verify.boxes)} boxes judged, {formatCount(
            result.verify.accepted,
          )} accepted, {formatCount(result.verify.rejected)} rejected, {formatCount(
            result.verify.locked_skipped,
          )} locked and skipped.
        </li>
      {/if}
      {#if segment}
        <li>
          Segment: {formatCount(segment.queued)} queued, {formatCount(segment.done)} done,
          {formatCount(segment.pending)} still pending, {formatCount(segment.failed)} failed.
        </li>
      {/if}
    </ul>
  {/if}
</div>
