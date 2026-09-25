<script lang="ts">
  /**
   * Wraps `<AutoLabelPanel>` with the ingest-aware gate
   * (docs/design/ingest-ui-and-acceptance-plan-2026-09-24.md §A.5):
   * blocked while the upload run is active, or while the served region
   * drain has unfinished work, or while the drain fetch itself is
   * failing. No client stability window — the operator decides, and the
   * note shows exactly when the served zero was observed.
   */
  import AutoLabelPanel from '$components/AutoLabelPanel.svelte';
  import type { IngestRunState } from '$lib/ingest/ingestRunController.svelte';
  import type { RegionDrain } from '$lib/types';

  interface Props {
    runState: IngestRunState;
    drain: RegionDrain | null;
    drainError: boolean;
    drainObservedAt: number | null;
  }
  let { runState, drain, drainError, drainObservedAt }: Props = $props();

  function formatTime(t: number | null): string {
    if (t === null) return '';
    return new Date(t).toLocaleTimeString();
  }

  const gate = $derived.by(() => {
    if (
      runState === 'uploading' ||
      runState === 'paused' ||
      runState === 'prefiltering'
    ) {
      return { blocked: true, reason: 'Finish or cancel the upload first' };
    }
    if (drainError) {
      return { blocked: true, reason: 'Worklog unavailable' };
    }
    if (drain && drain.total_unfinished > 0) {
      return {
        blocked: true,
        reason: `Region detection still has ${drain.total_unfinished} items queued (served worklog)`,
      };
    }
    return null;
  });

  const note = $derived(
    !gate && drainObservedAt !== null
      ? `Worklog empty as of ${formatTime(drainObservedAt)}`
      : null,
  );
</script>

<div class="space-y-2">
  {#if note}
    <p class="text-[11px] text-zinc-500">{note}</p>
  {/if}
  <AutoLabelPanel {gate} />
</div>
