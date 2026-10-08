<script lang="ts">
  /**
   * Wraps `<AutoLabelPanel>` with the ingest-aware gate
   * (docs/design/ingest-ui-and-acceptance-plan-2026-09-24.md §A.5):
   * blocked while the upload run is active, or while the served region
   * drain isn't `drained`, or while the drain fetch itself is failing.
   *
   * BA-3 (landed, OpenProcessor c5c606f): the gate reads the server's own
   * `drained` verdict (`total_unfinished` read 0 for
   * `IngestConfig.region_drain.stable_polls` consecutive polls) instead
   * of a raw `total_unfinished === 0` reading — no client-side stability
   * window either way, the server now computes the same one every client
   * used to have to invent independently.
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
    if (drain && !drain.drained) {
      return {
        blocked: true,
        reason:
          drain.total_unfinished > 0
            ? `Region detection still has ${drain.total_unfinished} items queued (served worklog)`
            : `Worklog just reached zero — waiting for the served stability verdict`,
      };
    }
    return null;
  });

  const note = $derived(
    !gate && drainObservedAt !== null
      ? `Worklog drained as of ${formatTime(drainObservedAt)}`
      : null,
  );
</script>

<div class="space-y-2">
  {#if note}
    <p class="text-[11px] text-zinc-500">{note}</p>
  {/if}
  <AutoLabelPanel {gate} />
</div>
