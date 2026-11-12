<script lang="ts">
  /**
   * The served region-detection worklog, as one block: the pending counts, the `drained` stability
   * verdict, a stall reason and the models that are not ready. Presentational: the page (or the
   * poller beside it) owns the read, so this adds no request.
   */
  import { formatTime } from '$lib/format/time';
  import type { RegionDrain } from '$lib/types_ingest';

  interface Props {
    drain: RegionDrain | null;
    error: string | null;
    lastChecked: number | null;
  }
  let { drain, error, lastChecked }: Props = $props();
</script>

<div>
  <h3 class="mb-2 text-sm font-semibold text-fg-1">Region detection worklog</h3>
  {#if error}
    <p class="text-xs text-danger-text">
      Detection worklog unavailable (backend: {error})
    </p>
  {:else if !drain}
    <p class="text-xs text-fg-3">Loading…</p>
  {:else}
    <div class="flex gap-4 text-xs">
      <div>
        <div class="text-fg-3">Pending detection</div>
        <div class="font-mono text-fg-1">{drain.pending_detection}</div>
      </div>
      <div>
        <div class="text-fg-3">Pending verification</div>
        <div class="font-mono text-fg-1">{drain.pending_verification}</div>
      </div>
      <div>
        <div class="text-fg-3">Total unfinished</div>
        <div
          class="font-mono {drain.total_unfinished === 0
            ? 'text-success-text'
            : 'text-fg-1'}"
          data-testid="region-drain-total-unfinished"
        >
          {drain.total_unfinished}
        </div>
      </div>
      <div>
        <div class="text-fg-3">Drained</div>
        <div
          class="font-mono {drain.drained ? 'text-success-text' : 'text-fg-1'}"
          data-testid="region-drain-drained"
        >
          {drain.drained ? 'yes' : 'no'}
        </div>
      </div>
    </div>
    {#if drain.stall_reason}
      <!-- V-1: why the worklog can't drain, served verbatim. -->
      <p
        class="mt-2 rounded border border-warn/40 bg-warn/10 px-2 py-1 text-xs text-warn-text"
        data-testid="region-drain-stall-reason"
      >
        Stalled: {drain.stall_reason}
      </p>
    {/if}
    {#if drain.region_dependencies && drain.region_dependencies.some((d) => !d.ready)}
      <ul class="mt-2 space-y-0.5 text-caption" data-testid="region-drain-dependencies">
        {#each drain.region_dependencies.filter((d) => !d.ready) as dep (dep.role + dep.model)}
          <li class="text-fg-2">
            <span class="text-fg-2">{dep.role}</span>
            <span class="font-mono">{dep.model}</span>: not ready — {dep.detail}
          </li>
        {/each}
      </ul>
    {/if}
    <p class="mt-1 text-caption text-fg-3">
      Last checked: {formatTime(lastChecked)}
      {#if drain.drained}
        · stable for {Math.round(drain.stable_for_s)}s
      {/if}
    </p>
  {/if}
</div>
