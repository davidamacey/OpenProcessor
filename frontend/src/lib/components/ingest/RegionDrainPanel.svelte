<script lang="ts">
  import { apiErrorText } from '$lib/api_errors';
  /**
   * Shows `pending_detection`/`pending_verification`/`total_unfinished`
   * plus the BA-3 `drained` stability verdict and "last checked"
   * (Ingest UI plan §A.2).
   * Polls at `config.regionDrainPollIntervalS` — BA-2's served
   * `region_drain.poll_interval_s` once `/ingest/config` is wired up by
   * the page, else the documented interim default (10s). Stops polling
   * while the page is hidden.
   *
   * §A.5: the drain gate that decides whether clustering can start reads
   * this same served value (including `drained`) through `onUpdate`, not
   * a second fetch.
   */
  import { onDestroy, onMount } from 'svelte';
  import { getRegionDrain } from '$lib/api_ingest';
  import { formatTime } from '$lib/format/time';
  import type { RegionDrain } from '$lib/types_ingest';

  interface Props {
    pollIntervalS?: number;
    onUpdate?: (drain: RegionDrain | null, observedAt: number) => void;
  }
  let { pollIntervalS = 10, onUpdate }: Props = $props();

  let drain = $state<RegionDrain | null>(null);
  let error = $state<string | null>(null);
  let lastChecked = $state<number | null>(null);
  let intervalId: ReturnType<typeof setInterval> | undefined;

  async function load(): Promise<void> {
    try {
      drain = await getRegionDrain();
      error = null;
      lastChecked = Date.now();
      onUpdate?.(drain, lastChecked);
    } catch (e) {
      error = apiErrorText(e);
      onUpdate?.(null, Date.now());
    }
  }

  function onVisibilityChange(): void {
    if (document.visibilityState === 'visible') void load();
  }

  onMount(() => {
    void load();
    intervalId = setInterval(() => {
      if (document.visibilityState === 'visible') void load();
    }, pollIntervalS * 1000);
    document.addEventListener('visibilitychange', onVisibilityChange);
  });

  onDestroy(() => {
    if (intervalId) clearInterval(intervalId);
    document.removeEventListener('visibilitychange', onVisibilityChange);
  });
</script>

<div>
  <h3 class="mb-2 text-sm font-semibold text-fg-1">Region detection worklog</h3>
  {#if error}
    <p class="text-xs text-danger">Detection worklog unavailable (backend: {error})</p>
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
          class="font-mono {drain.total_unfinished === 0 ? 'text-success' : 'text-fg-1'}"
          data-testid="region-drain-total-unfinished"
        >
          {drain.total_unfinished}
        </div>
      </div>
      <div>
        <div class="text-fg-3">Drained</div>
        <div
          class="font-mono {drain.drained ? 'text-success' : 'text-fg-1'}"
          data-testid="region-drain-drained"
        >
          {drain.drained ? 'yes' : 'no'}
        </div>
      </div>
    </div>
    {#if drain.stall_reason}
      <!-- V-1: why the worklog can't drain, served verbatim. -->
      <p
        class="mt-2 rounded border border-warn/40 bg-warn/10 px-2 py-1 text-xs text-warn"
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
