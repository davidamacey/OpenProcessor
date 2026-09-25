<script lang="ts">
  /**
   * Shows `pending_detection`/`pending_verification`/`total_unfinished`
   * plus the BA-3 `drained` stability verdict and "last checked"
   * (docs/design/ingest-ui-and-acceptance-plan-2026-09-24.md §A.2).
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
  import { ApiError, getRegionDrain } from '$lib/api';
  import type { RegionDrain } from '$lib/types';

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
      error = e instanceof ApiError ? (e.detail ?? e.message) : (e as Error).message;
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

  function formatTime(t: number | null): string {
    if (t === null) return '—';
    return new Date(t).toLocaleTimeString();
  }
</script>

<div>
  <h3 class="mb-2 text-sm font-semibold text-zinc-200">Region detection worklog</h3>
  {#if error}
    <p class="text-xs text-red-300">Detection worklog unavailable (backend: {error})</p>
  {:else if !drain}
    <p class="text-xs text-zinc-500">Loading…</p>
  {:else}
    <div class="flex gap-4 text-xs">
      <div>
        <div class="text-zinc-500">Pending detection</div>
        <div class="font-mono text-zinc-200">{drain.pending_detection}</div>
      </div>
      <div>
        <div class="text-zinc-500">Pending verification</div>
        <div class="font-mono text-zinc-200">{drain.pending_verification}</div>
      </div>
      <div>
        <div class="text-zinc-500">Total unfinished</div>
        <div
          class="font-mono {drain.total_unfinished === 0
            ? 'text-emerald-400'
            : 'text-zinc-200'}"
          data-testid="region-drain-total-unfinished"
        >
          {drain.total_unfinished}
        </div>
      </div>
      <div>
        <div class="text-zinc-500">Drained</div>
        <div
          class="font-mono {drain.drained ? 'text-emerald-400' : 'text-zinc-200'}"
          data-testid="region-drain-drained"
        >
          {drain.drained ? 'yes' : 'no'}
        </div>
      </div>
    </div>
    <p class="mt-1 text-[11px] text-zinc-500">
      Last checked: {formatTime(lastChecked)}
      {#if drain.drained}
        · stable for {Math.round(drain.stable_for_s)}s
      {/if}
    </p>
  {/if}
</div>
