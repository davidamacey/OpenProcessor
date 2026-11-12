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
  import RegionDrainDetails from '$lib/components/ingest/RegionDrainDetails.svelte';
  import type { RegionDrain } from '$lib/types_ingest';

  interface Props {
    pollIntervalS?: number;
    /** The rail's one-line view; the full worklog is `RegionDrainDetails` on the Activity tab. */
    compact?: boolean;
    onUpdate?: (drain: RegionDrain | null, observedAt: number) => void;
  }
  let { pollIntervalS = 10, compact = false, onUpdate }: Props = $props();

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

{#if compact}
  <div data-testid="region-drain-compact">
    <h3 class="mb-1 text-body font-semibold text-fg-1">Pending detection</h3>
    {#if error}
      <p class="text-meta text-danger-text">
        Detection worklog unavailable (backend: {error})
      </p>
    {:else if !drain}
      <p class="text-meta text-fg-3">Loading…</p>
    {:else}
      <p class="text-meta text-fg-2">
        <span class="font-mono text-title text-fg-1" data-testid="region-drain-pending"
          >{drain.pending_detection}</span
        >
        images waiting for the detector{drain.drained ? ' (drained)' : ''}
      </p>
    {/if}
  </div>
{:else}
  <RegionDrainDetails {drain} {error} {lastChecked} />
{/if}
