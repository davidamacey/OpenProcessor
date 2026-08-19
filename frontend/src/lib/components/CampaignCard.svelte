<script lang="ts">
  /**
   * Campaign status display: shows every run row in `runs[]` and a
   * "Promote best" action that hands the highest-mAP50 finished run
   * back to the parent.
   */
  import type { TrainJobStatus } from '$lib/types_train';

  interface Props {
    campaignId: string;
    runs: TrainJobStatus[];
    /** Cancel every non-terminal run. */
    onCancel?: (campaignId: string) => void | Promise<void>;
    /** Promote the highest-mAP finished run. */
    onPromoteBest?: (best: TrainJobStatus) => void;
    cancelling?: boolean;
  }

  let { campaignId, runs, onCancel, onPromoteBest, cancelling = false }: Props = $props();

  function statePillClass(s: string): string {
    switch (s) {
      case 'running':
      case 'starting':
        return 'bg-blue-500/20 text-blue-200 border-blue-500/40';
      case 'exporting':
        return 'bg-purple-500/20 text-purple-200 border-purple-500/40';
      case 'finished':
        return 'bg-green-500/20 text-green-200 border-green-500/40';
      case 'failed':
        return 'bg-red-500/20 text-red-200 border-red-500/40';
      case 'cancelled':
      case 'skipped':
        return 'bg-zinc-700 text-zinc-300 border-zinc-600';
      case 'lost':
        return 'bg-yellow-500/20 text-yellow-200 border-yellow-500/40';
      case 'queued':
      default:
        return 'bg-zinc-800 text-zinc-300 border-zinc-700';
    }
  }

  // Best run = finished + highest map50.
  const best = $derived.by(() => {
    let winner: TrainJobStatus | null = null;
    let winnerScore = -1;
    for (const r of runs) {
      if (r.state !== 'finished') continue;
      const m = r.best_metric?.map50;
      if (typeof m === 'number' && m > winnerScore) {
        winner = r;
        winnerScore = m;
      }
    }
    return winner;
  });

  const hasActive = $derived(
    runs.some((r) => ['queued', 'starting', 'running', 'exporting'].includes(r.state)),
  );

  // Sort: keep submission order within a campaign (ids are
  // `<campaign>_run00_<profile>`).
  const ordered = $derived([...runs].sort((a, b) => a.job_id.localeCompare(b.job_id)));
</script>

<section class="rounded-md border border-zinc-800 bg-zinc-900 p-4">
  <header class="mb-3 flex flex-wrap items-center gap-3">
    <h2 class="font-mono text-sm text-white">{campaignId}</h2>
    <span
      class="rounded-sm border border-blue-500/40 bg-blue-500/10 px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-blue-200"
    >
      campaign
    </span>
    <span class="grow"></span>
    {#if best && onPromoteBest}
      <button
        type="button"
        class="btn btn-primary"
        onclick={() => onPromoteBest(best)}
        disabled={cancelling}
        title="Promote {best.job_id} (best mAP50 {best.best_metric?.map50?.toFixed(3)})"
      >
        Promote best
      </button>
    {/if}
    {#if hasActive && onCancel}
      <button
        type="button"
        class="btn btn-danger"
        onclick={() => onCancel(campaignId)}
        disabled={cancelling}
      >
        {cancelling ? 'Cancelling…' : 'Cancel campaign'}
      </button>
    {/if}
  </header>

  <ul class="divide-y divide-zinc-800 rounded border border-zinc-800 bg-zinc-950">
    {#each ordered as r (r.job_id)}
      <li
        class="grid grid-cols-1 gap-2 px-3 py-2 text-sm sm:grid-cols-[auto_8rem_auto_1fr]"
      >
        <span
          class="rounded-sm border px-1.5 py-0.5 text-center text-[10px] font-medium uppercase tracking-wide {statePillClass(
            r.state,
          )}"
        >
          {r.state}
        </span>
        <span class="truncate font-mono text-xs text-zinc-400" title={r.job_id}>
          {r.job_id}
        </span>
        <span class="font-mono text-xs text-zinc-300">
          {r.current_epoch ?? '—'} / {r.total_epochs ?? '—'} epochs
        </span>
        <span class="font-mono text-xs text-zinc-300">
          mAP50 {r.best_metric?.map50?.toFixed(3) ?? '—'}
          <span class="text-zinc-500">·</span>
          mAP50-95 {r.best_metric?.map50_95?.toFixed(3) ?? '—'}
        </span>
      </li>
    {/each}
  </ul>

  {#if best}
    <p class="mt-3 text-xs text-zinc-400">
      Best so far:
      <span class="font-mono text-zinc-200">{best.job_id}</span>
      <span class="text-zinc-500">·</span>
      mAP50
      <span class="font-mono text-zinc-200">{best.best_metric?.map50?.toFixed(3)}</span>
    </p>
  {/if}
</section>
