<script lang="ts">
  /**
   * Campaign status display: shows every run row in `runs[]` and a
   * "Promote best" action that hands the highest-mAP50 finished run
   * back to the parent.
   */
  import type { TrainJobStatus } from '$lib/types_train';
  import { bestMapDisplay } from '$lib/trainRunsTable';

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
        return 'bg-accent/20 text-accent-text border-accent/40';
      case 'exporting':
        return 'bg-holdout/20 text-holdout-text border-holdout/40';
      case 'finished':
        return 'bg-success/20 text-success-text border-success/40';
      case 'failed':
        return 'bg-danger/20 text-danger-text border-danger/40';
      case 'cancelled':
      case 'skipped':
        return 'bg-surface-3 text-fg-2 border-line-strong';
      case 'lost':
        return 'bg-warn/20 text-warn-text border-warn/40';
      case 'queued':
      default:
        return 'bg-surface-2 text-fg-2 border-line-strong';
    }
  }

  // Best run = finished + highest served eval.map50 (the run's own
  // headline number — see trainRunsTable.ts's bestMapDisplay).
  const best = $derived.by(() => {
    let winner: TrainJobStatus | null = null;
    let winnerScore = -1;
    for (const r of runs) {
      if (r.state !== 'finished') continue;
      const m = bestMapDisplay(r).value;
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

<section class="rounded-md border border-line bg-surface-1 p-4">
  <header class="mb-3 flex flex-wrap items-center gap-3">
    <h2 class="text-body text-white"><code>{campaignId}</code></h2>
    <span
      class="rounded-sm border border-accent/40 bg-accent/10 px-1.5 py-0.5 text-caption text-accent-text"
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
        title="Promote {best.job_id} (mAP50 {bestMapDisplay(best).value?.toFixed(3) ??
          '—'})"
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

  <ul class="divide-y divide-line rounded border border-line bg-surface-0">
    {#each ordered as r (r.job_id)}
      <li
        class="grid grid-cols-1 gap-2 px-3 py-2 text-body sm:grid-cols-[auto_8rem_auto_1fr]"
      >
        <span
          class="rounded-sm border px-1.5 py-0.5 text-center text-caption font-medium capitalize {statePillClass(
            r.state,
          )}"
        >
          {r.state}
        </span>
        <code class="truncate text-meta text-fg-2" title={r.job_id}>
          {r.job_id}
        </code>
        <span class="text-meta text-fg-2 tabular-nums">
          {r.current_epoch ?? '—'} / {r.total_epochs ?? '—'} epochs
        </span>
        <span class="text-meta text-fg-2 tabular-nums">
          mAP50 {bestMapDisplay(r).value?.toFixed(3) ?? '—'}
          <span class="text-fg-3">·</span>
          mAP50-95 {r.eval?.map50_95?.toFixed(3) ?? '—'}
        </span>
      </li>
    {/each}
  </ul>

  {#if best}
    <p class="mt-3 text-meta text-fg-2">
      Best so far:
      <code class="text-fg-1">{best.job_id}</code>
      <span class="text-fg-3">·</span>
      mAP50
      <span class="text-fg-1 tabular-nums"
        >{bestMapDisplay(best).value?.toFixed(3) ?? '—'}</span
      >
    </p>
  {/if}
</section>
