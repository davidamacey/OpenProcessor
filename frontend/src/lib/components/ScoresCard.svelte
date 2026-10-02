<script lang="ts">
  /**
   * "Curation scores" card — `/settings` (docs/design/
   * frontend-coverage-audit-2026-09-24.md §G10). Per-scorer coverage
   * from `GET {API_PREFIX}/scores/coverage`, a "Compute" action (all
   * scorers or a per-scorer selection) that follows `EmbeddingPlot`'s
   * rebuild-job pattern (`EmbeddingPlot.svelte:294-346`: idempotent
   * job-poll, adopt-in-flight-on-mount, explicit Cancel), and
   * `/settings`' own confirm-before-write convention (a deployment-wide
   * operation, never fired straight off a click).
   *
   * A failed coverage load shows the error with a retry.
   *
   * Scorer ids are never hardcoded — every id this card can select or
   * display comes from `Object.keys(coverage)`, per the task brief.
   */

  import {
    ApiError,
    cancelScores,
    computeScores,
    getScoresCoverage,
    getScoresStatus,
    type ScoresCoverage,
    type ScoresJob,
  } from '$lib/api';
  import { focusOnMount } from '$lib/actions/focusOnMount';
  import { trapFocus } from '$lib/actions/trapFocus';
  import {
    classifyScoresPoll,
    formatCoverageCounts,
    formatCoveragePct,
  } from '$lib/scores';
  import { strategiesStore } from '$stores/strategies.svelte';
  import { toastStore } from '$stores/toast.svelte';

  const JOB_POLL_MS = 3000;

  let coverage = $state<ScoresCoverage>({});
  let loading = $state(false);
  let loadError = $state<string | null>(null);

  let selected = $state<Record<string, boolean>>({});
  let job = $state<ScoresJob | null>(null);
  let computeError = $state<string | null>(null);
  let starting = $state(false);
  let cancelling = $state(false);
  let jobPoll: ReturnType<typeof setInterval> | null = null;

  /** `null` = "Compute all" (sends `scorers: null`); an array = the
   *  operator's checked subset. Set by `openConfirm`. */
  let confirmScorers = $state<string[] | null | undefined>(undefined);

  const scorerIds = $derived(Object.keys(coverage));
  const selectedIds = $derived(scorerIds.filter((id) => selected[id]));

  async function loadCoverage(): Promise<void> {
    loading = true;
    try {
      coverage = await getScoresCoverage();
      loadError = null;
    } catch (e) {
      if ((e as Error)?.name === 'AbortError') return;
      loadError = (e as Error)?.message ?? 'failed to load curation-score coverage';
    } finally {
      loading = false;
    }
  }

  /** Adopt a compute job already in flight (another tab, or before a
   *  reload) — same shape as EmbeddingPlot's adoption effect. */
  async function adoptInFlightJob(): Promise<void> {
    try {
      const st = await getScoresStatus();
      if (st.status === 'running') {
        job = st;
        startJobPoll();
      }
    } catch {
      // Transient failure — nothing to adopt.
    }
  }

  function stopJobPoll(): void {
    if (jobPoll) clearInterval(jobPoll);
    jobPoll = null;
  }

  async function pollJob(): Promise<void> {
    let st: ScoresJob;
    try {
      st = await getScoresStatus();
    } catch {
      return; // transient — keep polling
    }
    const outcome = classifyScoresPoll(st.status);
    if (outcome === 'running') {
      job = st;
      return;
    }
    job = null;
    stopJobPoll();
    if (outcome === 'completed') {
      toastStore.success('Curation scores computed.');
      await loadCoverage();
      // /methods' field_coverage is derived from the same OpenSearch
      // fields this job just wrote — invalidate the cache so any
      // mounted StrategyBar picks the now-nonzero coverage up without a
      // full page reload (same call `/settings`' own save flow makes).
      strategiesStore.reset();
      void strategiesStore.init();
    } else if (outcome === 'failed') {
      computeError = st.error ?? 'scoring job failed';
    }
    // 'cancelled' — no toast/error, the operator asked for this.
  }

  function startJobPoll(): void {
    stopJobPoll();
    jobPoll = setInterval(() => void pollJob(), JOB_POLL_MS);
  }

  $effect(() => {
    void (async () => {
      await loadCoverage();
      if (loadError === null) await adoptInFlightJob();
    })();
    return stopJobPoll;
  });

  function toggleScorer(id: string): void {
    selected = { ...selected, [id]: !selected[id] };
  }

  function openConfirmAll(): void {
    confirmScorers = null;
  }

  function openConfirmSelected(): void {
    if (selectedIds.length === 0) return;
    confirmScorers = [...selectedIds];
  }

  function closeConfirm(): void {
    confirmScorers = undefined;
  }

  async function confirmCompute(): Promise<void> {
    if (confirmScorers === undefined || starting || job) return;
    const scorers = confirmScorers;
    starting = true;
    computeError = null;
    try {
      job = await computeScores(scorers);
      startJobPoll();
      toastStore.info(
        scorers === null
          ? 'Computing every enabled curation score…'
          : `Computing ${scorers.join(', ')}…`,
      );
    } catch (e) {
      const detail =
        e instanceof ApiError
          ? (e.detail ?? e.message)
          : ((e as Error)?.message ?? 'failed');
      computeError = detail;
      toastStore.error(`Compute failed: ${detail}`);
    } finally {
      starting = false;
      confirmScorers = undefined;
    }
  }

  async function cancelCompute(): Promise<void> {
    if (cancelling) return;
    cancelling = true;
    try {
      await cancelScores();
      toastStore.info('Curation-score compute cancelled.');
    } catch (e) {
      toastStore.error(`Cancel failed: ${(e as Error).message}`);
    } finally {
      job = null;
      stopJobPoll();
      cancelling = false;
    }
  }
</script>

<section class="surface flex flex-col gap-4 p-5">
  <div class="flex flex-wrap items-center gap-3">
    <h2 class="text-base font-semibold">Curation scores</h2>
    <span class="grow"></span>
    <button
      type="button"
      class="btn"
      onclick={() => void loadCoverage()}
      disabled={loading}
    >
      {loading ? 'Loading…' : 'Reload'}
    </button>
  </div>

  <!-- F8 D9: the Uncertainty and Model Disagreements queues are driven by
         probe predictions (run a probe on /train), not by these scorers. -->
  <p class="text-xs text-zinc-400" data-testid="scores-intro">
    Scores are computed on demand, not automatically. The score-based review sorts (for
    example mistakenness) have nothing to sort by until their scorer has run at least
    once. Mistakenness also needs model probe predictions; if those are missing, Compute
    shows the backend's own error below. The Uncertainty and Model Disagreements queues
    don't depend on these scorers: they fill from probe predictions (run a probe from a
    finished training run on /train).
  </p>

  {#if loadError}
    <p
      class="rounded border border-red-500/40 bg-red-500/10 px-3 py-2 text-xs text-red-200"
    >
      {loadError}
    </p>
  {/if}

  {#if scorerIds.length === 0 && !loading}
    <p class="text-sm text-zinc-500">No scorers advertised.</p>
  {:else}
    <table class="w-full text-left text-sm">
      <thead>
        <tr class="border-b border-zinc-800 text-xs text-zinc-500">
          <th class="w-8 py-1.5 font-normal"></th>
          <th class="py-1.5 font-normal">Scorer</th>
          <th class="py-1.5 font-normal">Field</th>
          <th class="py-1.5 font-normal">Scored / total</th>
          <th class="py-1.5 font-normal">Coverage</th>
        </tr>
      </thead>
      <tbody>
        {#each scorerIds as id (id)}
          {@const entry = coverage[id]}
          <tr class="border-b border-zinc-900">
            <td class="py-1.5">
              <input
                type="checkbox"
                checked={!!selected[id]}
                onchange={() => toggleScorer(id)}
                aria-label={`Select ${id}`}
              />
            </td>
            <td class="py-1.5 font-mono text-xs">{id}</td>
            <td class="py-1.5 font-mono text-xs text-zinc-500">{entry?.field ?? '—'}</td>
            <td class="py-1.5 text-zinc-300">{formatCoverageCounts(entry)}</td>
            <td class="py-1.5 text-zinc-300">{formatCoveragePct(entry)}</td>
          </tr>
        {/each}
      </tbody>
    </table>
  {/if}

  <div class="flex flex-wrap items-center gap-2">
    <button
      type="button"
      class="btn btn-primary"
      disabled={starting || job !== null || scorerIds.length === 0}
      onclick={openConfirmAll}
    >
      Compute all
    </button>
    <button
      type="button"
      class="btn"
      disabled={starting || job !== null || selectedIds.length === 0}
      onclick={openConfirmSelected}
    >
      Compute selected{selectedIds.length ? ` (${selectedIds.length})` : ''}
    </button>
    {#if job}
      <span class="text-xs text-zinc-400">
        Computing {job.scorers.join(', ') || 'scores'}
        {job.total
          ? ` — ${job.processed.toLocaleString()} / ${job.total.toLocaleString()}`
          : '…'}
      </span>
      <button
        type="button"
        class="rounded border border-zinc-700 px-2 py-0.5 text-xs text-zinc-300 hover:bg-zinc-800"
        onclick={() => void cancelCompute()}
        disabled={cancelling}
      >
        {cancelling ? 'Cancelling…' : 'Cancel'}
      </button>
    {/if}
  </div>

  {#if computeError}
    <p
      class="rounded border border-red-500/40 bg-red-500/10 px-3 py-2 text-xs text-red-200"
    >
      {computeError}
    </p>
  {/if}
</section>

{#if confirmScorers !== undefined}
  <!-- svelte-ignore a11y_click_events_have_key_events -->
  <div
    class="fixed inset-0 z-40 flex items-center justify-center bg-black/60 p-4"
    role="dialog"
    aria-modal="true"
    aria-label="Confirm compute curation scores"
    tabindex="-1"
    use:focusOnMount
    use:trapFocus={{ onEscape: closeConfirm }}
    onclick={(e) => {
      if (e.target === e.currentTarget) closeConfirm();
    }}
  >
    <div
      class="w-full max-w-md rounded-lg border border-zinc-800 bg-zinc-950 p-5 shadow-2xl"
    >
      <h3 class="mb-3 text-base font-semibold">Compute curation scores</h3>
      <p class="mb-3 text-sm text-zinc-300">
        {confirmScorers === null
          ? 'Runs every enabled scorer against the full pool.'
          : `Runs: ${confirmScorers?.join(', ')}.`}
      </p>
      <p class="mb-3 text-xs text-zinc-400">
        This is a deployment-wide, potentially long-running operation over the whole crop
        pool. Only one compute job can run at a time.
      </p>
      <div class="flex justify-end gap-2">
        <button type="button" class="btn" onclick={closeConfirm}>Cancel</button>
        <button
          type="button"
          class="btn btn-primary"
          onclick={() => void confirmCompute()}
        >
          Confirm
        </button>
      </div>
    </div>
  </div>
{/if}
