<script lang="ts">
  /**
   * "Run probe predictions" control — `/train`, embedded in `RunResults`
   * for a finished run (OpenProcessor #36 item 8). Populates
   * `probe_pred_*` on items, which is the one prerequisite `/review`'s
   * Uncertainty and Model-disagreements queues need (see item 9's
   * `empty_state.has_probe_predictions`).
   *
   * Follows `ScoresCard.svelte`'s job-poll pattern: idempotent
   * confirm-before-start, adopt-in-flight-on-mount, explicit Cancel,
   * served result/error rendered verbatim — never reworded or
   * recomputed. `POST {API_PREFIX}/probe/run` 409s when this run isn't
   * finished, has no exported checkpoint, or a probe is already running
   * (possibly from a DIFFERENT run — `canRunProbe` only guards the
   * obviously-unqualifying cases client-side; the 409 detail is shown
   * as-is for anything else).
   */
  import {
    ApiError,
    cancelProbe,
    getProbeStatus,
    runProbe,
    type ProbeStatusResponse,
  } from '$lib/api';
  import { canRunProbe, classifyProbePoll } from '$lib/probe';
  import { focusOnMount } from '$lib/actions/focusOnMount';
  import { trapFocus } from '$lib/actions/trapFocus';
  import { toastStore } from '$stores/toast.svelte';
  import type { TrainJobStatus } from '$lib/types_train';

  const JOB_POLL_MS = 3000;

  interface Props {
    status: TrainJobStatus;
  }

  let { status }: Props = $props();

  let job = $state<ProbeStatusResponse | null>(null);
  let starting = $state(false);
  let cancelling = $state(false);
  let confirmOpen = $state(false);
  let jobPoll: ReturnType<typeof setInterval> | null = null;
  /** The most recently finished/failed/cancelled result FOR THIS RUN,
   *  kept visible after the job clears so "Failed: <server message>"
   *  doesn't vanish the instant polling stops. Cleared on a fresh start. */
  let lastResult = $state<ProbeStatusResponse | null>(null);

  const eligible = $derived(canRunProbe(status));
  // A probe job started for a DIFFERENT run's job_id still shows as
  // "running" here — GET /probe/status is a single current/last-job
  // singleton, not scoped per train run — so the button disables and
  // says which run owns it, rather than silently offering a click that
  // will 409.
  const runningElsewhere = $derived(
    job !== null && job.train_job_id != null && job.train_job_id !== status.job_id,
  );
  const runningHere = $derived(
    job !== null && (job.train_job_id == null || job.train_job_id === status.job_id),
  );

  function stopJobPoll(): void {
    if (jobPoll) clearInterval(jobPoll);
    jobPoll = null;
  }

  async function pollJob(): Promise<void> {
    let st: ProbeStatusResponse;
    try {
      st = await getProbeStatus();
    } catch {
      return; // transient — keep polling
    }
    const outcome = classifyProbePoll(st.status);
    if (outcome === 'running') {
      job = st;
      return;
    }
    job = null;
    stopJobPoll();
    if (st.train_job_id === status.job_id || st.job_id === status.job_id) {
      lastResult = st;
    }
    if (outcome === 'completed') {
      toastStore.success(
        st.updated_count != null
          ? `Probe finished — ${st.updated_count.toLocaleString()} items updated.`
          : 'Probe finished.',
      );
    } else if (outcome === 'failed') {
      toastStore.error(`Probe failed: ${st.error ?? 'unknown error'}`);
    }
    // 'cancelled' — no toast, the operator asked for this.
  }

  function startJobPoll(): void {
    stopJobPoll();
    jobPoll = setInterval(() => void pollJob(), JOB_POLL_MS);
  }

  /** Adopt a probe job already running (another tab, or before a
   *  reload) — checked once per mount, same shape as ScoresCard's
   *  adoptInFlightJob. */
  async function adoptInFlightJob(): Promise<void> {
    try {
      const st = await getProbeStatus();
      if (st.status === 'running') {
        job = st;
        startJobPoll();
      }
    } catch {
      // No status endpoint / transient failure — nothing to adopt.
    }
  }

  $effect(() => {
    // Only poll for an in-flight job when this run could plausibly have
    // started one — a run RunResults renders for that never qualifies
    // (no checkpoint, not finished) has nothing of its own to adopt, and
    // the component itself renders nothing in that case, so firing a
    // GET here would be a probe-status request on every unrelated past
    // run's Results panel for no visible effect.
    if (eligible) void adoptInFlightJob();
    return stopJobPoll;
  });

  function openConfirm(): void {
    confirmOpen = true;
  }

  function closeConfirm(): void {
    confirmOpen = false;
  }

  async function confirmRun(): Promise<void> {
    if (starting || job) return;
    confirmOpen = false;
    starting = true;
    lastResult = null;
    try {
      job = await runProbe(status.job_id);
      startJobPoll();
      toastStore.info('Probe started…');
    } catch (e) {
      const detail =
        e instanceof ApiError
          ? (e.detail ?? e.message)
          : ((e as Error)?.message ?? 'failed');
      lastResult = { status: 'failed', error: String(detail) };
      toastStore.error(`Probe failed to start: ${detail}`);
    } finally {
      starting = false;
    }
  }

  async function cancelRun(): Promise<void> {
    if (cancelling) return;
    cancelling = true;
    try {
      await cancelProbe();
      toastStore.info('Probe cancelled.');
    } catch (e) {
      toastStore.error(`Cancel failed: ${(e as Error).message}`);
    } finally {
      job = null;
      stopJobPoll();
      cancelling = false;
    }
  }
</script>

{#if eligible || runningHere || lastResult}
  <div class="flex flex-wrap items-center gap-2 text-xs" data-testid="probe-control">
    <span class="font-semibold text-zinc-400">Probe predictions</span>
    {#if runningHere}
      <span class="text-zinc-400">
        {job?.status === 'running' ? 'Running…' : job?.status}
      </span>
      <button
        type="button"
        class="rounded border border-zinc-700 px-2 py-0.5 text-zinc-300 hover:bg-zinc-800"
        onclick={() => void cancelRun()}
        disabled={cancelling}
      >
        {cancelling ? 'Cancelling…' : 'Cancel'}
      </button>
    {:else if runningElsewhere}
      <span class="text-zinc-500" title={job?.train_job_id ?? undefined}>
        A probe is already running for another run.
      </span>
    {:else}
      <button
        type="button"
        class="rounded border border-zinc-700 bg-zinc-950 px-2 py-0.5 text-blue-300 hover:border-blue-500 hover:bg-blue-500/10"
        onclick={openConfirm}
        disabled={starting || !eligible}
        title={eligible
          ? undefined
          : 'This run has no exported checkpoint to probe from.'}
      >
        {starting ? 'Starting…' : 'Run probe predictions'}
      </button>
    {/if}
    {#if lastResult && !runningHere}
      {#if lastResult.status === 'failed'}
        <span class="text-red-300">Failed: {lastResult.error ?? 'unknown error'}</span>
      {:else if lastResult.status === 'cancelled'}
        <span class="text-zinc-500">Cancelled.</span>
      {:else}
        <span class="text-emerald-300">
          Done{lastResult.updated_count != null
            ? ` — ${lastResult.updated_count.toLocaleString()} items updated`
            : ''}.
        </span>
      {/if}
    {/if}
  </div>
{/if}

{#if confirmOpen}
  <!-- svelte-ignore a11y_click_events_have_key_events -->
  <div
    class="fixed inset-0 z-40 flex items-center justify-center bg-black/60 p-4"
    role="dialog"
    aria-modal="true"
    aria-label="Confirm run probe predictions"
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
      <h3 class="mb-3 text-base font-semibold">Run probe predictions</h3>
      <p class="mb-3 text-sm text-zinc-300">
        Runs this run's exported checkpoint over the crop pool to populate
        <code class="font-mono">probe_pred_*</code>. This is what the Uncertainty and
        Model-disagreements review queues need — they stay empty until a probe has run at
        least once.
      </p>
      <p class="mb-3 text-xs text-zinc-400">
        Deployment-wide, potentially long-running. Only one probe job can run at a time.
      </p>
      <div class="flex justify-end gap-2">
        <button type="button" class="btn" onclick={closeConfirm}>Cancel</button>
        <button type="button" class="btn btn-primary" onclick={() => void confirmRun()}>
          Confirm
        </button>
      </div>
    </div>
  </div>
{/if}
