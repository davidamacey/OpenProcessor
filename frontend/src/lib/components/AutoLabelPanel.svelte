<script lang="ts">
  import { apiErrorText } from '$lib/api_errors';
  /*
   * Auto-label control panel.
   *
   * Renders on the dashboard. Polls {API_PREFIX}/pipeline/auto_label/status — slowly
   * when idle (30s, just to detect a daemon-fired run), fast while the run is
   * live (3s). The run is a queued job: the panel prints the status the server
   * reports (queued, running, paused with its reason, cancelling), the stage,
   * progress bar, ETA, elapsed time, and the final stage-by-stage summary.
   *
   * The pipeline is embed_missing (opt-in) → vlm → finalize. Clustering is its
   * own run (`POST /clusters/runs`, <ClusterRunPanel>).
   *
   * Since 2026-09 the run can optionally be scoped: <AssistScopeBar>
   * picks one class (and, when the backend advertises them, a detection
   * profile and a prompt pack) and contributes `class_id` /
   * `detection_profile` / `prompt_pack` to the start call. The bar is
   * absent entirely unless `/methods` advertises the assist axes, and
   * contributes nothing when the operator leaves it alone — an unscoped
   * "Recluster now" is the same one-click, whole-dataset run it has
   * always been.
   */
  import { onMount, onDestroy } from 'svelte';
  import AssistScopeBar from './AssistScopeBar.svelte';
  import {
    AUTO_LABEL_LIVE_STATUSES,
    cancelAutoLabel,
    getAutoLabelStatus,
    pauseAutoLabel,
    resumeAutoLabel,
    startAutoLabel,
    unknownStrategyDetail,
    type AutoLabelJobState,
    type AutoLabelStatus,
  } from '$lib/api_autoLabel';
  import { configErrorDetail } from '$lib/api_profiles';
  import { createAssistScope } from '$lib/assistScope.svelte';
  import { resolveAutoLabelRunVlm } from '$lib/autoLabelRunVlm';
  import { isScopedAssistAvailable } from '$lib/strategies';
  import { formatDuration } from '$lib/format/duration';
  import { classesStore } from '$stores/classes.svelte';
  import { strategiesStore } from '$stores/strategies.svelte';
  import { toastStore } from '$stores/toast.svelte';
  import { summarizeRunStages } from '$lib/autoLabelRunSummary';
  import Modal from '$components/Modal.svelte';

  // Additive, optional (Ingest UI plan
  // §A.5): `/ingest`'s ClusteringHandoff passes a gate computed from the
  // upload run state + the served region drain; the dashboard passes
  // nothing, so its behavior is unchanged. `null`/absent === unblocked,
  // same as every other optional capability gate in this app.
  interface Props {
    gate?: { blocked: boolean; reason: string } | null;
  }
  let { gate = null }: Props = $props();

  let job: AutoLabelJobState | null = $state(null);
  let busy: boolean = $state(false);
  // p2 (2026-09-24 interactive pass): "Recluster now" used to fire on one
  // click with no confirmation — it's a dataset-wide, potentially
  // long-running pipeline run, so gate it behind an explicit confirm
  // step the way the dashboard's other one-shot actions already are.
  let confirmOpen: boolean = $state(false);
  let pollTimer: ReturnType<typeof setTimeout> | null = null;
  // G5: explicit control for pipeline.py's `run_vlm` (default False
  // server-side) — off by default here too, matching the backend. A
  // scoped run (AssistScopeBar picked a class/pack) always implies
  // run_vlm regardless of this checkbox, since a scoped run that skips
  // the VLM stage it claims to scope is a no-op (see start()).
  let runVlm: boolean = $state(false);
  // v0.4.0: embed the items without a vector first. Sent only when ticked.
  let embedMissing: boolean = $state(false);

  // -- VLM-assist scoping (this plan §4) ---------------------------------
  // The scope bar is absent, not disabled, until /methods advertises the
  // assist axes. `class_id` has no capability signal of its own and an
  // unknown query param is silently dropped server-side, so an ungated
  // picker would start an unscoped hours-long run while claiming it was
  // scoped — see isScopedAssistAvailable's doc comment.
  const scope = createAssistScope();

  // Idempotent, never-rejecting, cached one-shot (a failed load
  // leaves EMPTY_METHODS) — same call StrategyBar and /train
  // make. `scopeAvailable` is false for the first frames after mount;
  // that is correct (hide, then reveal) and must not be "fixed" with a
  // spinner or an await.
  $effect(() => {
    void strategiesStore.init();
  });
  const scopeAvailable = $derived(isScopedAssistAvailable(strategiesStore.methods));

  const scopeClassName = $derived(
    scope.classId == null
      ? null
      : (classesStore.byId(scope.classId)?.name ?? `class ${scope.classId}`),
  );

  // This component deliberately does not take its own subscription on
  // classesStore — src/routes/+layout.svelte already holds one for the
  // whole app lifetime, so classesStore.classes is populated and
  // refreshing on every route. A second ref-count here would be
  // redundant with no benefit.

  // Human-readable label per stage. Order matters — pipeline stages move
  // forward through this list. The percent indicator only renders for
  // 'vlm' because that's the only stage with a meaningful total.
  const STAGE_LABEL: Record<string, string> = {
    '': 'preparing…',
    embed_missing: 'embedding items without a vector',
    vlm: 'VLM sweep over unvalidated crops',
    finalize: 'final accounting',
  };

  const STAGE_ORDER = ['vlm', 'finalize'];

  // The embed stage runs first, and only when the run asked for it (the
  // job echoes its resolved args).
  function stageOrderFor(j: AutoLabelJobState | null): string[] {
    return j?.args?.embed_missing === true || j?.stage === 'embed_missing'
      ? ['embed_missing', ...STAGE_ORDER]
      : STAGE_ORDER;
  }

  function stageIndex(order: string[], s: string): number {
    const i = order.indexOf(s);
    return i < 0 ? 0 : i;
  }

  async function poll(): Promise<void> {
    try {
      job = await getAutoLabelStatus();
    } catch {
      /* network blip — keep last job, try again next tick */
    }
    schedule();
  }

  function schedule(): void {
    if (pollTimer) clearTimeout(pollTimer);
    // Fast poll while running, slow when idle. The slow poll catches
    // daemon-initiated runs without hammering the API.
    const ms = isLive ? 3000 : 30000;
    pollTimer = setTimeout(() => void poll(), ms);
  }

  function requestStart(): void {
    confirmOpen = true;
  }

  function closeConfirmStart(): void {
    if (busy) return;
    confirmOpen = false;
  }

  async function start(): Promise<void> {
    confirmOpen = false;
    busy = true;
    try {
      const sendRunVlm = resolveAutoLabelRunVlm(scope.toStartParams(), runVlm);
      job = await startAutoLabel({
        vlm_concurrency: 16,
        // 0 = process every unvalidated crop. Operators can ramp down
        // later if they want to time-box a run.
        max_vlm_crops: 0,
        // `{}` whenever nothing is scoped, so the composed URL stays
        // byte-identical to every request this panel has ever sent.
        ...scope.toStartParams(),
        // Omitted (not `false`) when off, so an unscoped/unchecked run's
        // URL stays byte-identical to every request this panel has ever
        // sent — the backend already defaults run_vlm to False.
        ...(sendRunVlm ? { run_vlm: true } : {}),
        ...(embedMissing ? { embed_missing: true } : {}),
      });
      toastStore.success(
        scopeClassName
          ? `Auto-label queued — VLM labeling limited to ${scopeClassName} (VLM stage will run).`
          : sendRunVlm
            ? 'Auto-label queued — running the VLM labeling stage.'
            : 'Auto-label queued.',
      );
      schedule();
    } catch (e) {
      const msg = apiErrorText(e);
      const refusal = configErrorDetail(e);
      if (unknownStrategyDetail(e)) {
        const d = unknownStrategyDetail(e)!;
        toastStore.error(
          `Start failed: unknown ${d.axis.replace('_', ' ')} "${d.requested}" — valid: ${d.valid_ids.join(', ') || 'none'}.`,
        );
      } else if (refusal) {
        // A served refusal (`vlm_external_not_acknowledged`,
        // `vlm_not_configured`, `vlm_endpoint_unavailable`, a pairing
        // `validation_failed`): its own message, never the generic 409.
        toastStore.error(`Start failed: ${refusal.message}`);
      } else if (msg.includes('409') || msg.includes('already in progress')) {
        toastStore.warn('An auto-label run is already queued, running or paused.');
        void poll();
      } else {
        toastStore.error(`Start failed: ${msg}`);
      }
    } finally {
      busy = false;
    }
  }

  async function cancel(): Promise<void> {
    busy = true;
    try {
      job = await cancelAutoLabel();
      toastStore.info('Cancel requested. Pipeline will stop at the next checkpoint.');
    } catch (e) {
      toastStore.error(`Cancel failed: ${apiErrorText(e)}`);
    } finally {
      busy = false;
    }
  }

  async function pause(): Promise<void> {
    busy = true;
    try {
      job = await pauseAutoLabel();
    } catch (e) {
      toastStore.error(`Pause failed: ${apiErrorText(e)}`);
    } finally {
      busy = false;
    }
  }

  async function resume(): Promise<void> {
    busy = true;
    try {
      job = await resumeAutoLabel();
      schedule();
    } catch (e) {
      toastStore.error(`Resume failed: ${apiErrorText(e)}`);
    } finally {
      busy = false;
    }
  }

  onMount(() => {
    void poll();
  });

  onDestroy(() => {
    if (pollTimer) clearTimeout(pollTimer);
  });

  // The server works on the run, or is stopping it (queued, running, cancelling).
  const isLive: boolean = $derived(
    AUTO_LABEL_LIVE_STATUSES.includes(
      ((job as AutoLabelJobState | null)?.status ?? 'idle') as AutoLabelStatus,
    ),
  );
  const isPaused: boolean = $derived(
    ((job as AutoLabelJobState | null)?.status ?? '') === 'paused',
  );
  // The run holds the project (live or paused): no new start until it ends.
  const hasRun: boolean = $derived(isLive || isPaused);
  const stageOrder: string[] = $derived(stageOrderFor(job as AutoLabelJobState | null));
  const stageIdx: number = $derived(
    stageIndex(stageOrder, (job as AutoLabelJobState | null)?.stage ?? ''),
  );
  const percent: number | null = $derived.by(() => {
    if (!job || job.total <= 0) return null;
    return Math.min(100, Math.round((job.processed / job.total) * 100));
  });

  const stageDurations: Array<[string, number]> = $derived.by(() => {
    const sd = (job as AutoLabelJobState | null)?.stage_durations;
    if (!sd) return [];
    return Object.entries(sd) as Array<[string, number]>;
  });
  const lastFinishedRel: string = $derived.by(() => {
    if (!job || !job.finished_at) return '';
    const seconds = Math.max(0, Date.now() / 1000 - job.finished_at);
    if (seconds < 60) return `${Math.round(seconds)}s ago`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
    return `${Math.floor(seconds / 86400)}d ago`;
  });
</script>

<section class="rounded-md border border-zinc-800 bg-zinc-950 p-4">
  <!-- D3 (visual audit 2026-09-24): at 800px the description was squeezed
       into a ~60px column beside the controls (one word per line). It now
       stacks above the controls until there's room for a side-by-side row. -->
  <div
    class="flex flex-col gap-3 xl:flex-row xl:items-center xl:justify-between"
    data-testid="recluster-header"
  >
    <div class="min-w-0 xl:max-w-md" data-testid="recluster-description">
      <h2 class="text-base font-semibold">Auto-label</h2>
      <p class="mt-0.5 text-xs text-zinc-400">
        Optionally embeds the items that have no vector, and runs the VLM over the
        unvalidated crops (off by default — see "Run VLM labeling stage" below), then
        finalizes. The classifier's confident labels and human validations are never
        overwritten. The run is queued, can be paused, and is safe to cancel. Grouping the
        items into clusters is a separate run (see Clusters below).
      </p>
    </div>
    <div class="flex flex-wrap items-center gap-3">
      {#if !hasRun}
        {#if scopeAvailable}
          <AssistScopeBar {scope} classes={classesStore.classes} disabled={busy} />
        {/if}
        <label
          class="flex items-center gap-1.5 text-xs text-zinc-300"
          title="pipeline.py's run_vlm — off by default, matching the backend. Implied automatically when a class, prompt pack or VLM endpoint is scoped above."
        >
          <input
            type="checkbox"
            class="h-3.5 w-3.5 accent-blue-500"
            checked={runVlm || !scope.isDefault}
            disabled={busy || !scope.isDefault}
            onchange={(e) => (runVlm = (e.currentTarget as HTMLInputElement).checked)}
          />
          Run VLM labeling stage
        </label>
        <label
          class="flex items-center gap-1.5 text-xs text-zinc-300"
          title="Embed the items that have no vector (a lazy ingest policy, a failed encoder) before clustering. Off by default."
        >
          <input
            type="checkbox"
            class="h-3.5 w-3.5 accent-blue-500"
            data-testid="embed-missing-checkbox"
            bind:checked={embedMissing}
            disabled={busy}
          />
          Embed missing vectors first
        </label>
      {/if}
      {#if hasRun}
        {#if isPaused}
          <button
            class="btn btn-primary"
            type="button"
            onclick={resume}
            disabled={busy}
            data-testid="auto-label-resume"
          >
            Resume
          </button>
        {:else if job?.status === 'queued' || job?.status === 'running'}
          <button
            class="btn"
            type="button"
            onclick={pause}
            disabled={busy}
            data-testid="auto-label-pause"
          >
            Pause
          </button>
        {/if}
        {#if job?.status !== 'cancelling'}
          <button class="btn btn-danger" type="button" onclick={cancel} disabled={busy}>
            Cancel
          </button>
        {/if}
      {:else}
        <button
          class="btn btn-primary"
          type="button"
          onclick={requestStart}
          disabled={busy || gate?.blocked}
          title={gate?.blocked ? gate.reason : undefined}
        >
          {scopeClassName ? `Auto-label · VLM: ${scopeClassName}` : 'Auto-label now'}
        </button>
      {/if}
    </div>
    {#if gate?.blocked}
      <p class="mt-1 text-[11px] text-amber-300">{gate.reason}</p>
    {/if}
  </div>

  {#if confirmOpen}
    <Modal label="Confirm auto-label" onclose={closeConfirmStart}>
      <h3 class="mb-3 text-base font-semibold">
        {scopeClassName ? `Auto-label · VLM: ${scopeClassName}` : 'Auto-label now'}
      </h3>
      <p class="mb-4 text-xs text-zinc-400">
        Queues the auto-label run{resolveAutoLabelRunVlm(scope.toStartParams(), runVlm)
          ? ' and runs the VLM labeling stage'
          : ''}. This can run for a while.
      </p>
      <div class="flex justify-end gap-2">
        <button type="button" class="btn" onclick={closeConfirmStart} disabled={busy}>
          Cancel
        </button>
        <button type="button" class="btn btn-primary" onclick={start} disabled={busy}>
          {busy ? 'Starting…' : 'Start'}
        </button>
      </div>
    </Modal>
  {/if}

  {#if job}
    <div class="mt-4 space-y-3">
      <!-- Status chip + stage label. Status drives the color so a glance
           tells the operator whether a previous run failed. -->
      <div class="flex flex-wrap items-center gap-2 text-xs">
        <span
          class="rounded px-2 py-0.5 font-mono uppercase
            {job.status === 'running'
            ? 'bg-blue-500/20 text-blue-200'
            : job.status === 'completed'
              ? 'bg-emerald-500/20 text-emerald-200'
              : job.status === 'failed'
                ? 'bg-red-500/20 text-red-200'
                : job.status === 'cancelled' ||
                    job.status === 'cancelling' ||
                    job.status === 'paused' ||
                    job.status === 'completed_with_errors' ||
                    job.status === 'interrupted'
                  ? 'bg-orange-500/20 text-orange-200'
                  : 'bg-zinc-700/40 text-zinc-300'}"
          data-testid="auto-label-status"
        >
          {job.status.replaceAll('_', ' ')}
        </span>
        {#if isPaused && job.pause_reason}
          <span class="text-zinc-300" data-testid="auto-label-pause-reason"
            >{job.pause_reason.replaceAll('_', ' ')}</span
          >
        {/if}
        {#if isLive}
          <span class="text-zinc-300">{STAGE_LABEL[job.stage] ?? job.stage}</span>
          <span class="font-mono text-zinc-500">
            stage {stageIdx + 1}/{stageOrder.length}
          </span>
        {:else if job.status !== 'idle' && !isPaused && job.finished_at}
          <span class="text-zinc-400">finished {lastFinishedRel}</span>
        {/if}
      </div>

      {#if isLive}
        <!-- Per-stage progress bar. Indeterminate when total=0; determinate
             for the VLM stage which is the dominant cost. -->
        <div class="h-2 overflow-hidden rounded bg-zinc-800">
          {#if percent != null}
            <div
              class="h-full bg-blue-500 transition-all"
              style="width: {percent}%"
            ></div>
          {:else}
            <div class="h-full w-1/3 animate-pulse bg-blue-500/50"></div>
          {/if}
        </div>
        <div
          class="flex flex-wrap items-center justify-between gap-2 font-mono text-[11px] text-zinc-400"
        >
          <span>
            {#if job.total > 0}
              {job.processed.toLocaleString()} / {job.total.toLocaleString()}
              {percent != null ? `(${percent}%)` : ''}
            {:else}
              working…
            {/if}
          </span>
          <span>
            elapsed {formatDuration(job.elapsed_seconds)}
            {#if job.eta_seconds != null}
              · ETA {formatDuration(job.eta_seconds)}
            {/if}
          </span>
        </div>
      {/if}

      {#if job.error}
        <p class="text-xs text-red-300">Error: {job.error}</p>
      {/if}
      {#if job.error_detail}
        <p class="text-xs text-red-300/80" data-testid="auto-label-error-detail">
          {job.error_detail}
        </p>
      {/if}

      {#if job.status === 'completed' || job.status === 'completed_with_errors' || job.status === 'failed'}
        <!-- Per-stage durations + peak VRAM (GPU runs only). Rendered as
             a compact table above the raw JSON so operators can answer
             "where did the wall time go?" without parsing the full result. -->
        {#if stageDurations.length > 0}
          <div class="space-y-1 text-xs text-zinc-300">
            {#if stageDurations.length > 0}
              <table class="font-mono text-[10px]">
                <tbody>
                  {#each stageDurations as [name, sec] (name)}
                    <tr>
                      <td class="pr-4 text-zinc-400">{name}</td>
                      <td class="text-zinc-200">{formatDuration(sec)}</td>
                    </tr>
                  {/each}
                </tbody>
              </table>
            {/if}
          </div>
        {/if}
        {#if job.result?.stages}
          <details class="text-xs text-zinc-300">
            <summary class="cursor-pointer text-zinc-400 hover:text-zinc-200">
              Last run summary
            </summary>
            <!-- D5 (visual audit 2026-09-24): a per-stage table of the
                 served result instead of a raw JSON dump; the JSON stays
                 one click away for the nested details. -->
            <table
              class="mt-2 w-full text-left text-[11px]"
              data-testid="last-run-stages"
            >
              <thead class="text-zinc-500">
                <tr>
                  <th class="py-1 pr-3 font-medium">Stage</th>
                  <th class="py-1 pr-3 font-medium">Status</th>
                  <th class="py-1 font-medium">Details</th>
                </tr>
              </thead>
              <tbody>
                {#each summarizeRunStages(job.result) as row (row.key)}
                  <tr
                    class="border-t border-zinc-800 align-top {row.status === 'error'
                      ? 'bg-red-500/10 text-red-200'
                      : ''}"
                    data-stage-status={row.status}
                  >
                    <td class="py-1 pr-3 text-zinc-200" title={row.key}>
                      {STAGE_LABEL[row.key] ?? row.key}
                    </td>
                    <td class="py-1 pr-3 font-mono text-zinc-300">{row.status}</td>
                    <td class="py-1 text-zinc-400">
                      {#if row.reason}<span class="mr-2">{row.reason}</span>{/if}
                      {#each row.fields as f (f.name)}
                        <span class="mr-2 inline-block font-mono text-[10px]"
                          ><span class="text-zinc-500">{f.name}</span> {f.value}</span
                        >
                      {/each}
                    </td>
                  </tr>
                {/each}
              </tbody>
            </table>
            <details class="mt-2">
              <summary
                class="cursor-pointer text-[10px] text-zinc-500 hover:text-zinc-300"
              >
                Raw JSON
              </summary>
              <pre
                class="mt-1 max-h-64 overflow-auto rounded bg-zinc-900 p-2 font-mono text-[10px] text-zinc-300">{JSON.stringify(
                  job.result,
                  null,
                  2,
                )}</pre>
            </details>
          </details>
        {/if}
      {/if}
    </div>
  {/if}
</section>
