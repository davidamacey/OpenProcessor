<script lang="ts">
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
  import AssistScopeBar from './AssistScopeBar.svelte';
  import { createAutoLabelController } from '$lib/autoLabelController.svelte';
  import { createAssistScope } from '$lib/assistScope.svelte';
  import { resolveAutoLabelRunVlm } from '$lib/autoLabelRunVlm';
  import { isScopedAssistAvailable } from '$lib/strategies';
  import { formatDuration } from '$lib/format/duration';
  import { classesStore } from '$stores/classes.svelte';
  import { strategiesStore } from '$stores/strategies.svelte';
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

  const ctl = createAutoLabelController(scope, () => scopeClassName);
  const job = $derived(ctl.job);
  const busy = $derived(ctl.busy);
  const confirmOpen = $derived(ctl.confirmOpen);
  const isLive = $derived(ctl.isLive);
  const isPaused = $derived(ctl.isPaused);
  const hasRun = $derived(ctl.hasRun);
  const stageOrder = $derived(ctl.stageOrder);
  const stageIdx = $derived(ctl.stageIdx);
  const percent = $derived(ctl.percent);
  const stageDurations = $derived(ctl.stageDurations);
  const lastFinishedRel = $derived(ctl.lastFinishedRel);
  const { requestStart, closeConfirmStart, start, cancel, pause, resume } = ctl;
</script>

<section class="rounded-md border border-line bg-surface-0 p-4">
  <!-- D3 (visual audit 2026-09-24): at 800px the description was squeezed
       into a ~60px column beside the controls (one word per line). It now
       stacks above the controls until there's room for a side-by-side row. -->
  <div
    class="flex flex-col gap-3 xl:flex-row xl:items-center xl:justify-between"
    data-testid="recluster-header"
  >
    <div class="min-w-0 xl:max-w-md" data-testid="recluster-description">
      <h2 class="text-title font-semibold">Auto-label</h2>
      <p class="mt-0.5 text-meta text-fg-2">
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
          class="flex items-center gap-1.5 text-meta text-fg-2"
          title="pipeline.py's run_vlm — off by default, matching the backend. Implied automatically when a class, prompt pack or VLM endpoint is scoped above."
        >
          <input
            type="checkbox"
            class="h-3.5 w-3.5 accent-accent"
            checked={ctl.runVlm || !scope.isDefault}
            disabled={busy || !scope.isDefault}
            onchange={(e) => (ctl.runVlm = (e.currentTarget as HTMLInputElement).checked)}
          />
          Run VLM labeling stage
        </label>
        <label
          class="flex items-center gap-1.5 text-meta text-fg-2"
          title="Embed the items that have no vector (a lazy ingest policy, a failed encoder) before clustering. Off by default."
        >
          <input
            type="checkbox"
            class="h-3.5 w-3.5 accent-accent"
            data-testid="embed-missing-checkbox"
            bind:checked={ctl.embedMissing}
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
      <p class="mt-1 text-caption text-warn-text">{gate.reason}</p>
    {/if}
  </div>

  {#if confirmOpen}
    <Modal label="Confirm auto-label" onclose={closeConfirmStart}>
      <h3 class="mb-3 text-title font-semibold">
        {scopeClassName ? `Auto-label · VLM: ${scopeClassName}` : 'Auto-label now'}
      </h3>
      <p class="mb-4 text-meta text-fg-2">
        Queues the auto-label run{resolveAutoLabelRunVlm(
          scope.toStartParams(),
          ctl.runVlm,
        )
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
      <div class="flex flex-wrap items-center gap-2 text-meta">
        <span
          class="rounded px-2 py-0.5 font-mono capitalize
            {job.status === 'running'
            ? 'bg-accent/20 text-accent-text'
            : job.status === 'completed'
              ? 'bg-success/20 text-success-text'
              : job.status === 'failed'
                ? 'bg-danger/20 text-danger-text'
                : job.status === 'cancelled' ||
                    job.status === 'cancelling' ||
                    job.status === 'paused' ||
                    job.status === 'completed_with_errors' ||
                    job.status === 'interrupted'
                  ? 'bg-warn/20 text-warn-text'
                  : 'bg-surface-3/40 text-fg-2'}"
          data-testid="auto-label-status"
        >
          {job.status.replaceAll('_', ' ')}
        </span>
        {#if isPaused && job.pause_reason}
          <span class="text-fg-2" data-testid="auto-label-pause-reason"
            >{job.pause_reason.replaceAll('_', ' ')}</span
          >
        {/if}
        {#if isLive}
          <span class="text-fg-2">{STAGE_LABEL[job.stage] ?? job.stage}</span>
          <span class="font-mono text-fg-3">
            stage {stageIdx + 1}/{stageOrder.length}
          </span>
        {:else if job.status !== 'idle' && !isPaused && job.finished_at}
          <span class="text-fg-2">finished {lastFinishedRel}</span>
        {/if}
      </div>

      {#if isLive}
        <!-- Per-stage progress bar. Indeterminate when total=0; determinate
             for the VLM stage which is the dominant cost. -->
        <div class="h-2 overflow-hidden rounded bg-surface-2">
          {#if percent != null}
            <div class="h-full bg-accent transition-all" style="width: {percent}%"></div>
          {:else}
            <div class="h-full w-1/3 animate-pulse bg-accent/50"></div>
          {/if}
        </div>
        <div
          class="flex flex-wrap items-center justify-between gap-2 font-mono text-caption text-fg-2"
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
        <p class="text-meta text-danger-text">Error: {job.error}</p>
      {/if}
      {#if job.error_detail}
        <p class="text-meta text-danger-text/80" data-testid="auto-label-error-detail">
          {job.error_detail}
        </p>
      {/if}

      {#if job.status === 'completed' || job.status === 'completed_with_errors' || job.status === 'failed'}
        <!-- Per-stage durations + peak VRAM (GPU runs only). Rendered as
             a compact table above the raw JSON so operators can answer
             "where did the wall time go?" without parsing the full result. -->
        {#if stageDurations.length > 0}
          <div class="space-y-1 text-meta text-fg-2">
            {#if stageDurations.length > 0}
              <table class="font-mono text-caption">
                <tbody>
                  {#each stageDurations as [name, sec] (name)}
                    <tr>
                      <td class="pr-4 text-fg-2">{name}</td>
                      <td class="text-fg-1">{formatDuration(sec)}</td>
                    </tr>
                  {/each}
                </tbody>
              </table>
            {/if}
          </div>
        {/if}
        {#if job.result?.stages}
          <details class="text-meta text-fg-2">
            <summary class="cursor-pointer text-fg-2 hover:text-fg-1">
              Last run summary
            </summary>
            <!-- D5 (visual audit 2026-09-24): a per-stage table of the
                 served result instead of a raw JSON dump; the JSON stays
                 one click away for the nested details. -->
            <table
              class="mt-2 w-full text-left text-caption"
              data-testid="last-run-stages"
            >
              <thead class="text-fg-3">
                <tr>
                  <th class="py-1 pr-3 font-medium">Stage</th>
                  <th class="py-1 pr-3 font-medium">Status</th>
                  <th class="py-1 font-medium">Details</th>
                </tr>
              </thead>
              <tbody>
                {#each summarizeRunStages(job.result) as row (row.key)}
                  <tr
                    class="border-t border-line align-top {row.status === 'error'
                      ? 'bg-danger/10 text-danger-text'
                      : ''}"
                    data-stage-status={row.status}
                  >
                    <td class="py-1 pr-3 text-fg-1" title={row.key}>
                      {STAGE_LABEL[row.key] ?? row.key}
                    </td>
                    <td class="py-1 pr-3 font-mono text-fg-2">{row.status}</td>
                    <td class="py-1 text-fg-2">
                      {#if row.reason}<span class="mr-2">{row.reason}</span>{/if}
                      {#each row.fields as f (f.name)}
                        <span class="mr-2 inline-block font-mono text-caption"
                          ><span class="text-fg-3">{f.name}</span> {f.value}</span
                        >
                      {/each}
                    </td>
                  </tr>
                {/each}
              </tbody>
            </table>
            <details class="mt-2">
              <summary class="cursor-pointer text-caption text-fg-3 hover:text-fg-2">
                Raw JSON
              </summary>
              <pre
                class="mt-1 max-h-64 overflow-auto rounded bg-surface-1 p-2 font-mono text-caption text-fg-2">{JSON.stringify(
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
