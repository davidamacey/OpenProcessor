<script lang="ts">
  import { apiErrorText } from '$lib/api';
  import { focusOnMount } from '$lib/actions/focusOnMount';
  /**
   * Promote-to-Triton modal. Phase 4 wiring; the labeler-side knob set
   * is intentionally minimal — Triton model name + max_batch_size +
   * fp16 + overwrite. The backend handles ONNX → JIT TensorRT compile.
   *
   * F-64: a promote-gate 422 renders the served message and every served
   * failure, and a "force" checkbox appears only when the server says
   * `force_allowed`.
   *
   * #87: promote is a background job. After submit the modal follows the
   * served phases (queued, exporting, loading, building, warming) until
   * `done` or `failed`; closing it mid-job leaves the job running, and
   * `attachJob` re-attaches to one already running (page reload).
   */
  import {
    PROMOTE_PHASES,
    PromoteJobController,
    isPromoteActive,
  } from '$lib/promoteJobController.svelte';
  import {
    gateFailureMessage,
    promoteGateDetail,
    promoteSuccessMessage,
    type PromoteGateDetail,
  } from '$lib/promote';
  import { toastStore } from '$stores/toast.svelte';
  import type {
    PromoteJobStatus,
    PromoteRequest,
    PromoteResponse,
  } from '$lib/types_train';

  interface Props {
    open: boolean;
    jobId: string | null;
    /** Default Triton model name (`defaultTritonName(job_id)`). */
    defaultName?: string;
    /** A promote already served for this run (`TrainJobStatus.promote`);
     *  an active one is followed instead of showing the form. */
    attachJob?: PromoteJobStatus | null;
    onclose: () => void;
    onpromoted?: (res: PromoteResponse) => void;
  }

  let { open, jobId, defaultName, attachJob, onclose, onpromoted }: Props = $props();

  let tritonName = $state<string>('');
  let maxBatch = $state<number>(8);
  let fp16 = $state<boolean>(true);
  let overwrite = $state<boolean>(false);
  let busy = $state<boolean>(false);
  let error = $state<string | null>(null);
  let gate = $state<PromoteGateDetail | null>(null);
  let force = $state<boolean>(false);

  const ctl = new PromoteJobController({
    onDone: (result, job) => {
      toastStore.success(
        result ? promoteSuccessMessage(result) : `Promoted ${job.triton_name} → Triton`,
      );
      if (result) onpromoted?.(result);
      onclose();
    },
  });
  const phaseIndex = $derived(
    ctl.job ? (PROMOTE_PHASES as readonly string[]).indexOf(ctl.job.status) : -1,
  );
  const following = $derived(ctl.job !== null && (ctl.active || ctl.failureText));

  $effect(() => () => ctl.stop());

  // Re-seed the form when the parent opens us with a new defaultName.
  $effect(() => {
    if (!open) return;
    tritonName = defaultName ?? '';
    maxBatch = 8;
    fp16 = true;
    overwrite = false;
    error = null;
    gate = null;
    force = false;
    ctl.reset();
    if (jobId && attachJob && isPromoteActive(attachJob)) ctl.attach(jobId, attachJob);
  });

  async function submit(): Promise<void> {
    if (!jobId || !tritonName.trim()) return;
    error = null;
    busy = true;
    try {
      const body: PromoteRequest = {
        triton_name: tritonName.trim(),
        max_batch_size: maxBatch,
        fp16,
        overwrite,
      };
      if (force && gate?.force_allowed) body.force = true;
      await ctl.start(jobId, body);
    } catch (e) {
      const served = promoteGateDetail(e);
      if (served) {
        gate = served;
        error = null;
        if (!served.force_allowed) force = false;
      } else {
        gate = null;
        error = apiErrorText(e);
      }
    } finally {
      busy = false;
    }
  }
</script>

{#if open}
  <div
    class="fixed inset-0 z-40 flex items-center justify-center bg-black/60 p-4"
    role="dialog"
    aria-modal="true"
    aria-label="Promote to Triton"
    use:focusOnMount
    tabindex="-1"
    onclick={(e) => {
      // Backdrop only: a click that bubbled up from the panel is not a
      // dismiss gesture.
      if (e.target === e.currentTarget) onclose();
    }}
    onkeydown={(e) => e.key === 'Escape' && onclose()}
  >
    <div
      class="w-full max-w-md rounded-lg border border-zinc-800 bg-zinc-950 p-5 shadow-2xl"
    >
      <h3 class="mb-3 text-base font-semibold">Promote to Triton</h3>
      {#if jobId}
        <p class="mb-3 truncate font-mono text-[11px] text-zinc-500" title={jobId}>
          {jobId}
        </p>
      {/if}

      {#if following && ctl.job}
        <div data-testid="promote-progress" class="mb-3 text-sm">
          <p class="mb-2 text-xs text-zinc-400">
            Promoting <span class="font-mono">{ctl.job.triton_name}</span> — takes 2 to 3 minutes.
            You can close this; it keeps running.
          </p>
          <ol class="space-y-1">
            {#each PROMOTE_PHASES as phase, i (phase)}
              <li
                data-testid={`promote-phase-${phase}`}
                aria-current={i === phaseIndex ? 'step' : undefined}
                class="flex items-center gap-2 text-xs {ctl.job.status === 'failed'
                  ? 'text-zinc-500'
                  : i < phaseIndex
                    ? 'text-emerald-300'
                    : i === phaseIndex
                      ? 'font-medium text-blue-300'
                      : 'text-zinc-500'}"
              >
                <span class="w-3 text-center">
                  {i < phaseIndex ? '✓' : i === phaseIndex ? '●' : '○'}
                </span>
                {phase}
              </li>
            {/each}
          </ol>
          {#if ctl.failureText}
            <p
              class="mt-3 rounded-md border border-red-500/40 bg-red-500/10 p-2 text-xs text-red-200"
              data-testid="promote-failed"
              role="alert"
            >
              {ctl.failureText}
            </p>
          {/if}
          {#if ctl.error}
            <p class="mt-3 text-xs text-red-300" role="alert">{ctl.error}</p>
          {/if}
        </div>
        <div class="flex items-center justify-end gap-2">
          {#if ctl.failureText}
            <button type="button" class="btn" onclick={() => ctl.reset()}>
              Edit and retry
            </button>
          {/if}
          <button type="button" class="btn" onclick={onclose}>
            {ctl.active ? 'Close (keeps running)' : 'Close'}
          </button>
        </div>
      {:else}
        <form
          onsubmit={async (e) => {
            e.preventDefault();
            await submit();
          }}
        >
          <label class="mb-3 block text-sm">
            <span class="mb-1 block text-xs text-zinc-400">Triton model name</span>
            <input
              type="text"
              bind:value={tritonName}
              required
              placeholder="model_name"
              pattern="[A-Za-z0-9_-]+"
              class="w-full rounded-md border border-zinc-700 bg-zinc-900 px-2 py-1.5 font-mono text-sm text-zinc-100 focus:border-blue-500 focus:outline-none"
            />
            <span class="mt-1 block text-[11px] text-zinc-500">
              Alphanumeric + underscore + hyphen. Becomes the Triton model directory name.
            </span>
          </label>

          <div class="mb-3 grid grid-cols-2 gap-3">
            <label class="block text-sm">
              <span class="mb-1 block text-xs text-zinc-400">Max batch size</span>
              <input
                type="number"
                min="1"
                max="64"
                bind:value={maxBatch}
                class="w-full rounded-md border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-sm text-zinc-100 focus:border-blue-500 focus:outline-none"
              />
            </label>
            <div class="flex flex-col gap-1.5 pt-5 text-xs text-zinc-300">
              <label class="flex items-center gap-2">
                <input
                  type="checkbox"
                  bind:checked={fp16}
                  class="h-4 w-4 cursor-pointer accent-blue-500"
                />
                FP16 precision
              </label>
              <label class="flex items-center gap-2">
                <input
                  type="checkbox"
                  bind:checked={overwrite}
                  class="h-4 w-4 cursor-pointer accent-blue-500"
                />
                Overwrite existing
              </label>
            </div>
          </div>

          {#if gate}
            <div
              class="mb-3 rounded-md border border-red-500/40 bg-red-500/10 p-2 text-xs text-red-200"
              data-testid="promote-gate"
            >
              <p class="font-medium">{gate.message}</p>
              {#if gate.failures.length > 0}
                <ul
                  class="mt-1 list-disc space-y-0.5 pl-4"
                  data-testid="promote-gate-failures"
                >
                  {#each gate.failures as f, i (f.code + (f.class_name ?? '') + i)}
                    <li>
                      {#if f.class_name}<span class="font-mono">{f.class_name}:</span
                        >{/if}
                      {gateFailureMessage(f)}
                    </li>
                  {/each}
                </ul>
              {/if}
              {#if gate.override}
                <p class="mt-1 text-red-200/80">{gate.override}</p>
              {/if}
            </div>
            {#if gate.force_allowed}
              <label class="mb-3 flex items-center gap-2 text-xs text-amber-200">
                <input
                  type="checkbox"
                  bind:checked={force}
                  data-testid="promote-force"
                  class="h-4 w-4 cursor-pointer accent-amber-500"
                />
                Promote anyway (bypass the gate)
              </label>
            {/if}
          {/if}
          {#if error}
            <p class="mb-3 text-xs text-red-300">{error}</p>
          {/if}
          {#if ctl.error}
            <p class="mb-3 text-xs text-red-300" role="alert">{ctl.error}</p>
          {/if}

          <div class="flex items-center justify-end gap-2">
            <button type="button" class="btn" onclick={onclose} disabled={busy}>
              Cancel
            </button>
            <button
              type="submit"
              class="btn btn-primary"
              disabled={busy || !tritonName.trim()}
            >
              {busy
                ? 'Promoting…'
                : force && gate?.force_allowed
                  ? 'Force promote'
                  : 'Promote'}
            </button>
          </div>
        </form>
      {/if}
    </div>
  </div>
{/if}
