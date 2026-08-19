<script lang="ts">
  import { focusOnMount } from '$lib/actions/focusOnMount';
  /**
   * Promote-to-Triton modal. Phase 4 wiring; the labeler-side knob set
   * is intentionally minimal — Triton model name + max_batch_size +
   * fp16 + overwrite. The backend handles ONNX → JIT TensorRT compile.
   */
  import { promoteTrainJob } from '$lib/api';
  import { toastStore } from '$stores/toast.svelte';
  import type { PromoteRequest, PromoteResponse } from '$lib/types_train';

  interface Props {
    open: boolean;
    jobId: string | null;
    /** Server suggests `<run_name>_v7`; we default to that. */
    defaultName?: string;
    onclose: () => void;
    onpromoted?: (res: PromoteResponse) => void;
  }

  let { open, jobId, defaultName, onclose, onpromoted }: Props = $props();

  let tritonName = $state<string>('');
  let maxBatch = $state<number>(8);
  let fp16 = $state<boolean>(true);
  let overwrite = $state<boolean>(false);
  let busy = $state<boolean>(false);
  let error = $state<string | null>(null);

  // Re-seed the form when the parent opens us with a new defaultName.
  $effect(() => {
    if (!open) return;
    tritonName = defaultName ?? '';
    maxBatch = 8;
    fp16 = true;
    overwrite = false;
    error = null;
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
      const res = await promoteTrainJob(jobId, body);
      toastStore.success(`Promoted ${res.triton_name} → Triton`);
      onpromoted?.(res);
      onclose();
    } catch (e) {
      error = (e as Error).message;
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
            placeholder="yolo26m_v7_2026-05-09"
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

        {#if error}
          <p class="mb-3 text-xs text-red-300">{error}</p>
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
            {busy ? 'Promoting…' : 'Promote'}
          </button>
        </div>
      </form>
    </div>
  </div>
{/if}
