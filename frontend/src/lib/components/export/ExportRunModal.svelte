<script lang="ts">
  /** /export's YOLO export progress and result modal. */
  import { getManifestUrl } from '$lib/api';
  import { formatCount } from '$lib/format/count';
  import Modal from '$components/Modal.svelte';
  import {
    downloadUrl,
    type ExportRunController,
  } from '$lib/export/exportRunController.svelte';

  interface Props {
    run: ExportRunController;
  }

  let { run }: Props = $props();
</script>

<!-- Export progress modal -->
{#if run.exportModalOpen}
  <Modal label="Export progress" onclose={run.closeExportModal}>
    <h3 class="mb-3 text-base font-semibold">YOLO export</h3>
    {#if run.exportState?.status === 'running' || run.exportRunning}
      <p class="mb-3 text-xs text-zinc-400">
        Running… polling every 5s. Safe to leave the page open.
      </p>
      {#if run.exportState?.progress != null}
        <div class="mb-2 h-2 w-full overflow-hidden rounded bg-zinc-900">
          <div
            class="h-full bg-blue-500"
            style:width="{Math.round((run.exportState.progress ?? 0) * 100)}%"
          ></div>
        </div>
      {/if}
      <p class="text-xs font-mono text-zinc-300">{run.exportState?.message ?? '…'}</p>
    {:else if run.exportState?.status === 'success'}
      <p class="mb-2 text-sm text-green-300">Export complete.</p>
      {#if run.exportState.export_dir}
        <p class="mb-3 break-all font-mono text-xs text-zinc-300">
          {run.exportState.export_dir}
        </p>
      {/if}
      {#if run.exportState.split_counts || run.exportState.image_count != null || run.exportState.object_count != null}
        <p class="mb-3 font-mono text-xs text-zinc-300">
          {formatCount(run.exportState.object_count)} objects in {formatCount(
            run.exportState.image_count,
          )} images
          {run.exportState.class_count != null
            ? `· ${run.exportState.class_count} classes`
            : ''}
          {#if run.exportState.split_counts}
            · images train {run.exportState.split_counts.train.toLocaleString()} · val {run.exportState.split_counts.val.toLocaleString()}
            · test {run.exportState.split_counts.test.toLocaleString()}
          {/if}
        </p>
      {/if}
      {#if run.exportState.images_dropped_not_fully_labeled}
        <p class="mb-3 text-xs text-orange-300">
          {formatCount(run.exportState.images_dropped_not_fully_labeled)} image(s) dropped —
          not fully labeled.
        </p>
      {/if}
      <div class="flex flex-wrap gap-2">
        <button
          type="button"
          class="btn"
          onclick={() => downloadUrl(getManifestUrl(), 'manifest.json')}
        >
          Download manifest.json
        </button>
      </div>
    {:else if run.exportState?.status === 'failed'}
      <p class="mb-2 text-sm text-red-300">Export failed.</p>
      <p class="font-mono text-xs text-red-200">
        {run.exportState.error ?? 'unknown error'}
      </p>
    {:else}
      <p class="text-xs text-zinc-400">No active export.</p>
    {/if}
    <div class="mt-4 flex justify-end">
      <button
        type="button"
        class="btn"
        onclick={run.closeExportModal}
        disabled={run.exportRunning}>Close</button
      >
    </div>
  </Modal>
{/if}
