<!--
  The audit queue: drawn crops still waiting for a human label, in the
  served order. Each row shows the crop, the class it carries now, the
  detector's own class and a link that opens it in Review to label it.
-->
<script lang="ts">
  import { resolve } from '$app/paths';
  import { getThumbUrl } from '$lib/api';
  import { projectHref } from '$lib/projectPaths';
  import { percentText } from '$lib/labelConfirmation';
  import { classSourcesStore } from '$stores/classSources.svelte';
  import type { AuditQueuePage } from '$lib/api_labelConfirmation';

  interface Props {
    queue: AuditQueuePage;
    ongoto: (page: number) => void;
  }
  let { queue, ongoto }: Props = $props();

  const lastPage = $derived(Math.max(1, Math.ceil(queue.total / queue.pageSize)));
</script>

<section class="surface flex flex-col gap-3 p-4" data-testid="audit-queue">
  <div class="flex flex-wrap items-center gap-2">
    <h2 class="text-base font-semibold">Waiting for a label</h2>
    <span class="text-xs text-zinc-400" data-testid="audit-queue-total"
      >{queue.total.toLocaleString()} crops</span
    >
  </div>
  {#if queue.items.length === 0}
    <p class="text-sm text-zinc-500">
      Nothing is waiting. Draw a sample to start an audit.
    </p>
  {:else}
    <ul class="grid grid-cols-1 gap-2 sm:grid-cols-2 xl:grid-cols-3">
      {#each queue.items as crop (crop.id)}
        <li
          class="flex min-w-0 items-center gap-3 rounded border border-zinc-800 bg-zinc-900 p-2"
          data-testid="audit-queue-item"
        >
          <img
            src={getThumbUrl(crop.id)}
            alt="crop {crop.id}"
            loading="lazy"
            class="h-16 w-16 shrink-0 rounded bg-zinc-950 object-contain"
          />
          <div class="flex min-w-0 grow flex-col gap-0.5 text-xs">
            <span class="truncate text-zinc-100" title={crop.class_name ?? ''}
              >{crop.class_name ?? 'No class'}</span
            >
            <span class="truncate text-zinc-500">
              {classSourcesStore.labelFor(crop.class_source) || crop.class_source || ''}
              {#if crop.class_confidence != null}
                · {percentText(crop.class_confidence)}
              {/if}
            </span>
            {#if crop.detector_class_name}
              <span class="truncate text-zinc-500" data-testid="audit-queue-detector"
                >Detector: <span class="text-zinc-300"
                  >{crop.detector_class_name}
                  {percentText(crop.detector_confidence)}</span
                ></span
              >
            {/if}
          </div>
          <a
            class="btn shrink-0"
            href={resolve(`${projectHref('/review')}?tab=all&crop_id=${crop.id}`)}
            data-testid="audit-queue-open">Label</a
          >
        </li>
      {/each}
    </ul>
    {#if lastPage > 1}
      <div class="flex items-center gap-2 text-xs text-zinc-400">
        <button
          type="button"
          class="btn"
          disabled={queue.page <= 1}
          onclick={() => ongoto(queue.page - 1)}>Previous</button
        >
        <span>Page {queue.page} of {lastPage}</span>
        <button
          type="button"
          class="btn"
          disabled={queue.page >= lastPage}
          onclick={() => ongoto(queue.page + 1)}>Next</button
        >
      </div>
    {/if}
  {/if}
</section>
