<script lang="ts">
  /**
   * The audit sample still waiting for a person: "300 drawn · 112 labelled · 188 waiting", the
   * way into Review (the one primary), and a read-only strip of the first waiting crops
   * (`GET /crops?review_tab=audit`). "See all" opens Detections on the same queue. The counts
   * are the served report's; the strip is what the served queue lists.
   */
  import { resolve } from '$app/paths';
  import { onMount } from 'svelte';
  import CropCard from '$components/CropCard.svelte';
  import { getCrops } from '$lib/api_cropRead';
  import { apiErrorText } from '$lib/api_errors';
  import { plural } from '$lib/format/plural';
  import { projectHref } from '$lib/projectPaths';
  import { reviewSearch } from '$lib/workspace/workspaceUrl';
  import type { Crop } from '$lib/types';

  interface Props {
    /** Crops a person has labelled. */
    audited: number;
    /** Crops still waiting. */
    pending: number;
  }
  let { audited, pending }: Props = $props();

  const STRIP = 24;
  let crops = $state<Crop[]>([]);
  let error = $state<string | null>(null);

  onMount(() => {
    const ctl = new AbortController();
    getCrops({ review_tab: 'audit', limit: STRIP }, null, ctl.signal)
      .then((page) => (crops = page.items))
      .catch((e: unknown) => {
        if ((e as Error)?.name !== 'AbortError') error = apiErrorText(e);
      });
    return () => ctl.abort();
  });

  const noop = (): void => {};
</script>

<section class="surface flex flex-col gap-3 p-4" data-testid="audit-pending-sample">
  <div class="flex flex-wrap items-center gap-x-4 gap-y-2">
    <p class="text-body text-fg-1" data-testid="audit-pending-summary">
      <span class="tabular-nums">{(audited + pending).toLocaleString()}</span> drawn ·
      <span class="tabular-nums">{audited.toLocaleString()}</span> labelled ·
      <span class="tabular-nums">{pending.toLocaleString()}</span> waiting
    </p>
    <span class="grow"></span>
    <a
      class="btn btn-primary"
      href={resolve(projectHref(`/workspace${reviewSearch({ queue: 'audit' })}`))}
      data-testid="audit-open-queue">Label them in Review</a
    >
  </div>
  {#if error}
    <p class="text-meta text-danger-text">{error}</p>
  {:else if crops.length > 0}
    <ul
      class="grid grid-cols-[repeat(auto-fill,minmax(7rem,1fr))] gap-2"
      data-testid="audit-pending-strip"
    >
      {#each crops as crop (crop.id)}
        <li>
          <CropCard
            {crop}
            onclick={noop}
            onacceptVlm={noop}
            onrejectVlm={noop}
            ondetail={noop}
          />
        </li>
      {/each}
    </ul>
    {#if pending > crops.length}
      <a
        class="text-meta text-accent-text hover:underline"
        href={resolve(projectHref('/workspace?view=detections&queue=audit'))}
        data-testid="audit-see-all">See all {plural(pending, 'waiting crop')}</a
      >
    {/if}
  {/if}
</section>
