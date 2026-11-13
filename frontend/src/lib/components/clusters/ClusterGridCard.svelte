<script lang="ts">
  import { goto } from '$app/navigation';
  import { resolve } from '$app/paths';
  import { resolveApiUrl } from '$lib/api';
  import { getThumbUrl } from '$lib/api_cropRead';
  import {
    cardHoverText,
    cardMeta,
    cardStatus,
    displayNameNote,
    displayNoText,
  } from '$lib/clusters/clusterCardText';
  import { clusterHref } from '$lib/clusters/clusterHref';
  import type { Cluster } from '$lib/types';
  import type { ClusterFlag } from '$lib/types_clusterList';

  // One card of the cluster grid: the served name and size, one meta line, at most one status.
  // A click or Enter opens the cluster (or the slot's gallery for a slot-bound class).
  let {
    c,
    flagVocabulary,
    tabindex = 0,
  }: { c: Cluster; flagVocabulary: ClusterFlag[]; tabindex?: number } = $props();

  function open(c: Cluster): void {
    void goto(resolve(clusterHref(c.uid, c.dominant_class_name)));
  }

  const numberText = $derived(displayNoText(c));
  const meta = $derived(cardMeta(c));
  const status = $derived(cardStatus(c, flagVocabulary));
  const metaLine = $derived(
    [
      c.display_name_kind === 'none'
        ? 'needs a label'
        : displayNameNote(c.display_name_kind),
      meta,
    ]
      .filter(Boolean)
      .join(' \u00b7 '),
  );
</script>

<li style="content-visibility:auto;contain-intrinsic-size:auto 250px">
  <button
    type="button"
    class="flex w-full flex-col rounded-md border-2 border-line-strong bg-surface-1 text-left transition hover:border-fg-3"
    title={cardHoverText(c)}
    data-grid-item
    {tabindex}
    onclick={() => open(c)}
  >
    <div class="grid grid-cols-2 gap-px overflow-hidden rounded-t bg-surface-0">
      {#each c.representative_crop_ids?.slice(0, 4) ?? [] as cropId, i (i)}
        <img
          src={c.representative_thumb_urls?.[i]
            ? resolveApiUrl(c.representative_thumb_urls[i])
            : getThumbUrl(cropId)}
          alt=""
          loading="lazy"
          onerror={(e) => {
            (e.currentTarget as HTMLImageElement).style.opacity = '0.2';
          }}
          class="aspect-square w-full bg-surface-0 object-contain"
        />
      {/each}
      {#each Array(Math.max(0, 4 - (c.representative_crop_ids?.length ?? 0))) as _, i (i)}
        <div class="aspect-square w-full bg-surface-1"></div>
      {/each}
    </div>
    <div class="flex flex-col gap-1 p-3">
      <div class="flex items-baseline gap-2">
        {#if c.display_name_kind !== 'none'}
          <span class="truncate text-body font-semibold" data-testid="cluster-name"
            >{c.display_name}</span
          >
        {:else}
          <span class="truncate text-body font-semibold text-warn-text"
            >Unlabeled <span title={c.uid} data-testid="cluster-uid">{numberText}</span
            ></span
          >
        {/if}
        {#if c.display_name_kind !== 'none' && numberText}
          <span
            class="shrink-0 text-meta text-fg-3 tabular-nums"
            title={c.uid}
            data-testid="cluster-uid">{numberText}</span
          >
        {/if}
        <span class="grow"></span>
        <span class="shrink-0 text-body text-fg-1 tabular-nums" data-testid="cluster-size"
          >{c.size.toLocaleString()}</span
        >
      </div>
      <p class="truncate text-meta text-fg-2" data-testid="cluster-meta">{metaLine}</p>
      {#if status}
        <span
          class="chip self-start {status.tone === 'warn' ? 'chip-warn' : 'chip-info'}"
          title={status.title}
          data-testid="cluster-status">{status.label}</span
        >
      {/if}
    </div>
  </button>
</li>
