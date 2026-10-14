<script lang="ts">
  import { goto } from '$app/navigation';
  import { resolve } from '$app/paths';
  import { resolveApiUrl } from '$lib/api';
  import { getThumbUrl } from '$lib/api_cropRead';
  import { slotForClassName } from '$lib/annotations/registeredSlots';
  import {
    dominantShareText,
    dominantShareTitle,
    cohesionText,
    COHESION_TOOLTIP,
  } from '$lib/clusters/clusterCardText';
  import { projectHref } from '$lib/projectPaths';
  import type { Cluster } from '$lib/types';

  // One card of the /clusters grid (a real cluster or a synthetic slot
  // inventory card); a click opens the cluster or the slot's inventory.
  let { c, unlabeledOnly }: { c: Cluster; unlabeledOnly: boolean } = $props();

  // Purity banding is served (`purity_tier`, `{API_PREFIX}/clusters`'
  // `purity_thresholds`) — no client 0.8/0.6 threshold. `has_subclusters`
  // is a separate, unrelated signal (AHC sub-clustering ran) and still
  // wins the border color outright.
  function borderColor(c: Cluster): string {
    // M4: a distinct, neutral border for the non-cluster inventory card —
    // the red "noisy" border would otherwise falsely imply a bad cluster.
    if (c.isSlotCard) return 'border-purple-500/60';
    if (c.has_subclusters) return 'border-blue-500/60';
    if (c.purity_tier === 'pure') return 'border-green-500/60';
    if (c.purity_tier === 'mixed') return 'border-orange-500/60';
    return 'border-red-500/60';
  }

  function purityBadge(c: Cluster): { color: string; text: string } | null {
    // M4: a synthetic slot inventory card is not a cluster —
    // it has no purity, so it gets no purity badge at all rather than
    // falling through to an invented "noisy 0%".
    if (c.isSlotCard) return null;
    if (c.purity_tier === 'pure')
      return { color: 'bg-green-500/20 text-green-300', text: 'pure' };
    if (c.purity_tier === 'mixed')
      return { color: 'bg-orange-500/20 text-orange-200', text: 'mixed' };
    return { color: 'bg-red-500/20 text-red-200', text: 'noisy' };
  }

  function open(c: Cluster): void {
    // Special-case: clicking a cluster whose dominant class is bound to
    // a registered slot should jump to that slot's browse view (which
    // surfaces every crop with the slot's sub-bbox), not the
    // single-cluster crop grid. Slot sub-bboxes live on other items, so
    // that class's cluster only contains the rare crops that were labeled
    // with it as their PRIMARY class — usually 1-2 mis-labels. The
    // operator's intent is "show me all the slot's items", so route them
    // to that inventory instead.
    //
    // P2.11 fix: resolve the CLICKED cluster's own dominant class, not
    // "the first slot-bound class" — the latter silently routed every
    // slot-bound cluster to the first registered slot's class filter,
    // which breaks the instant a second capable slot exists.
    if (slotForClassName(c.dominant_class_name) != null && c.dominant_class_id != null) {
      void goto(resolve(projectHref(`/clusters?class=${c.dominant_class_id}`)));
      return;
    }
    void goto(resolve(projectHref(`/clusters/${c.id}`)));
  }

  const pb = $derived(purityBadge(c));
</script>

<li style="content-visibility:auto;contain-intrinsic-size:auto 280px">
  <button
    type="button"
    class="flex w-full flex-col rounded-md border-2 bg-zinc-900 text-left transition hover:border-zinc-300 {borderColor(
      c,
    )}"
    onclick={() => open(c)}
  >
    <div class="grid grid-cols-2 gap-px overflow-hidden rounded-t bg-zinc-950">
      {#each c.representative_crop_ids?.slice(0, 4) ?? [] as cropId, i (i)}
        <img
          src={c.representative_thumb_urls?.[i]
            ? resolveApiUrl(c.representative_thumb_urls[i])
            : getThumbUrl(cropId)}
          alt="thumb"
          loading="lazy"
          class="aspect-square w-full bg-zinc-950 object-contain"
        />
      {/each}
      {#each Array(Math.max(0, 4 - (c.representative_crop_ids?.length ?? 0))) as _, i (i)}
        <div class="aspect-square w-full bg-zinc-900"></div>
      {/each}
    </div>
    <div class="p-3">
      <div class="mb-1 flex items-center gap-2">
        {#if c.isSlotCard}
          <span class="text-sm font-semibold" data-testid="slot-card-title"
            >{c.slotDisplayName ?? c.dominant_class_name}</span
          >
          <span
            class="rounded px-1.5 py-0.5 text-[10px] font-medium bg-purple-500/20 text-purple-200"
          >
            inventory
          </span>
        {:else}
          <span class="text-sm font-semibold">#{c.id}</span>
          {#if pb}
            <!-- DQ-M2 fix (dq-queues cutover, 2026-09-24): purity is now
                 nearest-centroid geometry purity, not the tautological
                 label-based number (always 1.0 for a class cluster) —
                 shown with its basis and n (it's noisy at low n), plus
                 label_purity/labelled_share in the tooltip so an
                 operator can tell the two apart. -->
            <span
              class="rounded px-1.5 py-0.5 text-[10px] font-medium {pb.color}"
              data-testid="cluster-cohesion"
              title="{COHESION_TOOLTIP} · label agreement {c.label_agreement != null
                ? `${(c.label_agreement * 100).toFixed(0)}%`
                : '—'} · labelled share {c.labelled_share != null
                ? `${(c.labelled_share * 100).toFixed(0)}%`
                : '—'}"
            >
              {pb.text} · {cohesionText(c) ?? 'cohesion —'}
            </span>
          {/if}
        {/if}
        {#if c.promotable}
          <span
            class="rounded border border-emerald-500/40 bg-emerald-500/20 px-1.5 py-0.5 text-[10px] text-emerald-200"
            title="Meets the server's auto-promote gate (purity + member count + labelled share)"
          >
            promotable
          </span>
        {/if}
        {#if c.has_subclusters}
          <span
            class="rounded border border-blue-500/40 bg-blue-500/20 px-1.5 py-0.5 text-[10px] text-blue-200"
          >
            AHC
          </span>
        {/if}
        <span class="grow"></span>
        {#if c.isSlotCard}
          <!-- C4: the inventory count's scope, see SlotGallery. -->
          <span
            class="font-mono text-xs text-zinc-400"
            title="Regions the browse endpoint lists: test-holdout items and active filters excluded. The sidebar's class count and the dashboard total include test-holdout items, and the review queue applies its own queue filters, so those can differ."
            >{c.size.toLocaleString()} listed</span
          >
        {:else}
          <span class="font-mono text-xs text-zinc-400">{c.size}</span>
        {/if}
      </div>
      <div
        class="truncate text-sm text-zinc-300"
        title={unlabeledOnly
          ? `Unlabeled cluster #${c.id}`
          : (c.dominant_class_name ?? `Unlabeled cluster #${c.id}`)}
      >
        {#if c.isSlotCard}
          <!-- M4: dominant_pct is meaningless for the inventory
               card too (there's no "dominant" anything — every
               item IS the slot's class) — no invented "· 0%".
               F8 D7: the title carries the served display name,
               so this line names the class once, as a class. -->
          <span class="text-zinc-500">class</span>
          <span class="font-mono">{c.dominant_class_name}</span>
        {:else if c.dominant_class_name && !unlabeledOnly}
          {c.dominant_class_name}
          {#if dominantShareText(c)}
            <span class="text-zinc-500" title={dominantShareTitle(c)}>
              · {dominantShareText(c)}
            </span>
          {/if}
        {:else}
          <span class="text-amber-300">Unlabeled #{c.id}</span>
          <span class="text-zinc-500">· needs label</span>
        {/if}
      </div>
    </div>
  </button>
</li>
