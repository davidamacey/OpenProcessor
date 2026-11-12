<script lang="ts">
  import { goto } from '$app/navigation';
  import { resolve } from '$app/paths';
  import { resolveApiUrl } from '$lib/api';
  import { getThumbUrl } from '$lib/api_cropRead';
  import { slotForClassName } from '$lib/annotations/registeredSlots';
  import {
    displayNoText,
    dominantShareText,
    dominantShareTitle,
    DETECTOR_TOOLTIP,
    VALIDATION_TEXT,
    detectorText,
    reviewCountsTitle,
  } from '$lib/clusters/clusterCardText';
  import { clusterScopeSearch } from '$lib/workspace/workspaceUrl';
  import { projectHref } from '$lib/projectPaths';
  import type { Cluster } from '$lib/types';
  import type { ClusterFlag } from '$lib/types_clusterList';

  // One card of the /clusters grid (a real cluster or a synthetic slot
  // inventory card); a click opens the cluster or the slot's inventory.
  let { c, flagVocabulary }: { c: Cluster; flagVocabulary: ClusterFlag[] } = $props();

  function borderColor(c: Cluster): string {
    // M4: a distinct, neutral border for the non-cluster inventory card.
    if (c.isSlotCard) return 'border-holdout/60';
    return 'border-line-strong';
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
    if (slotForClassName(c.dominant_class_name) != null && c.dominant_class_name) {
      void goto(
        resolve(
          projectHref(
            `/workspace?view=clusters&class_name=${encodeURIComponent(c.dominant_class_name)}`,
          ),
        ),
      );
      return;
    }
    void goto(resolve(projectHref(`/workspace${clusterScopeSearch(c.uid)}`)));
  }

  const numberText = $derived(displayNoText(c));
  const detector = $derived(c.isSlotCard ? null : detectorText(c));
  const flagById = $derived(new Map(flagVocabulary.map((f) => [f.id, f])));
</script>

<li style="content-visibility:auto;contain-intrinsic-size:auto 280px">
  <button
    type="button"
    class="flex w-full flex-col rounded-md border-2 bg-surface-1 text-left transition hover:border-fg-3 {borderColor(
      c,
    )}"
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
    <div class="p-3">
      <div class="mb-1 flex items-center gap-2">
        {#if c.isSlotCard}
          <span class="text-sm font-semibold" data-testid="slot-card-title"
            >{c.slotDisplayName ?? c.dominant_class_name}</span
          >
          <span
            class="rounded px-1.5 py-0.5 text-caption font-medium bg-holdout/20 text-holdout"
          >
            inventory
          </span>
        {:else}
          <span
            class="font-mono text-sm font-semibold"
            title={c.uid}
            data-testid="cluster-uid">{numberText}</span
          >
          {#if c.label_agreement != null}
            <span
              class="rounded bg-surface-2 px-1.5 py-0.5 text-caption font-medium text-fg-2"
              data-testid="cluster-agreement"
              title="Label agreement: the top class's share of the labelled members · labelled share {c.labelled_share !=
              null
                ? `${(c.labelled_share * 100).toFixed(0)}%`
                : '—'}"
            >
              agreement {(c.label_agreement * 100).toFixed(0)}%
            </span>
          {/if}
        {/if}
        <span class="grow"></span>
        {#if c.isSlotCard}
          <!-- C4: the inventory count's scope, see SlotGallery. -->
          <span
            class="font-mono text-xs text-fg-2"
            title="Regions the browse endpoint lists: test-holdout items and active filters excluded. The sidebar's class count and the dashboard total include test-holdout items, and the review queue applies its own queue filters, so those can differ."
            >{c.size.toLocaleString()} listed</span
          >
        {:else}
          <span class="font-mono text-xs text-fg-2">{c.size}</span>
        {/if}
      </div>
      <div
        class="truncate text-sm text-fg-2"
        title={c.dominant_class_name ?? `Unlabeled cluster ${c.uid}`}
      >
        {#if c.isSlotCard}
          <!-- M4: dominant_pct is meaningless for the inventory
               card too (there's no "dominant" anything — every
               item IS the slot's class) — no invented "· 0%".
               F8 D7: the title carries the served display name,
               so this line names the class once, as a class. -->
          <span class="text-fg-3">class</span>
          <span class="font-mono">{c.dominant_class_name}</span>
        {:else if c.dominant_class_name}
          {c.dominant_class_name}
          {#if dominantShareText(c)}
            <span class="text-fg-3" title={dominantShareTitle(c)}>
              · {dominantShareText(c)}
            </span>
          {/if}
        {:else}
          <span class="text-warn">Unlabeled {numberText}</span>
          <span class="text-fg-3">· needs label</span>
        {/if}
      </div>
      {#if !c.isSlotCard}
        <div class="mt-1.5 flex flex-wrap items-center gap-1 text-caption">
          {#if detector}
            <span
              class="rounded border border-line-strong px-1.5 py-0.5 text-fg-2"
              data-testid="cluster-detector"
              title={DETECTOR_TOOLTIP}>detector {detector}</span
            >
          {/if}
          {#if c.validation_state}
            <span
              class="rounded border border-line-strong px-1.5 py-0.5 text-fg-2"
              data-testid="cluster-validation"
              title="{c.validated_count} of {c.size} members have a human-validated class"
              >{VALIDATION_TEXT[c.validation_state]}</span
            >
          {/if}
          {#if c.needs_review_count}
            <span
              class="rounded border border-warn/40 bg-warn/10 px-1.5 py-0.5 text-warn"
              data-testid="cluster-needs-review"
              title={reviewCountsTitle(c.review_counts)}
              >{c.needs_review_count.toLocaleString()} to review</span
            >
          {/if}
          {#each c.flags ?? [] as id (id)}
            <span
              class="rounded border border-danger/30 bg-danger/10 px-1.5 py-0.5 text-danger"
              data-testid="cluster-flag"
              title={flagById.get(id)?.description}>{flagById.get(id)?.label ?? id}</span
            >
          {/each}
        </div>
      {/if}
    </div>
  </button>
</li>
