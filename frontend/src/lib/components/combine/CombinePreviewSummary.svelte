<script lang="ts">
  /**
   * Step 4 — the served preview, as served: the verdict, errors and
   * warnings, per-source counts, the target block, dedup counts and the
   * link/copy byte estimate. No count is computed here.
   */
  import CombineIssueList from '$components/combine/CombineIssueList.svelte';
  import { formatBytes } from '$lib/combine/combineText';
  import { formatCount } from '$lib/formatCount';
  import type { CombinePreview } from '$lib/types_combine';

  interface Props {
    preview: CombinePreview;
    /** The request changed since this preview was served. */
    stale?: boolean;
  }
  let { preview, stale = false }: Props = $props();

  const dedup = $derived(preview.dedup ?? {});
  const target = $derived(preview.target ?? {});
</script>

<section
  class="space-y-3"
  data-testid="combine-preview"
  data-ok={preview.ok}
  data-stale={stale}
>
  <h2 class="text-sm font-semibold text-zinc-200">4. Preview</h2>
  <p
    class="text-sm {preview.ok ? 'text-emerald-300' : 'text-red-300'}"
    data-testid="combine-preview-verdict"
  >
    {preview.ok ? 'No blocking problems.' : 'This combine cannot start yet.'}
    {#if stale}<span class="ml-2 text-xs text-amber-300">Updating…</span>{/if}
  </p>

  <CombineIssueList issues={preview.errors} testid="combine-errors" />
  <CombineIssueList issues={preview.warnings} testid="combine-warnings" />

  <div class="overflow-x-auto rounded border border-zinc-800">
    <table class="w-full text-xs" data-testid="combine-preview-sources">
      <thead class="bg-zinc-900 text-left text-zinc-400">
        <tr>
          <th class="px-2 py-1 font-normal">Source</th>
          <th class="px-2 py-1 text-right font-normal">Images</th>
          <th class="px-2 py-1 text-right font-normal">Items</th>
          <th class="px-2 py-1 text-right font-normal">Labeled</th>
          <th class="px-2 py-1 text-right font-normal">Test images</th>
        </tr>
      </thead>
      <tbody>
        {#each preview.sources as s (s.project)}
          <tr class="border-t border-zinc-800">
            <td class="px-2 py-1 font-mono text-zinc-100">{s.project}</td>
            <td class="px-2 py-1 text-right tabular-nums">{formatCount(s.images)}</td>
            <td class="px-2 py-1 text-right tabular-nums">{formatCount(s.items)}</td>
            <td class="px-2 py-1 text-right tabular-nums"
              >{formatCount(s.labeled_items)}</td
            >
            <td class="px-2 py-1 text-right tabular-nums"
              >{formatCount(s.holdout_images)}</td
            >
          </tr>
        {/each}
      </tbody>
    </table>
  </div>

  <div
    class="rounded border border-zinc-800 p-2 text-xs"
    data-testid="combine-preview-target"
  >
    <p class="mb-1 text-zinc-300">
      Target <span class="font-mono text-zinc-100">{target.slug ?? '—'}</span>
      {#if target.slug_available === false}
        <span class="ml-1 text-red-300" data-testid="combine-slug-unavailable"
          >slug not available</span
        >
      {:else if target.slug_available === true}
        <span class="ml-1 text-emerald-300">slug available</span>
      {/if}
    </p>
    <p class="text-zinc-400">
      {formatCount(target.projected_images)} images · {formatCount(
        target.projected_items,
      )} items ·
      {formatCount(target.unclassed_items)} unclassed ·
      {formatCount(target.holdout_images)} test images
    </p>
    {#if (target.classes ?? []).length > 0}
      <ul class="mt-1 space-y-0.5" data-testid="combine-preview-classes">
        {#each target.classes ?? [] as c (c.id)}
          <li class="text-zinc-300">
            <span class="text-zinc-100">{c.name}</span>
            <span class="tabular-nums text-zinc-400">· {formatCount(c.count)}</span>
            {#if (c.from ?? []).length > 0}
              <span class="text-zinc-500">
                from {(c.from ?? [])
                  .map((f) => `${f.project}/${f.class}`)
                  .join(', ')}</span
              >
            {/if}
          </li>
        {/each}
      </ul>
    {/if}
  </div>

  <div
    class="rounded border border-zinc-800 p-2 text-xs"
    data-testid="combine-preview-dedup"
  >
    <p class="mb-1 text-zinc-300">Duplicates</p>
    <p class="text-zinc-400">
      {formatCount(dedup.identical_images)} identical images ·
      {formatCount(dedup.merged_items)} merged boxes ·
      {formatCount(dedup.conflicts)} conflicts ·
      {dedup.near_duplicate_pairs_estimate == null
        ? 'near duplicates: not computed'
        : `${dedup.near_duplicate_pairs_estimate} near-duplicate pairs (estimate)`}
    </p>
    {#if (dedup.conflict_samples ?? []).length > 0}
      <details class="mt-1" data-testid="combine-conflict-samples">
        <summary class="cursor-pointer text-zinc-400">Conflict samples</summary>
        <pre
          class="mt-1 max-h-48 overflow-auto rounded bg-zinc-950 p-1 text-[10px] text-zinc-300">{JSON.stringify(
            dedup.conflict_samples,
            null,
            2,
          )}</pre>
      </details>
    {/if}
  </div>

  <p class="text-xs text-zinc-400" data-testid="combine-preview-bytes">
    Images to link: {formatBytes(preview.bytes?.to_link)} · to copy: {formatBytes(
      preview.bytes?.to_copy,
    )}
  </p>
</section>
