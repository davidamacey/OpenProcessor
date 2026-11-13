<script lang="ts">
  /**
   * Step 4 — the served preview, as served: the verdict, errors and
   * warnings, per-source counts, the target block, dedup counts and the
   * link/copy byte estimate. No count is computed here.
   */
  import CombineIssueList from '$components/combine/CombineIssueList.svelte';
  import { formatBytes } from '$lib/format/bytes';
  import { formatCount } from '$lib/format/count';
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
  <h2 class="text-body font-semibold text-fg-1">4. Preview</h2>
  <p
    class="text-body {preview.ok ? 'text-success-text' : 'text-danger-text'}"
    data-testid="combine-preview-verdict"
  >
    {preview.ok ? 'No blocking problems.' : 'This combine cannot start yet.'}
    {#if stale}<span class="ml-2 text-meta text-warn-text">Updating…</span>{/if}
  </p>

  <CombineIssueList issues={preview.errors} testid="combine-errors" />
  <CombineIssueList issues={preview.warnings} testid="combine-warnings" />

  <div class="overflow-x-auto rounded border border-line">
    <table class="w-full text-meta" data-testid="combine-preview-sources">
      <thead class="bg-surface-1 text-left text-fg-2">
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
          <tr class="border-t border-line">
            <td class="px-2 py-1 text-fg-1"><code>{s.project}</code></td>
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
    class="rounded border border-line p-2 text-meta"
    data-testid="combine-preview-target"
  >
    <p class="mb-1 text-fg-2">
      Target <code class="text-fg-1">{target.slug ?? '—'}</code>
      {#if target.slug_available === false}
        <span class="ml-1 text-danger-text" data-testid="combine-slug-unavailable"
          >slug not available</span
        >
      {:else if target.slug_available === true}
        <span class="ml-1 text-success-text">slug available</span>
      {/if}
    </p>
    <p class="text-fg-2">
      {formatCount(target.projected_images)} images · {formatCount(
        target.projected_items,
      )} items ·
      {formatCount(target.unclassed_items)} unclassed ·
      {formatCount(target.holdout_images)} test images
    </p>
    {#if (target.classes ?? []).length > 0}
      <ul class="mt-1 space-y-0.5" data-testid="combine-preview-classes">
        {#each target.classes ?? [] as c (c.id)}
          <li class="text-fg-2">
            <span class="text-fg-1">{c.name}</span>
            <span class="tabular-nums text-fg-2">· {formatCount(c.count)}</span>
            {#if (c.from ?? []).length > 0}
              <span class="text-fg-3">
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
    class="rounded border border-line p-2 text-meta"
    data-testid="combine-preview-dedup"
  >
    <p class="mb-1 text-fg-2">Duplicates</p>
    <p class="text-fg-2">
      {formatCount(dedup.identical_images)} identical images ·
      {formatCount(dedup.merged_items)} merged boxes ·
      {formatCount(dedup.conflicts)} conflicts ·
      {dedup.near_duplicate_pairs_estimate == null
        ? 'near duplicates: not computed'
        : `${dedup.near_duplicate_pairs_estimate} near-duplicate pairs (estimate)`}
    </p>
    {#if (dedup.conflict_samples ?? []).length > 0}
      <details class="mt-1" data-testid="combine-conflict-samples">
        <summary class="cursor-pointer text-fg-2">Conflict samples</summary>
        <pre
          class="mt-1 max-h-48 overflow-auto rounded bg-surface-0 p-1 text-caption text-fg-2">{JSON.stringify(
            dedup.conflict_samples,
            null,
            2,
          )}</pre>
      </details>
    {/if}
  </div>

  <p class="text-meta text-fg-2" data-testid="combine-preview-bytes">
    Images to link: {formatBytes(preview.bytes?.to_link)} · to copy: {formatBytes(
      preview.bytes?.to_copy,
    )}
  </p>
</section>
