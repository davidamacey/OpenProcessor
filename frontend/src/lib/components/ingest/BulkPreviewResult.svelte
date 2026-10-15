<script lang="ts">
  /**
   * What `POST /ingest/bulk/preview` served for a source: counts, bytes, extension
   * breakdown, the image-size sample, the skipped entries and the estimate with the
   * measured figures it rests on. Everything shown is the served value; nothing is
   * counted or estimated here.
   */
  import { formatBytes } from '$lib/format/bytes';
  import { formatDuration } from '$lib/format/duration';
  import type { BulkPreview, BulkPreviewConfig } from '$lib/types_bulkIngest';

  interface Props {
    preview: BulkPreview;
    config: BulkPreviewConfig;
  }
  let { preview, config }: Props = $props();

  const STOP_TEXT = {
    file_cap: (c: BulkPreviewConfig) => `the ${c.max_files}-file limit`,
    time_cap: (c: BulkPreviewConfig) => `the ${c.max_seconds} s time limit`,
    cancelled: () => 'a cancel',
  } as const;

  const est = $derived(preview.estimate);
  const duration = $derived(
    Math.round(est.duration_s_min) === Math.round(est.duration_s_max)
      ? `about ${formatDuration(est.duration_s_min)}`
      : `${formatDuration(est.duration_s_min)} to ${formatDuration(est.duration_s_max)}`,
  );
  const extensions = $derived(Object.entries(preview.by_extension).sort());
  const sizes = $derived(Object.entries(preview.megapixels.histogram));
  const skipped = $derived(
    [
      [preview.unreadable_dirs, 'unreadable directories'],
      [preview.symlinks_skipped, 'symlinks (not followed)'],
      [preview.missing, 'manifest entries that do not exist'],
      [preview.outside_roots, 'entries outside the source roots'],
      [preview.unreadable_files, 'files that could not be read'],
    ].filter(([n]) => (n as number) > 0) as [number, string][],
  );
</script>

<section
  class="space-y-2 rounded border border-zinc-800 p-3 text-xs"
  data-testid="bulk-preview"
>
  <p class="font-mono break-all text-zinc-400">{preview.target}</p>
  {#if preview.partial && preview.stop_reason}
    <p
      class="rounded border border-amber-700 px-2 py-1 text-amber-200"
      data-testid="bulk-preview-partial"
    >
      Partial: the scan stopped at {STOP_TEXT[preview.stop_reason](config)}, so these are
      at least the figures; the source holds more.
    </p>
  {/if}
  <p class="text-sm text-zinc-100" data-testid="bulk-preview-total">
    {preview.files} files · {formatBytes(preview.total_bytes)}
  </p>
  <table class="w-full max-w-sm text-zinc-300">
    <tbody>
      {#each extensions as [ext, totals] (ext)}
        <tr>
          <td class="font-mono">{ext}</td>
          <td class="text-right">{totals.files}</td>
          <td class="text-right">{formatBytes(totals.bytes)}</td>
        </tr>
      {/each}
    </tbody>
  </table>
  {#if preview.megapixels.sampled > 0}
    <p class="text-zinc-300" data-testid="bulk-preview-sizes">
      Image sizes (headers of {preview.megapixels.sampled} files, megapixels):
      {#each sizes as [label, n], i (label)}{i > 0 ? ' · ' : ' '}{label}: {n}{/each}.
      {#if preview.megapixels.share_at_or_above_gate != null}
        {Math.round(preview.megapixels.share_at_or_above_gate * 100)}% at or above
        {preview.megapixels.gate_megapixels} MP, the GPU-decode gate ({preview.gpu_decode
          ? 'on'
          : 'off'}).
      {/if}
      {#if preview.megapixels.unreadable > 0}
        {preview.megapixels.unreadable} headers could not be read.
      {/if}
    </p>
  {:else if preview.kind === 's3'}
    <p class="text-zinc-500">Image sizes are not read for S3 objects.</p>
  {/if}
  {#if skipped.length > 0}
    <ul class="text-amber-200" data-testid="bulk-preview-skipped">
      {#each skipped as [n, label] (label)}
        <li>{n} {label}</li>
      {/each}
      {#each preview.unreadable_samples as path (path)}
        <li class="font-mono break-all text-zinc-400">{path}</li>
      {/each}
    </ul>
  {/if}
  <dl
    class="grid grid-cols-1 gap-x-4 gap-y-1 sm:grid-cols-3"
    data-testid="bulk-preview-estimate"
  >
    <div>
      <dt class="text-zinc-500">Estimated ingest time</dt>
      <dd class="text-zinc-200">{duration}</dd>
    </div>
    <div>
      <dt class="text-zinc-500">Estimated index size</dt>
      <dd class="text-zinc-200">{formatBytes(est.opensearch_bytes)}</dd>
    </div>
    <div>
      <dt class="text-zinc-500">Estimated vectors</dt>
      <dd class="text-zinc-200">{Math.round(est.vectors)}</dd>
    </div>
  </dl>
  <details class="text-zinc-400">
    <summary class="cursor-pointer">
      Basis: measured figures from {config.source_document}
    </summary>
    <ul class="mt-1 list-disc space-y-1 pl-4">
      {#each config.rates as rate (rate.route)}
        <li>
          {rate.images_per_s.toFixed(1)} images/s, {rate.measured_on}
          ({rate.source_section})
        </li>
      {/each}
      <li>
        {config.embeddings_per_image.value}
        {config.embeddings_per_image.unit} ({config.embeddings_per_image.source_section})
      </li>
      <li>
        {config.opensearch_bytes_per_image.value}
        {config.opensearch_bytes_per_image.unit} ({config.opensearch_bytes_per_image
          .source_section})
      </li>
      <li>
        {est.basis === 'megapixel_sample'
          ? 'Time uses the sampled image sizes.'
          : 'Image sizes are unknown, so the time is the span from the slowest to the fastest measured route.'}
        Another host or detector profile will differ.
      </li>
    </ul>
  </details>
</section>
