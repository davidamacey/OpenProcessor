<script lang="ts">
  /**
   * The served dry run (`POST /datasets/preview`): per-split
   * counts, totals, the processing estimate, an OpenProcessor export's
   * facts, the region block and the issues. All numbers are served.
   */
  import DatasetIssueList from './DatasetIssueList.svelte';
  import type { DatasetFormatsResponse, DatasetPreview } from '$lib/types_import';

  interface Props {
    preview: DatasetPreview;
    formats: DatasetFormatsResponse;
  }

  let { preview, formats }: Props = $props();

  const formatLabel = $derived(
    formats.formats.find((f) => f.format === preview.format)?.label ?? preview.format,
  );
  const yesNo = (v: boolean): string => (v ? 'yes' : 'no');
</script>

<div class="space-y-4" data-testid="preview-summary">
  <p class="text-meta text-fg-2">
    <span class="text-fg-1">{formatLabel}</span> at
    <code class="font-mono">{preview.root}</code>
  </p>

  <div class="overflow-x-auto">
    <table class="w-full text-left text-meta" data-testid="preview-splits">
      <thead class="text-fg-3">
        <tr>
          <th class="px-2 py-1">Split</th>
          <th class="px-2 py-1 text-right">Images</th>
          <th class="px-2 py-1 text-right">Labeled</th>
          <th class="px-2 py-1 text-right">Negatives</th>
          <th class="px-2 py-1 text-right">Unlabeled</th>
          <th class="px-2 py-1 text-right">Boxes</th>
        </tr>
      </thead>
      <tbody class="font-mono">
        {#each preview.splits as s (s.split)}
          <tr class="border-t border-line">
            <td class="px-2 py-1 font-sans">{s.split}</td>
            <td class="px-2 py-1 text-right">{s.images.toLocaleString()}</td>
            <td class="px-2 py-1 text-right">{s.labeled.toLocaleString()}</td>
            <td class="px-2 py-1 text-right">{s.negatives.toLocaleString()}</td>
            <td class="px-2 py-1 text-right">{s.unlabeled.toLocaleString()}</td>
            <td class="px-2 py-1 text-right">{s.boxes.toLocaleString()}</td>
          </tr>
        {/each}
      </tbody>
    </table>
  </div>

  <dl class="grid grid-cols-2 gap-x-4 gap-y-1 text-meta sm:grid-cols-3">
    <div>
      <dt class="text-fg-3">Images</dt>
      <dd class="font-mono">{preview.totals.images.toLocaleString()}</dd>
    </div>
    <div>
      <dt class="text-fg-3">Boxes</dt>
      <dd class="font-mono">{preview.totals.boxes.toLocaleString()}</dd>
    </div>
    <div>
      <dt class="text-fg-3">Already in this project</dt>
      <dd class="font-mono">{preview.totals.images_already_indexed.toLocaleString()}</dd>
    </div>
    <div>
      <dt class="text-fg-3">To ingest</dt>
      <dd class="font-mono">{preview.totals.images_to_ingest.toLocaleString()}</dd>
    </div>
    <div>
      <dt class="text-fg-3">Detector runs</dt>
      <dd class="font-mono">{preview.estimate.detector_images.toLocaleString()}</dd>
    </div>
    <div>
      <dt class="text-fg-3">Embeddings</dt>
      <dd class="font-mono">{preview.estimate.embeddings.toLocaleString()}</dd>
    </div>
  </dl>

  {#if preview.op_export}
    {@const op = preview.op_export}
    <div class="rounded border border-line p-3 text-meta" data-testid="op-export-facts">
      <h4 class="mb-1 font-semibold text-fg-2">OpenProcessor export</h4>
      <dl class="grid grid-cols-2 gap-x-4 gap-y-0.5 sm:grid-cols-3">
        <div>
          <dt class="text-fg-3">Kind</dt>
          <dd>{op.dataset_kind ?? '—'}</dd>
        </div>
        <div>
          <dt class="text-fg-3">Box source</dt>
          <dd>{op.box_source ?? '—'}</dd>
        </div>
        <div>
          <dt class="text-fg-3">Image mode</dt>
          <dd>{op.image_mode ?? '—'}</dd>
        </div>
        <div>
          <dt class="text-fg-3">Frozen test split</dt>
          <dd>
            {yesNo(op.test_frozen.present)}{op.test_frozen.present
              ? `, verified: ${yesNo(op.test_frozen.verified)}`
              : ''}
          </dd>
        </div>
        <div>
          <dt class="text-fg-3">Stratum map</dt>
          <dd>
            {op.stratum_map.present
              ? `${op.stratum_map.entries.toLocaleString()} entries`
              : 'none'}
          </dd>
        </div>
      </dl>
    </div>
  {/if}

  {#if preview.region}
    {@const r = preview.region}
    <div class="rounded border border-line p-3 text-meta" data-testid="preview-region">
      <h4 class="mb-1 font-semibold text-fg-2">Region labels</h4>
      <p class="text-fg-2">
        Profile <span class="text-fg-1">{r.profile.name}</span>, region class
        <span class="text-fg-1">{r.region_class_name}</span>, parents from
        <span class="text-fg-1"
          >{formats.parents_modes.find((p) => p.value === r.parents_mode)?.label ??
            r.parents_mode}</span
        >; {r.standalone_boxes.toLocaleString()} boxes will become standalone items.
      </p>
    </div>
  {/if}

  <DatasetIssueList issues={preview.issues} catalog={formats.issues} />
</div>
