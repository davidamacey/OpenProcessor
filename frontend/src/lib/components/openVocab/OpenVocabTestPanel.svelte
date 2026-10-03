<script lang="ts">
  /**
   * Test one open-vocabulary target on one image
   * (`POST /open_vocab/test`; nothing is written). The target is any row of
   * the draft (or of the saved revision on screen); the image is a stored
   * crop's (its served `image_id` is sent) or an upload. Shows what the
   * server returned: what the gate decided, the validation, and every
   * candidate hit with what selection did with it, drawn over the image as
   * served. A segmenter failure is an error banner, never "no hits".
   */
  import { onDestroy } from 'svelte';
  import ConfigIssueList from '$components/config/ConfigIssueList.svelte';
  import SourceImageOverlay from '$components/SourceImageOverlay.svelte';
  import { dropReasonText, hitShapes } from '$lib/openVocab/hitShapes';
  import { createOpenVocabTest } from '$lib/openVocab/openVocabTestController.svelte';
  import type { OpenVocabTargetBody } from '$lib/types_openVocab';
  import HitOverlay from './HitOverlay.svelte';

  interface Props {
    /** The targets of the draft, or of the saved revision being viewed. */
    targets: OpenVocabTargetBody[];
    imageMaxSide?: number;
    dedupIou?: number;
  }

  let { targets, imageMaxSide, dedupIou }: Props = $props();

  const t = createOpenVocabTest();
  let index = $state(0);
  let uploadUrl = $state<string | null>(null);

  const current = $derived(targets[Math.min(index, targets.length - 1)]);

  onDestroy(() => {
    t.stop();
    if (uploadUrl) URL.revokeObjectURL(uploadUrl);
  });

  function run(): void {
    if (!current) return;
    void t.run({
      target: current,
      image_max_side: imageMaxSide,
      dedup_iou: dedupIou,
    });
  }

  function readBase64(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const r = new FileReader();
      r.onload = () => resolve(String(r.result).replace(/^data:[^,]*,/, ''));
      r.onerror = () => reject(r.error);
      r.readAsDataURL(file);
    });
  }

  async function pick(file: File | undefined): Promise<void> {
    if (uploadUrl) URL.revokeObjectURL(uploadUrl);
    uploadUrl = null;
    t.upload = null;
    if (!file) return;
    uploadUrl = URL.createObjectURL(file);
    t.upload = { name: file.name, base64: await readBase64(file) };
  }

  const shapes = $derived(t.result ? hitShapes(t.result.hits) : []);
</script>

<section
  class="surface flex flex-col gap-3 p-4 text-sm"
  data-testid="open-vocab-test-panel"
  aria-label="Test a target"
>
  <h2 class="text-base font-semibold">Test a target</h2>
  <p class="text-xs text-zinc-400">
    Runs one target (as drafted, saved or not) on one image and shows every candidate the
    segmenter found and what selection kept. Nothing is written.
  </p>

  {#if targets.length === 0}
    <p class="text-xs text-zinc-500" data-testid="test-no-targets">
      Add a target to test it.
    </p>
  {:else}
    <div class="grid gap-2 sm:grid-cols-2">
      <label class="flex flex-col gap-1 text-xs text-zinc-400">
        Target
        <select class="select select-sm" bind:value={index} data-testid="ov-test-target">
          {#each targets as tg, i (i)}
            <option value={i}>#{i} {tg.prompt || 'untitled target'}</option>
          {/each}
        </select>
      </label>
      <label class="flex flex-col gap-1 text-xs text-zinc-400">
        Image
        <select
          class="select select-sm"
          bind:value={t.source}
          data-testid="ov-test-source"
        >
          <option value="crop">A stored crop's image</option>
          <option value="upload">Upload an image</option>
        </select>
      </label>
      {#if t.source === 'crop'}
        <label class="flex flex-col gap-1 text-xs text-zinc-400 sm:col-span-2">
          Crop id (from review or browse)
          <input
            class="input input-sm font-mono"
            bind:value={t.cropId}
            placeholder="c_123"
            data-testid="ov-test-crop-id"
          />
        </label>
      {:else}
        <label class="flex flex-col gap-1 text-xs text-zinc-400 sm:col-span-2">
          JPEG or PNG
          <input
            type="file"
            accept="image/jpeg,image/png"
            data-testid="ov-test-file"
            onchange={(e) => void pick((e.currentTarget as HTMLInputElement).files?.[0])}
          />
        </label>
      {/if}
      <label class="flex items-center gap-2 text-xs text-zinc-300 sm:col-span-2">
        <input type="checkbox" bind:checked={t.precheck} data-testid="ov-test-precheck" />
        Run the VLM pre-check
      </label>
    </div>
    <div>
      <button
        type="button"
        class="btn btn-primary btn-sm"
        disabled={!t.canRun}
        data-testid="ov-test-run"
        onclick={run}>{t.running ? 'Testing…' : 'Run test'}</button
      >
    </div>
  {/if}

  {#if t.error}
    <p
      class="rounded border border-red-500/40 bg-red-500/10 px-2 py-1 text-red-200"
      data-testid="ov-test-error"
    >
      {t.segmenterError ? `Segmenter error: ${t.error}` : t.error}
    </p>
  {/if}

  {#if t.result}
    {@const r = t.result}
    <div
      class="flex flex-col gap-3 border-t border-zinc-800 pt-3"
      data-testid="ov-test-result"
    >
      <p class="text-xs text-zinc-400">
        <span class="text-zinc-500">Prompt</span>
        <span class="font-mono text-zinc-200">{r.prompt}</span>
        <span class="ml-2 text-zinc-500">Class</span>
        <span class="font-mono text-zinc-200" data-testid="ov-test-class"
          >{r.class_name || 'discovery'}</span
        >
        <span class="ml-2 text-zinc-500">Took</span>
        <span class="font-mono text-zinc-200">{Math.round(r.elapsed_ms)} ms</span>
      </p>
      <p
        class="text-xs {r.gate.run ? 'text-zinc-300' : 'text-amber-300'}"
        data-testid="ov-test-gate"
      >
        {#if r.gate.run}
          The gate let this run{r.gate.reason ? `: ${r.gate.reason}` : ''}.
        {:else}
          Skipped by tier {r.gate.tier ?? '?'}: {r.gate.reason ?? 'no reason served'}
        {/if}
      </p>
      {#if r.validation}
        <ConfigIssueList
          issues={[...r.validation.errors, ...r.validation.warnings]}
          showField
        />
      {/if}

      {#if r.hits.length === 0}
        <p class="text-xs text-zinc-400" data-testid="ov-test-no-hits">
          The segmenter returned no candidates.
        </p>
      {:else}
        <div class="overflow-x-auto">
          <table class="w-full text-left text-xs" data-testid="ov-test-hits">
            <thead class="text-zinc-500">
              <tr>
                <th class="py-1 pr-3 font-normal">Hit</th>
                <th class="py-1 pr-3 font-normal">Score</th>
                <th class="py-1 pr-3 font-normal">Selected</th>
                <th class="py-1 font-normal">Dropped because</th>
              </tr>
            </thead>
            <tbody>
              {#each r.hits as h, i (i)}
                <tr
                  class="border-t border-zinc-800 {h.selected ? '' : 'opacity-60'}"
                  data-testid="ov-test-hit"
                  data-selected={h.selected}
                >
                  <td class="py-1 pr-3 font-mono">#{i}</td>
                  <td class="py-1 pr-3 font-mono">{h.score.toFixed(2)}</td>
                  <td class="py-1 pr-3">{h.selected ? 'yes' : 'no'}</td>
                  <td class="py-1" title={h.drop_reason ?? ''}
                    >{dropReasonText(h.drop_reason)}</td
                  >
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      {/if}

      <div class="max-w-xl">
        {#if t.ranOnCropId}
          <SourceImageOverlay cropId={t.ranOnCropId} extraShapes={shapes} />
        {:else if uploadUrl}
          <HitOverlay src={uploadUrl} {shapes} />
        {/if}
      </div>
    </div>
  {/if}
</section>
