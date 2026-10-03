<!--
  The ingest-policy draft as a form: embedding (always open), the detect
  filter and the detector override (collapsed advanced sections). It writes
  the editor's draft in place; nothing is validated here (the server's 422
  is shown on save and its preview error under the cost preview). Help text
  is frontend copy quoting the backend's documented behavior.
-->
<script lang="ts">
  import type { IngestPolicyEditor } from '$lib/detector/ingestPolicyController.svelte';
  import {
    CLASS_RESOLUTIONS,
    EMBEDDING_MODES,
    type ClassResolution,
    type EmbeddingMode,
  } from '$lib/types_detector';
  import ClassNamePicker from './ClassNamePicker.svelte';

  let { editor }: { editor: IngestPolicyEditor } = $props();

  const MODE_HELP: Record<EmbeddingMode, string> = {
    all: 'Every detection gets a vector at ingest.',
    selected: 'Only detections matching the criteria below get a vector.',
    lazy: 'No detection gets a vector at ingest; embed them later.',
  };
  const RESOLUTION_HELP: Record<ClassResolution, string> = {
    proposal: 'The detector label is kept as a proposal; no class is assigned.',
    by_name:
      'Sets the class when a registry class has the detector label’s name. It is a machine label.',
  };

  function num(v: string): number | null {
    return v.trim() === '' ? null : Number(v);
  }

  const draft = $derived(editor.draft);
  const embedding = $derived(draft.embedding ?? {});
  const detect = $derived(draft.detect ?? {});

  function setEmbedding(patch: Record<string, unknown>): void {
    editor.draft.embedding = { ...(editor.draft.embedding ?? {}), ...patch };
  }
  function setDetect(patch: Record<string, unknown>): void {
    editor.draft.detect = { ...(editor.draft.detect ?? {}), ...patch };
  }
  function setDetector(patch: Record<string, unknown>): void {
    editor.draft.detector = {
      model: '',
      ...(editor.draft.detector ?? {}),
      ...patch,
    };
  }
</script>

<div class="space-y-4" data-testid="ingest-policy-form">
  <section class="surface space-y-3 p-4">
    <h2 class="text-sm font-semibold">Embedding</h2>
    <label class="flex flex-col gap-1 text-xs">
      <span class="text-zinc-400">Mode</span>
      <select
        class="input w-48"
        data-testid="policy-mode"
        value={embedding.mode ?? 'all'}
        onchange={(e) => setEmbedding({ mode: e.currentTarget.value })}
      >
        {#each EMBEDDING_MODES as m (m)}
          <option value={m}>{m}</option>
        {/each}
      </select>
      <span class="text-zinc-500">{MODE_HELP[embedding.mode ?? 'all']}</span>
    </label>

    {#if embedding.mode === 'selected'}
      <ClassNamePicker
        label="Classes to embed"
        options={editor.labelNames}
        value={embedding.classes ?? []}
        onchange={(next) => setEmbedding({ classes: next })}
      />
      <div class="flex flex-wrap gap-3 text-xs">
        <label class="flex flex-col gap-1">
          <span class="text-zinc-400">Min confidence</span>
          <input
            class="input w-28"
            type="number"
            step="any"
            data-testid="policy-embed-min-confidence"
            value={embedding.min_confidence ?? ''}
            oninput={(e) => setEmbedding({ min_confidence: num(e.currentTarget.value) })}
          />
        </label>
        <label class="flex flex-col gap-1">
          <span class="text-zinc-400">Min box area (fraction)</span>
          <input
            class="input w-28"
            type="number"
            step="any"
            value={embedding.min_box_area_frac ?? ''}
            oninput={(e) =>
              setEmbedding({ min_box_area_frac: num(e.currentTarget.value) })}
          />
        </label>
        <label class="flex flex-col gap-1">
          <span class="text-zinc-400">Max per image</span>
          <input
            class="input w-28"
            type="number"
            step="1"
            value={embedding.max_per_image ?? ''}
            oninput={(e) => setEmbedding({ max_per_image: num(e.currentTarget.value) })}
          />
        </label>
      </div>
    {/if}
  </section>

  <details class="surface p-4" data-testid="policy-detect-section">
    <summary class="cursor-pointer text-sm font-semibold">
      Detect filter (advanced)
    </summary>
    <div class="mt-3 space-y-3">
      <label class="flex items-center gap-2 text-xs">
        <input
          type="checkbox"
          data-testid="policy-limit-classes"
          checked={detect.classes != null}
          onchange={(e) => setDetect({ classes: e.currentTarget.checked ? [] : null })}
        />
        <span class="text-zinc-200">Limit to these classes</span>
      </label>
      {#if detect.classes != null}
        <ClassNamePicker
          label="Keep only"
          options={editor.labelNames}
          value={detect.classes}
          onchange={(next) => setDetect({ classes: next })}
        />
      {/if}
      <ClassNamePicker
        label="Exclude"
        options={editor.labelNames}
        value={detect.exclude_classes ?? []}
        onchange={(next) => setDetect({ exclude_classes: next })}
      />
      <div class="flex flex-wrap gap-3 text-xs">
        <label class="flex flex-col gap-1">
          <span class="text-zinc-400">Min confidence</span>
          <input
            class="input w-28"
            type="number"
            step="any"
            value={detect.min_confidence ?? ''}
            oninput={(e) => setDetect({ min_confidence: num(e.currentTarget.value) })}
          />
        </label>
        <label class="flex flex-col gap-1">
          <span class="text-zinc-400">Min box area (fraction)</span>
          <input
            class="input w-28"
            type="number"
            step="any"
            value={detect.min_box_area_frac ?? ''}
            oninput={(e) => setDetect({ min_box_area_frac: num(e.currentTarget.value) })}
          />
        </label>
        <label class="flex flex-col gap-1">
          <span class="text-zinc-400">Max per image</span>
          <input
            class="input w-28"
            type="number"
            step="1"
            value={detect.max_per_image ?? ''}
            oninput={(e) => setDetect({ max_per_image: num(e.currentTarget.value) })}
          />
        </label>
      </div>
      <label class="flex flex-col gap-1 text-xs">
        <span class="text-zinc-400">Class resolution</span>
        <select
          class="input w-48"
          data-testid="policy-class-resolution"
          value={detect.class_resolution ?? 'proposal'}
          onchange={(e) => setDetect({ class_resolution: e.currentTarget.value })}
        >
          {#each CLASS_RESOLUTIONS as r (r)}
            <option value={r}>{r}</option>
          {/each}
        </select>
        <span class="text-zinc-500"
          >{RESOLUTION_HELP[detect.class_resolution ?? 'proposal']}</span
        >
      </label>
    </div>
  </details>

  <details class="surface p-4" data-testid="policy-detector-section">
    <summary class="cursor-pointer text-sm font-semibold">
      Detector override (advanced)
    </summary>
    <div class="mt-3 space-y-3 text-xs">
      <label class="flex items-center gap-2">
        <input
          type="checkbox"
          data-testid="policy-use-deployment-detector"
          checked={draft.detector == null}
          onchange={(e) =>
            e.currentTarget.checked
              ? (editor.draft.detector = null)
              : setDetector({ model: '' })}
        />
        <span class="text-zinc-200">Use the deployment detector</span>
      </label>
      {#if draft.detector != null}
        <div class="flex flex-wrap gap-3">
          <label class="flex flex-col gap-1">
            <span class="text-zinc-400">Model</span>
            <input
              class="input w-56"
              value={draft.detector.model}
              oninput={(e) => setDetector({ model: e.currentTarget.value })}
            />
          </label>
          <label class="flex flex-col gap-1">
            <span class="text-zinc-400">Version</span>
            <input
              class="input w-24"
              value={draft.detector.version ?? ''}
              oninput={(e) => setDetector({ version: e.currentTarget.value })}
            />
          </label>
          <label class="flex flex-col gap-1">
            <span class="text-zinc-400">Input size</span>
            <input
              class="input w-24"
              type="number"
              step="1"
              value={draft.detector.input_size ?? ''}
              oninput={(e) => setDetector({ input_size: num(e.currentTarget.value) })}
            />
          </label>
          <label class="flex flex-col gap-1">
            <span class="text-zinc-400">Labels path</span>
            <input
              class="input w-64"
              value={draft.detector.labels_path ?? ''}
              oninput={(e) => setDetector({ labels_path: e.currentTarget.value })}
            />
          </label>
        </div>
      {/if}
    </div>
  </details>
</div>
