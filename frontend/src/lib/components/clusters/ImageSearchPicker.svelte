<script lang="ts">
  /**
   * Query by picture: choose or drop a file, optionally drag a region on its preview, search.
   * Emits the file and the region as a normalized `[x1, y1, x2, y2]`; the backend decodes,
   * crops and ranks (nothing is computed here beyond turning the dragged rectangle into
   * fractions of the preview).
   */
  import { onDestroy } from 'svelte';
  import type { ImageSearchBox } from '$lib/api_cropRead';

  interface Props {
    busy?: boolean;
    onSearch: (file: File, box: ImageSearchBox | null) => void;
  }
  let { busy = false, onSearch }: Props = $props();

  let open = $state(false);
  let file = $state<File | null>(null);
  let previewUrl = $state<string | null>(null);
  let box = $state<ImageSearchBox | null>(null);
  let dragStart: { x: number; y: number } | null = null;
  let dragOver = $state(false);

  function setFile(next: File | null): void {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    file = next;
    previewUrl = next ? URL.createObjectURL(next) : null;
    box = null;
  }

  onDestroy(() => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
  });

  function pick(files: FileList | null | undefined): void {
    const f = files?.[0];
    if (f && f.type.startsWith('image/')) setFile(f);
  }

  function fraction(e: PointerEvent): { x: number; y: number } {
    const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
    const clamp = (v: number) => Math.min(1, Math.max(0, v));
    return {
      x: clamp((e.clientX - r.left) / r.width),
      y: clamp((e.clientY - r.top) / r.height),
    };
  }

  function down(e: PointerEvent): void {
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    dragStart = fraction(e);
    box = null;
  }

  function move(e: PointerEvent): void {
    if (!dragStart) return;
    const p = fraction(e);
    box = [
      Math.min(dragStart.x, p.x),
      Math.min(dragStart.y, p.y),
      Math.max(dragStart.x, p.x),
      Math.max(dragStart.y, p.y),
    ];
  }

  function up(): void {
    dragStart = null;
    // A click without a drag is not a region.
    if (box && (box[2] - box[0] < 0.01 || box[3] - box[1] < 0.01)) box = null;
  }

  function submit(): void {
    if (!file) return;
    onSearch(file, box);
    open = false;
  }
</script>

<div class="relative">
  <button
    type="button"
    class="btn-sm border border-zinc-700 bg-zinc-900 text-zinc-300 hover:bg-zinc-800"
    aria-expanded={open}
    data-testid="image-search-toggle"
    onclick={() => (open = !open)}
  >
    Search by image
  </button>
  {#if open}
    <div
      class="absolute left-0 top-full z-30 mt-1 w-80 rounded-md border border-zinc-700 bg-zinc-900 p-3 text-xs shadow-lg"
      data-testid="image-search-panel"
    >
      {#if previewUrl}
        <div
          class="relative cursor-crosshair touch-none select-none"
          onpointerdown={down}
          onpointermove={move}
          onpointerup={up}
          data-testid="image-search-preview"
        >
          <img
            src={previewUrl}
            alt="Query preview"
            class="block w-full"
            draggable="false"
          />
          {#if box}
            <div
              class="pointer-events-none absolute border-2 border-blue-400 bg-blue-400/20"
              style="left:{box[0] * 100}%;top:{box[1] * 100}%;width:{(box[2] - box[0]) *
                100}%;height:{(box[3] - box[1]) * 100}%"
            ></div>
          {/if}
        </div>
        <p class="mt-1 text-zinc-500">
          {box
            ? 'Searching by the marked region.'
            : 'Drag on the picture to search by a region.'}
        </p>
      {:else}
        <label
          class="flex h-24 cursor-pointer items-center justify-center rounded border border-dashed text-zinc-400 {dragOver
            ? 'border-blue-400 bg-blue-500/10'
            : 'border-zinc-600'}"
          ondragover={(e) => {
            e.preventDefault();
            dragOver = true;
          }}
          ondragleave={() => (dragOver = false)}
          ondrop={(e) => {
            e.preventDefault();
            dragOver = false;
            pick(e.dataTransfer?.files);
          }}
        >
          Drop a picture or click to choose
          <input
            type="file"
            accept="image/*"
            class="sr-only"
            data-testid="image-search-file"
            onchange={(e) => pick((e.currentTarget as HTMLInputElement).files)}
          />
        </label>
      {/if}
      <div class="mt-2 flex items-center gap-2">
        <button
          type="button"
          class="btn-sm border border-blue-500/50 bg-blue-500/10 text-blue-200 hover:bg-blue-500/20 disabled:opacity-50"
          disabled={!file || busy}
          data-testid="image-search-run"
          onclick={submit}
        >
          Search
        </button>
        {#if box}
          <button
            type="button"
            class="btn-sm border border-zinc-700 text-zinc-300 hover:bg-zinc-800"
            onclick={() => (box = null)}
          >
            Clear region
          </button>
        {/if}
        {#if file}
          <button
            type="button"
            class="btn-sm border border-zinc-700 text-zinc-300 hover:bg-zinc-800"
            onclick={() => setFile(null)}
          >
            Change picture
          </button>
        {/if}
      </div>
    </div>
  {/if}
</div>
