<script lang="ts">
  import { focusOnMount } from '$lib/actions/focusOnMount';

  interface Props {
    open: boolean;
    onclose: () => void;
    appName: string;
  }

  let { open, onclose, appName }: Props = $props();
</script>

{#if open}
  <div
    class="fixed inset-0 z-40 flex items-center justify-center bg-black/60 p-4"
    role="dialog"
    aria-modal="true"
    aria-label="About {appName}"
    use:focusOnMount
    tabindex="-1"
    onclick={(e) => {
      if (e.target === e.currentTarget) onclose();
    }}
    onkeydown={(e) => e.key === 'Escape' && onclose()}
  >
    <div
      class="w-full max-w-sm rounded-lg border border-zinc-800 bg-zinc-950 p-5 shadow-2xl"
    >
      <div class="mb-3 flex items-center gap-2">
        <svg
          viewBox="0 0 128 128"
          class="h-8 w-8 shrink-0 rounded border border-zinc-700"
        >
          <rect width="128" height="128" rx="24" fill="#09090b" />
          <rect x="26" y="70" width="30" height="30" rx="5" fill="#3f3f46" />
          <rect x="60" y="70" width="30" height="30" rx="5" fill="#3f3f46" />
          <rect x="26" y="34" width="30" height="30" rx="5" fill="#60a5fa" />
          <rect x="60" y="34" width="30" height="30" rx="5" fill="#f59e0b" />
          <path
            d="M33 49 l6 6 l12 -12"
            fill="none"
            stroke="#09090b"
            stroke-width="4"
            stroke-linecap="round"
            stroke-linejoin="round"
          />
        </svg>
        <h3 class="text-base font-semibold">{appName}</h3>
      </div>

      <p class="mb-2 text-sm font-medium text-zinc-100">
        Sort your crops smarter — cluster, label, and review with AI-assisted suggestions.
      </p>
      <p class="mb-3 text-sm text-zinc-300">
        Any object class can be configured as an "annotation slot" — secondary bbox
        detection, OCR/text fields, detector provenance, and review-queue lifecycle — so
        the same app works across domains, not just one dataset.
      </p>

      <dl class="mb-4 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs text-zinc-400">
        <dt class="text-zinc-500">License</dt>
        <dd>AGPL-3.0-or-later</dd>
        <dt class="text-zinc-500">Stack</dt>
        <dd>SvelteKit 2 · Svelte 5 runes · TypeScript</dd>
      </dl>

      <div class="flex justify-end">
        <button type="button" class="btn" onclick={onclose}>Close</button>
      </div>
    </div>
  </div>
{/if}
