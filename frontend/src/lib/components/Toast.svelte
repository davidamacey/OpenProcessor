<script lang="ts">
  import { toastStore } from '$stores/toast.svelte';

  const kindClass = (k: 'info' | 'success' | 'warn' | 'error'): string => {
    switch (k) {
      case 'success':
        return 'border-green-500/40 bg-green-500/10 text-green-200';
      case 'warn':
        return 'border-orange-500/40 bg-orange-500/10 text-orange-100';
      case 'error':
        return 'border-red-500/40 bg-red-500/10 text-red-100';
      default:
        return 'border-blue-500/40 bg-blue-500/10 text-blue-100';
    }
  };
</script>

<div
  class="pointer-events-none fixed right-4 bottom-4 z-50 flex w-[360px] max-w-[95vw] flex-col gap-2"
  aria-live="polite"
  aria-atomic="true"
>
  {#each toastStore.toasts as t (t.id)}
    <div
      class="pointer-events-auto rounded-md border px-3 py-2 text-sm shadow-lg backdrop-blur {kindClass(
        t.kind,
      )}"
    >
      <div class="flex items-start gap-2">
        <span class="grow whitespace-pre-wrap break-words">{t.text}</span>
        <button
          type="button"
          class="text-zinc-300 hover:text-white"
          aria-label="Dismiss"
          onclick={() => toastStore.dismiss(t.id)}
        >
          ×
        </button>
      </div>
    </div>
  {/each}
</div>
