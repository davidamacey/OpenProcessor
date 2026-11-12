<script lang="ts">
  import Icon from '$components/ui/Icon.svelte';
  import { toastStore } from '$stores/toast.svelte';

  const kindClass = (k: 'info' | 'success' | 'warn' | 'error'): string => {
    switch (k) {
      case 'success':
        return 'border-success/40 bg-success/10 text-success-text';
      case 'warn':
        return 'border-warn/40 bg-warn/10 text-warn-text';
      case 'error':
        return 'border-danger/40 bg-danger/10 text-danger-text';
      default:
        return 'border-accent/40 bg-accent/10 text-accent-text';
    }
  };
</script>

<div
  class="pointer-events-none fixed right-4 bottom-4 z-80 flex w-[360px] max-w-[95vw] flex-col gap-2"
  aria-live="polite"
  aria-atomic="true"
>
  {#each toastStore.toasts as t (t.id)}
    <div
      class="pointer-events-auto rounded-md border px-3 py-2 text-body shadow-lg backdrop-blur {kindClass(
        t.kind,
      )}"
    >
      <div class="flex items-start gap-2">
        <span class="grow whitespace-pre-wrap break-words">{t.text}</span>
        <button
          type="button"
          class="text-fg-2 hover:text-white"
          aria-label="Dismiss"
          onclick={() => toastStore.dismiss(t.id)}
        >
          <Icon name="x" size={14} />
        </button>
      </div>
    </div>
  {/each}
</div>
