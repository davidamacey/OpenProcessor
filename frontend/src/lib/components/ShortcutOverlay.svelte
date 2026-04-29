<script lang="ts">
  import { keyboardStore } from '$stores/keyboard.svelte';

  const shortcuts = $derived(keyboardStore.shortcutsForCurrentScope());
</script>

{#if keyboardStore.overlayOpen}
  <div
    class="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4"
    role="dialog"
    aria-modal="true"
    aria-label="Keyboard shortcuts"
    tabindex="-1"
    onclick={() => keyboardStore.closeOverlay()}
    onkeydown={(e) => e.key === 'Escape' && keyboardStore.closeOverlay()}
  >
    <!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
    <div
      role="document"
      class="w-full max-w-2xl rounded-lg border border-zinc-700 bg-zinc-950 p-6 shadow-2xl"
      onclick={(e) => e.stopPropagation()}
      onkeydown={(e) => e.stopPropagation()}
      tabindex="-1"
    >
      <div class="mb-4 flex items-center justify-between">
        <h2 class="text-lg font-semibold text-white">Keyboard Shortcuts</h2>
        <span class="text-xs text-zinc-500">
          scope: <code class="text-zinc-300">{keyboardStore.scope}</code>
        </span>
      </div>

      {#if shortcuts.length === 0}
        <p class="text-sm text-zinc-400">No shortcuts registered for this scope.</p>
      {:else}
        <ul class="grid grid-cols-1 gap-2 sm:grid-cols-2">
          {#each shortcuts as s (s.scope + ':' + s.key)}
            <li class="flex items-center justify-between gap-3 text-sm">
              <span class="text-zinc-300">{s.description}</span>
              <kbd>{s.key}</kbd>
            </li>
          {/each}
        </ul>
      {/if}

      <div class="mt-5 border-t border-zinc-800 pt-3 text-xs text-zinc-500">
        Press <kbd>~</kbd> to toggle this panel, <kbd>Esc</kbd> to close.
      </div>
    </div>
  </div>
{/if}
