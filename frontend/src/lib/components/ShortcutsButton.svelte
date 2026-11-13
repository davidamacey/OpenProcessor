<script lang="ts">
  // Lives on each sorting/labeling page that actually has keyboard
  // shortcuts (/clusters, /clusters/[id], /review) rather than in the
  // global top bar — a page with no shortcuts has nothing to show.
  import { keyboardStore } from '$stores/keyboard.svelte';
  import { keymapStore } from '$stores/keymap.svelte';
</script>

<button
  type="button"
  class="btn-sm gap-1.5 border-line-strong bg-surface-1 text-fg-2 hover:bg-surface-2 hover:text-white"
  onclick={() => keyboardStore.toggleOverlay()}
  title="Keyboard shortcuts ({keymapStore.glyph('global.shortcuts_overlay')})"
>
  <span>?</span>
  <span class="hidden sm:inline">shortcuts</span>
</button>
