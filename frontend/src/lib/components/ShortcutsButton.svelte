<script lang="ts">
  // Lives on each sorting/labeling page that actually has keyboard
  // shortcuts (/clusters, /clusters/[id], /review) rather than in the
  // global top bar — a page with no shortcuts has nothing to show.
  import { keyboardStore } from '$stores/keyboard.svelte';
</script>

<button
  type="button"
  class="btn-sm gap-1.5 border-zinc-700 bg-zinc-900 text-zinc-300 hover:bg-zinc-800 hover:text-white"
  onclick={() => keyboardStore.toggleOverlay()}
  title="Keyboard shortcuts (~)"
>
  <span class="font-mono">?</span>
  <span class="hidden sm:inline">shortcuts</span>
</button>
