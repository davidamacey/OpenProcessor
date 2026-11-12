<script lang="ts">
  /** A key hint. Styling is the `kbd` rule in app.css; this keeps one spelling in markup. */
  import type { Snippet } from 'svelte';

  let { children }: { children: Snippet } = $props();
</script>

<kbd>{@render children()}</kbd>
