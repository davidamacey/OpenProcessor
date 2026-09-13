<!--
  Shared dropdown-caret icon. Replaces the plain "▾" text glyph, which
  rendered inconsistently small/thin across browsers/fonts and was
  flagged as "way too small" on the Ignore-reason and StrategyBar
  dropdown toggles. Stroke-based SVG matching the convention already
  used in SemanticSearchBox.svelte (currentColor, round caps, 1.6
  stroke-width) so it inherits the button's text color and scales
  cleanly at any size.
-->
<script lang="ts">
  let { size = 12 }: { size?: number } = $props();
</script>

<svg
  width={size}
  height={size}
  viewBox="0 0 16 16"
  fill="none"
  stroke="currentColor"
  stroke-width="1.8"
  stroke-linecap="round"
  stroke-linejoin="round"
  aria-hidden="true"
>
  <polyline points="3.5,6 8,11 12.5,6" />
</svg>
