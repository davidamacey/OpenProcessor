<script lang="ts">
  /**
   * The one way to draw an icon: an SVG from the Lucide set by name (`iconRegistry.ts`).
   * Decorative by default (`aria-hidden`); give `label` when the icon alone carries meaning.
   * Colour is `currentColor`, so a text class colours it.
   */
  import { ICONS, type IconName } from './iconRegistry';

  interface Props {
    name: IconName;
    /** Pixel size; 16 sits in a 13px control, 14 in a 12px one. */
    size?: number;
    strokeWidth?: number;
    class?: string;
    /** Accessible name; omit for a decorative icon. */
    label?: string;
  }
  let {
    name,
    size = 16,
    strokeWidth = 2,
    class: className = '',
    label,
  }: Props = $props();
  // Lucide hides an icon from assistive tech only when no aria prop is passed at all, so the
  // label is spread in rather than passed as `undefined`.
  const Glyph = $derived(ICONS[name]);
</script>

<Glyph
  {size}
  {strokeWidth}
  class="shrink-0 {className}"
  {...label ? { 'aria-label': label, role: 'img' } : {}}
/>
