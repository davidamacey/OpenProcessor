<script lang="ts">
  /**
   * Clarity (blur_lap_ratio) floor slider, shared by /clusters,
   * /clusters/[id] and /review.
   *
   * `value` is the live drag position; `oncommit` fires on release
   * (change, not input) so dragging doesn't fire a request per pixel —
   * the caller owns the committed `minBlurRatio` state.
   */
  interface Props {
    value: number;
    oncommit: () => void;
    max?: number;
    label?: string;
    labelClass?: string;
    title?: string;
    /** Tailwind width class for the range track. */
    width?: string;
    /** Render the v1.1.9 sale-quality stop marks (1.1 / 1.3 / 1.4). */
    stops?: boolean;
  }

  let {
    value = $bindable(),
    oncommit,
    max = 2,
    label = 'clarity ≥',
    labelClass = 'text-zinc-400',
    title,
    width = 'w-32',
    stops = false,
  }: Props = $props();
</script>

<label class="flex shrink-0 items-center gap-1.5" {title}>
  <span class={labelClass}>{label}</span>
  <input
    type="range"
    min="0"
    {max}
    step="0.05"
    list={stops ? 'blur-stops' : undefined}
    bind:value
    onchange={oncommit}
    class="h-1 {width} cursor-pointer accent-blue-500"
  />
  {#if stops}
    <datalist id="blur-stops">
      <option value="1.1"></option>
      <option value="1.3"></option>
      <option value="1.4"></option>
    </datalist>
  {/if}
  <span class="w-10 tabular-nums text-zinc-400">
    {value > 0 ? value.toFixed(2) : 'off'}
  </span>
</label>
