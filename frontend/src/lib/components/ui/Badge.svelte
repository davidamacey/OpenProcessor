<script lang="ts">
  /**
   * A count or status pill (round, never interactive): the number beside a nav item or a
   * tab. Shares the `.chip-*` tone colours with `Chip`.
   */
  import type { Snippet } from 'svelte';

  interface Props {
    tone?: 'neutral' | 'accent' | 'success' | 'warn' | 'danger' | 'info' | 'holdout';
    children: Snippet;
  }
  let { tone = 'neutral', children }: Props = $props();
</script>

<span
  class="inline-flex min-w-5 items-center justify-center rounded-full border px-1.5 text-caption font-medium tabular-nums {tone ===
  'neutral'
    ? 'border-line-strong bg-surface-2 text-fg-2'
    : `chip-${tone}`}"
>
  {@render children()}
</span>
