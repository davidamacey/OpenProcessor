<script lang="ts">
  /**
   * A one-of-N switch (view modes, split names): a radiogroup of buttons with the radio
   * keyboard pattern (arrows move and select, Home and End jump; only the checked one is a tab
   * stop). The checked option is the accent. `orientation="vertical"` stacks it.
   */
  import Icon from './Icon.svelte';
  import type { IconName } from './iconRegistry';

  export interface SegmentOption {
    value: string;
    label?: string;
    icon?: IconName;
    title?: string;
    disabled?: boolean;
  }
  interface Props {
    options: SegmentOption[];
    value: string;
    label: string;
    onchange: (value: string) => void;
    size?: 'sm' | 'md';
    orientation?: 'horizontal' | 'vertical';
  }
  let {
    options,
    value,
    label,
    onchange,
    size = 'md',
    orientation = 'horizontal',
  }: Props = $props();

  function onkeydown(e: KeyboardEvent): void {
    const enabled = options.filter((o) => !o.disabled);
    const at = enabled.findIndex((o) => o.value === value);
    const fwd = orientation === 'vertical' ? 'ArrowDown' : 'ArrowRight';
    const back = orientation === 'vertical' ? 'ArrowUp' : 'ArrowLeft';
    let next = -1;
    if (e.key === fwd) next = (at + 1) % enabled.length;
    else if (e.key === back) next = (at - 1 + enabled.length) % enabled.length;
    else if (e.key === 'Home') next = 0;
    else if (e.key === 'End') next = enabled.length - 1;
    if (next < 0) return;
    e.preventDefault();
    const host = e.currentTarget as HTMLElement;
    const target = enabled[next]!.value;
    onchange(target);
    queueMicrotask(() =>
      host.querySelector<HTMLElement>(`[data-value="${target}"]`)?.focus(),
    );
  }
</script>

<div
  class="inline-flex gap-0.5 rounded-control border border-line-strong bg-surface-2 p-0.5 {orientation ===
  'vertical'
    ? 'flex-col'
    : ''}"
  role="radiogroup"
  aria-label={label}
  tabindex="-1"
  {onkeydown}
>
  {#each options as o (o.value)}
    <button
      type="button"
      role="radio"
      data-value={o.value}
      aria-checked={o.value === value}
      tabindex={o.value === value ? 0 : -1}
      disabled={o.disabled}
      title={o.title}
      aria-label={o.label ? undefined : (o.title ?? o.value)}
      class="inline-flex items-center justify-center gap-1.5 rounded-chip px-2.5 whitespace-nowrap disabled:opacity-50 {size ===
      'sm'
        ? 'h-6 text-meta'
        : 'h-7 text-control'} {o.value === value
        ? 'bg-accent-soft text-accent-text shadow-[inset_0_0_0_1px_var(--color-accent-line)]'
        : 'text-fg-2 hover:bg-surface-3 hover:text-fg-1'}"
      onclick={() => onchange(o.value)}
    >
      {#if o.icon}<Icon name={o.icon} size={size === 'sm' ? 14 : 16} />{/if}
      {#if o.label}{o.label}{/if}
    </button>
  {/each}
</div>
