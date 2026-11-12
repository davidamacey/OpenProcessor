<script lang="ts">
  /**
   * A small tag (the xs control tier). Static by default; with `onclick` it is a button, and
   * `pressed` makes it a filter toggle (accent when on). `onremove` adds a close button named
   * by `removeLabel`. Tones are the `.chip-*` classes in app.css.
   */
  import type { Snippet } from 'svelte';
  import Icon from './Icon.svelte';
  import type { IconName } from './iconRegistry';

  type Tone = 'neutral' | 'accent' | 'success' | 'warn' | 'danger' | 'info' | 'holdout';

  interface Props {
    tone?: Tone;
    icon?: IconName;
    pressed?: boolean;
    disabled?: boolean;
    title?: string;
    onclick?: () => void;
    onremove?: () => void;
    removeLabel?: string;
    children: Snippet;
  }
  let {
    tone = 'neutral',
    icon,
    pressed,
    disabled,
    title,
    onclick,
    onremove,
    removeLabel = 'Remove',
    children,
  }: Props = $props();

  const effective = $derived(pressed ? 'accent' : tone);
  const cls = $derived(`chip ${effective === 'neutral' ? '' : `chip-${effective}`}`);
</script>

{#snippet content()}
  {#if icon}<Icon name={icon} size={12} />{/if}
  {@render children()}
{/snippet}

{#if onclick}
  <button type="button" class={cls} aria-pressed={pressed} {disabled} {title} {onclick}>
    {@render content()}
  </button>
{:else}
  <span class={cls} {title}>
    {@render content()}
    {#if onremove}
      <button
        type="button"
        class="-mr-1 rounded-chip hover:text-fg-1"
        aria-label={removeLabel}
        onclick={onremove}
      >
        <Icon name="x" size={12} />
      </button>
    {/if}
  </span>
{/if}
