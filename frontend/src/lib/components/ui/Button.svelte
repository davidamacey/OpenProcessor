<script lang="ts">
  /**
   * The one button. It renders the control-tier classes in app.css (`.btn`, `.btn-sm`,
   * `.btn-primary`, ...), so the 390 hand-written `class="btn"` sites and this component are
   * the same styling, not two copies. One `primary` per view. `loading` disables the button,
   * sets aria-busy and swaps the leading icon for a spinner. An icon-only button needs an
   * `aria-label`.
   */
  import type { Snippet } from 'svelte';
  import type { HTMLButtonAttributes } from 'svelte/elements';
  import Icon from './Icon.svelte';
  import type { IconName } from './iconRegistry';

  interface Props extends Omit<HTMLButtonAttributes, 'children' | 'class'> {
    variant?: 'primary' | 'secondary' | 'ghost' | 'danger';
    size?: 'sm' | 'md';
    loading?: boolean;
    /** A leading icon, or the only content when there are no children. */
    icon?: IconName;
    class?: string;
    children?: Snippet;
  }
  let {
    variant = 'secondary',
    size = 'md',
    loading = false,
    icon,
    class: className = '',
    type = 'button',
    disabled,
    children,
    ...rest
  }: Props = $props();

  const VARIANT = {
    primary: 'btn-primary',
    secondary: '',
    ghost: 'btn-ghost',
    danger: 'btn-danger',
  } as const;
  const iconOnly = $derived(!!icon && !children);
</script>

<button
  {...rest}
  {type}
  disabled={disabled || loading}
  aria-busy={loading || undefined}
  class="{size === 'sm' ? 'btn-sm' : 'btn'} {VARIANT[variant]} {iconOnly
    ? 'btn-icon'
    : ''} {className}"
>
  {#if loading}
    <Icon name="loader-circle" size={size === 'sm' ? 14 : 16} class="animate-spin" />
  {:else if icon}
    <Icon name={icon} size={size === 'sm' ? 14 : 16} />
  {/if}
  {@render children?.()}
</button>
