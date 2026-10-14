<script lang="ts">
  /**
   * The one modal shell: a fixed scrim, a labelled `role="dialog"`
   * (`aria-modal`), focus moved in on open and trapped while open, Escape and
   * a scrim click calling `onclose`, and focus restored to the opener on
   * close. Callers supply the panel content (and, through `panelClass`, its
   * size); nothing else hand-rolls a backdrop.
   *
   * Omit `onclose` for a dialog that must not be dismissed from the scrim or
   * the keyboard (a busy step); its own buttons are then the only way out.
   * Extra attributes (`data-testid`) land on the dialog element.
   */
  import type { Snippet } from 'svelte';
  import type { HTMLAttributes } from 'svelte/elements';
  import { trapFocus } from '$lib/actions/trapFocus';

  interface Props extends Omit<HTMLAttributes<HTMLDivElement>, 'children' | 'class'> {
    /** The dialog's accessible name. */
    label: string;
    onclose?: () => void;
    /** Where the panel sits: centered, or near the top (pickers). */
    align?: 'center' | 'top';
    /** Stacking layer; a dialog opened over another dialog or the nav uses `top`. */
    layer?: 'base' | 'top';
    /** The scrim colour; lightboxes use a darker one. */
    scrim?: string;
    /** The panel's box. Defaults to the standard medium form dialog. */
    panelClass?: string;
    children: Snippet;
  }

  // Focus the dialog itself unless a control inside already took focus (an
  // autofocused input), so the keyboard starts inside it either way.
  function enter(node: HTMLElement): void {
    queueMicrotask(() => {
      if (!node.contains(document.activeElement)) node.focus();
    });
  }

  let {
    label,
    onclose,
    align = 'center',
    layer = 'base',
    scrim = 'bg-black/60',
    panelClass = 'w-full max-w-md rounded-lg border border-zinc-800 bg-zinc-950 p-5 shadow-2xl',
    children,
    ...rest
  }: Props = $props();
</script>

<div
  {...rest}
  class="fixed inset-0 flex justify-center p-4 {layer === 'top'
    ? 'z-50'
    : 'z-40'} {align === 'top' ? 'items-start pt-24' : 'items-center'}"
  role="dialog"
  aria-modal="true"
  aria-label={label}
  tabindex="-1"
  use:enter
  use:trapFocus={{ onEscape: onclose }}
>
  <!-- The scrim is a separate, screen-reader-hidden layer rather than a click
       handler on the dialog itself: the keyboard route to dismiss is Escape,
       which the dialog handles, so the click has no key twin to announce. -->
  <div
    class="absolute inset-0 {scrim}"
    aria-hidden="true"
    data-modal-backdrop
    onclick={onclose}
  ></div>
  <div class="relative {panelClass}">
    {@render children()}
  </div>
</div>
