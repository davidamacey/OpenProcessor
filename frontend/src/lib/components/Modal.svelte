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
    /** Where the panel sits: centered, near the top (pickers), or flush to the left or right
     *  edge (a drawer; `ui/Drawer.svelte`). */
    align?: 'center' | 'top' | 'start' | 'end';
    /** Stacking layer: `drawer` (50), a modal (70), or `top` (75) for a dialog opened over
     *  another dialog. The nav is 40, popovers 60, toasts 80. */
    layer?: 'drawer' | 'base' | 'top';
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

  const LAYER = { drawer: 'z-50', base: 'z-70', top: 'z-[75]' } as const;
  const ALIGN = {
    center: 'items-center',
    top: 'items-start pt-24',
    start: 'justify-start',
    end: 'justify-end',
  } as const;
  const EDGE = { center: false, top: false, start: true, end: true } as const;

  let {
    label,
    onclose,
    align = 'center',
    layer = 'base',
    scrim = 'bg-black/60',
    panelClass = 'w-full max-w-md rounded-modal border border-line bg-surface-3 p-5 shadow-e3',
    children,
    ...rest
  }: Props = $props();
</script>

<div
  {...rest}
  class="fixed inset-0 flex {EDGE[align] ? '' : 'justify-center p-4'} {LAYER[
    layer
  ]} {ALIGN[align]}"
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
