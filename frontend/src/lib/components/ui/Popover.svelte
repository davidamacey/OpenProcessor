<script lang="ts">
  /**
   * A floating panel anchored under its trigger: opens on the trigger, closes on Escape
   * (focus returns to the trigger), on a pointer press outside, and on `close()`. The panel
   * is surface-3 with elevation e3 at the popover layer (60). `trigger` receives the state and
   * the props to spread on the button; `children` receives `close`.
   */
  import type { Snippet } from 'svelte';

  interface TriggerProps {
    open: boolean;
    toggle: () => void;
    attrs: {
      'aria-haspopup': 'true';
      'aria-expanded': boolean;
      onclick: () => void;
      onkeydown: (e: KeyboardEvent) => void;
    };
  }
  interface Props {
    trigger: Snippet<[TriggerProps]>;
    children: Snippet<[() => void]>;
    /** The panel's accessible name. */
    label: string;
    align?: 'start' | 'end';
    /** Panel box classes (width, padding). */
    panelClass?: string;
    open?: boolean;
  }
  let {
    trigger,
    children,
    label,
    align = 'start',
    panelClass = 'w-64 py-1',
    open = $bindable(false),
  }: Props = $props();

  let root = $state<HTMLDivElement | null>(null);
  let panel = $state<HTMLDivElement | null>(null);
  // The side the panel really opens to: the requested one unless it would leave the viewport.
  let side = $state<'start' | 'end'>('start');

  $effect(() => {
    if (!open || !panel || !root) return;
    const anchor = root.getBoundingClientRect();
    const width = panel.offsetWidth;
    if (align === 'end') side = anchor.right - width < 0 ? 'start' : 'end';
    else side = anchor.left + width > window.innerWidth ? 'end' : 'start';
  });

  function close(): void {
    open = false;
  }
  function toggle(): void {
    open = !open;
  }
  function onkeydown(e: KeyboardEvent): void {
    if (e.key !== 'Escape' || !open) return;
    e.stopPropagation();
    open = false;
    root?.querySelector<HTMLElement>('[aria-haspopup]')?.focus();
  }
  function onWindowPointer(e: PointerEvent): void {
    if (open && root && !root.contains(e.target as Node)) open = false;
  }
</script>

<svelte:window onpointerdown={onWindowPointer} />

<div class="relative shrink-0" bind:this={root}>
  {@render trigger({
    open,
    toggle,
    attrs: { 'aria-haspopup': 'true', 'aria-expanded': open, onclick: toggle, onkeydown },
  })}
  {#if open}
    <!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
    <div
      bind:this={panel}
      class="absolute top-full z-60 mt-1 max-w-[calc(100vw-1rem)] rounded-card border border-line bg-surface-3 shadow-e3 {side ===
      'end'
        ? 'right-0'
        : 'left-0'} {panelClass}"
      role="group"
      aria-label={label}
      tabindex="-1"
      {onkeydown}
    >
      {@render children(close)}
    </div>
  {/if}
</div>
