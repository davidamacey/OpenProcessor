<script lang="ts">
  /**
   * A button that opens a list of actions or links (`Popover` underneath). Items are either
   * links (`href`, an app path built with `projectHref`) or actions (`onselect`); arrow keys
   * move between them, Home and End jump, Escape closes. `current` marks the active link.
   */
  import { resolve } from '$app/paths';
  import Icon from './Icon.svelte';
  import type { IconName } from './iconRegistry';
  import Popover from './Popover.svelte';

  export interface MenuItem {
    id: string;
    label: string;
    href?: string;
    onselect?: () => void;
    icon?: IconName;
    tone?: 'danger';
    disabled?: boolean;
  }
  interface Props {
    label: string;
    items: MenuItem[];
    current?: string;
    align?: 'start' | 'end';
    triggerClass?: string;
  }
  let {
    label,
    items,
    current,
    align = 'start',
    triggerClass = 'btn-sm btn-ghost',
  }: Props = $props();

  let open = $state(false);
  let list = $state<HTMLDivElement | null>(null);

  const ITEM =
    'flex w-full items-center gap-2 px-3 py-1.5 text-left text-body hover:bg-surface-2 focus-visible:bg-surface-2';
  const toneClass = (item: MenuItem): string =>
    item.tone === 'danger' ? 'text-danger-text' : 'text-fg-1';

  function choose(item: MenuItem): void {
    open = false;
    item.onselect?.();
  }

  function onListKeydown(e: KeyboardEvent): void {
    const els = Array.from(
      list?.querySelectorAll<HTMLElement>('[role="menuitem"]:not([disabled])') ?? [],
    );
    if (!els.length) return;
    const at = els.indexOf(document.activeElement as HTMLElement);
    let next = -1;
    if (e.key === 'ArrowDown') next = (at + 1) % els.length;
    else if (e.key === 'ArrowUp') next = (at - 1 + els.length) % els.length;
    else if (e.key === 'Home') next = 0;
    else if (e.key === 'End') next = els.length - 1;
    if (next < 0) return;
    e.preventDefault();
    els[next]!.focus();
  }

  function focusFirst(node: HTMLElement): void {
    queueMicrotask(() =>
      node.querySelector<HTMLElement>('[role="menuitem"]:not([disabled])')?.focus(),
    );
  }
</script>

<Popover {label} {align} bind:open panelClass="min-w-48 py-1">
  {#snippet trigger({ open: isOpen, attrs })}
    <button type="button" class={triggerClass} {...attrs}>
      {label}
      <Icon name={isOpen ? 'chevron-up' : 'chevron-down'} size={14} />
    </button>
  {/snippet}
  <div
    role="menu"
    tabindex="-1"
    aria-label={label}
    bind:this={list}
    onkeydown={onListKeydown}
    use:focusFirst
  >
    {#each items as item (item.id)}
      {#if item.href}
        <a
          role="menuitem"
          class="{ITEM} {toneClass(item)} {current === item.id ? 'font-medium' : ''}"
          href={resolve(item.href as '/')}
          aria-current={current === item.id ? 'page' : undefined}
          onclick={() => choose(item)}
        >
          {#if item.icon}<Icon name={item.icon} size={14} />{/if}
          {item.label}
        </a>
      {:else}
        <button
          type="button"
          role="menuitem"
          class="{ITEM} {toneClass(item)} disabled:opacity-50"
          disabled={item.disabled}
          onclick={() => choose(item)}
        >
          {#if item.icon}<Icon name={item.icon} size={14} />{/if}
          {item.label}
        </button>
      {/if}
    {/each}
  </div>
</Popover>
