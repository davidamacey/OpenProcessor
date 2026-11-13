<script lang="ts">
  /**
   * Tabs with a 2px accent underline on the active one. Two modes, chosen by the items:
   * - route / URL tabs: every item has an `href` (an app path, from `projectHref`; a query
   *   string such as `?tab=audit` is fine). Rendered as links in a `nav`, the active one
   *   `aria-current="page"`. The URL is the state, so reload and back work.
   * - in-page tabs: no `href`; `value` and `onselect` drive it, with the tablist keyboard
   *   pattern (arrows, Home, End; the tab panel is the caller's).
   * Both sit in a `ScrollStrip`: on a narrow screen the row scrolls sideways, a button shows
   * on each side that has hidden tabs, and the active tab is kept in view.
   */
  import { resolve } from '$app/paths';
  import ScrollStrip from '$components/ScrollStrip.svelte';
  import Badge from './Badge.svelte';
  import Icon from './Icon.svelte';
  import type { IconName } from './iconRegistry';

  export interface TabItem {
    id: string;
    label: string;
    href?: string;
    icon?: IconName;
    count?: number | string;
    disabled?: boolean;
  }
  interface Props {
    items: TabItem[];
    /** The id of the active tab. */
    value: string;
    label: string;
    onselect?: (id: string) => void;
  }
  let { items, value, label, onselect }: Props = $props();

  const asLinks = $derived(items.length > 0 && items.every((i) => i.href));
  const BASE =
    'relative inline-flex shrink-0 items-center gap-1.5 whitespace-nowrap border-b-2 px-3 py-2 text-control';
  const cls = (item: TabItem): string =>
    `${BASE} ${
      item.id === value
        ? 'border-accent text-fg-1'
        : 'border-transparent text-fg-2 hover:text-fg-1'
    } ${item.disabled ? 'pointer-events-none opacity-50' : ''}`;

  function onkeydown(e: KeyboardEvent): void {
    const enabled = items.filter((i) => !i.disabled);
    const at = enabled.findIndex((i) => i.id === value);
    let next = -1;
    if (e.key === 'ArrowRight') next = (at + 1) % enabled.length;
    else if (e.key === 'ArrowLeft') next = (at - 1 + enabled.length) % enabled.length;
    else if (e.key === 'Home') next = 0;
    else if (e.key === 'End') next = enabled.length - 1;
    if (next < 0) return;
    e.preventDefault();
    const host = e.currentTarget as HTMLElement;
    const id = enabled[next]!.id;
    onselect?.(id);
    queueMicrotask(() => host.querySelector<HTMLElement>(`[data-tab="${id}"]`)?.focus());
  }
</script>

{#snippet inner(item: TabItem)}
  {#if item.icon}<Icon name={item.icon} size={14} />{/if}
  {item.label}
  {#if item.count !== undefined}<Badge>{item.count}</Badge>{/if}
{/snippet}

<div class="mb-4 border-b border-line" data-testid="tabs">
  {#if asLinks}
    <ScrollStrip navLabel={label} activeKey={value} testId="tabs-strip">
      {#each items as item (item.id)}
        <a
          class={cls(item)}
          href={resolve(item.href as '/')}
          data-tab={item.id}
          aria-current={item.id === value ? 'page' : undefined}
          aria-disabled={item.disabled || undefined}
        >
          {@render inner(item)}
        </a>
      {/each}
    </ScrollStrip>
  {:else}
    <ScrollStrip activeKey={value} testId="tabs-strip">
      <div
        class="flex min-w-max"
        role="tablist"
        aria-label={label}
        {onkeydown}
        tabindex="-1"
      >
        {#each items as item (item.id)}
          <button
            type="button"
            role="tab"
            class={cls(item)}
            data-tab={item.id}
            data-active={item.id === value ? 'true' : undefined}
            aria-selected={item.id === value}
            tabindex={item.id === value ? 0 : -1}
            disabled={item.disabled}
            onclick={() => onselect?.(item.id)}
          >
            {@render inner(item)}
          </button>
        {/each}
      </div>
    </ScrollStrip>
  {/if}
</div>
